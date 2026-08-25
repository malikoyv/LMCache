# SPDX-License-Identifier: Apache-2.0
"""Reuse-based write admission for wear-sensitive storage backends.

LMCache offloads every KV chunk it produces to every configured backend. For
RAM-backed tiers that is free, but for flash-backed tiers every offload costs
NAND endurance. In most serving traces the large majority of chunks are never
read back, so those writes buy nothing and only consume the drive's write
budget (DWPD).

This module implements a *doorkeeper* admission filter: a chunk is not written
to a guarded backend the first time it is offered for storage, only once it
has been offered ``admission_min_reuse`` further times. A store offer happens
when a request prefills a prefix LMCache did not have, so a repeat offer is
direct evidence that the prefix is reused with a reuse distance longer than
RAM residency -- exactly the population the flash tier exists to serve.

The cost model is deliberately simple: each admitted chunk costs exactly one
extra prefill (the recompute that proves its reuse), and single-use chunks --
typically the bulk of the trace -- are never written at all. Reuse that stays
within RAM residency is served by the RAM tier as before and neither needs nor
triggers a flash write.

See ``docs/design/v1/storage_backend/write-admission.md`` for the underlying
model, the measured trade-off curve, and guidance on choosing ``K``.
"""

# Standard
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)

# Backends guarded by default when the filter is enabled. Both are
# unconditionally file/flash-backed (GdsBackend shares LocalDiskBackend's
# PathSharder), so both carry the same NAND endurance cost. Memory-speed
# tiers are deliberately excluded: they have no endurance cost, and demoting
# them would hurt hit rate for no gain.
_DEFAULT_GUARDED_BACKENDS = ("LocalDiskBackend", "GdsBackend")

# Writes to these backends are transfer mechanisms rather than cache fills: a
# peer is already waiting for the data, so suppressing a write loses it.
_TRANSFER_BACKENDS = frozenset({"PDBackend", "P2PBackend"})

# These tiers live at RAM or CXL-memory speed and have no endurance budget to
# protect. MaruBackend stores directly in CXL mmap memory (no flash involved),
# so unlike NixlStorageBackend its wear cost is never configuration-dependent.
_STAGING_BACKENDS = frozenset({"LocalCPUBackend", "MaruBackend"})

_NEVER_GUARDED_BACKENDS = _TRANSFER_BACKENDS | _STAGING_BACKENDS

# Number of history slots. Each slot costs 5 bytes (4-byte tag + 1-byte
# counter), so the default table is 5 MiB.
_DEFAULT_HISTORY_SIZE = 1 << 20

_HASH_MASK = (1 << 64) - 1
_TAG_MASK = (1 << 32) - 1

# Counter saturation value.
_MAX_COUNT = 0xFF

# How often to emit a cumulative effectiveness summary, in offered chunks.
_LOG_EVERY_N_OFFERS = 100_000


def _mix64(value: int) -> int:
    """Scramble a 64-bit value so its low and high halves are independent.

    The slot index and the fingerprint are taken from different halves of the
    digest, so they must be decorrelated or collision detection silently stops
    working. ``CacheEngineKey.__hash__`` delegates to Python's tuple hash,
    which makes no such guarantee -- for small integer chunk hashes the high
    32 bits are simply zero. This is the splitmix64 finalizer.

    :param value: An arbitrary 64-bit unsigned integer.
    :returns: A 64-bit unsigned integer with entropy spread across all bits.
    """
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _HASH_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _HASH_MASK
    return value ^ (value >> 31)


def _require_int(config: LMCacheEngineConfig, key: str, default: int) -> int:
    """Read an integer setting from ``extra_config``.

    :param config: The engine configuration to read from.
    :param key: The ``extra_config`` key to read.
    :param default: Value to use when the key is absent.
    :returns: The configured integer.
    :raises ValueError: If the value is present but is not an integer.
        ``bool`` is rejected despite subclassing ``int``.
    """
    value = config.get_extra_config_value(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    return value


@dataclass(frozen=True)
class AdmissionStats:
    """Cumulative counters describing how selective the filter has been.

    :param offered: Number of chunks offered for storage. This is the write
        count a filterless engine would have produced, i.e. the baseline.
    :param written: Number of chunks admitted to guarded backends. Guarded
        backends dedupe chunks they already hold, so the real write count can
        be lower.
    """

    offered: int
    written: int

    @property
    def suppressed(self) -> int:
        """Number of chunk writes the filter avoided."""
        return self.offered - self.written

    @property
    def write_fraction(self) -> float:
        """Fraction of the baseline write volume that was actually written.

        This is the multiplier that applies to the guarded backend's write
        volume, and therefore to its DWPD. A value of ``1.0`` means the
        filter saved nothing; ``0.06`` means writes dropped ~16x.
        """
        if self.offered == 0:
            return 1.0
        return self.written / self.offered


class ReuseAdmissionPolicy:
    """Admit a chunk to guarded backends only after repeated store offers.

    The filter keeps a fixed-size, direct-mapped history of recently offered
    chunks. Each slot holds a 32-bit fingerprint of the chunk identity plus a
    saturating 8-bit offer counter. A chunk is admitted once its counter
    exceeds ``min_reuse``. Because the table is direct-mapped, a colliding
    chunk simply takes over the slot; the victim loses its history and must
    re-earn admission. This bounds memory to ``history_size * 5`` bytes with no
    eviction bookkeeping, at the price of an approximate (never fatal) answer.

    The policy is disabled - ``guarded_backends`` is empty and every key is
    admitted - when ``min_reuse`` is zero.

    Thread safety: all public methods are safe to call concurrently.
    """

    def __init__(
        self,
        min_reuse: int,
        history_size: int = _DEFAULT_HISTORY_SIZE,
        guarded_backends: Sequence[str] = _DEFAULT_GUARDED_BACKENDS,
    ) -> None:
        """Build a reuse admission filter.

        :param min_reuse: Number of offers a chunk must accumulate beyond the
            first before it is written. ``0`` disables the filter entirely;
            ``1`` means "write on the second offer".
        :param history_size: Requested number of history slots. Rounded up to
            the next power of two.
        :param guarded_backends: Class names of the backends this filter
            applies to, e.g. ``["LocalDiskBackend"]``.
        :raises ValueError: If ``min_reuse`` is negative or exceeds the
            counter's saturation point, if ``history_size`` is not positive,
            or if a backend is named that must never be guarded.
        """
        if min_reuse < 0:
            raise ValueError(f"admission_min_reuse must be >= 0, got {min_reuse}")
        if min_reuse >= _MAX_COUNT:
            raise ValueError(
                f"admission_min_reuse must be < {_MAX_COUNT}, got {min_reuse}"
            )
        if history_size <= 0:
            raise ValueError(f"admission_history_size must be > 0, got {history_size}")

        forbidden = _NEVER_GUARDED_BACKENDS.intersection(guarded_backends)
        if forbidden:
            raise ValueError(
                f"Backends {sorted(forbidden)} must not be guarded by the "
                f"admission filter. {sorted(_TRANSFER_BACKENDS)} transfer KV to a "
                f"waiting peer, so a suppressed write is lost data; "
                f"{sorted(_STAGING_BACKENDS)} run at RAM/CXL-memory speed and "
                f"have no endurance cost to protect."
            )

        self._min_reuse = min_reuse
        self._guarded_backends: frozenset[str] = (
            frozenset(guarded_backends) if min_reuse > 0 else frozenset()
        )

        self._capacity = 1 << max(1, (history_size - 1).bit_length())
        self._slot_mask = self._capacity - 1
        # "I" is at least 4 bytes and "B" is exactly 1 byte; both are
        # zero-initialised from a zero-filled bytes object.
        self._tags = array("I", bytes(self._capacity * array("I").itemsize))
        self._counts = array("B", bytes(self._capacity))

        self._lock = threading.Lock()
        self._offered = 0
        self._written = 0
        self._last_logged_offers = 0

        if self._guarded_backends:
            logger.info(
                "Reuse admission filter enabled: min_reuse=%d, history_slots=%d, "
                "guarded_backends=%s. Chunks are written only after %d further "
                "store offer(s); expect one extra prefill per admitted chunk.",
                min_reuse,
                self._capacity,
                sorted(self._guarded_backends),
                min_reuse,
            )

    @property
    def guarded_backends(self) -> frozenset[str]:
        """Backend class names this filter gates. Empty when disabled."""
        return self._guarded_backends

    def admit(self, keys: Sequence[CacheEngineKey]) -> set[CacheEngineKey]:
        """Record a store offer per key and return the writable subset.

        Call this exactly once per store operation, no matter how many guarded
        backends consume the result, so that a chunk's offer counter advances
        once per offer rather than once per backend.

        :param keys: The cache keys being offered for storage.
        :returns: The subset of ``keys`` that may be written to guarded
            backends. All keys are returned when the filter is disabled.
        """
        if not self._guarded_backends:
            return set(keys)

        admitted: set[CacheEngineKey] = set()
        with self._lock:
            for key in keys:
                digest = _mix64(hash(key) & _HASH_MASK)
                slot = digest & self._slot_mask
                tag = (digest >> 32) & _TAG_MASK

                count = self._counts[slot]
                if count == 0 or self._tags[slot] != tag:
                    # First sighting, or the slot belonged to another chunk.
                    self._tags[slot] = tag
                    count = 1
                    self._counts[slot] = count
                elif count < _MAX_COUNT:
                    count += 1
                    self._counts[slot] = count

                if count > self._min_reuse:
                    admitted.add(key)

            self._offered += len(keys)
            self._written += len(admitted)
            snapshot = AdmissionStats(self._offered, self._written)
            should_log = (
                snapshot.offered - self._last_logged_offers >= _LOG_EVERY_N_OFFERS
            )
            if should_log:
                self._last_logged_offers = snapshot.offered

        if should_log:
            logger.info(
                "Reuse admission filter: wrote %d of %d offered chunks (%.1f%%), "
                "avoiding %d writes to %s.",
                snapshot.written,
                snapshot.offered,
                100.0 * snapshot.write_fraction,
                snapshot.suppressed,
                sorted(self._guarded_backends),
            )

        return admitted

    def stats(self) -> AdmissionStats:
        """Return a consistent snapshot of the cumulative counters."""
        with self._lock:
            return AdmissionStats(self._offered, self._written)


def create_admission_policy(config: LMCacheEngineConfig) -> ReuseAdmissionPolicy:
    """Build the admission policy described by ``config.extra_config``.

    Recognised ``extra_config`` keys:

    - ``admission_min_reuse`` (int, default ``0``): store offers required
      beyond the first before a chunk is written. ``0`` disables the filter.
    - ``admission_backends`` (list of str, default
      ``["LocalDiskBackend", "GdsBackend"]``): backend class names to guard.
      Transfer backends and memory-speed tiers are rejected.
    - ``admission_history_size`` (int, default ``1048576``): history slots,
      rounded up to a power of two, costing 5 bytes each.

    :param config: The engine configuration to read the settings from.
    :returns: A policy that is disabled unless ``admission_min_reuse`` is set.
    :raises ValueError: If any of the settings has an invalid value or type.
    """
    min_reuse = _require_int(config, "admission_min_reuse", 0)
    history_size = _require_int(config, "admission_history_size", _DEFAULT_HISTORY_SIZE)

    backends = config.get_extra_config_value(
        "admission_backends", list(_DEFAULT_GUARDED_BACKENDS)
    )
    # A bare string would otherwise decompose into a list of single characters
    # and silently guard nothing.
    if not isinstance(backends, (list, tuple)) or not all(
        isinstance(name, str) for name in backends
    ):
        raise ValueError(
            f"admission_backends must be a list of backend class names, "
            f"got {backends!r}"
        )

    return ReuseAdmissionPolicy(
        min_reuse=min_reuse,
        history_size=history_size,
        guarded_backends=list(backends),
    )
