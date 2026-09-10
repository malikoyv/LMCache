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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional
import enum
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

# Request-scoped key a client sets to declare, in seconds, how long its KV is
# worth keeping. Must not start with "lmcache.tag.": that prefix is folded into
# CacheEngineKey.tags and would fork the cache namespace per class, destroying
# the cross-request prefix sharing this feature depends on.
_REQUEST_TTL_KEY = "lmcache.kv_ttl_s"


class VetoMode(enum.Enum):
    """How the ephemeral write veto acts on a short-lived store.

    :cvar OFF: The veto is inert and records nothing.
    :cvar OBSERVE: Declarations are counted but every store is written. This is
        what makes the feature measurable before it is trusted: it answers
        "what share of flash writes would this remove?" without changing a
        single byte of behaviour, so the decision to enforce is made on the
        fleet's own numbers rather than on a corpus.
    :cvar ENFORCE: Short-lived stores skip the guarded backends.
    """

    OFF = "off"
    OBSERVE = "observe"
    ENFORCE = "enforce"


def _declared_ttl_seconds(
    request_configs: Optional[Mapping[str, Any]],
) -> Optional[int]:
    """Read the caller-declared KV lifetime from a request's config dict.

    Values arrive as integers in-process but as strings after a round trip
    through the lookup-client wire format, so both are accepted.

    :param request_configs: The request's ``lmcache.*`` config dict, or None.
    :returns: The declared lifetime in seconds, or None when the request made
        no declaration or the declaration was unusable.
    """
    if not request_configs:
        return None
    raw = request_configs.get(_REQUEST_TTL_KEY)
    if raw is None:
        return None

    ttl: Optional[int] = None
    if isinstance(raw, int) and not isinstance(raw, bool):
        ttl = raw
    elif isinstance(raw, str):
        try:
            ttl = int(raw)
        except ValueError:
            ttl = None

    if ttl is None or ttl < 0:
        logger.debug("Ignoring unusable %s=%r; storing durably.", _REQUEST_TTL_KEY, raw)
        return None
    return ttl


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


def _read_guarded_backends(config: LMCacheEngineConfig) -> list[str]:
    """Read the shared ``admission_backends`` list from ``extra_config``.

    Which tiers are wear-sensitive is one fact about a deployment, so the
    reuse filter and the ephemeral veto read the same setting rather than
    each carrying its own copy.

    :param config: The engine configuration to read the setting from.
    :returns: Backend class names to guard.
    :raises ValueError: If the value is not a list of strings. A bare string
        is rejected explicitly: it would otherwise decompose into a list of
        single characters and silently guard nothing.
    """
    backends = config.get_extra_config_value(
        "admission_backends", list(_DEFAULT_GUARDED_BACKENDS)
    )
    if not isinstance(backends, (list, tuple)) or not all(
        isinstance(name, str) for name in backends
    ):
        raise ValueError(
            f"admission_backends must be a list of backend class names, "
            f"got {backends!r}"
        )
    return list(backends)


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
    return ReuseAdmissionPolicy(
        min_reuse=_require_int(config, "admission_min_reuse", 0),
        history_size=_require_int(
            config, "admission_history_size", _DEFAULT_HISTORY_SIZE
        ),
        guarded_backends=_read_guarded_backends(config),
    )


@dataclass(frozen=True)
class EphemeralVetoStats:
    """Cumulative counters describing how much traffic declared itself short-lived.

    Counted in *puts*, not logical stores: a layerwise store issues one put per
    layer, and each is a separate write to a guarded backend, so this is the
    unit that maps to NAND bytes.

    :param mode: The mode these counters were collected under. In
        :attr:`VetoMode.OBSERVE` the matched counters are what *would* have
        been skipped; in :attr:`VetoMode.ENFORCE` they are what was skipped.
    :param puts: Put operations evaluated by the veto.
    :param chunks: Chunks contained in those puts.
    :param matched_puts: Puts whose declared lifetime was below the threshold.
    :param matched_chunks: Chunks in those puts, i.e. the guarded-backend
        writes avoided (or avoidable, when observing).
    :param malformed: Puts carrying a declaration that could not be parsed.
        These were stored durably; a non-zero value means a client is tagging
        incorrectly and its savings are silently not happening.
    """

    mode: VetoMode
    puts: int
    chunks: int
    matched_puts: int
    matched_chunks: int
    malformed: int

    @property
    def matched_chunk_fraction(self) -> float:
        """Fraction of offered chunks that skipped, or would skip, guarded backends."""
        if self.chunks == 0:
            return 0.0
        return self.matched_chunks / self.chunks


class EphemeralWriteVeto:
    """Skip wear-sensitive backends for requests that declare short-lived KV.

    A reuse filter infers a chunk's worth from the chunk's own past, so it
    cannot separate KV that is read for weeks from KV that is read forty times
    in ten minutes and never again -- both look equally hot, because the
    difference is *when*, not *how often*. The producer knows in advance: an
    agent scaffold's subagent context dies with the subagent. This veto is the
    channel for saying so.

    A request declares a lifetime with ``lmcache.kv_ttl_s``; a store whose
    declared lifetime is below ``min_ttl_s`` skips the guarded backends and is
    served from the tiers above them. Set ``min_ttl_s`` to the RAM tier's
    residency: below that, a flash write can never be read back *from flash*,
    because the chunk's entire horizon fits in RAM.

    This is deliberately one backend finer than the existing
    ``lmcache.skip_save``, which suppresses the save to every tier including
    ``LocalCPUBackend``. Ephemeral chunks are read many times and must stay in
    DRAM; only the flash write is worthless.

    The veto is per *store*, never per chunk. Two requests of different classes
    can produce the same chunk -- a shared system prompt is produced by main
    threads and subagents alike -- so a chunk one request declined to persist
    is written normally when the next request computes it. That is self-healing
    and makes it impossible for one tenant's tag to deny another tenant's data.

    Undeclared, malformed and negative values fail open: the store is durable.
    A scaffold that forgets to tag pays today's bill, never a worse one.

    Start in :attr:`VetoMode.OBSERVE` to measure what the fleet would save
    before changing behaviour, and only then enforce.

    Thread safety: all public methods are safe to call concurrently.
    """

    def __init__(
        self,
        mode: VetoMode,
        min_ttl_s: int,
        guarded_backends: Sequence[str] = _DEFAULT_GUARDED_BACKENDS,
    ) -> None:
        """Build an ephemeral write veto.

        :param mode: Whether to be inert, count only, or skip writes.
        :param min_ttl_s: Declared lifetime, in seconds, below which a put
            skips the guarded backends.
        :param guarded_backends: Class names of the backends the veto applies
            to, e.g. ``["LocalDiskBackend"]``.
        :raises ValueError: If ``min_ttl_s`` is not positive while the veto is
            active, if ``min_ttl_s`` is negative, or if a backend is named that
            must never be guarded.
        """
        if min_ttl_s < 0:
            raise ValueError(f"admission_flash_min_ttl_s must be >= 0, got {min_ttl_s}")
        if mode is not VetoMode.OFF and min_ttl_s == 0:
            raise ValueError(
                f"admission_flash_min_ttl_s must be > 0 when "
                f"admission_flash_veto_mode is '{mode.value}'; a zero threshold "
                f"can never match a declaration."
            )

        forbidden = _NEVER_GUARDED_BACKENDS.intersection(guarded_backends)
        if forbidden:
            raise ValueError(
                f"Backends {sorted(forbidden)} must not be vetoed. "
                f"{sorted(_TRANSFER_BACKENDS)} transfer KV to a waiting peer, so a "
                f"suppressed write is lost data; {sorted(_STAGING_BACKENDS)} is "
                f"where ephemeral chunks are meant to be served from."
            )

        self._mode = mode
        self._min_ttl_s = min_ttl_s
        self._guarded_backends: frozenset[str] = (
            frozenset(guarded_backends) if mode is VetoMode.ENFORCE else frozenset()
        )

        self._lock = threading.Lock()
        self._puts = 0
        self._chunks = 0
        self._matched_puts = 0
        self._matched_chunks = 0
        self._malformed = 0

        if mode is VetoMode.ENFORCE:
            logger.info(
                "Ephemeral write veto enforcing: requests declaring %s below %d "
                "seconds skip %s and are served from the tiers above.",
                _REQUEST_TTL_KEY,
                min_ttl_s,
                sorted(frozenset(guarded_backends)),
            )
        elif mode is VetoMode.OBSERVE:
            logger.info(
                "Ephemeral write veto observing only: counting requests that "
                "declare %s below %d seconds; no writes are suppressed.",
                _REQUEST_TTL_KEY,
                min_ttl_s,
            )

    @property
    def mode(self) -> VetoMode:
        """The configured mode."""
        return self._mode

    @property
    def guarded_backends(self) -> frozenset[str]:
        """Backend class names a matched put skips. Empty unless enforcing."""
        return self._guarded_backends

    def evaluate_put(
        self,
        request_configs: Optional[Mapping[str, Any]],
        num_chunks: int,
    ) -> bool:
        """Record one put offer and report whether it must skip guarded backends.

        Call this exactly once per put, before the backend fan-out, so the
        counters track puts rather than backends. Recording happens in
        :attr:`VetoMode.OBSERVE` too; that is the point of the mode.

        :param request_configs: The originating request's ``lmcache.*`` config
            dict, or None when the put has no request context.
        :param num_chunks: Number of chunks in this put, used for the
            avoided-write counters.
        :returns: True only when enforcing and the declared lifetime is below
            the threshold.
        """
        if self._mode is VetoMode.OFF:
            return False

        ttl = _declared_ttl_seconds(request_configs)
        declared = request_configs is not None and _REQUEST_TTL_KEY in request_configs
        matched = ttl is not None and ttl < self._min_ttl_s

        with self._lock:
            self._puts += 1
            self._chunks += num_chunks
            if declared and ttl is None:
                self._malformed += 1
            if matched:
                self._matched_puts += 1
                self._matched_chunks += num_chunks

        return matched and self._mode is VetoMode.ENFORCE

    def stats(self) -> EphemeralVetoStats:
        """Return a consistent snapshot of the cumulative counters."""
        with self._lock:
            return EphemeralVetoStats(
                mode=self._mode,
                puts=self._puts,
                chunks=self._chunks,
                matched_puts=self._matched_puts,
                matched_chunks=self._matched_chunks,
                malformed=self._malformed,
            )


def create_ephemeral_veto(config: LMCacheEngineConfig) -> EphemeralWriteVeto:
    """Build the ephemeral write veto described by ``config.extra_config``.

    Recognised ``extra_config`` keys:

    - ``admission_flash_veto_mode`` (str, default ``"off"``): one of ``off``,
      ``observe`` or ``enforce``. See :class:`VetoMode`.
    - ``admission_flash_min_ttl_s`` (int, default ``0``): declared lifetime in
      seconds below which a put is matched. Required when the mode is not
      ``off``.
    - ``admission_backends``: shared with :func:`create_admission_policy`.

    The veto and the reuse filter are independent: either can run without the
    other.

    :param config: The engine configuration to read the settings from.
    :returns: A veto that is inert unless ``admission_flash_veto_mode`` is set.
    :raises ValueError: If any of the settings has an invalid value or type.
    """
    raw_mode = config.get_extra_config_value(
        "admission_flash_veto_mode", VetoMode.OFF.value
    )
    try:
        mode = VetoMode(raw_mode)
    except ValueError:
        raise ValueError(
            f"admission_flash_veto_mode must be one of "
            f"{[m.value for m in VetoMode]}, got {raw_mode!r}"
        ) from None

    return EphemeralWriteVeto(
        mode=mode,
        min_ttl_s=_require_int(config, "admission_flash_min_ttl_s", 0),
        guarded_backends=_read_guarded_backends(config),
    )
