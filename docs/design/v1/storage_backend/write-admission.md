# Reuse-Based Write Admission (Doorkeeper)

Source: [`lmcache/v1/storage_backend/admission_policy.py`](../../../../lmcache/v1/storage_backend/admission_policy.py)

## The problem

LMCache offloads every KV chunk it produces to every configured backend. For a
RAM tier that is free. For a flash tier it is not: every offloaded chunk is a
NAND write, and flash endurance is a consumable budget measured in
**DWPD** (drive writes per day) that the drive is warrantied for.

The pathology is that **most chunks are never read back**. A chunk written for a
prefix that is never reused costs full write endurance and returns zero cache
hits. Measured across production-shaped traces, the fraction of chunks that are
ever re-read is often in the single digits:

| Trace | Blocks | Reads | Blocks with >=1 future read | Baseline DWPD (70B) |
|---|---:|---:|---:|---:|
| Qwen to-B API | 3,830,390 | 6,125,747 | **6.0%** | 16.11 |
| Qwen thinking | 1,735,015 | 1,488,975 | **14.3%** | 7.31 |
| Mooncake toolagent | 183,300 | 226,316 | **21.5%** | 50.23 |
| Mooncake conversation | 182,790 | 105,710 | **24.2%** | 50.09 |
| Qwen coder | 5,202,517 | 10,268,390 | **39.0%** | 21.89 |
| Qwen to-C chat | 2,530,337 | 3,763,639 | **46.8%** | 10.64 |
| WEKA agentic | 4,231,491 | 125,698,180 | **94.0%** | 1.16 |

Read the "blocks with >=1 future read" column as the write volume a
*clairvoyant* admission policy would produce while still serving 100% of the
reads. For the Qwen to-B API trace that is a **16x** reduction in NAND writes.
That headroom is the motivation for this module.

## The policy: a doorkeeper

"Will this chunk be read again?" is not knowable at write time. What *is*
observable is a **repeat store offer**. A store offer happens when a request
prefills a prefix LMCache did not have, so a repeat offer for the same chunk
means the prefix was reused after falling out of every tier -- with a reuse
distance longer than RAM residency, which is exactly the population the flash
tier exists to serve.

That gives the policy (the doorkeeper pattern, as in TinyLFU):

> Do not write a chunk on its first offer. Write it on offer ``K + 1``.

```
                     chunk C, K = 1, reuse distance > RAM residency

  request 1   miss -> prefill -> offer #1        RAM only, NOT written
    ...RAM evicts C...
  request 2   miss -> prefill -> offer #2        WRITTEN to disk
  request 3+  disk HIT                           no write, no recompute

  chunk D (never reused)

  request 1   miss -> prefill -> offer #1        RAM only, NOT written
  (never seen again -- zero flash bytes written)
```

Reuse that stays *within* RAM residency needs no flash at all: the RAM tier
serves it, and the vLLM connector masks already-cached tokens out of `store()`,
so no repeat offer is generated and no write happens. The flash write budget is
spent only on chunks that demonstrably outlive RAM.

## The cost model, honestly

Let `f(K)` be the fraction of chunks offered more than `K` times.

- **Writes** scale by `f(K)`; DWPD scales identically, being linear in bytes
  written.
- **Hits lost**: exactly **one recompute per admitted chunk** -- the prefill on
  offer `K+1` that would have been a disk hit without the filter. As a fraction
  of all reads that is `f(K) x blocks / reads`.

Applying that at `K = 1`:

| Trace | Write reduction | Reads lost | Verdict |
|---|---:|---:|---|
| Qwen to-B API | 16.6x | ~3.8% | Strongly worth it |
| Qwen thinking | 7.0x | ~16.6% | Worth it if TTFT headroom exists |
| Mooncake toolagent | 4.6x | ~17.4% | Situational |
| WEKA agentic | 1.06x | ~3.2% | **Not worth it** -- leave disabled |

Unlike a clairvoyant policy, the doorkeeper *does* lose hits -- the table's
"100% of reads retained" row is an upper bound, not what this ships. The
decision rule is workload-specific:

> **Enable this when `f(1)` is small.** If most chunks are reused anyway (WEKA
> agentic, at 94%), the filter suppresses almost no writes while still costing
> one recompute per chunk. There is no universally good `K`; that is why the
> default is *off*.

Raising `K` above 1 is almost always a bad trade: each admitted chunk then
costs `K` recomputes, and the trace data shows reads retained collapsing far
faster than writes shrink.

### Why this matters beyond endurance

- **Drive class and price.** A sustained 16 DWPD requirement forces
  write-intensive enterprise NVMe plus heavy over-provisioning. Under 1 DWPD
  the same workload runs on read-intensive TLC/QLC at a fraction of the cost
  per TB.
- **Read latency under load.** Writes and reads contend for the same NAND
  channels, and garbage collection scales with write volume. Cutting writes
  16x removes that interference from the retrieval path, which is what shows
  up as TTFT.
- **Effective cache capacity.** The disk tier holds only chunks with
  demonstrated reuse, so the same capacity delivers a higher hit rate.

## Which tiers to guard

| Backend | Guard it? | Why |
|---|---|---|
| `LocalDiskBackend`, `GdsBackend` | **Yes - default** | Both are unconditionally file/flash-backed (`GdsBackend` shares `LocalDiskBackend`'s `PathSharder`), so both carry the same NAND endurance cost. |
| `NixlStorageBackend` | **Only if the configured Nixl plugin is flash-backed** | Nixl can run over UCX (RDMA to remote memory) or over GDS/POSIX (real files). The class doesn't tell you which, so it's opt-in, not default. |
| `RemoteBackend` | **Only if single-tenant** | See below. |
| `LocalCPUBackend`, `MaruBackend` | **Never** - rejected | Both run at RAM/CXL-memory speed with no endurance cost. `MaruBackend` stores directly in CXL mmap memory rather than flash, so unlike `NixlStorageBackend` its cost profile isn't configuration-dependent. |
| `PDBackend`, `P2PBackend` | **Never** - rejected | These are *transfer mechanisms*, not caches. A peer is blocked waiting for the KV, so a suppressed write is lost data and a broken request, not a cache miss. |

The last two rows are enforced: naming them in `admission_backends` raises a
`ValueError` at startup rather than silently corrupting a disagg-prefill
deployment.

### On the remote tier

Guarding `RemoteBackend` also saves network bandwidth and, on object stores,
per-PUT request charges. But the remote tier is usually **shared across
replicas**, and the filter is per-process: a chunk that looks single-use
locally may be the hot shared prefix another replica is about to request.
Suppressing that write removes the cross-instance sharing that justifies having
a remote tier. Guard it only in single-instance deployments; for shared stores,
admission control belongs at the store, where aggregate reuse is observable.

## Implementation

The filter is a fixed-size, direct-mapped table of recently offered chunks.
Each slot holds a 32-bit fingerprint plus a saturating 8-bit offer counter --
5 bytes per slot, 5 MiB at the default 2^20 slots, with no eviction
bookkeeping.

```
digest = splitmix64(hash(key))
slot   = digest & (capacity - 1)
tag    = (digest >> 32) & 0xFFFFFFFF

if counts[slot] == 0 or tags[slot] != tag:   # new chunk, or slot stolen
    tags[slot], counts[slot] = tag, 1
else:
    counts[slot] = min(counts[slot] + 1, 255)

admit = counts[slot] > min_reuse
```

Three properties worth calling out:

- **Approximate by construction.** A colliding chunk takes over the slot and
  the victim must re-earn admission. That is a bounded, self-correcting error:
  the worst case is one redundant prefill, never a correctness problem.
  Trading exactness for `O(1)` memory is the right call here.
- **The digest is mixed before it is split.** `CacheEngineKey.__hash__`
  delegates to Python's tuple hash, which does not guarantee entropy in the
  high bits; for small integer chunk hashes they are zero. Since the slot and
  the fingerprint come from different halves, using the raw hash would
  silently disable collision detection. The splitmix64 finalizer removes that
  coupling.
- **One decision per store, not per backend.** `StorageManager.batched_put`
  calls `admit()` once and applies the verdict to every guarded backend, so a
  chunk's counter advances once per offer regardless of how many backends are
  configured.

### Where it hooks in

Exactly one call site: `StorageManager.batched_put` filters the
`(keys, memory_objs)` pair for guarded backends. The storage backend interface
is untouched -- no backend knows the filter exists -- and the retrieve path is
untouched, so the filter adds zero work to TTFT. Unguarded backends see exactly
the traffic they saw before, and reference counting is unaffected: the manager
still owns and releases every allocated object.

### Considered and rejected: promotion on read hit

An earlier revision also wrote a chunk to flash when it was served from RAM
("promotion"), reasoning that a RAM hit proves reuse without costing a
recompute. It was removed:

- It targets the wrong population. A RAM hit proves reuse *within* RAM
  residency -- reuse the RAM tier is already serving. The flash tier exists for
  reuse distances *beyond* RAM residency, and for those the promotion signal
  never fires; the store-offer path handles them anyway.
- It put disk work (allocator matching, put submission, and the disk backend's
  inline eviction loop) on the retrieve path, which is TTFT.
- It required a promote-once marker, an allocator-compatibility check, and had
  a known gap on the async-loading path. Removing it deleted all three.

The doorkeeper alone is strictly simpler and spends the flash budget on the
population flash is for, at a bounded, predictable cost of one recompute per
admitted chunk.

### Considered and rejected: demote on RAM eviction

Writing a chunk to flash at the moment RAM evicts it (if it saw hits while
resident) writes exactly the used-and-about-to-be-lost chunks. It was rejected
because RAM eviction runs inside the allocator's memory-pressure loop;
deferring the free until a disk write completes risks deadlocking allocation
precisely when memory is scarce.

### Related: duplicate-write elimination

Independently of the filter, `LocalDiskBackend.submit_put_task` now skips
chunks it already holds. Chunks are content-addressed by key, so rewriting one
produces byte-identical data and buys nothing but wear. This applies
unconditionally and needs no configuration.

## Configuration

All settings live under `extra_config`. The filter is disabled by default.

```yaml
local_disk: "/local/disk/"
max_local_disk_size: 500

extra_config:
  # Store offers beyond the first required before a chunk is written. 0
  # (default) disables the filter; 1 means "write on the second offer".
  admission_min_reuse: 1

  # Backend class names to guard. Transfer backends and the RAM tier are
  # rejected at startup.
  admission_backends: ["LocalDiskBackend"]

  # History slots, rounded up to a power of two, 5 bytes each.
  admission_history_size: 1048576
```

Size `admission_history_size` to cover the reuse distance you care about: a
chunk must still be in the table when it is offered again. As a rule of thumb,
set it to a few times the number of distinct chunks offered within your reuse
window; the default 2^20 slots covers ~1M chunks for 5 MiB.

## Measuring the effect

The filter logs a cumulative summary every 100,000 store offers:

```
Reuse admission filter: wrote 12043 of 200000 offered chunks (6.0%),
avoiding 187957 writes to ['LocalDiskBackend'].
```

That percentage is directly comparable to the "blocks with >=1 future read"
column above, and it is the multiplier to apply to your measured baseline DWPD.
`StorageManager.admission_policy.stats()` exposes the same counters
programmatically as an `AdmissionStats`. `written` counts submissions; guarded
backends dedupe chunks they already hold, so real write volume can be slightly
lower.

To validate end to end, run the same workload twice -- once with
`admission_min_reuse: 0` and once with `1` -- and compare host write bytes
(`/sys/block/<dev>/stat` field 7, or `nvme smart-log` data units written)
against the cache hit rate and TTFT reported by LMCache. Ship the filter
enabled only if the write reduction is large and the hit-rate cost is small on
*your* trace.
