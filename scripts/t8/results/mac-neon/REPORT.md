# T8 leaves on Plonky3: Apple silicon, NEON and SHA-2

Each hash's standard Merkle leaf against its T8 leaf, on the same compression kernel, inside an unchanged Plonky3 tree.

## How to reproduce

From the repository root, on the `t8-leaves` branch:

```sh
# Everything below, then this report, in about 8 minutes.
scripts/t8/run.sh
```

Or one table at a time, reading the numbers criterion prints:

```sh
export RUSTFLAGS=-Ctarget-cpu=native
cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_leaf
cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_verify
RAYON_NUM_THREADS=1 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_commit
RAYON_NUM_THREADS=10 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_commit
```

- Every command in this report assumes `RUSTFLAGS=-Ctarget-cpu=native`, as exported above.
- A trailing regex selects benchmarks, for example `-- 'leaf/sha256'`.
- macOS has no core pinning, so single-threaded runs are left to the scheduler, which keeps a busy thread on a performance core.
- `--profile optimized` is Plonky3's own profile: thin LTO and one codegen unit.
- Criterion prints `time: [low median high]`, the 95% interval of the median.
- It also keeps each estimate in `target/criterion/<group>/<scheme>/<size>/new/estimates.json`.
- The script saves them as the baseline `native`, copies them to `scripts/t8/results/mac-neon/`, and builds this report from them.

## Reading the tables

- **Standard, T8**: criterion's median time, with the half-width of its 95% interval.
- **GB/s**: record bytes hashed per second, 1 GB = 10^9 bytes.
- **Speedup**: standard time over T8 time, so above 1 means T8 is faster.
- **Call ratio**: standard calls over T8 calls, what counting compressions alone predicts.

## Setup

**native build**

```
date: 2026-09-29T17:07:20+02:00
commit: 997732a5360817c23c0e3ed01f5e000f6bf9790b
cpu: Apple M1 Pro
kernel: 25.6.0
rustc: rustc 1.98.0-nightly (9e2abe0c6 2026-06-16)
RUSTFLAGS: -Ctarget-cpu=native
governor: n/a (macOS)
boost: n/a (macOS)
```

## Compression calls per record

Counted, not timed. A tree of N records adds the same N - 1 node calls to both leaves.

| Hash | Record | Standard calls | T8 calls | T8 against standard |
|---|---:|---:|---:|---:|
| BLAKE3 | 256 B | 4 | 3 | 25.0% fewer |
| BLAKE3 | 480 B | 8 | 6 | 25.0% fewer |
| BLAKE3 | 65,664 B | 1,090 | 879 | 19.4% fewer |
| BLAKE2s | 256 B | 4 | 3 | 25.0% fewer |
| BLAKE2s | 480 B | 8 | 6 | 25.0% fewer |
| BLAKE2s | 65,664 B | 1,026 | 879 | 14.3% fewer |
| SHA-256 | 256 B | 5 | 3 | 40.0% fewer |
| SHA-256 | 480 B | 8 | 6 | 25.0% fewer |
| SHA-256 | 65,664 B | 1,027 | 879 | 14.4% fewer |
| Keccak-256 | 256 B | 2 | 3 | 50.0% more |
| Keccak-256 | 480 B | 4 | 6 | 50.0% more |
| Keccak-256 | 65,664 B | 483 | 879 | 82.0% more |

Keccak-256 is slower with T8, and SHA3-256 would be the same: both run Keccak-f with a 136-byte rate.

- One permutation already takes in 136 fresh bytes, so a plain hash covers 256 B in 2 calls.
- Each T8 call is a 96-byte function, so it uses only 96 of those 136 bytes and needs 3 calls.
- T8 pays off only when one call takes about 96 bytes, as the BLAKE3, BLAKE2s and SHA-256 compressions do.

## Measured

### Leaf hashing, batched, one core

Command: `cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_leaf -- 'leaf/.*/batch'`

- Code: `merkle-tree/benches/t8_leaf.rs`, one `hash_many` call over the whole batch, on one thread.
- Batches: 2^16 records of 256 B, 2^15 of 480 B, 2^8 of 65,664 B, so 16 MiB each.
- Input: deterministic xorshift bytes, generated before timing.
- Criterion: 1 s warm-up, 3 s measurement, 100 samples.

| Hash | Record | Records | Standard | T8 | Standard GB/s | T8 GB/s | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 2^16 | 7.18 ms ±0.1% | 5.94 ms ±0.2% | 2.34 | 2.82 | **1.208x** | 1.333x |
| BLAKE3 | 480 B | 2^15 | 7.14 ms ±0.1% | 5.74 ms ±0.2% | 2.20 | 2.74 | **1.244x** | 1.333x |
| BLAKE3 | 65,664 B | 2^8 | 7.42 ms ±0.1% | 6.35 ms ±0.1% | 2.27 | 2.65 | **1.169x** | 1.240x |
| BLAKE2s | 256 B | 2^16 | 11.4 ms ±0.1% | 9.41 ms ±0.2% | 1.47 | 1.78 | **1.214x** | 1.333x |
| BLAKE2s | 480 B | 2^15 | 11.3 ms ±0.1% | 9.21 ms ±0.2% | 1.39 | 1.71 | **1.224x** | 1.333x |
| BLAKE2s | 65,664 B | 2^8 | 11.1 ms ±0.1% | 10.3 ms ±0.2% | 1.51 | 1.64 | **1.087x** | 1.167x |
| SHA-256 | 256 B | 2^16 | 8.63 ms ±0.1% | 5.19 ms ±0.0% | 1.94 | 3.23 | **1.662x** | 1.667x |
| SHA-256 | 480 B | 2^15 | 6.95 ms ±0.2% | 5.19 ms ±0.0% | 2.26 | 3.03 | **1.339x** | 1.333x |
| SHA-256 | 65,664 B | 2^8 | 6.93 ms ±0.1% | 5.93 ms ±0.1% | 2.43 | 2.84 | **1.169x** | 1.168x |

### Leaf hashing, one record, one core

Command: `cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_leaf -- 'leaf/.*/single'`

- Code: `merkle-tree/benches/t8_leaf.rs`, one `hash_slice` call on one record, the path a verifier takes.
- Records: 256 B, 480 B and 65,664 B.
- Criterion: 0.5 s warm-up, 2 s measurement, 100 samples.

| Hash | Record | Standard | T8 | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 387 ns ±0.1% | 284 ns ±0.0% | **1.363x** | 1.333x |
| BLAKE3 | 480 B | 768 ns ±0.1% | 582 ns ±0.1% | **1.321x** | 1.333x |
| BLAKE3 | 65,664 B | 46.8 µs ±0.0% | 87.3 µs ±0.1% | **0.536x** | 1.240x |
| BLAKE2s | 256 B | 528 ns ±0.1% | 414 ns ±0.1% | **1.275x** | 1.333x |
| BLAKE2s | 480 B | 1.05 µs ±0.0% | 840 ns ±0.1% | **1.253x** | 1.333x |
| BLAKE2s | 65,664 B | 134 µs ±0.0% | 125 µs ±0.0% | **1.074x** | 1.167x |
| SHA-256 | 256 B | 179 ns ±0.1% | 86.3 ns ±0.1% | **2.078x** | 1.667x |
| SHA-256 | 480 B | 294 ns ±0.1% | 175 ns ±0.1% | **1.685x** | 1.333x |
| SHA-256 | 65,664 B | 39.2 µs ±0.0% | 25.6 µs ±0.0% | **1.529x** | 1.168x |

### Commitment: one full tree

Commands:

- one thread: `RAYON_NUM_THREADS=1 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_commit`
- every hardware thread: `RAYON_NUM_THREADS=10 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_commit`

- Code: `merkle-tree/benches/t8_commit.rs`, one `MerkleTree::new` over a borrowed byte matrix, one record per row.
- Timing covers leaf hashing, node hashing, and the digest layers' allocation and release.
- Trees: 2^16 and 2^20 records of 256 B, and 2^10 records of 65,664 B.
- Nodes: BLAKE3 or BLAKE2s of the 64 child bytes, or one SHA-256 compression, the same in both columns.
- Criterion: 1 s warm-up, 3 s measurement, 10 samples.

| Hash | Record | Records | Threads | Standard | T8 | Standard GB/s | T8 GB/s | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 2^16 | 1 | 9.27 ms ±0.2% | 7.9 ms ±0.1% | 1.81 | 2.12 | **1.173x** | 1.250x |
| BLAKE3 | 256 B | 2^20 | 1 | 145 ms ±0.1% | 126 ms ±0.1% | 1.85 | 2.13 | **1.155x** | 1.250x |
| BLAKE3 | 256 B | 2^16 | 10 | 1.47 ms ±1.6% | 1.31 ms ±1.0% | 11.43 | 12.85 | **1.124x** | 1.250x |
| BLAKE3 | 256 B | 2^20 | 10 | 21.1 ms ±4.8% | 18.2 ms ±0.3% | 12.73 | 14.71 | **1.156x** | 1.250x |
| BLAKE3 | 65,664 B | 2^10 | 1 | 29.7 ms ±0.2% | 25.6 ms ±0.1% | 2.26 | 2.63 | **1.161x** | 1.240x |
| BLAKE3 | 65,664 B | 2^10 | 10 | 4.45 ms ±0.5% | 3.89 ms ±0.7% | 15.10 | 17.27 | **1.144x** | 1.240x |
| BLAKE2s | 256 B | 2^16 | 1 | 14.6 ms ±0.1% | 12.2 ms ±0.1% | 1.15 | 1.37 | **1.192x** | 1.250x |
| BLAKE2s | 256 B | 2^20 | 1 | 232 ms ±0.9% | 194 ms ±0.1% | 1.16 | 1.38 | **1.194x** | 1.250x |
| BLAKE2s | 256 B | 2^16 | 10 | 2.16 ms ±1.0% | 1.86 ms ±0.7% | 7.76 | 9.00 | **1.161x** | 1.250x |
| BLAKE2s | 256 B | 2^20 | 10 | 29.9 ms ±3.5% | 26.8 ms ±0.3% | 8.98 | 10.03 | **1.117x** | 1.250x |
| BLAKE2s | 65,664 B | 2^10 | 1 | 44.7 ms ±0.1% | 40.1 ms ±0.1% | 1.50 | 1.68 | **1.116x** | 1.167x |
| BLAKE2s | 65,664 B | 2^10 | 10 | 6.26 ms ±0.4% | 5.63 ms ±0.7% | 10.73 | 11.94 | **1.112x** | 1.167x |
| SHA-256 | 256 B | 2^16 | 1 | 10.5 ms ±0.1% | 7.02 ms ±0.1% | 1.61 | 2.39 | **1.489x** | 1.500x |
| SHA-256 | 256 B | 2^20 | 1 | 167 ms ±0.1% | 112 ms ±0.1% | 1.61 | 2.40 | **1.492x** | 1.500x |
| SHA-256 | 256 B | 2^16 | 10 | 1.5 ms ±1.5% | 1.08 ms ±0.8% | 11.18 | 15.49 | **1.386x** | 1.500x |
| SHA-256 | 256 B | 2^20 | 10 | 20.6 ms ±2.0% | 14 ms ±0.4% | 13.03 | 19.23 | **1.475x** | 1.500x |
| SHA-256 | 65,664 B | 2^10 | 1 | 27.8 ms ±0.3% | 23.8 ms ±0.1% | 2.42 | 2.82 | **1.167x** | 1.168x |
| SHA-256 | 65,664 B | 2^10 | 10 | 3.42 ms ±0.4% | 2.99 ms ±0.6% | 19.63 | 22.50 | **1.146x** | 1.168x |

### Verification: one opening, one core

Command: `cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_verify`

- Code: `merkle-tree/benches/t8_verify.rs`, `MerkleTreeMmcs::verify_batch` on prepared openings, cap height 0.
- Each sample verifies 64 openings at distinct random positions, the same for both columns.
- The table divides by 64, so it shows one opening: the record's leaf hash plus the path to the root.
- Trees: 2^10 and 2^20 records of 256 B, and 2^10 records of 65,664 B.
- Criterion: 0.5 s warm-up, 2 s measurement, 100 samples.

| Hash | Record | Records | Standard per opening | T8 per opening | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 2^10 | 2.06 µs ±0.1% | 1.97 µs ±0.0% | **1.044x** | 1.077x |
| BLAKE3 | 256 B | 2^20 | 3.64 µs ±0.1% | 3.55 µs ±0.1% | **1.025x** | 1.043x |
| BLAKE3 | 65,664 B | 2^10 | 48.7 µs ±0.1% | 89.1 µs ±0.0% | **0.546x** | 1.237x |
| BLAKE2s | 256 B | 2^10 | 2.57 µs ±0.2% | 2.49 µs ±0.1% | **1.033x** | 1.077x |
| BLAKE2s | 256 B | 2^20 | 4.54 µs ±0.1% | 4.47 µs ±0.1% | **1.015x** | 1.043x |
| BLAKE2s | 65,664 B | 2^10 | 136 µs ±0.1% | 127 µs ±0.1% | **1.072x** | 1.165x |
| SHA-256 | 256 B | 2^10 | 807 ns ±0.1% | 733 ns ±0.0% | **1.101x** | 1.154x |
| SHA-256 | 256 B | 2^20 | 1.37 µs ±0.1% | 1.29 µs ±0.0% | **1.057x** | 1.087x |
| SHA-256 | 65,664 B | 2^10 | 39.9 µs ±0.1% | 26.4 µs ±0.0% | **1.515x** | 1.166x |

## Notes

- BLAKE3 keeps T8's three roles apart with its counter and flags: the construction as analysed.
- BLAKE2s keeps them apart with counters 1, 2 and 3 and the final flag clear, which no plain BLAKE2s call uses.
- SHA-256's compression takes exactly 96 bytes, so its three calls are one function. The security analysis does not cover that instantiation.
- Keccak-256 absorbs 136 bytes per permutation, so T8 costs more calls than the plain hash and is not measured.
- The build targets the host CPU: SHA-256 batches four streams of the ARMv8 SHA-2 extension, BLAKE3 and BLAKE2s sixteen NEON lanes.
- Batched SHA-256 leaves follow the call ratio exactly: both leaves wait on the same SHA-2 unit, four streams deep.
- One SHA-256 record runs its calls a and b as two streams, so a stage waits on two calls, not three.
- So single SHA-256 records, and verification of 64 KiB-class records, beat the call ratio.
- Batched BLAKE3 and BLAKE2s leaves fall short of the call ratio: each out-of-line T8 call costs a little more than a call inside the standard chunk loop.
- Their loads and transposes are hidden: with them removed, the BLAKE3 T8 batch time does not change.
- On one thread, SHA-256 commitment follows the call ratio, and BLAKE3 and BLAKE2s land a little below it, as their batched leaves do.
- On all 10 cores, trees of 256-byte records keep most of the call saving, unlike on the 32 threads of the x86 host.
- A single long BLAKE3 record hashes its chunks in parallel; T8's chained stages cannot, hence the 0.5x rows.
