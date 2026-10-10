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
- On AArch64, add `p3-blake3/neon` to `--features`, so one standard BLAKE3 message runs NEON code, not portable code.
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
date: 2026-09-29T20:52:19+02:00
commit: bee40458b9ddae25f25fc8899c8df161cf4e71ca
cpu: Apple M1 Pro
kernel: 25.6.0
rustc: rustc 1.98.0-nightly (9e2abe0c6 2026-06-16)
RUSTFLAGS: -Ctarget-cpu=native
features: parallel,p3-blake3/neon
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
| SHA-256, T253 | 253 B | 5 | 3 | 40.0% fewer |
| SHA-256, T253 | 474 B | 8 | 6 | 25.0% fewer |
| SHA-256, T253 | 65,448 B | 1,023 | 888 | 13.2% fewer |
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
| BLAKE3 | 256 B | 2^16 | 7.17 ms ±0.1% | 5.93 ms ±0.1% | 2.34 | 2.83 | **1.209x** | 1.333x |
| BLAKE3 | 480 B | 2^15 | 7.12 ms ±0.1% | 5.72 ms ±0.1% | 2.21 | 2.75 | **1.245x** | 1.333x |
| BLAKE3 | 65,664 B | 2^8 | 7.38 ms ±0.1% | 6.36 ms ±0.1% | 2.28 | 2.64 | **1.161x** | 1.240x |
| BLAKE2s | 256 B | 2^16 | 11.4 ms ±0.1% | 9.06 ms ±0.0% | 1.47 | 1.85 | **1.264x** | 1.333x |
| BLAKE2s | 480 B | 2^15 | 11.3 ms ±0.1% | 8.91 ms ±0.1% | 1.39 | 1.77 | **1.267x** | 1.333x |
| BLAKE2s | 65,664 B | 2^8 | 11.1 ms ±0.1% | 10 ms ±0.1% | 1.51 | 1.68 | **1.113x** | 1.167x |
| SHA-256 | 256 B | 2^16 | 8.62 ms ±0.0% | 5.19 ms ±0.1% | 1.95 | 3.23 | **1.661x** | 1.667x |
| SHA-256 | 480 B | 2^15 | 6.9 ms ±0.1% | 5.19 ms ±0.1% | 2.28 | 3.03 | **1.330x** | 1.333x |
| SHA-256 | 65,664 B | 2^8 | 6.94 ms ±0.1% | 5.93 ms ±0.1% | 2.42 | 2.83 | **1.170x** | 1.168x |
| SHA-256, T253 | 253 B | 2^16 | 8.84 ms ±0.1% | 5.2 ms ±0.1% | 1.88 | 3.19 | **1.700x** | 1.667x |
| SHA-256, T253 | 474 B | 2^15 | 6.9 ms ±0.1% | 5.19 ms ±0.1% | 2.25 | 2.99 | **1.329x** | 1.333x |
| SHA-256, T253 | 65,448 B | 2^8 | 6.89 ms ±0.1% | 5.99 ms ±0.1% | 2.43 | 2.79 | **1.150x** | 1.152x |

### Leaf hashing, one record, one core

Command: `cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_leaf -- 'leaf/.*/single'`

- Code: `merkle-tree/benches/t8_leaf.rs`, one `hash_slice` call on one record, the path a verifier takes.
- Records: 256 B, 480 B and 65,664 B.
- Criterion: 0.5 s warm-up, 2 s measurement, 100 samples.

| Hash | Record | Standard | T8 | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 387 ns ±0.1% | 284 ns ±0.1% | **1.360x** | 1.333x |
| BLAKE3 | 480 B | 769 ns ±0.1% | 582 ns ±0.1% | **1.321x** | 1.333x |
| BLAKE3 | 65,664 B | 46.8 µs ±0.1% | 87.3 µs ±0.1% | **0.536x** | 1.240x |
| BLAKE2s | 256 B | 527 ns ±0.1% | 414 ns ±0.1% | **1.274x** | 1.333x |
| BLAKE2s | 480 B | 1.05 µs ±0.0% | 840 ns ±0.1% | **1.251x** | 1.333x |
| BLAKE2s | 65,664 B | 134 µs ±0.0% | 125 µs ±0.0% | **1.073x** | 1.167x |
| SHA-256 | 256 B | 203 ns ±0.8% | 86.3 ns ±0.1% | **2.357x** | 1.667x |
| SHA-256 | 480 B | 321 ns ±0.1% | 175 ns ±0.1% | **1.841x** | 1.333x |
| SHA-256 | 65,664 B | 39.3 µs ±0.0% | 25.7 µs ±0.1% | **1.531x** | 1.168x |
| SHA-256, T253 | 253 B | 202 ns ±0.6% | 86.2 ns ±0.1% | **2.339x** | 1.667x |
| SHA-256, T253 | 474 B | 319 ns ±0.6% | 200 ns ±0.1% | **1.596x** | 1.333x |
| SHA-256, T253 | 65,448 B | 39.1 µs ±0.0% | 32.5 µs ±0.0% | **1.203x** | 1.152x |

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
| BLAKE3 | 256 B | 2^16 | 1 | 9.26 ms ±0.2% | 7.9 ms ±0.1% | 1.81 | 2.12 | **1.173x** | 1.250x |
| BLAKE3 | 256 B | 2^20 | 1 | 145 ms ±0.1% | 126 ms ±0.1% | 1.85 | 2.14 | **1.157x** | 1.250x |
| BLAKE3 | 256 B | 2^16 | 10 | 1.49 ms ±1.3% | 1.3 ms ±0.4% | 11.28 | 12.87 | **1.141x** | 1.250x |
| BLAKE3 | 256 B | 2^20 | 10 | 20.8 ms ±5.4% | 18.5 ms ±1.1% | 12.94 | 14.54 | **1.124x** | 1.250x |
| BLAKE3 | 65,664 B | 2^10 | 1 | 29.7 ms ±0.1% | 25.5 ms ±0.1% | 2.27 | 2.63 | **1.162x** | 1.240x |
| BLAKE3 | 65,664 B | 2^10 | 10 | 4.48 ms ±1.9% | 3.93 ms ±1.1% | 15.02 | 17.11 | **1.139x** | 1.240x |
| BLAKE2s | 256 B | 2^16 | 1 | 14.6 ms ±0.3% | 12.2 ms ±0.2% | 1.15 | 1.37 | **1.192x** | 1.250x |
| BLAKE2s | 256 B | 2^20 | 1 | 232 ms ±0.3% | 194 ms ±0.4% | 1.16 | 1.38 | **1.192x** | 1.250x |
| BLAKE2s | 256 B | 2^16 | 10 | 2.21 ms ±2.6% | 1.92 ms ±1.3% | 7.58 | 8.74 | **1.154x** | 1.250x |
| BLAKE2s | 256 B | 2^20 | 10 | 29.7 ms ±0.3% | 27 ms ±0.4% | 9.04 | 9.94 | **1.100x** | 1.250x |
| BLAKE2s | 65,664 B | 2^10 | 1 | 44.7 ms ±0.1% | 40 ms ±0.1% | 1.50 | 1.68 | **1.117x** | 1.167x |
| BLAKE2s | 65,664 B | 2^10 | 10 | 6.31 ms ±0.3% | 5.73 ms ±0.8% | 10.65 | 11.73 | **1.101x** | 1.167x |
| SHA-256 | 256 B | 2^16 | 1 | 10.4 ms ±0.1% | 6.99 ms ±0.1% | 1.61 | 2.40 | **1.489x** | 1.500x |
| SHA-256 | 256 B | 2^20 | 1 | 166 ms ±0.1% | 111 ms ±0.0% | 1.62 | 2.41 | **1.493x** | 1.500x |
| SHA-256 | 256 B | 2^16 | 10 | 1.63 ms ±3.6% | 1.09 ms ±0.7% | 10.27 | 15.40 | **1.500x** | 1.500x |
| SHA-256 | 256 B | 2^20 | 10 | 20.5 ms ±1.9% | 14.1 ms ±0.5% | 13.11 | 19.06 | **1.454x** | 1.500x |
| SHA-256 | 65,664 B | 2^10 | 1 | 27.8 ms ±0.1% | 23.8 ms ±0.1% | 2.42 | 2.83 | **1.168x** | 1.168x |
| SHA-256 | 65,664 B | 2^10 | 10 | 3.54 ms ±1.9% | 3.08 ms ±1.2% | 19.00 | 21.86 | **1.151x** | 1.168x |
| SHA-256, T253 | 253 B | 2^16 | 1 | 10.6 ms ±0.1% | 7 ms ±0.2% | 1.56 | 2.37 | **1.520x** | 1.500x |
| SHA-256, T253 | 253 B | 2^20 | 1 | 170 ms ±0.1% | 111 ms ±0.1% | 1.56 | 2.38 | **1.523x** | 1.500x |
| SHA-256, T253 | 253 B | 2^16 | 10 | 1.61 ms ±4.6% | 1.08 ms ±0.6% | 10.27 | 15.33 | **1.493x** | 1.500x |
| SHA-256, T253 | 253 B | 2^20 | 10 | 21.2 ms ±4.0% | 14.2 ms ±3.1% | 12.53 | 18.66 | **1.489x** | 1.500x |
| SHA-256, T253 | 65,448 B | 2^10 | 1 | 27.6 ms ±0.1% | 24.1 ms ±0.1% | 2.42 | 2.79 | **1.149x** | 1.152x |
| SHA-256, T253 | 65,448 B | 2^10 | 10 | 3.6 ms ±4.4% | 3.09 ms ±1.0% | 18.62 | 21.66 | **1.163x** | 1.152x |

### Verification: one opening, one core

Command: `cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_verify`

- Code: `merkle-tree/benches/t8_verify.rs`, `MerkleTreeMmcs::verify_batch` on prepared openings, cap height 0.
- Each sample verifies 64 openings at distinct random positions, the same for both columns.
- The table divides by 64, so it shows one opening: the record's leaf hash plus the path to the root.
- Trees: 2^10 and 2^20 records of 256 B, and 2^10 records of 65,664 B.
- Criterion: 0.5 s warm-up, 2 s measurement, 100 samples.

| Hash | Record | Records | Standard per opening | T8 per opening | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 2^10 | 2.06 µs ±0.1% | 1.96 µs ±0.0% | **1.049x** | 1.077x |
| BLAKE3 | 256 B | 2^20 | 3.64 µs ±0.1% | 3.55 µs ±0.0% | **1.025x** | 1.043x |
| BLAKE3 | 65,664 B | 2^10 | 48.6 µs ±0.0% | 88.9 µs ±0.0% | **0.546x** | 1.237x |
| BLAKE2s | 256 B | 2^10 | 2.57 µs ±0.1% | 2.48 µs ±0.1% | **1.034x** | 1.077x |
| BLAKE2s | 256 B | 2^20 | 4.53 µs ±0.1% | 4.46 µs ±0.1% | **1.015x** | 1.043x |
| BLAKE2s | 65,664 B | 2^10 | 136 µs ±0.0% | 127 µs ±0.0% | **1.072x** | 1.165x |
| SHA-256 | 256 B | 2^10 | 806 ns ±0.0% | 732 ns ±0.0% | **1.102x** | 1.154x |
| SHA-256 | 256 B | 2^20 | 1.37 µs ±0.0% | 1.29 µs ±0.1% | **1.057x** | 1.087x |
| SHA-256 | 65,664 B | 2^10 | 39.9 µs ±0.0% | 26.3 µs ±0.0% | **1.517x** | 1.166x |
| SHA-256, T253 | 253 B | 2^10 | 812 ns ±0.1% | 732 ns ±0.1% | **1.109x** | 1.154x |
| SHA-256, T253 | 253 B | 2^20 | 1.37 µs ±0.1% | 1.29 µs ±0.0% | **1.060x** | 1.087x |
| SHA-256, T253 | 65,448 B | 2^10 | 39.8 µs ±0.0% | 33.2 µs ±0.0% | **1.198x** | 1.150x |

## Wider tree nodes

Command: `RAYON_NUM_THREADS=1 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_arity`, then again on every thread.

- Code: `merkle-tree/benches/t8_arity.rs`, the same Plonky3 tree with arity 2 and arity 4.
- Leaves: the standard hash of 256-byte records, identical in both columns.
- A 4-ary Keccak-256 or SHA3-256 node fits in one permutation; a 4-ary BLAKE3 node takes two compressions.
- Verification times one opening; each sample verifies 64.
- Proof digests: siblings in one opening, 32 bytes each.

| Hash | Records | Operation | Binary | 4-ary | Speedup | Calls, binary / 4-ary | Proof digests, binary / 4-ary |
|---|---:|---|---:|---:|---:|---:|---:|
| Keccak-256 | 2^16 | commit, 1 thread | 21 ms | 16.5 ms | **1.270x** | 196,607 / 152,917 | 16 / 24 |
| Keccak-256 | 2^16 | commit, 10 threads | 2.92 ms | 2.3 ms | **1.270x** | 196,607 / 152,917 | 16 / 24 |
| Keccak-256 | 2^16 | verify | 6.43 µs | 3.94 µs | **1.630x** | 18 / 10 | 16 / 24 |
| Keccak-256 | 2^20 | commit, 1 thread | 334 ms | 263 ms | **1.270x** | 3,145,727 / 2,446,677 | 20 / 30 |
| Keccak-256 | 2^20 | commit, 10 threads | 41.4 ms | 33 ms | **1.256x** | 3,145,727 / 2,446,677 | 20 / 30 |
| Keccak-256 | 2^20 | verify | 7.87 µs | 4.81 µs | **1.639x** | 22 / 12 | 20 / 30 |
| SHA3-256 | 2^16 | commit, 1 thread | 21 ms | 16.5 ms | **1.270x** | 196,607 / 152,917 | 16 / 24 |
| SHA3-256 | 2^16 | commit, 10 threads | 2.97 ms | 2.37 ms | **1.252x** | 196,607 / 152,917 | 16 / 24 |
| SHA3-256 | 2^16 | verify | 6.42 µs | 3.92 µs | **1.637x** | 18 / 10 | 16 / 24 |
| SHA3-256 | 2^20 | commit, 1 thread | 334 ms | 263 ms | **1.269x** | 3,145,727 / 2,446,677 | 20 / 30 |
| SHA3-256 | 2^20 | commit, 10 threads | 41.3 ms | 32.7 ms | **1.263x** | 3,145,727 / 2,446,677 | 20 / 30 |
| SHA3-256 | 2^20 | verify | 7.88 µs | 4.77 µs | **1.650x** | 22 / 12 | 20 / 30 |
| BLAKE3 | 2^16 | commit, 1 thread | 9.26 ms | 8.59 ms | **1.078x** | 327,679 / 305,834 | 16 / 24 |
| BLAKE3 | 2^16 | commit, 10 threads | 1.52 ms | 1.39 ms | **1.093x** | 327,679 / 305,834 | 16 / 24 |
| BLAKE3 | 2^16 | verify | 2.95 µs | 2.79 µs | **1.056x** | 20 / 20 | 16 / 24 |
| BLAKE3 | 2^20 | commit, 1 thread | 152 ms | 142 ms | **1.067x** | 5,242,879 / 4,893,354 | 20 / 30 |
| BLAKE3 | 2^20 | commit, 10 threads | 21.2 ms | 19.7 ms | **1.076x** | 5,242,879 / 4,893,354 | 20 / 30 |
| BLAKE3 | 2^20 | verify | 3.6 µs | 3.4 µs | **1.059x** | 24 / 24 | 20 / 30 |

## Notes

- BLAKE3 keeps T8's three roles apart with its counter and flags: the construction as analysed.
- BLAKE2s keeps them apart with counters 1, 2 and 3 and the final flag clear, which no plain BLAKE2s call uses.
- SHA-256's compression takes exactly 96 bytes, so T8's three calls there are one function. The security analysis does not cover that instantiation.
- T253 is T8 on SHA-256 with a role byte at the top of each call's chaining value, so its three calls are distinct functions.
- The tree's node hash starts from the SHA-256 initial value, whose top byte is 0x6a, so no role meets it either.
- The price is one byte per tagged block: records of 32 + 221 k bytes, compared with plain SHA-256 on the same bytes.
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
- BLAKE3 pairs a and b on x86 only: two BLAKE3 calls in one NEON pass cost about as much as two scalar calls.
- T253 runs on the same ARM SHA-2 kernel as T8, so its batches follow the call ratio and one record waits on two calls per stage.
- Past its first stage, one T253 record costs about 20 ns more per stage than one T8 record, which the call counts do not explain.
- The standard one-record SHA-256 rows moved from 180 to 203 ns at 256 B, and from 294 to 321 ns at 480 B, when the T253 benches joined the binary.
- Their code is unchanged, so the shift is code layout, and it flatters the one-record SHA-256 speedups by about 10%.
