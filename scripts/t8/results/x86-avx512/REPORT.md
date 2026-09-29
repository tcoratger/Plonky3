# T8 leaves on Plonky3

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
taskset -c 4 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_leaf
taskset -c 4 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_verify
RAYON_NUM_THREADS=1 taskset -c 4 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_commit
RAYON_NUM_THREADS=32 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_commit
```

- Every command in this report assumes `RUSTFLAGS=-Ctarget-cpu=native`, as exported above.
- On AArch64, add `p3-blake3/neon` to `--features`, so one standard BLAKE3 message runs NEON code, not portable code.
- A trailing regex selects benchmarks, for example `-- 'leaf/sha256'`.
- `taskset -c 4` pins the process to one core, so single-threaded runs do not migrate.
- `--profile optimized` is Plonky3's own profile: thin LTO and one codegen unit.
- Criterion prints `time: [low median high]`, the 95% interval of the median.
- It also keeps each estimate in `target/criterion/<group>/<scheme>/<size>/new/estimates.json`.
- The script saves them as the baseline `native`, copies them to `scripts/t8/results/x86-avx512/`, and builds this report from them.

## Reading the tables

- **Standard, T8**: criterion's median time, with the half-width of its 95% interval.
- **GB/s**: record bytes hashed per second, 1 GB = 10^9 bytes.
- **Speedup**: standard time over T8 time, so above 1 means T8 is faster.
- **Call ratio**: standard calls over T8 calls, what counting compressions alone predicts.

## Setup

**native build**

```
date: 2026-09-29T18:30:07+02:00
commit: dd692539c5b656b2611e7ebdae13d9cf7732ebb8
cpu: AMD Ryzen 9 9950X3D 16-Core Processor
kernel: 7.0.0-34-generic
rustc: rustc 1.98.0 (88d9e12ae 2026-08-18)
RUSTFLAGS: -Ctarget-cpu=native
features: parallel
governor: powersave
boost: 1
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

Command: `taskset -c 4 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_leaf -- 'leaf/.*/batch'`

- Code: `merkle-tree/benches/t8_leaf.rs`, one `hash_many` call over the whole batch, on one thread.
- Batches: 2^16 records of 256 B, 2^15 of 480 B, 2^8 of 65,664 B, so 16 MiB each.
- Input: deterministic xorshift bytes, generated before timing.
- Criterion: 1 s warm-up, 3 s measurement, 100 samples.

| Hash | Record | Records | Standard | T8 | Standard GB/s | T8 GB/s | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 2^16 | 1.21 ms ±0.0% | 917 µs ±0.0% | 13.86 | 18.30 | **1.320x** | 1.333x |
| BLAKE3 | 480 B | 2^15 | 1.16 ms ±0.0% | 844 µs ±0.0% | 13.57 | 18.63 | **1.373x** | 1.333x |
| BLAKE3 | 65,664 B | 2^8 | 1.05 ms ±0.1% | 924 µs ±0.0% | 16.02 | 18.20 | **1.136x** | 1.240x |
| BLAKE2s | 256 B | 2^16 | 2.03 ms ±0.0% | 1.41 ms ±0.0% | 8.27 | 11.88 | **1.436x** | 1.333x |
| BLAKE2s | 480 B | 2^15 | 1.84 ms ±0.0% | 1.39 ms ±0.0% | 8.53 | 11.28 | **1.322x** | 1.333x |
| BLAKE2s | 65,664 B | 2^8 | 1.57 ms ±0.0% | 1.48 ms ±0.0% | 10.74 | 11.39 | **1.061x** | 1.167x |
| SHA-256 | 256 B | 2^16 | 2.19 ms ±0.0% | 1.66 ms ±0.1% | 7.68 | 10.12 | **1.318x** | 1.667x |
| SHA-256 | 480 B | 2^15 | 1.92 ms ±0.0% | 1.62 ms ±0.0% | 8.20 | 9.71 | **1.183x** | 1.333x |
| SHA-256 | 65,664 B | 2^8 | 1.91 ms ±0.0% | 1.73 ms ±0.0% | 8.82 | 9.74 | **1.104x** | 1.168x |
| SHA-256, T253 | 253 B | 2^16 | 2.23 ms ±0.0% | 1.68 ms ±0.0% | 7.45 | 9.85 | **1.322x** | 1.667x |
| SHA-256, T253 | 474 B | 2^15 | 1.91 ms ±0.2% | 1.64 ms ±0.0% | 8.13 | 9.49 | **1.168x** | 1.333x |
| SHA-256, T253 | 65,448 B | 2^8 | 1.89 ms ±0.0% | 1.79 ms ±0.0% | 8.85 | 9.37 | **1.058x** | 1.152x |

### Leaf hashing, one record, one core

Command: `taskset -c 4 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_leaf -- 'leaf/.*/single'`

- Code: `merkle-tree/benches/t8_leaf.rs`, one `hash_slice` call on one record, the path a verifier takes.
- Records: 256 B, 480 B and 65,664 B.
- Criterion: 0.5 s warm-up, 2 s measurement, 100 samples.

| Hash | Record | Standard | T8 | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 263 ns ±0.0% | 111 ns ±0.0% | **2.361x** | 1.333x |
| BLAKE3 | 480 B | 515 ns ±0.0% | 244 ns ±0.0% | **2.108x** | 1.333x |
| BLAKE3 | 65,664 B | 5.29 µs ±0.0% | 38.9 µs ±0.0% | **0.136x** | 1.240x |
| BLAKE2s | 256 B | 385 ns ±0.0% | 283 ns ±0.0% | **1.359x** | 1.333x |
| BLAKE2s | 480 B | 776 ns ±0.1% | 587 ns ±0.0% | **1.322x** | 1.333x |
| BLAKE2s | 65,664 B | 100 µs ±0.0% | 89 µs ±0.0% | **1.125x** | 1.167x |
| SHA-256 | 256 B | 131 ns ±0.1% | 69.4 ns ±0.0% | **1.881x** | 1.667x |
| SHA-256 | 480 B | 203 ns ±0.1% | 138 ns ±0.0% | **1.469x** | 1.333x |
| SHA-256 | 65,664 B | 24.2 µs ±0.0% | 20.1 µs ±0.0% | **1.202x** | 1.168x |
| SHA-256, T253 | 253 B | 142 ns ±0.0% | 72.8 ns ±0.1% | **1.943x** | 1.667x |
| SHA-256, T253 | 474 B | 203 ns ±0.1% | 158 ns ±0.2% | **1.286x** | 1.333x |
| SHA-256, T253 | 65,448 B | 24.1 µs ±0.0% | 25.3 µs ±0.3% | **0.954x** | 1.152x |

### Commitment: one full tree

Commands:

- one thread: `RAYON_NUM_THREADS=1 taskset -c 4 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_commit`
- every hardware thread: `RAYON_NUM_THREADS=32 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_commit`

- Code: `merkle-tree/benches/t8_commit.rs`, one `MerkleTree::new` over a borrowed byte matrix, one record per row.
- Timing covers leaf hashing, node hashing, and the digest layers' allocation and release.
- Trees: 2^16 and 2^20 records of 256 B, and 2^10 records of 65,664 B.
- Nodes: BLAKE3 or BLAKE2s of the 64 child bytes, or one SHA-256 compression, the same in both columns.
- Criterion: 1 s warm-up, 3 s measurement, 10 samples.

| Hash | Record | Records | Threads | Standard | T8 | Standard GB/s | T8 GB/s | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 2^16 | 1 | 1.9 ms ±2.2% | 1.63 ms ±1.4% | 8.82 | 10.27 | **1.164x** | 1.250x |
| BLAKE3 | 256 B | 2^20 | 1 | 41.9 ms ±4.3% | 38.7 ms ±6.5% | 6.41 | 6.94 | **1.082x** | 1.250x |
| BLAKE3 | 256 B | 2^16 | 32 | 516 µs ±0.9% | 517 µs ±0.3% | 32.51 | 32.48 | **0.999x** | 1.250x |
| BLAKE3 | 256 B | 2^20 | 32 | 11.7 ms ±1.1% | 11.5 ms ±2.2% | 22.89 | 23.36 | **1.020x** | 1.250x |
| BLAKE3 | 65,664 B | 2^10 | 1 | 4.6 ms ±1.3% | 4.19 ms ±4.5% | 14.63 | 16.04 | **1.096x** | 1.240x |
| BLAKE3 | 65,664 B | 2^10 | 32 | 624 µs ±1.4% | 585 µs ±1.2% | 107.80 | 114.91 | **1.066x** | 1.240x |
| BLAKE2s | 256 B | 2^16 | 1 | 2.53 ms ±0.1% | 1.97 ms ±0.0% | 6.64 | 8.51 | **1.281x** | 1.250x |
| BLAKE2s | 256 B | 2^20 | 1 | 60.8 ms ±0.1% | 52.8 ms ±0.6% | 4.41 | 5.09 | **1.152x** | 1.250x |
| BLAKE2s | 256 B | 2^16 | 32 | 541 µs ±0.5% | 543 µs ±4.3% | 30.99 | 30.92 | **0.998x** | 1.250x |
| BLAKE2s | 256 B | 2^20 | 32 | 11.5 ms ±0.6% | 11.9 ms ±0.9% | 23.31 | 22.60 | **0.969x** | 1.250x |
| BLAKE2s | 65,664 B | 2^10 | 1 | 6.67 ms ±0.1% | 6.61 ms ±12.8% | 10.08 | 10.18 | **1.009x** | 1.167x |
| BLAKE2s | 65,664 B | 2^10 | 32 | 769 µs ±0.4% | 727 µs ±1.9% | 87.49 | 92.44 | **1.057x** | 1.167x |
| SHA-256 | 256 B | 2^16 | 1 | 2.74 ms ±1.6% | 2.16 ms ±3.2% | 6.13 | 7.76 | **1.266x** | 1.500x |
| SHA-256 | 256 B | 2^20 | 1 | 62.1 ms ±0.4% | 54.7 ms ±0.0% | 4.32 | 4.91 | **1.136x** | 1.500x |
| SHA-256 | 256 B | 2^16 | 32 | 538 µs ±0.2% | 535 µs ±0.4% | 31.21 | 31.35 | **1.004x** | 1.500x |
| SHA-256 | 256 B | 2^20 | 32 | 11.8 ms ±1.2% | 11.8 ms ±0.9% | 22.77 | 22.74 | **0.999x** | 1.500x |
| SHA-256 | 65,664 B | 2^10 | 1 | 8.27 ms ±5.1% | 7.52 ms ±5.8% | 8.13 | 8.94 | **1.100x** | 1.168x |
| SHA-256 | 65,664 B | 2^10 | 32 | 803 µs ±0.8% | 754 µs ±1.0% | 83.74 | 89.20 | **1.065x** | 1.168x |
| SHA-256, T253 | 253 B | 2^16 | 1 | 2.74 ms ±0.1% | 3.15 ms ±4.2% | 6.04 | 5.27 | **0.872x** | 1.500x |
| SHA-256, T253 | 253 B | 2^20 | 1 | 78 ms ±17.2% | 55.3 ms ±0.1% | 3.40 | 4.80 | **1.412x** | 1.500x |
| SHA-256, T253 | 253 B | 2^16 | 32 | 536 µs ±0.4% | 534 µs ±0.2% | 30.94 | 31.03 | **1.003x** | 1.500x |
| SHA-256, T253 | 253 B | 2^20 | 32 | 12 ms ±2.0% | 12 ms ±0.7% | 22.11 | 22.15 | **1.002x** | 1.500x |
| SHA-256, T253 | 65,448 B | 2^10 | 1 | 7.74 ms ±0.0% | 7.39 ms ±0.2% | 8.66 | 9.06 | **1.047x** | 1.152x |
| SHA-256, T253 | 65,448 B | 2^10 | 32 | 783 µs ±0.8% | 768 µs ±0.3% | 85.58 | 87.31 | **1.020x** | 1.152x |

### Verification: one opening, one core

Command: `taskset -c 4 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_verify`

- Code: `merkle-tree/benches/t8_verify.rs`, `MerkleTreeMmcs::verify_batch` on prepared openings, cap height 0.
- Each sample verifies 64 openings at distinct random positions, the same for both columns.
- The table divides by 64, so it shows one opening: the record's leaf hash plus the path to the root.
- Trees: 2^10 and 2^20 records of 256 B, and 2^10 records of 65,664 B.
- Criterion: 0.5 s warm-up, 2 s measurement, 100 samples.

| Hash | Record | Records | Standard per opening | T8 per opening | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 2^10 | 1.23 µs ±0.4% | 1.1 µs ±0.1% | **1.121x** | 1.077x |
| BLAKE3 | 256 B | 2^20 | 2.16 µs ±0.4% | 2.03 µs ±0.1% | **1.065x** | 1.043x |
| BLAKE3 | 65,664 B | 2^10 | 6.45 µs ±0.1% | 39.9 µs ±0.0% | **0.162x** | 1.237x |
| BLAKE2s | 256 B | 2^10 | 1.74 µs ±0.2% | 1.64 µs ±0.2% | **1.061x** | 1.077x |
| BLAKE2s | 256 B | 2^20 | 3.04 µs ±0.1% | 2.91 µs ±0.3% | **1.044x** | 1.043x |
| BLAKE2s | 65,664 B | 2^10 | 102 µs ±0.0% | 90.4 µs ±0.0% | **1.124x** | 1.165x |
| SHA-256 | 256 B | 2^10 | 575 ns ±0.1% | 523 ns ±0.1% | **1.100x** | 1.154x |
| SHA-256 | 256 B | 2^20 | 987 ns ±0.1% | 907 ns ±0.2% | **1.088x** | 1.087x |
| SHA-256 | 65,664 B | 2^10 | 24.7 µs ±0.1% | 20.7 µs ±0.0% | **1.196x** | 1.166x |
| SHA-256, T253 | 253 B | 2^10 | 588 ns ±0.1% | 525 ns ±0.1% | **1.119x** | 1.154x |
| SHA-256, T253 | 253 B | 2^20 | 999 ns ±0.1% | 927 ns ±0.3% | **1.078x** | 1.087x |
| SHA-256, T253 | 65,448 B | 2^10 | 24.6 µs ±0.0% | 25.7 µs ±0.1% | **0.959x** | 1.150x |

## Wider tree nodes

Command: `RAYON_NUM_THREADS=1 taskset -c 4 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_arity`, then again on every thread.

- Code: `merkle-tree/benches/t8_arity.rs`, the same Plonky3 tree with arity 2 and arity 4.
- Leaves: the standard hash of 256-byte records, identical in both columns.
- A 4-ary Keccak-256 or SHA3-256 node fits in one permutation; a 4-ary BLAKE3 node takes two compressions.
- Verification times one opening; each sample verifies 64.
- Proof digests: siblings in one opening, 32 bytes each.

| Hash | Records | Operation | Binary | 4-ary | Speedup | Calls, binary / 4-ary | Proof digests, binary / 4-ary |
|---|---:|---|---:|---:|---:|---:|---:|
| Keccak-256 | 2^16 | commit, 1 thread | 6.83 ms | 4.71 ms | **1.451x** | 196,607 / 152,917 | 16 / 24 |
| Keccak-256 | 2^16 | commit, 32 threads | 603 µs | 511 µs | **1.180x** | 196,607 / 152,917 | 16 / 24 |
| Keccak-256 | 2^16 | verify | 7.63 µs | 4.33 µs | **1.761x** | 18 / 10 | 16 / 24 |
| Keccak-256 | 2^20 | commit, 1 thread | 146 ms | 121 ms | **1.208x** | 3,145,727 / 2,446,677 | 20 / 30 |
| Keccak-256 | 2^20 | commit, 32 threads | 11.5 ms | 9.63 ms | **1.193x** | 3,145,727 / 2,446,677 | 20 / 30 |
| Keccak-256 | 2^20 | verify | 9.32 µs | 5.18 µs | **1.797x** | 22 / 12 | 20 / 30 |
| SHA3-256 | 2^16 | commit, 1 thread | 6.18 ms | 4.66 ms | **1.326x** | 196,607 / 152,917 | 16 / 24 |
| SHA3-256 | 2^16 | commit, 32 threads | 604 µs | 504 µs | **1.199x** | 196,607 / 152,917 | 16 / 24 |
| SHA3-256 | 2^16 | verify | 7.56 µs | 4.33 µs | **1.747x** | 18 / 10 | 16 / 24 |
| SHA3-256 | 2^20 | commit, 1 thread | 144 ms | 115 ms | **1.252x** | 3,145,727 / 2,446,677 | 20 / 30 |
| SHA3-256 | 2^20 | commit, 32 threads | 11.3 ms | 9.51 ms | **1.185x** | 3,145,727 / 2,446,677 | 20 / 30 |
| SHA3-256 | 2^20 | verify | 9.31 µs | 5.22 µs | **1.784x** | 22 / 12 | 20 / 30 |
| BLAKE3 | 2^16 | commit, 1 thread | 1.57 ms | 1.47 ms | **1.066x** | 327,679 / 305,834 | 16 / 24 |
| BLAKE3 | 2^16 | commit, 32 threads | 571 µs | 468 µs | **1.221x** | 327,679 / 305,834 | 16 / 24 |
| BLAKE3 | 2^16 | verify | 1.85 µs | 1.61 µs | **1.146x** | 20 / 20 | 16 / 24 |
| BLAKE3 | 2^20 | commit, 1 thread | 42.5 ms | 40.3 ms | **1.056x** | 5,242,879 / 4,893,354 | 20 / 30 |
| BLAKE3 | 2^20 | commit, 32 threads | 11.7 ms | 9.72 ms | **1.207x** | 5,242,879 / 4,893,354 | 20 / 30 |
| BLAKE3 | 2^20 | verify | 2.16 µs | 1.95 µs | **1.112x** | 24 / 24 | 20 / 30 |

## Notes

- BLAKE3 keeps T8's three roles apart with its counter and flags: the construction as analysed.
- BLAKE2s keeps them apart with counters 1, 2 and 3 and the final flag clear, which no plain BLAKE2s call uses.
- SHA-256's compression takes exactly 96 bytes, so T8's three calls there are one function. The security analysis does not cover that instantiation.
- T253 is T8 on SHA-256 with a role byte at the top of each call's chaining value, so its three calls are distinct functions.
- The tree's node hash starts from the SHA-256 initial value, whose top byte is 0x6a, so no role meets it either.
- The price is one byte per tagged block: records of 32 + 221 k bytes, compared with plain SHA-256 on the same bytes.
- Keccak-256 absorbs 136 bytes per permutation, so T8 costs more calls than the plain hash and is not measured.
- The build targets the host CPU, so all three hashes batch on AVX-512.
- Both leaves share the transpose of every record into vector lanes, which the call count does not see.
- Batched leaves read every record once, 64 bytes at a time, exactly as the standard leaf does.
- So the transpose into vector lanes costs both leaves the same, and dilutes the call saving a little.
- On one thread, commitment follows the call ratio.
- On 32 threads, trees of 256-byte records wait on memory, so fewer calls barely shows.
- Trees of 64 KiB-class records stay compute-bound on 32 threads, and keep most of the call saving.
- A single long BLAKE3 record hashes its chunks in parallel; T8's chained stages cannot, hence the 0.1x rows.
