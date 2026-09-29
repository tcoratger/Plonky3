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
- A trailing regex selects benchmarks, for example `-- 'leaf/sha256'`.
- `taskset -c 4` pins the process to one core, so single-threaded runs do not migrate.
- `--profile optimized` is Plonky3's own profile: thin LTO and one codegen unit.
- Criterion prints `time: [low median high]`, the 95% interval of the median.
- It also keeps each estimate in `target/criterion/<group>/<scheme>/<size>/new/estimates.json`.
- The script saves them as the baseline `native`, copies them to `scripts/t8/results/`, and builds this report from them.

## Reading the tables

- **Standard, T8**: criterion's median time, with the half-width of its 95% interval.
- **GB/s**: record bytes hashed per second, 1 GB = 10^9 bytes.
- **Speedup**: standard time over T8 time, so above 1 means T8 is faster.
- **Call ratio**: standard calls over T8 calls, what counting compressions alone predicts.

## Setup

**native build**

```
date: 2026-09-29T16:29:31+02:00
commit: e3cae250e17d9ab1a060aae2fd3b0762a32dff62
cpu: AMD Ryzen 9 9950X3D 16-Core Processor
kernel: 7.0.0-34-generic
rustc: rustc 1.98.0 (88d9e12ae 2026-08-18)
RUSTFLAGS: -Ctarget-cpu=native
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
| BLAKE3 | 256 B | 2^16 | 1.21 ms ±0.0% | 921 µs ±0.0% | 13.89 | 18.22 | **1.312x** | 1.333x |
| BLAKE3 | 480 B | 2^15 | 1.14 ms ±0.0% | 871 µs ±0.1% | 13.82 | 18.05 | **1.306x** | 1.333x |
| BLAKE3 | 65,664 B | 2^8 | 1.11 ms ±0.2% | 963 µs ±0.1% | 15.18 | 17.45 | **1.149x** | 1.240x |
| BLAKE2s | 256 B | 2^16 | 2.07 ms ±0.0% | 1.47 ms ±0.0% | 8.11 | 11.41 | **1.406x** | 1.333x |
| BLAKE2s | 480 B | 2^15 | 1.85 ms ±0.0% | 1.4 ms ±0.0% | 8.50 | 11.24 | **1.322x** | 1.333x |
| BLAKE2s | 65,664 B | 2^8 | 1.63 ms ±0.0% | 1.51 ms ±0.0% | 10.29 | 11.12 | **1.080x** | 1.167x |
| SHA-256 | 256 B | 2^16 | 2.22 ms ±0.0% | 1.68 ms ±0.0% | 7.56 | 9.98 | **1.321x** | 1.667x |
| SHA-256 | 480 B | 2^15 | 1.92 ms ±0.0% | 1.62 ms ±0.0% | 8.19 | 9.73 | **1.189x** | 1.333x |
| SHA-256 | 65,664 B | 2^8 | 1.9 ms ±0.0% | 1.74 ms ±0.0% | 8.86 | 9.68 | **1.093x** | 1.168x |

### Leaf hashing, one record, one core

Command: `taskset -c 4 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_leaf -- 'leaf/.*/single'`

- Code: `merkle-tree/benches/t8_leaf.rs`, one `hash_slice` call on one record, the path a verifier takes.
- Records: 256 B, 480 B and 65,664 B.
- Criterion: 0.5 s warm-up, 2 s measurement, 100 samples.

| Hash | Record | Standard | T8 | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 263 ns ±0.0% | 160 ns ±0.0% | **1.647x** | 1.333x |
| BLAKE3 | 480 B | 515 ns ±0.0% | 341 ns ±0.0% | **1.511x** | 1.333x |
| BLAKE3 | 65,664 B | 5.32 µs ±0.0% | 53 µs ±0.0% | **0.100x** | 1.240x |
| BLAKE2s | 256 B | 385 ns ±0.0% | 283 ns ±0.0% | **1.359x** | 1.333x |
| BLAKE2s | 480 B | 777 ns ±0.1% | 587 ns ±0.0% | **1.323x** | 1.333x |
| BLAKE2s | 65,664 B | 100 µs ±0.0% | 89.1 µs ±0.0% | **1.125x** | 1.167x |
| SHA-256 | 256 B | 130 ns ±0.1% | 69.9 ns ±0.0% | **1.868x** | 1.667x |
| SHA-256 | 480 B | 203 ns ±0.1% | 145 ns ±0.0% | **1.399x** | 1.333x |
| SHA-256 | 65,664 B | 24.2 µs ±0.0% | 21.8 µs ±0.1% | **1.110x** | 1.168x |

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
| BLAKE3 | 256 B | 2^16 | 1 | 1.92 ms ±1.1% | 1.64 ms ±0.1% | 8.74 | 10.20 | **1.168x** | 1.250x |
| BLAKE3 | 256 B | 2^20 | 1 | 42.5 ms ±4.7% | 40.6 ms ±4.9% | 6.31 | 6.61 | **1.047x** | 1.250x |
| BLAKE3 | 256 B | 2^16 | 32 | 642 µs ±5.6% | 617 µs ±3.5% | 26.13 | 27.17 | **1.040x** | 1.250x |
| BLAKE3 | 256 B | 2^20 | 32 | 14.3 ms ±8.1% | 13.4 ms ±3.9% | 18.74 | 20.02 | **1.069x** | 1.250x |
| BLAKE3 | 65,664 B | 2^10 | 1 | 4.4 ms ±0.5% | 3.88 ms ±0.3% | 15.30 | 17.32 | **1.132x** | 1.240x |
| BLAKE3 | 65,664 B | 2^10 | 32 | 725 µs ±3.4% | 607 µs ±4.3% | 92.80 | 110.72 | **1.193x** | 1.240x |
| BLAKE2s | 256 B | 2^16 | 1 | 2.54 ms ±0.1% | 1.98 ms ±0.1% | 6.61 | 8.45 | **1.278x** | 1.250x |
| BLAKE2s | 256 B | 2^20 | 1 | 60.3 ms ±0.1% | 56 ms ±4.1% | 4.45 | 4.79 | **1.077x** | 1.250x |
| BLAKE2s | 256 B | 2^16 | 32 | 669 µs ±14.0% | 657 µs ±4.8% | 25.09 | 25.52 | **1.017x** | 1.250x |
| BLAKE2s | 256 B | 2^20 | 32 | 13.1 ms ±9.1% | 13.5 ms ±3.0% | 20.42 | 19.81 | **0.970x** | 1.250x |
| BLAKE2s | 65,664 B | 2^10 | 1 | 6.86 ms ±1.7% | 6.31 ms ±0.3% | 9.81 | 10.66 | **1.087x** | 1.167x |
| BLAKE2s | 65,664 B | 2^10 | 32 | 857 µs ±4.0% | 786 µs ±2.2% | 78.48 | 85.59 | **1.091x** | 1.167x |
| SHA-256 | 256 B | 2^16 | 1 | 2.76 ms ±0.6% | 2.16 ms ±0.3% | 6.09 | 7.75 | **1.274x** | 1.500x |
| SHA-256 | 256 B | 2^20 | 1 | 62.9 ms ±0.3% | 54.5 ms ±6.5% | 4.26 | 4.92 | **1.154x** | 1.500x |
| SHA-256 | 256 B | 2^16 | 32 | 555 µs ±0.8% | 556 µs ±1.3% | 30.24 | 30.17 | **0.998x** | 1.500x |
| SHA-256 | 256 B | 2^20 | 32 | 12.4 ms ±2.1% | 12.4 ms ±1.0% | 21.70 | 21.57 | **0.994x** | 1.500x |
| SHA-256 | 65,664 B | 2^10 | 1 | 9.04 ms ±4.7% | 7.63 ms ±4.4% | 7.44 | 8.81 | **1.184x** | 1.168x |
| SHA-256 | 65,664 B | 2^10 | 32 | 831 µs ±2.1% | 792 µs ±0.8% | 80.96 | 84.92 | **1.049x** | 1.168x |

### Verification: one opening, one core

Command: `taskset -c 4 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_verify`

- Code: `merkle-tree/benches/t8_verify.rs`, `MerkleTreeMmcs::verify_batch` on prepared openings, cap height 0.
- Each sample verifies 64 openings at distinct random positions, the same for both columns.
- The table divides by 64, so it shows one opening: the record's leaf hash plus the path to the root.
- Trees: 2^10 and 2^20 records of 256 B, and 2^10 records of 65,664 B.
- Criterion: 0.5 s warm-up, 2 s measurement, 100 samples.

| Hash | Record | Records | Standard per opening | T8 per opening | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 2^10 | 1.25 µs ±0.0% | 1.15 µs ±0.4% | **1.082x** | 1.077x |
| BLAKE3 | 256 B | 2^20 | 2.19 µs ±0.1% | 2.1 µs ±0.3% | **1.043x** | 1.043x |
| BLAKE3 | 65,664 B | 2^10 | 6.67 µs ±0.0% | 54 µs ±0.1% | **0.123x** | 1.237x |
| BLAKE2s | 256 B | 2^10 | 1.78 µs ±1.3% | 1.63 µs ±0.2% | **1.095x** | 1.077x |
| BLAKE2s | 256 B | 2^20 | 3.13 µs ±1.7% | 3.13 µs ±0.3% | **1.000x** | 1.043x |
| BLAKE2s | 65,664 B | 2^10 | 102 µs ±0.0% | 90.5 µs ±0.0% | **1.125x** | 1.165x |
| SHA-256 | 256 B | 2^10 | 581 ns ±0.4% | 522 ns ±0.1% | **1.112x** | 1.154x |
| SHA-256 | 256 B | 2^20 | 991 ns ±0.3% | 922 ns ±0.1% | **1.075x** | 1.087x |
| SHA-256 | 65,664 B | 2^10 | 25.1 µs ±0.2% | 23.1 µs ±0.4% | **1.084x** | 1.166x |

## Notes

- BLAKE3 keeps T8's three roles apart with its counter and flags: the construction as analysed.
- BLAKE2s keeps them apart with counters 1, 2 and 3 and the final flag clear, which no plain BLAKE2s call uses.
- SHA-256's compression takes exactly 96 bytes, so its three calls are one function. The security analysis does not cover that instantiation.
- Keccak-256 absorbs 136 bytes per permutation, so T8 costs more calls than the plain hash and is not measured.
- The build targets the host CPU, so all three hashes batch on AVX-512.
- Both leaves share the transpose of every record into vector lanes, which the call count does not see.
- Batched leaves read every record once, 64 bytes at a time, exactly as the standard leaf does.
- So the transpose into vector lanes costs both leaves the same, and dilutes the call saving a little.
- On one thread, commitment follows the call ratio.
- On 32 threads, trees of 256-byte records wait on memory, so fewer calls barely shows.
- Trees of 64 KiB-class records stay compute-bound on 32 threads, and keep most of the call saving.
- A single long BLAKE3 record hashes its chunks in parallel; T8's chained stages cannot, hence the 0.1x rows.
