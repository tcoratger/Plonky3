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
date: 2026-09-29T16:11:06+02:00
commit: d8978e6653000473480bf5742384a77a232b6037 + local changes
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
| BLAKE3 | 256 B | 2^16 | 1.24 ms ±0.0% | 925 µs ±0.0% | 13.54 | 18.14 | **1.340x** | 1.333x |
| BLAKE3 | 480 B | 2^15 | 1.18 ms ±0.0% | 850 µs ±0.0% | 13.34 | 18.51 | **1.388x** | 1.333x |
| BLAKE3 | 65,664 B | 2^8 | 1.06 ms ±0.0% | 931 µs ±0.0% | 15.81 | 18.06 | **1.142x** | 1.240x |
| BLAKE2s | 256 B | 2^16 | 2.04 ms ±0.0% | 1.43 ms ±0.1% | 8.23 | 11.77 | **1.430x** | 1.333x |
| BLAKE2s | 480 B | 2^15 | 1.84 ms ±0.0% | 1.4 ms ±0.0% | 8.56 | 11.20 | **1.307x** | 1.333x |
| BLAKE2s | 65,664 B | 2^8 | 1.57 ms ±0.0% | 1.48 ms ±0.0% | 10.71 | 11.35 | **1.060x** | 1.167x |
| SHA-256 | 256 B | 2^16 | 2.24 ms ±0.0% | 1.67 ms ±0.1% | 7.49 | 10.06 | **1.342x** | 1.667x |
| SHA-256 | 480 B | 2^15 | 1.92 ms ±0.0% | 1.61 ms ±0.0% | 8.21 | 9.78 | **1.192x** | 1.333x |
| SHA-256 | 65,664 B | 2^8 | 1.83 ms ±0.1% | 1.71 ms ±0.0% | 9.19 | 9.83 | **1.070x** | 1.168x |

### Leaf hashing, one record, one core

Command: `taskset -c 4 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_leaf -- 'leaf/.*/single'`

- Code: `merkle-tree/benches/t8_leaf.rs`, one `hash_slice` call on one record, the path a verifier takes.
- Records: 256 B, 480 B and 65,664 B.
- Criterion: 0.5 s warm-up, 2 s measurement, 100 samples.

| Hash | Record | Standard | T8 | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 267 ns ±1.3% | 160 ns ±0.0% | **1.672x** | 1.333x |
| BLAKE3 | 480 B | 518 ns ±0.1% | 341 ns ±0.0% | **1.521x** | 1.333x |
| BLAKE3 | 65,664 B | 5.31 µs ±0.0% | 53.2 µs ±0.2% | **0.100x** | 1.240x |
| BLAKE2s | 256 B | 386 ns ±0.0% | 284 ns ±0.6% | **1.359x** | 1.333x |
| BLAKE2s | 480 B | 786 ns ±0.1% | 591 ns ±0.3% | **1.329x** | 1.333x |
| BLAKE2s | 65,664 B | 100 µs ±0.0% | 90 µs ±0.0% | **1.114x** | 1.167x |
| SHA-256 | 256 B | 131 ns ±0.2% | 69.7 ns ±0.0% | **1.881x** | 1.667x |
| SHA-256 | 480 B | 204 ns ±1.7% | 145 ns ±0.0% | **1.407x** | 1.333x |
| SHA-256 | 65,664 B | 24.2 µs ±0.0% | 21.8 µs ±0.0% | **1.111x** | 1.168x |

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
| BLAKE3 | 256 B | 2^16 | 1 | 2.09 ms ±0.0% | 1.78 ms ±0.1% | 8.04 | 9.43 | **1.173x** | 1.250x |
| BLAKE3 | 256 B | 2^20 | 1 | 40.3 ms ±5.0% | 34.1 ms ±6.0% | 6.66 | 7.88 | **1.182x** | 1.250x |
| BLAKE3 | 256 B | 2^16 | 32 | 543 µs ±0.7% | 546 µs ±1.8% | 30.88 | 30.73 | **0.995x** | 1.250x |
| BLAKE3 | 256 B | 2^20 | 32 | 11.4 ms ±1.3% | 11.7 ms ±1.3% | 23.49 | 23.04 | **0.981x** | 1.250x |
| BLAKE3 | 65,664 B | 2^10 | 1 | 5.86 ms ±0.1% | 4.8 ms ±0.3% | 11.48 | 14.00 | **1.220x** | 1.240x |
| BLAKE3 | 65,664 B | 2^10 | 32 | 2.04 ms ±4.4% | 2.19 ms ±5.4% | 32.89 | 30.71 | **0.934x** | 1.240x |
| BLAKE2s | 256 B | 2^16 | 1 | 2.69 ms ±0.0% | 2.11 ms ±0.1% | 6.25 | 7.94 | **1.271x** | 1.250x |
| BLAKE2s | 256 B | 2^20 | 1 | 53.6 ms ±0.6% | 46.1 ms ±0.4% | 5.01 | 5.83 | **1.164x** | 1.250x |
| BLAKE2s | 256 B | 2^16 | 32 | 574 µs ±1.9% | 558 µs ±1.5% | 29.24 | 30.05 | **1.028x** | 1.250x |
| BLAKE2s | 256 B | 2^20 | 32 | 11.6 ms ±1.0% | 11.4 ms ±1.2% | 23.24 | 23.55 | **1.013x** | 1.250x |
| BLAKE2s | 65,664 B | 2^10 | 1 | 7.69 ms ±0.1% | 7.33 ms ±0.3% | 8.75 | 9.18 | **1.049x** | 1.167x |
| BLAKE2s | 65,664 B | 2^10 | 32 | 2.02 ms ±1.4% | 2.04 ms ±1.7% | 33.22 | 32.94 | **0.992x** | 1.167x |
| SHA-256 | 256 B | 2^16 | 1 | 2.86 ms ±0.1% | 2.27 ms ±0.0% | 5.86 | 7.39 | **1.260x** | 1.500x |
| SHA-256 | 256 B | 2^20 | 1 | 57.2 ms ±0.1% | 48.9 ms ±0.2% | 4.69 | 5.49 | **1.171x** | 1.500x |
| SHA-256 | 256 B | 2^16 | 32 | 557 µs ±1.9% | 573 µs ±1.6% | 30.12 | 29.27 | **0.972x** | 1.500x |
| SHA-256 | 256 B | 2^20 | 32 | 12.1 ms ±1.8% | 12.2 ms ±3.0% | 22.22 | 22.00 | **0.990x** | 1.500x |
| SHA-256 | 65,664 B | 2^10 | 1 | 8.86 ms ±0.4% | 8.22 ms ±0.2% | 7.59 | 8.18 | **1.078x** | 1.168x |
| SHA-256 | 65,664 B | 2^10 | 32 | 2.13 ms ±3.6% | 2.23 ms ±2.4% | 31.56 | 30.10 | **0.954x** | 1.168x |

### Verification: one opening, one core

Command: `taskset -c 4 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_verify`

- Code: `merkle-tree/benches/t8_verify.rs`, `MerkleTreeMmcs::verify_batch` on prepared openings, cap height 0.
- Each sample verifies 64 openings at distinct random positions, the same for both columns.
- The table divides by 64, so it shows one opening: the record's leaf hash plus the path to the root.
- Trees: 2^10 and 2^20 records of 256 B, and 2^10 records of 65,664 B.
- Criterion: 0.5 s warm-up, 2 s measurement, 100 samples.

| Hash | Record | Records | Standard per opening | T8 per opening | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 2^10 | 1.24 µs ±0.4% | 1.17 µs ±0.3% | **1.057x** | 1.077x |
| BLAKE3 | 256 B | 2^20 | 2.16 µs ±0.4% | 2.11 µs ±0.1% | **1.024x** | 1.043x |
| BLAKE3 | 65,664 B | 2^10 | 6.49 µs ±0.1% | 54.1 µs ±0.0% | **0.120x** | 1.237x |
| BLAKE2s | 256 B | 2^10 | 1.7 µs ±0.2% | 1.6 µs ±0.1% | **1.060x** | 1.077x |
| BLAKE2s | 256 B | 2^20 | 2.96 µs ±0.2% | 2.88 µs ±0.1% | **1.027x** | 1.043x |
| BLAKE2s | 65,664 B | 2^10 | 102 µs ±0.0% | 90.4 µs ±0.0% | **1.123x** | 1.165x |
| SHA-256 | 256 B | 2^10 | 578 ns ±0.1% | 520 ns ±0.2% | **1.111x** | 1.154x |
| SHA-256 | 256 B | 2^20 | 991 ns ±0.1% | 916 ns ±0.1% | **1.082x** | 1.087x |
| SHA-256 | 65,664 B | 2^10 | 24.7 µs ±0.0% | 22.8 µs ±0.2% | **1.083x** | 1.166x |

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
- On 32 threads it does not: every thread waits on memory, so fewer calls barely shows.
- On this Plonky3 commit the tree still copies each row before hashing it, which doubles that memory traffic.
- A single long BLAKE3 record hashes its chunks in parallel; T8's chained stages cannot, hence the 0.1x rows.
