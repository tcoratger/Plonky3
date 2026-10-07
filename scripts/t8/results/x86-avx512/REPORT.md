# T8 leaves on Plonky3

Each hash's standard Merkle leaf against its T8 leaf, on the same compression kernel, inside an unchanged Plonky3 tree.

## How to reproduce

From the repository root, on the `t8-leaves` branch:

```sh
# Run all benchmark suites, then generate this report.
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
- Criterion prints `time: [low estimate high]`; this report reads the median and its 95% interval from the saved estimates.
- The script keeps each estimate in `target/bench-native/criterion/<group>/<scheme>/<size>/native/estimates.json`.
- The script saves them as the baseline `native`, copies them to `scripts/t8/results/x86-avx512/`, and builds this report from them.

## Reading the tables

- **Standard, T8**: criterion's median time, with the half-width of its 95% interval.
- **GB/s**: record bytes hashed per second, 1 GB = 10^9 bytes.
- **Speedup**: standard time over T8 time, so above 1 means T8 is faster.
- **Call ratio**: standard calls over T8 calls, what counting compressions alone predicts.

## Setup

**native build**

```
date: 2026-10-07T12:08:14+02:00
commit: c4e1531e43625626997f920a7e0ee389dfa4a047
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
| BLAKE3 | 256 B | 2^16 | 1.2 ms ±0.1% | 928 µs ±0.4% | 13.97 | 18.09 | **1.294x** | 1.333x |
| BLAKE3 | 480 B | 2^15 | 1.15 ms ±0.0% | 841 µs ±0.1% | 13.71 | 18.69 | **1.364x** | 1.333x |
| BLAKE3 | 65,664 B | 2^8 | 1.03 ms ±0.1% | 952 µs ±0.1% | 16.25 | 17.66 | **1.087x** | 1.240x |
| BLAKE2s | 256 B | 2^16 | 2.03 ms ±0.0% | 1.47 ms ±0.0% | 8.27 | 11.39 | **1.377x** | 1.333x |
| BLAKE2s | 480 B | 2^15 | 1.83 ms ±0.0% | 1.39 ms ±0.0% | 8.60 | 11.33 | **1.317x** | 1.333x |
| BLAKE2s | 65,664 B | 2^8 | 1.61 ms ±0.0% | 1.5 ms ±0.0% | 10.41 | 11.24 | **1.080x** | 1.167x |
| SHA-256 | 256 B | 2^16 | 2.11 ms ±0.1% | 1.66 ms ±0.0% | 7.95 | 10.10 | **1.270x** | 1.667x |
| SHA-256 | 480 B | 2^15 | 1.89 ms ±0.0% | 1.6 ms ±0.0% | 8.33 | 9.80 | **1.176x** | 1.333x |
| SHA-256 | 65,664 B | 2^8 | 1.8 ms ±0.0% | 1.75 ms ±0.1% | 9.34 | 9.62 | **1.030x** | 1.168x |
| SHA-256, T253 | 253 B | 2^16 | 2.28 ms ±0.1% | 1.67 ms ±0.0% | 7.28 | 9.92 | **1.362x** | 1.667x |
| SHA-256, T253 | 474 B | 2^15 | 1.91 ms ±0.0% | 1.63 ms ±0.0% | 8.12 | 9.53 | **1.174x** | 1.333x |
| SHA-256, T253 | 65,448 B | 2^8 | 1.79 ms ±0.0% | 1.78 ms ±0.0% | 9.34 | 9.39 | **1.005x** | 1.152x |

### Leaf hashing, one record, one core

Command: `taskset -c 4 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_leaf -- 'leaf/.*/single'`

- Code: `merkle-tree/benches/t8_leaf.rs`, one `hash_slice` call on one record, the path a verifier takes.
- Records: 256 B, 480 B and 65,664 B.
- Criterion: 0.5 s warm-up, 2 s measurement, 100 samples.

| Hash | Record | Standard | T8 | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 262 ns ±0.0% | 111 ns ±0.0% | **2.359x** | 1.333x |
| BLAKE3 | 480 B | 514 ns ±0.0% | 244 ns ±0.0% | **2.108x** | 1.333x |
| BLAKE3 | 65,664 B | 5.24 µs ±0.0% | 38.8 µs ±0.0% | **0.135x** | 1.240x |
| BLAKE2s | 256 B | 384 ns ±0.0% | 283 ns ±0.0% | **1.359x** | 1.333x |
| BLAKE2s | 480 B | 774 ns ±0.1% | 586 ns ±0.0% | **1.321x** | 1.333x |
| BLAKE2s | 65,664 B | 100 µs ±0.0% | 88.9 µs ±0.0% | **1.125x** | 1.167x |
| SHA-256 | 256 B | 130 ns ±0.2% | 68.8 ns ±0.0% | **1.894x** | 1.667x |
| SHA-256 | 480 B | 203 ns ±0.0% | 137 ns ±0.0% | **1.479x** | 1.333x |
| SHA-256 | 65,664 B | 24.2 µs ±0.0% | 20 µs ±0.0% | **1.208x** | 1.168x |
| SHA-256, T253 | 253 B | 142 ns ±0.1% | 72.2 ns ±0.1% | **1.961x** | 1.667x |
| SHA-256, T253 | 474 B | 203 ns ±0.0% | 157 ns ±0.2% | **1.293x** | 1.333x |
| SHA-256, T253 | 65,448 B | 24.1 µs ±0.0% | 25.2 µs ±0.2% | **0.957x** | 1.152x |

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
| BLAKE3 | 256 B | 2^16 | 1 | 1.8 ms ±0.0% | 1.58 ms ±0.1% | 9.32 | 10.60 | **1.137x** | 1.250x |
| BLAKE3 | 256 B | 2^20 | 1 | 43.3 ms ±0.1% | 39.1 ms ±3.3% | 6.20 | 6.87 | **1.107x** | 1.250x |
| BLAKE3 | 256 B | 2^16 | 32 | 521 µs ±0.6% | 519 µs ±0.9% | 32.22 | 32.33 | **1.003x** | 1.250x |
| BLAKE3 | 256 B | 2^20 | 32 | 11.6 ms ±0.6% | 11.6 ms ±1.6% | 23.20 | 23.20 | **1.000x** | 1.250x |
| BLAKE3 | 65,664 B | 2^10 | 1 | 4.36 ms ±0.5% | 3.78 ms ±1.6% | 15.44 | 17.78 | **1.152x** | 1.240x |
| BLAKE3 | 65,664 B | 2^10 | 32 | 633 µs ±0.6% | 595 µs ±0.9% | 106.25 | 112.93 | **1.063x** | 1.240x |
| BLAKE2s | 256 B | 2^16 | 1 | 2.52 ms ±0.0% | 1.98 ms ±0.0% | 6.65 | 8.47 | **1.274x** | 1.250x |
| BLAKE2s | 256 B | 2^20 | 1 | 59.4 ms ±0.1% | 51 ms ±0.2% | 4.52 | 5.27 | **1.166x** | 1.250x |
| BLAKE2s | 256 B | 2^16 | 32 | 535 µs ±0.5% | 538 µs ±3.7% | 31.34 | 31.20 | **0.995x** | 1.250x |
| BLAKE2s | 256 B | 2^20 | 32 | 11.4 ms ±1.8% | 11.9 ms ±0.9% | 23.46 | 22.49 | **0.959x** | 1.250x |
| BLAKE2s | 65,664 B | 2^10 | 1 | 6.55 ms ±0.0% | 6.09 ms ±0.0% | 10.26 | 11.03 | **1.075x** | 1.167x |
| BLAKE2s | 65,664 B | 2^10 | 32 | 786 µs ±1.4% | 728 µs ±1.8% | 85.51 | 92.37 | **1.080x** | 1.167x |
| SHA-256 | 256 B | 2^16 | 1 | 2.57 ms ±0.1% | 2.12 ms ±0.0% | 6.52 | 7.91 | **1.213x** | 1.500x |
| SHA-256 | 256 B | 2^20 | 1 | 60.7 ms ±0.2% | 53.6 ms ±0.2% | 4.42 | 5.01 | **1.132x** | 1.500x |
| SHA-256 | 256 B | 2^16 | 32 | 542 µs ±1.8% | 542 µs ±0.5% | 30.93 | 30.93 | **1.000x** | 1.500x |
| SHA-256 | 256 B | 2^20 | 32 | 12 ms ±1.1% | 12.1 ms ±3.0% | 22.36 | 22.21 | **0.993x** | 1.500x |
| SHA-256 | 65,664 B | 2^10 | 1 | 7.32 ms ±0.0% | 7.13 ms ±0.1% | 9.18 | 9.43 | **1.027x** | 1.168x |
| SHA-256 | 65,664 B | 2^10 | 32 | 789 µs ±1.1% | 761 µs ±1.7% | 85.25 | 88.33 | **1.036x** | 1.168x |
| SHA-256, T253 | 253 B | 2^16 | 1 | 2.76 ms ±0.1% | 2.16 ms ±0.2% | 6.01 | 7.69 | **1.280x** | 1.500x |
| SHA-256, T253 | 253 B | 2^20 | 1 | 62.9 ms ±0.2% | 56.8 ms ±0.0% | 4.22 | 4.67 | **1.106x** | 1.500x |
| SHA-256, T253 | 253 B | 2^16 | 32 | 543 µs ±0.7% | 540 µs ±0.6% | 30.55 | 30.68 | **1.004x** | 1.500x |
| SHA-256, T253 | 253 B | 2^20 | 32 | 12.1 ms ±1.9% | 12.1 ms ±2.0% | 21.93 | 21.95 | **1.001x** | 1.500x |
| SHA-256, T253 | 65,448 B | 2^10 | 1 | 7.33 ms ±0.1% | 7.31 ms ±0.0% | 9.14 | 9.17 | **1.004x** | 1.152x |
| SHA-256, T253 | 65,448 B | 2^10 | 32 | 758 µs ±1.5% | 775 µs ±0.7% | 88.42 | 86.47 | **0.978x** | 1.152x |

### Verification: one opening, one core

Command: `taskset -c 4 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_verify`

- Code: `merkle-tree/benches/t8_verify.rs`, `MerkleTreeMmcs::verify_batch` on prepared openings, cap height 0.
- Each sample verifies 64 openings at distinct random positions, the same for both columns.
- The table divides by 64, so it shows one opening: the record's leaf hash plus the path to the root.
- Trees: 2^10 and 2^20 records of 256 B, and 2^10 records of 65,664 B.
- Criterion: 0.5 s warm-up, 2 s measurement, 100 samples.

| Hash | Record | Records | Standard per opening | T8 per opening | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 2^10 | 1.15 µs ±0.1% | 1.02 µs ±0.2% | **1.125x** | 1.077x |
| BLAKE3 | 256 B | 2^20 | 1.98 µs ±0.1% | 1.83 µs ±0.2% | **1.078x** | 1.043x |
| BLAKE3 | 65,664 B | 2^10 | 6.33 µs ±0.0% | 39.8 µs ±0.0% | **0.159x** | 1.237x |
| BLAKE2s | 256 B | 2^10 | 1.64 µs ±0.1% | 1.52 µs ±0.0% | **1.082x** | 1.077x |
| BLAKE2s | 256 B | 2^20 | 2.78 µs ±0.2% | 2.66 µs ±0.0% | **1.047x** | 1.043x |
| BLAKE2s | 65,664 B | 2^10 | 101 µs ±0.0% | 90.2 µs ±0.0% | **1.123x** | 1.165x |
| SHA-256 | 256 B | 2^10 | 648 ns ±0.2% | 590 ns ±1.4% | **1.098x** | 1.154x |
| SHA-256 | 256 B | 2^20 | 1.07 µs ±0.1% | 989 ns ±0.1% | **1.078x** | 1.087x |
| SHA-256 | 65,664 B | 2^10 | 24.8 µs ±0.0% | 20.6 µs ±0.0% | **1.201x** | 1.166x |
| SHA-256, T253 | 253 B | 2^10 | 646 ns ±0.2% | 580 ns ±0.2% | **1.114x** | 1.154x |
| SHA-256, T253 | 253 B | 2^20 | 1.07 µs ±0.4% | 1.01 µs ±0.5% | **1.060x** | 1.087x |
| SHA-256, T253 | 65,448 B | 2^10 | 24.6 µs ±0.0% | 25.7 µs ±0.1% | **0.960x** | 1.150x |

## Wider tree nodes

Command: `RAYON_NUM_THREADS=1 taskset -c 4 cargo bench -p p3-merkle-tree --features parallel --profile optimized --bench t8_arity`, then again on every thread.

- Code: `merkle-tree/benches/t8_arity.rs`, the same Plonky3 tree with arity 2 and arity 4.
- Leaves: the standard hash of 256-byte records, identical in both columns.
- A 4-ary Keccak-256 or SHA3-256 node fits in one permutation; a 4-ary BLAKE3 node takes two compressions.
- Verification times one opening; each sample verifies 64.
- Proof digests: siblings in one opening, 32 bytes each.

| Hash | Records | Operation | Binary | 4-ary | Speedup | Calls, binary / 4-ary | Proof digests, binary / 4-ary |
|---|---:|---|---:|---:|---:|---:|---:|
| Keccak-256 | 2^16 | commit, 1 thread | 5.44 ms | 3.94 ms | **1.381x** | 196,607 / 152,917 | 16 / 24 |
| Keccak-256 | 2^16 | commit, 32 threads | 592 µs | 500 µs | **1.184x** | 196,607 / 152,917 | 16 / 24 |
| Keccak-256 | 2^16 | verify | 7.32 µs | 4.08 µs | **1.795x** | 18 / 10 | 16 / 24 |
| Keccak-256 | 2^20 | commit, 1 thread | 119 ms | 101 ms | **1.171x** | 3,145,727 / 2,446,677 | 20 / 30 |
| Keccak-256 | 2^20 | commit, 32 threads | 11.4 ms | 9.73 ms | **1.176x** | 3,145,727 / 2,446,677 | 20 / 30 |
| Keccak-256 | 2^20 | verify | 8.95 µs | 4.95 µs | **1.809x** | 22 / 12 | 20 / 30 |
| SHA3-256 | 2^16 | commit, 1 thread | 5.04 ms | 3.95 ms | **1.276x** | 196,607 / 152,917 | 16 / 24 |
| SHA3-256 | 2^16 | commit, 32 threads | 586 µs | 482 µs | **1.216x** | 196,607 / 152,917 | 16 / 24 |
| SHA3-256 | 2^16 | verify | 7.31 µs | 4.07 µs | **1.795x** | 18 / 10 | 16 / 24 |
| SHA3-256 | 2^20 | commit, 1 thread | 119 ms | 102 ms | **1.173x** | 3,145,727 / 2,446,677 | 20 / 30 |
| SHA3-256 | 2^20 | commit, 32 threads | 12.2 ms | 10.5 ms | **1.165x** | 3,145,727 / 2,446,677 | 20 / 30 |
| SHA3-256 | 2^20 | verify | 8.91 µs | 4.91 µs | **1.815x** | 22 / 12 | 20 / 30 |
| BLAKE3 | 2^16 | commit, 1 thread | 2.3 ms | 2.49 ms | **0.924x** | 327,679 / 305,834 | 16 / 24 |
| BLAKE3 | 2^16 | commit, 32 threads | 555 µs | 494 µs | **1.124x** | 327,679 / 305,834 | 16 / 24 |
| BLAKE3 | 2^16 | verify | 1.63 µs | 1.5 µs | **1.088x** | 20 / 20 | 16 / 24 |
| BLAKE3 | 2^20 | commit, 1 thread | 55.8 ms | 57.1 ms | **0.977x** | 5,242,879 / 4,893,354 | 20 / 30 |
| BLAKE3 | 2^20 | commit, 32 threads | 12.2 ms | 10.1 ms | **1.206x** | 5,242,879 / 4,893,354 | 20 / 30 |
| BLAKE3 | 2^20 | verify | 1.98 µs | 1.83 µs | **1.083x** | 24 / 24 | 20 / 30 |

## Notes

- BLAKE3 keeps T8's three roles apart with its counter and flags: the construction as analysed.
- BLAKE2s keeps them apart with counters 1, 2 and 3 and the final flag clear, which no plain BLAKE2s call uses.
- SHA-256's compression takes exactly 96 bytes, so T8's three calls there are one function. The security analysis does not cover that instantiation.
- T253 is T8 on SHA-256 with a role byte at the top of each call's chaining value, so its three calls are distinct functions.
- The tree's node hash starts from the SHA-256 initial value, whose top byte is 0x6a, so no role meets it either.
- The price is one byte per tagged block: records of 32 + 221 k bytes, compared with plain SHA-256 on the same bytes.
- Keccak-256 absorbs 136 bytes per permutation, so T8 costs more calls than the plain hash and is not measured.
- The build targets the host CPU; full batches of all three hashes use AVX-512.
- Standard hashes use the current main branch kernels and batch scheduling; T8 and T253 use the adapted research-branch drivers.
- BLAKE3, BLAKE2s and SHA-256 T8 batches load 64-byte rows and transpose them into vector lanes.
- T253 uses overlapping rows to load its tagged 31-byte fields.
- Loads and transposes add work beyond the compression calls counted in the tables.
- A single long BLAKE3 record can hash independent chunks in parallel; T8 must chain its stages.
- The host uses the powersave governor with boost enabled; small timing differences should be read alongside the reported intervals.
