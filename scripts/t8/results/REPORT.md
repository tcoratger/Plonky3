# T8 leaves on Plonky3

Each hash's standard leaf against T8 on the same compression kernel, inside an unchanged Plonky3 tree.

Times are criterion medians with the half-width of their 95% interval.

Speedup is standard time over T8 time; the call ratio is what counting compressions predicts.

## Setup

**native build**

```
date: 2026-09-29T15:02:40+02:00
plonky3: e1b98e91f64979f63a26f172d041f088e8e49274 + the uncommitted T8 files
cpu: AMD Ryzen 9 9950X3D 16-Core Processor
kernel: 7.0.0-34-generic
rustc: rustc 1.98.0 (88d9e12ae 2026-08-18)
RUSTFLAGS: -Ctarget-cpu=native
governor: powersave
boost: 1
```

## Compression calls per record

Counted, not timed. A tree of N records adds the same N - 1 node calls to both leaves.

| Hash | Record | Standard calls | T8 calls | Reduction |
|---|---:|---:|---:|---:|
| BLAKE3 | 256 B | 4 | 3 | +25.0% |
| BLAKE3 | 480 B | 8 | 6 | +25.0% |
| BLAKE3 | 65,664 B | 1,090 | 879 | +19.4% |
| BLAKE2s | 256 B | 4 | 3 | +25.0% |
| BLAKE2s | 480 B | 8 | 6 | +25.0% |
| BLAKE2s | 65,664 B | 1,026 | 879 | +14.3% |
| SHA-256 | 256 B | 5 | 3 | +40.0% |
| SHA-256 | 480 B | 8 | 6 | +25.0% |
| SHA-256 | 65,664 B | 1,027 | 879 | +14.4% |
| Keccak-256 | 256 B | 2 | 3 | -50.0% |
| Keccak-256 | 480 B | 4 | 6 | -50.0% |
| Keccak-256 | 65,664 B | 483 | 879 | -82.0% |

## Measured, native build

**Leaf hashing, batched, one core** (GiB/s counts record bytes; 1 GiB/s = 1.074 B/ns)

| Hash | Record | Records | Standard | T8 | Standard GiB/s | T8 GiB/s | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 2^16 | 1.21 ms ±0.0% | 1.01 ms ±0.1% | 13.86 | 16.54 | **1.194x** | 1.333x |
| BLAKE3 | 480 B | 2^15 | 1.17 ms ±0.0% | 965 µs ±0.0% | 13.46 | 16.31 | **1.211x** | 1.333x |
| BLAKE3 | 65,664 B | 2^8 | 1.07 ms ±0.0% | 1.04 ms ±0.0% | 15.69 | 16.14 | **1.028x** | 1.240x |
| BLAKE2s | 256 B | 2^16 | 2.03 ms ±0.1% | 1.76 ms ±0.0% | 8.25 | 9.54 | **1.156x** | 1.333x |
| BLAKE2s | 480 B | 2^15 | 1.84 ms ±0.0% | 1.76 ms ±0.0% | 8.57 | 8.93 | **1.042x** | 1.333x |
| BLAKE2s | 65,664 B | 2^8 | 1.56 ms ±0.0% | 1.92 ms ±0.0% | 10.74 | 8.76 | **0.815x** | 1.167x |
| SHA-256 | 256 B | 2^16 | 2.2 ms ±0.0% | 1.84 ms ±0.0% | 7.62 | 9.10 | **1.194x** | 1.667x |
| SHA-256 | 480 B | 2^15 | 1.91 ms ±0.0% | 1.81 ms ±0.0% | 8.25 | 8.70 | **1.054x** | 1.333x |
| SHA-256 | 65,664 B | 2^8 | 1.89 ms ±0.0% | 2 ms ±0.1% | 8.90 | 8.40 | **0.943x** | 1.168x |

**Leaf hashing, one record, one core** (the verifier's leaf)

| Hash | Record | Standard | T8 | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 263 ns ±0.0% | 160 ns ±0.0% | **1.647x** | 1.333x |
| BLAKE3 | 480 B | 515 ns ±0.0% | 341 ns ±0.0% | **1.511x** | 1.333x |
| BLAKE3 | 65,664 B | 5.3 µs ±1.3% | 53 µs ±0.1% | **0.100x** | 1.240x |
| BLAKE2s | 256 B | 382 ns ±0.0% | 283 ns ±0.0% | **1.348x** | 1.333x |
| BLAKE2s | 480 B | 783 ns ±0.0% | 587 ns ±0.0% | **1.333x** | 1.333x |
| BLAKE2s | 65,664 B | 100 µs ±0.1% | 89 µs ±0.0% | **1.126x** | 1.167x |
| SHA-256 | 256 B | 130 ns ±0.0% | 69.7 ns ±0.0% | **1.868x** | 1.667x |
| SHA-256 | 480 B | 203 ns ±0.1% | 145 ns ±0.0% | **1.401x** | 1.333x |
| SHA-256 | 65,664 B | 24.2 µs ±0.0% | 21.8 µs ±0.0% | **1.112x** | 1.168x |

**Commitment: one full tree**

| Hash | Record | Records | Threads | Standard | T8 | Standard GiB/s | T8 GiB/s | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 2^16 | 1 | 2.01 ms ±0.3% | 1.97 ms ±0.1% | 8.33 | 8.51 | **1.021x** | 1.250x |
| BLAKE3 | 256 B | 2^20 | 1 | 37.3 ms ±5.1% | 40.1 ms ±4.7% | 7.19 | 6.70 | **0.932x** | 1.250x |
| BLAKE3 | 256 B | 2^16 | 32 | 609 µs ±26.0% | 592 µs ±3.5% | 27.53 | 28.36 | **1.030x** | 1.250x |
| BLAKE3 | 256 B | 2^20 | 32 | 12.8 ms ±3.9% | 11.4 ms ±14.2% | 21.03 | 23.50 | **1.118x** | 1.250x |
| BLAKE3 | 65,664 B | 2^10 | 1 | 5.34 ms ±2.4% | 5.19 ms ±0.2% | 12.59 | 12.95 | **1.028x** | 1.240x |
| BLAKE3 | 65,664 B | 2^10 | 32 | 2.94 ms ±17.8% | 1.96 ms ±5.1% | 22.89 | 34.26 | **1.497x** | 1.240x |
| BLAKE2s | 256 B | 2^16 | 1 | 2.68 ms ±0.0% | 2.5 ms ±0.0% | 6.26 | 6.71 | **1.072x** | 1.250x |
| BLAKE2s | 256 B | 2^20 | 1 | 53.4 ms ±0.8% | 51.3 ms ±0.3% | 5.03 | 5.23 | **1.041x** | 1.250x |
| BLAKE2s | 256 B | 2^16 | 32 | 773 µs ±35.2% | 783 µs ±13.1% | 21.71 | 21.42 | **0.987x** | 1.250x |
| BLAKE2s | 256 B | 2^20 | 32 | 15.9 ms ±20.8% | 12 ms ±2.9% | 16.91 | 22.32 | **1.320x** | 1.250x |
| BLAKE2s | 65,664 B | 2^10 | 1 | 7.77 ms ±0.4% | 9.04 ms ±1.2% | 8.66 | 7.44 | **0.859x** | 1.167x |
| BLAKE2s | 65,664 B | 2^10 | 32 | 2.67 ms ±11.2% | 2.77 ms ±10.4% | 25.19 | 24.27 | **0.964x** | 1.167x |
| SHA-256 | 256 B | 2^16 | 1 | 2.85 ms ±0.0% | 2.5 ms ±0.1% | 5.88 | 6.70 | **1.139x** | 1.500x |
| SHA-256 | 256 B | 2^20 | 1 | 58.6 ms ±14.6% | 54.1 ms ±2.0% | 4.58 | 4.96 | **1.083x** | 1.500x |
| SHA-256 | 256 B | 2^16 | 32 | 671 µs ±4.6% | 738 µs ±5.4% | 25.02 | 22.73 | **0.908x** | 1.500x |
| SHA-256 | 256 B | 2^20 | 32 | 12.4 ms ±4.5% | 12.1 ms ±1.5% | 21.63 | 22.18 | **1.025x** | 1.500x |
| SHA-256 | 65,664 B | 2^10 | 1 | 13.4 ms ±17.4% | 9.98 ms ±5.8% | 5.01 | 6.74 | **1.346x** | 1.168x |
| SHA-256 | 65,664 B | 2^10 | 32 | 2.12 ms ±2.9% | 2.16 ms ±9.0% | 31.65 | 31.13 | **0.984x** | 1.168x |

**Verification: one opening, one core** (64 openings per sample)

| Hash | Record | Records | Standard per opening | T8 per opening | Speedup | Call ratio |
|---|---:|---:|---:|---:|---:|---:|
| BLAKE3 | 256 B | 2^10 | 1.25 µs ±0.0% | 1.16 µs ±0.0% | **1.077x** | 1.077x |
| BLAKE3 | 256 B | 2^20 | 2.21 µs ±0.5% | 2.12 µs ±0.1% | **1.044x** | 1.043x |
| BLAKE3 | 65,664 B | 2^10 | 6.48 µs ±0.1% | 54 µs ±0.0% | **0.120x** | 1.237x |
| BLAKE2s | 256 B | 2^10 | 1.7 µs ±0.1% | 1.62 µs ±0.3% | **1.052x** | 1.077x |
| BLAKE2s | 256 B | 2^20 | 2.97 µs ±0.0% | 2.88 µs ±0.4% | **1.031x** | 1.043x |
| BLAKE2s | 65,664 B | 2^10 | 102 µs ±0.0% | 90.4 µs ±0.0% | **1.124x** | 1.165x |
| SHA-256 | 256 B | 2^10 | 578 ns ±0.0% | 522 ns ±0.1% | **1.108x** | 1.154x |
| SHA-256 | 256 B | 2^20 | 995 ns ±0.5% | 915 ns ±0.1% | **1.087x** | 1.087x |
| SHA-256 | 65,664 B | 2^10 | 24.7 µs ±0.0% | 22.8 µs ±0.2% | **1.083x** | 1.166x |

## Reading the numbers

- BLAKE3 keeps T8's three roles apart with its counter and flags: the construction as analysed.
- BLAKE2s keeps them apart with counters 1, 2 and 3 and the final flag clear, which no plain BLAKE2s call uses.
- SHA-256's compression takes exactly 96 bytes, so its three calls are one function. The security analysis does not cover that instantiation.
- Keccak-256 absorbs 136 bytes per permutation, so T8 costs more calls than the plain hash and is not measured.
- The build targets the host CPU, so all three hashes batch on AVX-512.
- Both leaves share the transpose of every record into vector lanes, which the call count does not see.
- A tree that streams from DRAM on every thread is bound by memory, not by calls.
