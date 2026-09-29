# T8 results

One report per platform, each from `scripts/t8/run.sh` on that host.

| Platform | Host | Batched kernels | Report |
|---|---|---|---|
| x86-64, AVX-512 | AMD Ryzen 9 9950X3D, Linux | AVX-512 for all three hashes | [x86-avx512/REPORT.md](x86-avx512/REPORT.md) |
| AArch64, NEON and SHA-2 | Apple M1 Pro, macOS | NEON for BLAKE3 and BLAKE2s, the SHA-2 extension for SHA-256 | [mac-neon/REPORT.md](mac-neon/REPORT.md) |

`run.sh` picks the platform from the host, and writes into its folder.

Leaf hashing on one core, 256-byte records, standard time over T8 time:

| | BLAKE3 | BLAKE2s | SHA-256 |
|---|---:|---:|---:|
| Call ratio | 1.333x | 1.333x | 1.667x |
| Batch, x86 AVX-512 | 1.31x | 1.41x | 1.32x |
| Batch, M1 Pro | 1.21x | 1.21x | 1.66x |
| One record, x86 | 1.65x | 1.36x | 1.87x |
| One record, M1 Pro | 1.36x | 1.28x | 2.08x |
