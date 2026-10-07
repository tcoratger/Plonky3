# T8 results

One report per platform, each from `scripts/t8/run.sh` on that host.

| Platform | Host | Batched kernels | Measurement date | Report |
|---|---|---|---|---|
| x86-64, AVX-512 | AMD Ryzen 9 9950X3D, Linux | AVX-512 for full batches | 2026-10-07 | [x86-avx512/REPORT.md](x86-avx512/REPORT.md) |
| AArch64, NEON and SHA-2 | Apple M1 Pro, macOS | NEON for BLAKE3 and BLAKE2s, SHA-2 for SHA-256 | 2026-09-29 | [mac-neon/REPORT.md](mac-neon/REPORT.md) |

`run.sh` picks the platform from the host and writes into its folder.

Leaf hashing on one core, 256-byte records (253 bytes for T253), standard time over T8 time:

| | BLAKE3 | BLAKE2s | SHA-256 | T253 |
|---|---:|---:|---:|---:|
| Call ratio | 1.333x | 1.333x | 1.667x | 1.667x |
| Batch, x86 AVX-512 | 1.294x | 1.377x | 1.270x | 1.362x |
| Batch, M1 Pro | 1.209x | 1.264x | 1.661x | 1.700x |
| One record, x86 | 2.359x | 1.359x | 1.894x | 1.961x |
| One record, M1 Pro | 1.360x | 1.274x | 2.357x | 2.339x |

The x86 report measures the rebased branch at `c4e1531e43625626997f920a7e0ee389dfa4a047`.
The Apple report is historical and has not been rerun on the new base; rerun `run.sh` on that host to refresh it.
Its single-record SHA-256 and T253 speedups include the code-layout caveat documented in that report.

Beside T8, each report also measures:

- **T253**: T8 on SHA-256 with role-separated calls, on 253-byte records against plain SHA-256 on the same bytes.
- **Wider nodes**: binary against 4-ary trees over the same standard leaves, with the proof size of each.
