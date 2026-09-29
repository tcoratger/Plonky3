#!/usr/bin/env python3
"""Write the T8 report from criterion results: counted calls, then measured time and speed."""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# What differs between the hosts a report is written for.
#
# - pin: the prefix that keeps a one-thread run on one core, empty where the OS has none.
# - threads: every hardware thread of the host.
# - pin_note: how single-threaded runs stay put.
# - notes: host-specific lines of the closing notes.
PLATFORMS = {
    "x86-avx512": {
        "title": "T8 leaves on Plonky3",
        "pin": "taskset -c 4 ",
        "threads": 32,
        "pin_note": "`taskset -c 4` pins the process to one core, so single-threaded runs do not migrate.",
        "notes": [
            "The build targets the host CPU, so all three hashes batch on AVX-512.",
            "Both leaves share the transpose of every record into vector lanes, which the call count does not see.",
            "Batched leaves read every record once, 64 bytes at a time, exactly as the standard leaf does.",
            "So the transpose into vector lanes costs both leaves the same, and dilutes the call saving a little.",
            "On one thread, commitment follows the call ratio.",
            "On 32 threads, trees of 256-byte records wait on memory, so fewer calls barely shows.",
            "Trees of 64 KiB-class records stay compute-bound on 32 threads, and keep most of the call saving.",
            "A single long BLAKE3 record hashes its chunks in parallel; T8's chained stages cannot, hence the 0.1x rows.",
        ],
    },
    "mac-neon": {
        "title": "T8 leaves on Plonky3: Apple silicon, NEON and SHA-2",
        "pin": "",
        "threads": 10,
        "pin_note": "macOS has no core pinning, so single-threaded runs are left to the scheduler, which keeps a busy thread on a performance core.",
        "notes": [
            "The build targets the host CPU: SHA-256 batches four streams of the ARMv8 SHA-2 extension, BLAKE3 and BLAKE2s sixteen NEON lanes.",
            "Batched SHA-256 leaves follow the call ratio exactly: both leaves wait on the same SHA-2 unit, four streams deep.",
            "One SHA-256 record runs its calls a and b as two streams, so a stage waits on two calls, not three.",
            "So single SHA-256 records, and verification of 64 KiB-class records, beat the call ratio.",
            "Batched BLAKE3 and BLAKE2s leaves fall short of the call ratio: each out-of-line T8 call costs a little more than a call inside the standard chunk loop.",
            "Their loads and transposes are hidden: with them removed, the BLAKE3 T8 batch time does not change.",
            "On one thread, SHA-256 commitment follows the call ratio, and BLAKE3 and BLAKE2s land a little below it, as their batched leaves do.",
            "On all 10 cores, trees of 256-byte records keep most of the call saving, unlike on the 32 threads of the x86 host.",
            "A single long BLAKE3 record hashes its chunks in parallel; T8's chained stages cannot, hence the 0.5x rows.",
        ],
    },
}

# The host this report is written for, set by `main`.
PLATFORM = PLATFORMS["x86-avx512"]

HASHES = {"blake3": "BLAKE3", "blake2s": "BLAKE2s", "sha256": "SHA-256", "sha256-t253": "SHA-256, T253", "keccak": "Keccak-256"}
RECORDS = [256, 480, 65_664]
T253_RECORDS = [253, 474, 65_448]


def standard_leaf(hash_name: str, length: int) -> int:
    """Native calls of the standard hash on one record of `length` bytes."""
    if hash_name == "blake3":
        # One compression per 64-byte block, plus one parent per extra 1 KiB chunk.
        return max(1, math.ceil(length / 64)) + max(1, math.ceil(length / 1024)) - 1
    if hash_name == "blake2s":
        # One compression per 64-byte block, the last one flagged final.
        return max(1, math.ceil(length / 64))
    if hash_name in ("sha256", "sha256-t253"):
        # One compression per 64-byte block, after at least 9 bytes of padding.
        return math.ceil((length + 9) / 64)
    # One permutation per 136-byte block, after at least one byte of padding.
    return length // 136 + 1


def t8_leaf(length: int, hash_name: str = "") -> int:
    """Native calls of the T8 leaf on one record: three per stage.

    A T8 stage adds 224 fresh bytes, a T253 stage 221.
    """
    stride = 221 if hash_name == "sha256-t253" else 224
    return 3 * ((length - 32) // stride)


def tree_calls(hash_name: str, length: int, log: int, kind: str) -> tuple[int, int]:
    """Native calls of (standard, T8) to commit a tree of 2^log records, or to verify one opening."""
    s, t = standard_leaf(hash_name, length), t8_leaf(length, hash_name)
    if kind == "commit":
        n = 1 << log
        return n * s + n - 1, n * t + n - 1
    return s + log, t + log


def fmt_time(ns: float) -> str:
    """Nanoseconds in the unit that keeps three significant digits readable."""
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3)):
        if ns >= scale:
            return f"{ns / scale:.3g} {unit}"
    return f"{ns:.3g} ns"


def load(root: Path) -> dict:
    """Map (build, group, value) -> scheme -> (median ns, 95% half-width as a fraction)."""
    out: dict = defaultdict(dict)
    for bench in root.rglob("benchmark.json"):
        build = bench.parent.name
        if build not in ("default", "native"):
            continue
        meta = json.loads(bench.read_text())
        median = json.loads((bench.parent / "estimates.json").read_text())["median"]
        ci = median["confidence_interval"]
        half = (ci["upper_bound"] - ci["lower_bound"]) / 2 / median["point_estimate"]
        out[(build, meta["group_id"], meta["value_str"])][meta["function_id"]] = (
            median["point_estimate"],
            half,
        )
    return out


def env(results: Path) -> str:
    """The recorded environment of each build."""
    lines = []
    for build in ("default", "native"):
        path = results / f"env-{build}.txt"
        if path.exists():
            lines.append(f"**{build} build**\n\n```\n{path.read_text().strip()}\n```\n")
    return "\n".join(lines)


def calls_table() -> str:
    """Counted calls per record, for every hash including those T8 does not help."""
    rows = [
        "| Hash | Record | Standard calls | T8 calls | T8 against standard |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, name in HASHES.items():
        for length in (T253_RECORDS if key == "sha256-t253" else RECORDS):
            s, t = standard_leaf(key, length), t8_leaf(length, key)
            change = f"{100 * (1 - t / s):.1f}% fewer" if t <= s else f"{100 * (t / s - 1):.1f}% more"
            rows.append(f"| {name} | {length:,} B | {s:,} | {t:,} | {change} |")
    return "\n".join(rows)


# Every benchmark command shares this prefix, run from the repository root.
BENCH = "cargo bench -p p3-merkle-tree --features parallel --profile optimized"

# What produced each table: the command, then its parameters.
def leaf_batch() -> str:
    return f"""Command: `{PLATFORM['pin']}{BENCH} --bench t8_leaf -- 'leaf/.*/batch'`

- Code: `merkle-tree/benches/t8_leaf.rs`, one `hash_many` call over the whole batch, on one thread.
- Batches: 2^16 records of 256 B, 2^15 of 480 B, 2^8 of 65,664 B, so 16 MiB each.
- Input: deterministic xorshift bytes, generated before timing.
- Criterion: 1 s warm-up, 3 s measurement, 100 samples.
"""

def leaf_single() -> str:
    return f"""Command: `{PLATFORM['pin']}{BENCH} --bench t8_leaf -- 'leaf/.*/single'`

- Code: `merkle-tree/benches/t8_leaf.rs`, one `hash_slice` call on one record, the path a verifier takes.
- Records: 256 B, 480 B and 65,664 B.
- Criterion: 0.5 s warm-up, 2 s measurement, 100 samples.
"""

def commit() -> str:
    return f"""Commands:

- one thread: `RAYON_NUM_THREADS=1 {PLATFORM['pin']}{BENCH} --bench t8_commit`
- every hardware thread: `RAYON_NUM_THREADS={PLATFORM['threads']} {BENCH} --bench t8_commit`

- Code: `merkle-tree/benches/t8_commit.rs`, one `MerkleTree::new` over a borrowed byte matrix, one record per row.
- Timing covers leaf hashing, node hashing, and the digest layers' allocation and release.
- Trees: 2^16 and 2^20 records of 256 B, and 2^10 records of 65,664 B.
- Nodes: BLAKE3 or BLAKE2s of the 64 child bytes, or one SHA-256 compression, the same in both columns.
- Criterion: 1 s warm-up, 3 s measurement, 10 samples.
"""

def verify() -> str:
    return f"""Command: `{PLATFORM['pin']}{BENCH} --bench t8_verify`

- Code: `merkle-tree/benches/t8_verify.rs`, `MerkleTreeMmcs::verify_batch` on prepared openings, cap height 0.
- Each sample verifies 64 openings at distinct random positions, the same for both columns.
- The table divides by 64, so it shows one opening: the record's leaf hash plus the path to the root.
- Trees: 2^10 and 2^20 records of 256 B, and 2^10 records of 65,664 B.
- Criterion: 0.5 s warm-up, 2 s measurement, 100 samples.
"""


def measured(data: dict, build: str) -> str:
    """The measured tables of one build."""
    out = []

    def cell(entry: tuple[float, float]) -> str:
        return f"{fmt_time(entry[0])} ±{100 * entry[1]:.1f}%"

    # Leaf hashing, batched.
    rows = [
        "| Hash | Record | Records | Standard | T8 | Standard GB/s | T8 GB/s | Speedup | Call ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in HASHES:
        for (b, group, value), per in sorted(data.items()):
            if b != build or not group.startswith(f"leaf/{key}/batch/") or len(per) < 2:
                continue
            length = int(group.split("/")[-1][:-1])
            log = int(value.split("^")[1])
            bytes_ = length << log
            s, t = per["standard"], per["t8"]
            ratio = standard_leaf(key, length) / t8_leaf(length, key)
            rows.append(
                f"| {HASHES[key]} | {length:,} B | 2^{log} | {cell(s)} | {cell(t)} |"
                f" {bytes_ / s[0]:.2f} | {bytes_ / t[0]:.2f} | **{s[0] / t[0]:.3f}x** | {ratio:.3f}x |"
            )
    out.append("### Leaf hashing, batched, one core\n\n" + leaf_batch() + "\n" + "\n".join(rows))

    # Leaf hashing, one record at a time.
    rows = [
        "| Hash | Record | Standard | T8 | Speedup | Call ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in HASHES:
        for (b, group, value), per in sorted(data.items(), key=lambda kv: (kv[0][1], int(kv[0][2][:-1]) if kv[0][2].endswith("B") else 0)):
            if b != build or group != f"leaf/{key}/single" or len(per) < 2:
                continue
            length = int(value[:-1])
            s, t = per["standard"], per["t8"]
            ratio = standard_leaf(key, length) / t8_leaf(length, key)
            rows.append(f"| {HASHES[key]} | {length:,} B | {cell(s)} | {cell(t)} | **{s[0] / t[0]:.3f}x** | {ratio:.3f}x |")
    out.append("### Leaf hashing, one record, one core\n\n" + leaf_single() + "\n" + "\n".join(rows))

    # Commitment.
    rows = [
        "| Hash | Record | Records | Threads | Standard | T8 | Standard GB/s | T8 GB/s | Speedup | Call ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in HASHES:
        entries = []
        for (b, group, value), per in data.items():
            if b != build or not group.startswith(f"commit/{key}/") or len(per) < 2:
                continue
            _, _, rec, threads = group.split("/")
            entries.append((int(rec[:-1]), int(threads[:-1]), int(value.split("^")[1]), per))
        for length, threads, log, per in sorted(entries):
            s, t = per["standard"], per["t8"]
            cs, ct = tree_calls(key, length, log, "commit")
            bytes_ = length << log
            rows.append(
                f"| {HASHES[key]} | {length:,} B | 2^{log} | {threads} | {cell(s)} | {cell(t)} |"
                f" {bytes_ / s[0]:.2f} | {bytes_ / t[0]:.2f} | **{s[0] / t[0]:.3f}x** | {cs / ct:.3f}x |"
            )
    out.append("### Commitment: one full tree\n\n" + commit() + "\n" + "\n".join(rows))

    # Verification.
    rows = [
        "| Hash | Record | Records | Standard per opening | T8 per opening | Speedup | Call ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in HASHES:
        entries = []
        for (b, group, value), per in data.items():
            if b != build or not group.startswith(f"verify/{key}/") or len(per) < 2:
                continue
            entries.append((int(group.split("/")[-1][:-1]), int(value.split("^")[1]), per))
        for length, log, per in sorted(entries):
            s, t = per["standard"], per["t8"]
            cs, ct = tree_calls(key, length, log, "verify")
            per_open = lambda e: (e[0] / 64, e[1])
            rows.append(
                f"| {HASHES[key]} | {length:,} B | 2^{log} | {cell(per_open(s))} | {cell(per_open(t))} |"
                f" **{s[0] / t[0]:.3f}x** | {cs / ct:.3f}x |"
            )
    out.append("### Verification: one opening, one core\n\n" + verify() + "\n" + "\n".join(rows))
    return "\n\n".join(out)


def main(results: Path, platform: str) -> None:
    global PLATFORM
    PLATFORM = PLATFORMS[platform]
    data: dict = {}
    for build in ("default", "native"):
        root = results / f"criterion-{build}"
        if root.exists():
            data.update(load(root))

    print(f"# {PLATFORM['title']}\n")
    print("Each hash's standard Merkle leaf against its T8 leaf, on the same compression kernel, inside an unchanged Plonky3 tree.\n")
    print("## How to reproduce\n")
    print("From the repository root, on the `t8-leaves` branch:\n")
    print("```sh")
    print("# Everything below, then this report, in about 8 minutes.")
    print("scripts/t8/run.sh")
    print("```\n")
    print("Or one table at a time, reading the numbers criterion prints:\n")
    print("```sh")
    print("export RUSTFLAGS=-Ctarget-cpu=native")
    pin = PLATFORM["pin"]
    print(f"{pin}{BENCH} --bench t8_leaf")
    print(f"{pin}{BENCH} --bench t8_verify")
    print(f"RAYON_NUM_THREADS=1 {pin}{BENCH} --bench t8_commit")
    print(f"RAYON_NUM_THREADS={PLATFORM['threads']} {BENCH} --bench t8_commit")
    print("```\n")
    print("- Every command in this report assumes `RUSTFLAGS=-Ctarget-cpu=native`, as exported above.")
    print("- On AArch64, add `p3-blake3/neon` to `--features`, so one standard BLAKE3 message runs NEON code, not portable code.")
    print("- A trailing regex selects benchmarks, for example `-- 'leaf/sha256'`.")
    print(f"- {PLATFORM['pin_note']}")
    print("- `--profile optimized` is Plonky3's own profile: thin LTO and one codegen unit.")
    print("- Criterion prints `time: [low median high]`, the 95% interval of the median.")
    print("- It also keeps each estimate in `target/criterion/<group>/<scheme>/<size>/new/estimates.json`.")
    print(f"- The script saves them as the baseline `native`, copies them to `scripts/t8/results/{platform}/`, and builds this report from them.\n")
    print("## Reading the tables\n")
    print("- **Standard, T8**: criterion's median time, with the half-width of its 95% interval.")
    print("- **GB/s**: record bytes hashed per second, 1 GB = 10^9 bytes.")
    print("- **Speedup**: standard time over T8 time, so above 1 means T8 is faster.")
    print("- **Call ratio**: standard calls over T8 calls, what counting compressions alone predicts.\n")
    print("## Setup\n")
    print(env(results))
    print("## Compression calls per record\n")
    print("Counted, not timed. A tree of N records adds the same N - 1 node calls to both leaves.\n")
    print(calls_table())
    print()
    print("Keccak-256 is slower with T8, and SHA3-256 would be the same: both run Keccak-f with a 136-byte rate.\n")
    print("- One permutation already takes in 136 fresh bytes, so a plain hash covers 256 B in 2 calls.")
    print("- Each T8 call is a 96-byte function, so it uses only 96 of those 136 bytes and needs 3 calls.")
    print("- T8 pays off only when one call takes about 96 bytes, as the BLAKE3, BLAKE2s and SHA-256 compressions do.\n")
    for build in ("default", "native"):
        if any(k[0] == build for k in data):
            print("## Measured\n")
            print(measured(data, build))
            print()
    print("## Notes\n")
    print("- BLAKE3 keeps T8's three roles apart with its counter and flags: the construction as analysed.")
    print("- BLAKE2s keeps them apart with counters 1, 2 and 3 and the final flag clear, which no plain BLAKE2s call uses.")
    print("- SHA-256's compression takes exactly 96 bytes, so T8's three calls there are one function. The security analysis does not cover that instantiation.")
    print("- T253 is T8 on SHA-256 with a role byte at the top of each call's chaining value, so its three calls are distinct functions.")
    print("- The tree's node hash starts from the SHA-256 initial value, whose top byte is 0x6a, so no role meets it either.")
    print("- The price is one byte per tagged block: records of 32 + 221 k bytes, compared with plain SHA-256 on the same bytes.")
    print("- Keccak-256 absorbs 136 bytes per permutation, so T8 costs more calls than the plain hash and is not measured.")
    for line in PLATFORM["notes"]:
        print(f"- {line}")

if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "x86-avx512")
