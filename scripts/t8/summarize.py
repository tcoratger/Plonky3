#!/usr/bin/env python3
"""Write the T8 report from criterion results: counted calls, then measured time and speed."""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HASHES = {"blake3": "BLAKE3", "blake2s": "BLAKE2s", "sha256": "SHA-256", "keccak": "Keccak-256"}
RECORDS = [256, 480, 65_664]


def standard_leaf(hash_name: str, length: int) -> int:
    """Native calls of the standard hash on one record of `length` bytes."""
    if hash_name == "blake3":
        # One compression per 64-byte block, plus one parent per extra 1 KiB chunk.
        return max(1, math.ceil(length / 64)) + max(1, math.ceil(length / 1024)) - 1
    if hash_name == "blake2s":
        # One compression per 64-byte block, the last one flagged final.
        return max(1, math.ceil(length / 64))
    if hash_name == "sha256":
        # One compression per 64-byte block, after at least 9 bytes of padding.
        return math.ceil((length + 9) / 64)
    # One permutation per 136-byte block, after at least one byte of padding.
    return length // 136 + 1


def t8_leaf(length: int) -> int:
    """Native calls of T8 on one record: three per stage of 224 fresh bytes."""
    return 3 * ((length - 32) // 224)


def tree_calls(hash_name: str, length: int, log: int, kind: str) -> tuple[int, int]:
    """Native calls of (standard, T8) to commit a tree of 2^log records, or to verify one opening."""
    s, t = standard_leaf(hash_name, length), t8_leaf(length)
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
        "| Hash | Record | Standard calls | T8 calls | Reduction |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, name in HASHES.items():
        for length in RECORDS:
            s, t = standard_leaf(key, length), t8_leaf(length)
            rows.append(f"| {name} | {length:,} B | {s:,} | {t:,} | {100 * (1 - t / s):+.1f}% |")
    return "\n".join(rows)


def measured(data: dict, build: str) -> str:
    """The measured tables of one build."""
    out = []

    def cell(entry: tuple[float, float]) -> str:
        return f"{fmt_time(entry[0])} ±{100 * entry[1]:.1f}%"

    # Leaf hashing, batched.
    rows = [
        "| Hash | Record | Records | Standard | T8 | Standard GiB/s | T8 GiB/s | Speedup | Call ratio |",
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
            ratio = standard_leaf(key, length) / t8_leaf(length)
            rows.append(
                f"| {HASHES[key]} | {length:,} B | 2^{log} | {cell(s)} | {cell(t)} |"
                f" {bytes_ / s[0]:.2f} | {bytes_ / t[0]:.2f} | **{s[0] / t[0]:.3f}x** | {ratio:.3f}x |"
            )
    out.append("**Leaf hashing, batched, one core** (GiB/s counts record bytes; 1 GiB/s = 1.074 B/ns)\n\n" + "\n".join(rows))

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
            ratio = standard_leaf(key, length) / t8_leaf(length)
            rows.append(f"| {HASHES[key]} | {length:,} B | {cell(s)} | {cell(t)} | **{s[0] / t[0]:.3f}x** | {ratio:.3f}x |")
    out.append("**Leaf hashing, one record, one core** (the verifier's leaf)\n\n" + "\n".join(rows))

    # Commitment.
    rows = [
        "| Hash | Record | Records | Threads | Standard | T8 | Standard GiB/s | T8 GiB/s | Speedup | Call ratio |",
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
    out.append("**Commitment: one full tree**\n\n" + "\n".join(rows))

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
    out.append("**Verification: one opening, one core** (64 openings per sample)\n\n" + "\n".join(rows))
    return "\n\n".join(out)


def main(results: Path) -> None:
    data: dict = {}
    for build in ("default", "native"):
        root = results / f"criterion-{build}"
        if root.exists():
            data.update(load(root))

    print("# T8 leaves on Plonky3\n")
    print("Each hash's standard leaf against T8 on the same compression kernel, inside an unchanged Plonky3 tree.\n")
    print("Times are criterion medians with the half-width of their 95% interval.\n")
    print("Speedup is standard time over T8 time; the call ratio is what counting compressions predicts.\n")
    print("## Setup\n")
    print(env(results))
    print("## Compression calls per record\n")
    print("Counted, not timed. A tree of N records adds the same N - 1 node calls to both leaves.\n")
    print(calls_table())
    print()
    for build in ("default", "native"):
        if any(k[0] == build for k in data):
            print(f"## Measured, {build} build\n")
            print(measured(data, build))
            print()
    print("## Reading the numbers\n")
    print("- BLAKE3 keeps T8's three roles apart with its counter and flags: the construction as analysed.")
    print("- BLAKE2s keeps them apart with counters 1, 2 and 3 and the final flag clear, which no plain BLAKE2s call uses.")
    print("- SHA-256's compression takes exactly 96 bytes, so its three calls are one function. The security analysis does not cover that instantiation.")
    print("- Keccak-256 absorbs 136 bytes per permutation, so T8 costs more calls than the plain hash and is not measured.")
    print("- The build targets the host CPU, so all three hashes batch on AVX-512.")
    print("- Both leaves share the transpose of every record into vector lanes, which the call count does not see.")
    print("- A tree that streams from DRAM on every thread is bound by memory, not by calls.")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
