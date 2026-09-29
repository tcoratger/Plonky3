#!/usr/bin/env bash
# Measure each hash's standard Merkle leaves against T8 leaves, then write the report.
#
# Every benchmark runs both leaves of a family back to back on the same input.
#
# One build: `-Ctarget-cpu=native` with Plonky3's `optimized` profile.
#
# Usage: scripts/t8/run.sh
#
# Results go to scripts/t8/results: the environment, the raw criterion estimates and REPORT.md.

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WT=$(cd "$HERE/../.." && pwd)
OUT=$HERE/results

# Benchmarks take this lock exclusively and builds take it shared, so no build disturbs a measurement.
LOCK=${LOCK:-$OUT/.lock}

# The core that single-threaded benchmarks are pinned to.
CORE=${CORE:-4}

mkdir -p "$OUT"
touch "$LOCK"
cd "$WT"

for build in native; do
    export RUSTFLAGS="-Ctarget-cpu=native"
    export CARGO_TARGET_DIR="$WT/target/bench-$build"
    bench=(cargo bench -p p3-merkle-tree --features parallel --profile optimized)

    # The environment of this build.
    {
        echo "date: $(date -Iseconds)"
        echo "plonky3: $(git rev-parse HEAD) + the uncommitted T8 files"
        echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)"
        echo "kernel: $(uname -r)"
        echo "rustc: $(rustc --version)"
        echo "RUSTFLAGS: ${RUSTFLAGS:-<none>}"
        echo "governor: $(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
        echo "boost: $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo unknown)"
    } > "$OUT/env-$build.txt"

    # Build first, under the shared lock.
    flock -s "$LOCK" "${bench[@]}" --bench t8_leaf --bench t8_verify --bench t8_commit --no-run

    # One core: leaf hashing and verification.
    flock -x "$LOCK" taskset -c "$CORE" "${bench[@]}" --bench t8_leaf -- --noplot --save-baseline "$build"
    flock -x "$LOCK" taskset -c "$CORE" "${bench[@]}" --bench t8_verify -- --noplot --save-baseline "$build"

    # Commitment on one pinned thread, then on every hardware thread.
    RAYON_NUM_THREADS=1 flock -x "$LOCK" taskset -c "$CORE" \
        "${bench[@]}" --bench t8_commit -- --noplot --save-baseline "$build"
    RAYON_NUM_THREADS=$(nproc) flock -x "$LOCK" \
        "${bench[@]}" --bench t8_commit -- --noplot --save-baseline "$build"

    # Keep the raw estimates next to the report.
    rm -rf "$OUT/criterion-$build"
    cp -r "$CARGO_TARGET_DIR/criterion" "$OUT/criterion-$build"
done

python3 "$HERE/summarize.py" "$OUT" > "$OUT/REPORT.md"
echo "wrote $OUT/REPORT.md"
