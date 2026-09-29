#!/usr/bin/env bash
# Measure each hash's standard Merkle leaves against T8 leaves, then write the report.
#
# Every benchmark runs both leaves of a family back to back on the same input.
#
# One build: `-Ctarget-cpu=native` with Plonky3's `optimized` profile.
#
# Usage: scripts/t8/run.sh
#
# The host picks the platform: x86-avx512 on Linux, mac-neon on macOS.
#
# Results go to scripts/t8/results/<platform>: the environment, the raw criterion estimates and REPORT.md.

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WT=$(cd "$HERE/../.." && pwd)

# The core that single-threaded benchmarks are pinned to.
CORE=${CORE:-4}

# Linux pins single-threaded runs with taskset and serializes runs with flock.
#
# macOS has neither, so runs there go unpinned and unlocked.
case "$(uname -s)" in
    Darwin)
        PLATFORM=${PLATFORM:-mac-neon}
        PIN=()
        locked() { shift; "$@"; }
        cpu() { sysctl -n machdep.cpu.brand_string; }
        threads() { sysctl -n hw.ncpu; }
        governor() { echo "n/a (macOS)"; }
        boost() { echo "n/a (macOS)"; }
        ;;
    *)
        PLATFORM=${PLATFORM:-x86-avx512}
        PIN=(taskset -c "$CORE")
        locked() { local mode=$1; shift; flock "$mode" "$LOCK" "$@"; }
        cpu() { grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs; }
        threads() { nproc; }
        governor() { cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null || echo unknown; }
        boost() { cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo unknown; }
        ;;
esac

OUT=$HERE/results/$PLATFORM

# Benchmarks take this lock exclusively and builds take it shared, so no build disturbs a measurement.
LOCK=${LOCK:-$OUT/.lock}


mkdir -p "$OUT"
touch "$LOCK"
cd "$WT"

for build in native; do
    export RUSTFLAGS="-Ctarget-cpu=native"
    export CARGO_TARGET_DIR="$WT/target/bench-$build"
    bench=(cargo bench -p p3-merkle-tree --features parallel --profile optimized)

    # The environment of this build.
    {
        echo "date: $(date -Iseconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S%z)"
        echo "commit: $(git rev-parse HEAD)$(git diff --quiet HEAD -- . ":!scripts/t8/results" || echo " + local changes")"
        echo "cpu: $(cpu)"
        echo "kernel: $(uname -r)"
        echo "rustc: $(rustc --version)"
        echo "RUSTFLAGS: ${RUSTFLAGS:-<none>}"
        echo "governor: $(governor)"
        echo "boost: $(boost)"
    } > "$OUT/env-$build.txt"

    # Build first, under the shared lock.
    locked -s "${bench[@]}" --bench t8_leaf --bench t8_verify --bench t8_commit --no-run

    # One core: leaf hashing and verification.
    locked -x ${PIN[@]+"${PIN[@]}"} "${bench[@]}" --bench t8_leaf -- --noplot --save-baseline "$build"
    locked -x ${PIN[@]+"${PIN[@]}"} "${bench[@]}" --bench t8_verify -- --noplot --save-baseline "$build"

    # Commitment on one pinned thread, then on every hardware thread.
    locked -x env RAYON_NUM_THREADS=1 ${PIN[@]+"${PIN[@]}"} \
        "${bench[@]}" --bench t8_commit -- --noplot --save-baseline "$build"
    locked -x env RAYON_NUM_THREADS="$(threads)" \
        "${bench[@]}" --bench t8_commit -- --noplot --save-baseline "$build"

    # Keep the raw estimates next to the report.
    rm -rf "$OUT/criterion-$build"
    cp -r "$CARGO_TARGET_DIR/criterion" "$OUT/criterion-$build"
done

python3 "$HERE/summarize.py" "$OUT" "$PLATFORM" > "$OUT/REPORT.md"
echo "wrote $OUT/REPORT.md"
