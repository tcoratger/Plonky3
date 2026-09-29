//! Leaf hashing alone, on one thread: each hash's standard leaf against its T8 leaf.
//!
//! Both leaves of a family hash the same records, on the same kernel.
//!
//! Run pinned to one core, for example:
//! `taskset -c 4 cargo bench -p p3-merkle-tree --bench t8_leaf`

use core::hint::black_box;
use core::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_blake2s::{Blake2s256, T8Blake2s};
use p3_blake3::{Blake3, T8Blake3};
use p3_sha256::{Sha256, T8Sha256};
use p3_symmetric::CryptographicHasher;

/// Batches as (record bytes, log2 records), each 16 MiB: past the private caches, inside L3.
const BATCHES: [(usize, u32); 3] = [(256, 16), (480, 15), (65_664, 8)];

/// Single records: eight blocks, fifteen blocks, and the 64 KiB class.
const SINGLES: [usize; 3] = [256, 480, 65_664];

/// Deterministic bytes, generated once outside timing.
fn input(len: usize) -> Vec<u8> {
    let mut x = 0x2545_f491_4f6c_dd1du64 ^ len as u64;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

/// Time one family's standard and T8 leaves, batched and one record at a time.
fn family<S, T>(c: &mut Criterion, name: &str, standard: &S, t8: &T)
where
    S: CryptographicHasher<u8, [u8; 32]>,
    T: CryptographicHasher<u8, [u8; 32]>,
{
    for (len, log) in BATCHES {
        let bytes = input(len << log);
        let mut out = vec![[0u8; 32]; 1 << log];
        let mut group = c.benchmark_group(format!("leaf/{name}/batch/{len}B"));
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(3));
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        let id = format!("2^{log}");
        group.bench_function(BenchmarkId::new("standard", &id), |b| {
            b.iter(|| standard.hash_many(black_box(&bytes), black_box(&mut out)));
        });
        group.bench_function(BenchmarkId::new("t8", &id), |b| {
            b.iter(|| t8.hash_many(black_box(&bytes), black_box(&mut out)));
        });
        group.finish();
    }

    // One record, as a verifier hashes it.
    let mut group = c.benchmark_group(format!("leaf/{name}/single"));
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    for len in SINGLES {
        let record = input(len);
        group.throughput(Throughput::Bytes(len as u64));
        let id = format!("{len}B");
        group.bench_function(BenchmarkId::new("standard", &id), |b| {
            b.iter(|| standard.hash_slice(black_box(&record)));
        });
        group.bench_function(BenchmarkId::new("t8", &id), |b| {
            b.iter(|| t8.hash_slice(black_box(&record)));
        });
    }
    group.finish();
}

fn bench_leaf(c: &mut Criterion) {
    family(c, "blake3", &Blake3, &T8Blake3);
    family(c, "blake2s", &Blake2s256, &T8Blake2s);
    family(c, "sha256", &Sha256, &T8Sha256);
}

criterion_group!(benches, bench_leaf);
criterion_main!(benches);
