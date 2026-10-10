//! Commitment: one full `MerkleTree::new`, each hash's standard leaves against its T8 leaves.
//!
//! The tree builder, the node hash and the thread schedule are the same code for both leaves.
//!
//! Timing covers the digest layers' allocation, hashing and release, not the input.
//!
//! Set the worker count explicitly, for example:
//! `RAYON_NUM_THREADS=1 taskset -c 4 cargo bench -p p3-merkle-tree --features parallel --bench t8_commit`

use core::hint::black_box;
use core::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_blake2s::{Blake2s256, T8Blake2s};
use p3_blake3::{Blake3, T8Blake3};
use p3_matrix::dense::RowMajorMatrixView;
use p3_maybe_rayon::prelude::current_num_threads;
use p3_merkle_tree::MerkleTree;
use p3_sha256::{Sha256, Sha256Compress, T8Sha256, T253Sha256};
use p3_symmetric::{CompressionFunctionFromHasher, CryptographicHasher, PseudoCompressionFunction};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Tree shapes as (record bytes, log2 records): 16 MiB, 256 MiB, and 64 MiB of 64 KiB-class records.
const SHAPES: [(usize, u32); 3] = [(256, 16), (256, 20), (65_664, 10)];

/// The same shapes for T253, whose records are `32 + 221 k` bytes.
const T253_SHAPES: [(usize, u32); 3] = [(253, 16), (253, 20), (65_448, 10)];

/// Build one tree over borrowed records and return its root.
fn commit<H, C>(h: &H, c: &C, records: &[u8], len: usize) -> [u8; 32]
where
    H: CryptographicHasher<u8, [u8; 32]> + Sync,
    C: PseudoCompressionFunction<[u8; 32], 2> + Sync,
{
    let matrix = RowMajorMatrixView::new(records, len);
    MerkleTree::<u8, u8, _, 2, 32>::new::<u8, u8, _, _>(h, c, vec![matrix])
        .root()
        .into()
}

/// Time one family's trees with standard and with T8 leaves, over the same node hash.
fn family<S, T, C>(
    c: &mut Criterion,
    name: &str,
    standard: &S,
    t8: &T,
    node: &C,
    shapes: &[(usize, u32)],
) where
    S: CryptographicHasher<u8, [u8; 32]> + Sync,
    T: CryptographicHasher<u8, [u8; 32]> + Sync,
    C: PseudoCompressionFunction<[u8; 32], 2> + Sync,
{
    let threads = current_num_threads();
    for &(len, log) in shapes {
        let mut rng = SmallRng::seed_from_u64(u64::from(log) ^ len as u64);
        let bytes: Vec<u8> = (0..len << log).map(|_| rng.random()).collect();

        let mut group = c.benchmark_group(format!("commit/{name}/{len}B/{threads}t"));
        group.sample_size(10);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(3));
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        let id = format!("2^{log}");
        group.bench_function(BenchmarkId::new("standard", &id), |b| {
            b.iter(|| commit(standard, node, black_box(&bytes), len));
        });
        group.bench_function(BenchmarkId::new("t8", &id), |b| {
            b.iter(|| commit(t8, node, black_box(&bytes), len));
        });
        group.finish();
    }
}

fn bench_commit(c: &mut Criterion) {
    family(
        c,
        "blake3",
        &Blake3,
        &T8Blake3,
        &CompressionFunctionFromHasher::<_, 2, 32>::new(Blake3),
        &SHAPES,
    );
    family(
        c,
        "blake2s",
        &Blake2s256,
        &T8Blake2s,
        &CompressionFunctionFromHasher::<_, 2, 32>::new(Blake2s256),
        &SHAPES,
    );
    family(c, "sha256", &Sha256, &T8Sha256, &Sha256Compress, &SHAPES);
    family(
        c,
        "sha256-t253",
        &Sha256,
        &T253Sha256,
        &Sha256Compress,
        &T253_SHAPES,
    );
}

criterion_group!(benches, bench_commit);
criterion_main!(benches);
