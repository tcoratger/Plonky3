//! Verification: the MMCS checking record openings, each hash's standard leaves against its T8 leaves.
//!
//! Each iteration verifies 64 prepared openings at distinct random positions.
//!
//! So one opening costs the time divided by 64, with warm caches.
//!
//! Run pinned to one core, for example:
//! `taskset -c 4 cargo bench -p p3-merkle-tree --bench t8_verify`

use core::hint::black_box;
use core::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_blake2s::{Blake2s256, T8Blake2s};
use p3_blake3::{Blake3, T8Blake3};
use p3_commit::{BatchOpening, Mmcs};
use p3_matrix::Dimensions;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sha256::{Sha256, Sha256Compress, T8Sha256, T253Sha256};
use p3_symmetric::{CompressionFunctionFromHasher, CryptographicHasher, PseudoCompressionFunction};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Tree shapes as (record bytes, log2 records).
const SHAPES: [(usize, u32); 3] = [(256, 10), (256, 20), (65_664, 10)];

/// The same shapes for T253, whose records are `32 + 221 k` bytes.
const T253_SHAPES: [(usize, u32); 3] = [(253, 10), (253, 20), (65_448, 10)];

/// Openings verified per iteration.
const OPENINGS: usize = 64;

/// Plonky3's Merkle commitment over byte records.
type ByteMmcs<H, C> = MerkleTreeMmcs<u8, u8, H, C, 2, 32>;

/// One prepared opening: its position and what the prover sends.
type Query<H, C> = (usize, BatchOpening<u8, ByteMmcs<H, C>>);

/// Everything a verifier holds: the root, the shape, and prepared openings.
struct Fixture<H, C>
where
    H: CryptographicHasher<u8, [u8; 32]> + Sync,
    C: PseudoCompressionFunction<[u8; 32], 2> + Sync,
{
    mmcs: ByteMmcs<H, C>,
    commit: <ByteMmcs<H, C> as Mmcs<u8>>::Commitment,
    dims: [Dimensions; 1],
    queries: Vec<Query<H, C>>,
}

/// Commit to `records` and open the same positions for every scheme.
fn fixture<H, C>(h: H, node: C, records: &RowMajorMatrix<u8>, positions: &[usize]) -> Fixture<H, C>
where
    H: CryptographicHasher<u8, [u8; 32]> + Sync,
    C: PseudoCompressionFunction<[u8; 32], 2> + Sync,
{
    let mmcs = ByteMmcs::new(h, node, 0);
    let dims = [p3_matrix::Matrix::dimensions(records)];
    let (commit, data) = mmcs.commit(vec![records.clone()]);
    let queries = positions
        .iter()
        .map(|&i| (i, mmcs.open_batch(i, &data)))
        .collect();
    Fixture {
        mmcs,
        commit,
        dims,
        queries,
    }
}

/// Verify every prepared opening once.
fn verify_all<H, C>(f: &Fixture<H, C>)
where
    H: CryptographicHasher<u8, [u8; 32]> + Sync,
    C: PseudoCompressionFunction<[u8; 32], 2> + Sync,
{
    for (index, opening) in &f.queries {
        f.mmcs
            .verify_batch(black_box(&f.commit), &f.dims, *index, opening.into())
            .expect("an honest opening verifies");
    }
}

/// Time one family's openings with standard and with T8 leaves, over the same node hash.
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
    C: PseudoCompressionFunction<[u8; 32], 2> + Sync + Clone,
{
    for &(len, log) in shapes {
        let rows = 1usize << log;
        let mut rng = SmallRng::seed_from_u64(u64::from(log) ^ len as u64);
        let records = RowMajorMatrix::new((0..rows * len).map(|_| rng.random()).collect(), len);

        // Distinct random positions, the same for both leaves.
        let mut positions: Vec<usize> = Vec::with_capacity(OPENINGS);
        while positions.len() < OPENINGS {
            let i = rng.random_range(0..rows);
            if !positions.contains(&i) {
                positions.push(i);
            }
        }
        let standard = fixture(standard.clone(), node.clone(), &records, &positions);
        let t8 = fixture(t8.clone(), node.clone(), &records, &positions);

        let mut group = c.benchmark_group(format!("verify/{name}/{len}B"));
        group.warm_up_time(Duration::from_millis(500));
        group.measurement_time(Duration::from_secs(2));
        group.throughput(Throughput::Elements(OPENINGS as u64));
        let id = format!("2^{log}");
        group.bench_function(BenchmarkId::new("standard", &id), |b| {
            b.iter(|| verify_all(&standard));
        });
        group.bench_function(BenchmarkId::new("t8", &id), |b| {
            b.iter(|| verify_all(&t8));
        });
        group.finish();
    }
}

fn bench_verify(c: &mut Criterion) {
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

criterion_group!(benches, bench_verify);
criterion_main!(benches);
