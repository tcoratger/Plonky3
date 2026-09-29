//! Wider tree nodes: binary against 4-ary trees, both over the same standard leaves.
//!
//! A 4-ary node hashes four 32-byte children, 128 bytes.
//!
//! - Keccak-256 and SHA3-256 absorb 136 bytes per permutation, so a 4-ary node costs one call, as a binary one does.
//! - BLAKE3 hashes 128 bytes in two compressions, where the three binary nodes it replaces cost three.
//!
//! A proof then carries three siblings per level over half as many levels: 1.5x the digests.
//!
//! Run, for example:
//! `RAYON_NUM_THREADS=1 taskset -c 4 cargo bench -p p3-merkle-tree --features parallel --bench t8_arity`

use core::hint::black_box;
use core::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_blake3::Blake3;
use p3_commit::{BatchOpening, Mmcs};
use p3_keccak::{Keccak256Hash, Sha3_256Hash};
use p3_matrix::Dimensions;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::current_num_threads;
use p3_merkle_tree::{MerkleTree, MerkleTreeMmcs};
use p3_symmetric::{CompressionFunctionFromHasher, CryptographicHasher, PseudoCompressionFunction};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Record bytes: the eight-block records of the T8 comparison.
const RECORD: usize = 256;

/// Log2 of the leaf counts: both are powers of four, so the 4-ary tree needs no binary bridge.
const LOGS: [u32; 2] = [16, 20];

/// Openings verified per sample.
const OPENINGS: usize = 64;

/// Plonky3's Merkle commitment over byte records, with arity `N`.
type ByteMmcs<H, C, const N: usize> = MerkleTreeMmcs<u8, u8, H, C, N, 32>;

/// One prepared opening: its position and what the prover sends.
type Query<H, C, const N: usize> = (usize, BatchOpening<u8, ByteMmcs<H, C, N>>);

/// Time one commitment with arity `N`.
fn commit<H, C, const N: usize>(h: &H, c: &C, bytes: &[u8]) -> [u8; 32]
where
    H: CryptographicHasher<u8, [u8; 32]> + Sync,
    C: PseudoCompressionFunction<[u8; 32], N> + Sync,
{
    let matrix = RowMajorMatrixView::new(bytes, RECORD);
    MerkleTree::<u8, u8, _, N, 32>::new::<u8, u8, _, _>(h, c, vec![matrix])
        .root()
        .into()
}

/// Commit once and open the given positions, for the verification timing.
fn openings<H, C, const N: usize>(
    h: H,
    c: C,
    records: &RowMajorMatrix<u8>,
    positions: &[usize],
) -> impl Fn()
where
    H: CryptographicHasher<u8, [u8; 32]> + Sync,
    C: PseudoCompressionFunction<[u8; 32], N> + Sync,
{
    let mmcs = ByteMmcs::<H, C, N>::new(h, c, 0);
    let dims: [Dimensions; 1] = [p3_matrix::Matrix::dimensions(records)];
    let (root, data) = mmcs.commit(vec![records.clone()]);
    let queries: Vec<Query<H, C, N>> = positions
        .iter()
        .map(|&i| (i, mmcs.open_batch(i, &data)))
        .collect();
    move || {
        for (index, opening) in &queries {
            mmcs.verify_batch(black_box(&root), &dims, *index, opening.into())
                .expect("an honest opening verifies");
        }
    }
}

/// Time one hash's binary and 4-ary trees, for commitment and verification.
fn family<H>(c: &mut Criterion, name: &str, h: &H)
where
    H: CryptographicHasher<u8, [u8; 32]> + Sync,
{
    let binary = CompressionFunctionFromHasher::<_, 2, 32>::new(h.clone());
    let quaternary = CompressionFunctionFromHasher::<_, 4, 32>::new(h.clone());
    let threads = current_num_threads();

    for log in LOGS {
        let rows = 1usize << log;
        let mut rng = SmallRng::seed_from_u64(u64::from(log));
        let bytes: Vec<u8> = (0..rows * RECORD).map(|_| rng.random()).collect();
        let id = format!("2^{log}");

        // Commitment: one full tree.
        {
            let mut group = c.benchmark_group(format!("arity/{name}/commit/{threads}t"));
            group.sample_size(10);
            group.warm_up_time(Duration::from_secs(1));
            group.measurement_time(Duration::from_secs(3));
            group.throughput(Throughput::Bytes(bytes.len() as u64));
            group.bench_function(BenchmarkId::new("binary", &id), |b| {
                b.iter(|| commit::<_, _, 2>(h, &binary, black_box(&bytes)));
            });
            group.bench_function(BenchmarkId::new("quaternary", &id), |b| {
                b.iter(|| commit::<_, _, 4>(h, &quaternary, black_box(&bytes)));
            });
            group.finish();
        }

        // Verification, only on one thread: the verifier checks one opening at a time.
        if threads != 1 {
            continue;
        }
        let records = RowMajorMatrix::new(bytes, RECORD);
        let mut positions = Vec::with_capacity(OPENINGS);
        while positions.len() < OPENINGS {
            let i = rng.random_range(0..rows);
            if !positions.contains(&i) {
                positions.push(i);
            }
        }
        let verify_binary = openings::<_, _, 2>(h.clone(), binary.clone(), &records, &positions);
        let verify_quaternary =
            openings::<_, _, 4>(h.clone(), quaternary.clone(), &records, &positions);
        let mut group = c.benchmark_group(format!("arity/{name}/verify"));
        group.warm_up_time(Duration::from_millis(500));
        group.measurement_time(Duration::from_secs(2));
        group.throughput(Throughput::Elements(OPENINGS as u64));
        group.bench_function(BenchmarkId::new("binary", &id), |b| b.iter(&verify_binary));
        group.bench_function(BenchmarkId::new("quaternary", &id), |b| {
            b.iter(&verify_quaternary);
        });
        group.finish();
    }
}

fn bench_arity(c: &mut Criterion) {
    family(c, "keccak", &Keccak256Hash);
    family(c, "sha3", &Sha3_256Hash);
    family(c, "blake3", &Blake3);
}

criterion_group!(benches, bench_arity);
criterion_main!(benches);
