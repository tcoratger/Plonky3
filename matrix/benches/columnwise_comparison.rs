use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::strided_poc::StridedMatrix;
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

fn columnwise_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("columnwise_dot_product_comparison");

    // Test configurations: (rows, cols, description)
    let configs = [
        (64, 64, "small"),      // Small: 64×64
        (512, 512, "medium"),   // Medium: 512×512
        (2048, 2048, "large"),  // Large: 2048×2048
    ];

    for (rows, cols, size) in configs.iter() {
        let mut rng = SmallRng::seed_from_u64(0);

        // Benchmark 1: DenseMatrix with row-major layout (current implementation)
        // This is the baseline - optimized parallel packed implementation
        group.bench_with_input(
            BenchmarkId::new("DenseMatrix_RowMajor", size),
            &(rows, cols),
            |b, &(rows, cols)| {
                let matrix = RowMajorMatrix::<F>::rand_nonzero(&mut rng, *rows, *cols);
                let vector = RowMajorMatrix::<EF>::rand_nonzero(&mut rng, *rows, 1).values;

                b.iter(|| {
                    matrix.columnwise_dot_product(&vector)
                });
            }
        );

        // Benchmark 2: StridedMatrix starting from row-major (naive transpose)
        // This shows the cost of strided column access even with transpose view
        group.bench_with_input(
            BenchmarkId::new("StridedMatrix_RowMajor_Transpose", size),
            &(rows, cols),
            |b, &(rows, cols)| {
                let dense = RowMajorMatrix::<F>::rand_nonzero(&mut rng, *rows, *cols);
                let matrix = StridedMatrix::new(dense.values.clone(), *cols);
                let vector = RowMajorMatrix::<EF>::rand_nonzero(&mut rng, *rows, 1).values;

                b.iter(|| {
                    matrix.columnwise_dot_product(&vector)
                });
            }
        );

        // Benchmark 3: DenseMatrix physical transpose + columnwise_dot_product
        // This shows the cost of O(n) transpose followed by computation
        group.bench_with_input(
            BenchmarkId::new("DenseMatrix_PhysicalTranspose", size),
            &(rows, cols),
            |b, &(rows, cols)| {
                let matrix = RowMajorMatrix::<F>::rand_nonzero(&mut rng, *rows, *cols);
                let vector = RowMajorMatrix::<EF>::rand_nonzero(&mut rng, *rows, 1).values;

                b.iter(|| {
                    // Materialize transpose + compute (both are O(n))
                    let transposed = matrix.transpose();
                    transposed.columnwise_dot_product(&vector)
                });
            }
        );
    }

    group.finish();
}

// Separate benchmark group for demonstrating the O(1) transpose benefit
fn transpose_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose_overhead");

    let configs = [
        (64, 64, "small"),
        (512, 512, "medium"),
        (2048, 2048, "large"),
    ];

    for (rows, cols, size) in configs.iter() {
        let mut rng = SmallRng::seed_from_u64(0);

        // Physical transpose (O(n))
        group.bench_with_input(
            BenchmarkId::new("Physical_Transpose", size),
            &(rows, cols),
            |b, &(rows, cols)| {
                let matrix = RowMajorMatrix::<F>::rand_nonzero(&mut rng, *rows, *cols);

                b.iter(|| {
                    matrix.transpose()
                });
            }
        );

        // View transpose (O(1))
        group.bench_with_input(
            BenchmarkId::new("View_Transpose", size),
            &(rows, cols),
            |b, &(rows, cols)| {
                let dense = RowMajorMatrix::<F>::rand_nonzero(&mut rng, *rows, *cols);
                let matrix = StridedMatrix::new(dense.values, *cols);

                b.iter(|| {
                    matrix.transpose_view()
                });
            }
        );
    }

    group.finish();
}

criterion_group!(benches, columnwise_comparison, transpose_overhead);
criterion_main!(benches);
