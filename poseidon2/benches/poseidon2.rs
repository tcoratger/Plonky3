use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_bn254::{Bn254, Poseidon2Bn254};
use p3_field::{Algebra, Field, InjectiveMonomial, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_mersenne_31::{Mersenne31, Poseidon2Mersenne31};
use p3_symmetric::Permutation;
use p3_util::pretty_name;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

/// Dense matrix-vector product: state = matrix * state (O(WIDTH²) muls).
#[inline]
fn dense_matmul<F: Field, A: Algebra<F>, const WIDTH: usize>(
    state: &mut [A; WIDTH],
    matrix: &[[F; WIDTH]; WIDTH],
) {
    let input = state.clone();
    for i in 0..WIDTH {
        state[i] = A::ZERO;
        for j in 0..WIDTH {
            let mut term = input[j].clone();
            term *= matrix[i][j];
            state[i] += term;
        }
    }
}

/// Same round structure as Poseidon2 but with dense random WIDTH×WIDTH matrices
/// instead of the structured linear layers. Estimates Poseidon1 cost.
#[derive(Clone)]
struct DensePoseidon1Estimate<F, const WIDTH: usize, const D: u64> {
    initial_external_constants: Vec<[F; WIDTH]>,
    terminal_external_constants: Vec<[F; WIDTH]>,
    internal_constants: Vec<F>,
    external_matrix: [[F; WIDTH]; WIDTH],
    internal_matrix: [[F; WIDTH]; WIDTH],
}

impl<F: PrimeField64, const WIDTH: usize, const D: u64> DensePoseidon1Estimate<F, WIDTH, D>
where
    StandardUniform: Distribution<F>,
{
    fn new_from_rng_128<R: Rng>(rng: &mut R) -> Self {
        let (rounds_f, rounds_p) =
            p3_poseidon2::poseidon2_round_numbers_128::<F>(WIDTH, D).unwrap();
        let half_f = rounds_f / 2;
        let initial_external_constants = (0..half_f)
            .map(|_| core::array::from_fn(|_| rng.random()))
            .collect();
        let terminal_external_constants = (0..half_f)
            .map(|_| core::array::from_fn(|_| rng.random()))
            .collect();
        let internal_constants = (0..rounds_p).map(|_| rng.random()).collect();
        let external_matrix = core::array::from_fn(|_| core::array::from_fn(|_| rng.random()));
        let internal_matrix = core::array::from_fn(|_| core::array::from_fn(|_| rng.random()));
        Self {
            initial_external_constants,
            terminal_external_constants,
            internal_constants,
            external_matrix,
            internal_matrix,
        }
    }
}

impl<F, A, const WIDTH: usize, const D: u64> Permutation<[A; WIDTH]>
    for DensePoseidon1Estimate<F, WIDTH, D>
where
    F: Field + InjectiveMonomial<D>,
    A: Algebra<F> + Sync + InjectiveMonomial<D>,
{
    fn permute_mut(&self, state: &mut [A; WIDTH]) {
        dense_matmul(state, &self.external_matrix);
        for rc in &self.initial_external_constants {
            for (s, &c) in state.iter_mut().zip(rc.iter()) {
                *s += c;
                *s = s.injective_exp_n();
            }
            dense_matmul(state, &self.external_matrix);
        }
        for &rc in &self.internal_constants {
            state[0] += rc;
            state[0] = state[0].injective_exp_n();
            dense_matmul(state, &self.internal_matrix);
        }
        for rc in &self.terminal_external_constants {
            for (s, &c) in state.iter_mut().zip(rc.iter()) {
                *s += c;
                *s = s.injective_exp_n();
            }
            dense_matmul(state, &self.external_matrix);
        }
    }
}

fn bench_poseidon2(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(1);

    // let poseidon2_bb_16 = Poseidon2BabyBear::<16>::new_from_rng_128(&mut rng);
    // poseidon2::<BabyBear, Poseidon2BabyBear<16>, 16>(c, &poseidon2_bb_16);
    // let poseidon2_bb_24 = Poseidon2BabyBear::<24>::new_from_rng_128(&mut rng);
    // poseidon2::<BabyBear, Poseidon2BabyBear<24>, 24>(c, &poseidon2_bb_24);

    let poseidon2_kb_16 = Poseidon2KoalaBear::<16>::new_from_rng_128(&mut rng);
    poseidon2::<KoalaBear, Poseidon2KoalaBear<16>, 16>(c, &poseidon2_kb_16);
    let poseidon2_kb_24 = Poseidon2KoalaBear::<24>::new_from_rng_128(&mut rng);
    poseidon2::<KoalaBear, Poseidon2KoalaBear<24>, 24>(c, &poseidon2_kb_24);

    // let poseidon2_m31_16 = Poseidon2Mersenne31::<16>::new_from_rng_128(&mut rng);
    // poseidon2::<Mersenne31, Poseidon2Mersenne31<16>, 16>(c, &poseidon2_m31_16);
    // let poseidon2_m31_24 = Poseidon2Mersenne31::<24>::new_from_rng_128(&mut rng);
    // poseidon2::<Mersenne31, Poseidon2Mersenne31<24>, 24>(c, &poseidon2_m31_24);

    // let poseidon2_gold_8 = Poseidon2Goldilocks::<8>::new_from_rng_128(&mut rng);
    // poseidon2::<Goldilocks, Poseidon2Goldilocks<8>, 8>(c, &poseidon2_gold_8);
    // let poseidon2_gold_12 = Poseidon2Goldilocks::<12>::new_from_rng_128(&mut rng);
    // poseidon2::<Goldilocks, Poseidon2Goldilocks<12>, 12>(c, &poseidon2_gold_12);
    // let poseidon2_gold_16 = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rng);
    // poseidon2::<Goldilocks, Poseidon2Goldilocks<16>, 16>(c, &poseidon2_gold_16);

    // // We hard code the round numbers for Bn254Fr.
    // let poseidon2_bn254 = Poseidon2Bn254::<3>::new_from_rng(8, 22, &mut rng);
    // poseidon2::<Bn254, Poseidon2Bn254<3>, 3>(c, &poseidon2_bn254);

    // ---- Poseidon1 estimate (dense matrices) on KoalaBear ----
    // Comment/uncomment these to compare against the Poseidon2 results above.
    let p1_kb_16 = DensePoseidon1Estimate::<KoalaBear, 16, 3>::new_from_rng_128(&mut rng);
    poseidon2::<KoalaBear, DensePoseidon1Estimate<KoalaBear, 16, 3>, 16>(c, &p1_kb_16);
    let p1_kb_24 = DensePoseidon1Estimate::<KoalaBear, 24, 3>::new_from_rng_128(&mut rng);
    poseidon2::<KoalaBear, DensePoseidon1Estimate<KoalaBear, 24, 3>, 24>(c, &p1_kb_24);
}

fn poseidon2<F, Perm, const WIDTH: usize>(c: &mut Criterion, poseidon2: &Perm)
where
    F: Field,
    Perm: Permutation<[F::Packing; WIDTH]>,
{
    let input = [F::Packing::ZERO; WIDTH];
    let name = format!("poseidon2::<{}, {}>", pretty_name::<F::Packing>(), WIDTH);
    let id = BenchmarkId::new(name, WIDTH);
    c.bench_with_input(id, &input, |b, &input| b.iter(|| poseidon2.permute(input)));
}

criterion_group!(benches, bench_poseidon2);
criterion_main!(benches);
