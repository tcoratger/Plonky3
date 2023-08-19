use p3_air::{Air, AirBuilder};
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2BowersFft;
use p3_field::AbstractField;
use p3_fri::{FriBasedPcs, FriConfigImpl, FriLdt};
use p3_goldilocks::Goldilocks;
use p3_ldt::QuotientMmcs;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::MatrixRowSlices;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon::Poseidon;
use p3_symmetric::compression::TruncatedPermutation;
use p3_symmetric::mds::NaiveMDSMatrix;
use p3_symmetric::sponge::PaddingFreeSponge;
use p3_uni_stark::{prove, StarkConfigImpl};
use rand::thread_rng;

struct MulAir;

impl<AB: AirBuilder> Air<AB> for MulAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let main_local = main.row_slice(0);
        let diff = main_local[0] * main_local[1] - main_local[2];
        builder.assert_zero(diff);
    }
}

#[test]
fn test_prove_goldilocks() {
    type Val = Goldilocks;
    type Dom = Goldilocks;
    type Challenge = Goldilocks; // TODO

    type MyMds = NaiveMDSMatrix<Val, 8>;
    let mds = MyMds::new([[Val::ONE; 8]; 8]); // TODO: Use a real MDS matrix

    type Perm = Poseidon<Val, MyMds, 8, 7>;
    let perm = Perm::new(5, 5, vec![Val::ONE; 120], mds);

    type H4 = PaddingFreeSponge<Val, Perm, { 4 + 4 }>;
    let h4 = H4::new(perm.clone());

    type C = TruncatedPermutation<Val, Perm, 2, 4, { 2 * 4 }>;
    let c = C::new(perm.clone());

    type MyMmcs = MerkleTreeMmcs<Val, [Val; 4], H4, C>;
    let mmcs = MyMmcs::new(h4, c);

    type Dft = Radix2BowersFft;
    let dft = Dft::default();

    type Challenger = DuplexChallenger<Val, Perm, 8>;

    type Quotient = QuotientMmcs<Dom, MyMmcs>;
    type MyFriConfig = FriConfigImpl<Val, Challenge, Quotient, MyMmcs, Challenger>;
    let fri_config = MyFriConfig::new(40, mmcs.clone());
    let ldt = FriLdt { config: fri_config };

    type Pcs = FriBasedPcs<MyFriConfig, MyMmcs, Dft>;
    type MyConfig = StarkConfigImpl<Val, Dom, Challenge, Pcs, Dft, Challenger>;

    let mut rng = thread_rng();
    let trace = RowMajorMatrix::rand(&mut rng, 256, 10);
    let pcs = Pcs::new(dft, 1, mmcs, ldt);
    let config = StarkConfigImpl::new(pcs, Dft::default());
    let mut challenger = Challenger::new(perm);
    prove::<MyConfig, _>(&MulAir, &config, &mut challenger, trace);
}

#[test]
#[ignore] // TODO: Not ready yet.
fn test_prove_mersenne_31() {
    todo!()
}
