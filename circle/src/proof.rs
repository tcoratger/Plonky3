use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::Field;

#[derive(Debug, Clone)]
pub(crate) struct CircleCommitPhaseProofStep<F: Field, M: Mmcs<F>> {
    /// The log2 of the folding arity used for this step.
    pub(crate) log_arity: u8,
    /// The openings of the commit phase codeword at the sibling locations.
    /// For arity k, this contains k-1 sibling values.
    pub(crate) sibling_values: Vec<F>,

    pub(crate) opening_proof: M::Proof,
}
