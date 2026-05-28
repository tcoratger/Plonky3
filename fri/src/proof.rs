use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::Field;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct CommitPhaseProofStep<F: Field, M: Mmcs<F>> {
    /// The log2 of the folding arity used for this step.
    pub log_arity: u8,
    /// The openings of the commit phase codeword at the sibling locations.
    /// For arity k, this contains k-1 sibling values.
    pub sibling_values: Vec<F>,

    pub opening_proof: M::Proof,
}
