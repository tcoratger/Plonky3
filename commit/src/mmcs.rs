use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::ops::Deref;

use p3_challenger::fs::{
    DefaultCodec, Interaction, Label, ProverState, TranscriptBound, TranscriptError, VerifierState,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// Matrix dimensions for one MMCS commitment batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchDimensions(Vec<Dimensions>);

impl BatchDimensions {
    /// Build batch dimensions from matrix dimensions in commitment order.
    #[must_use]
    pub const fn new(dimensions: Vec<Dimensions>) -> Self {
        Self(dimensions)
    }

    /// Build batch dimensions for a batch containing one matrix.
    #[must_use]
    pub fn single(dimensions: Dimensions) -> Self {
        Self(vec![dimensions])
    }

    /// Borrow the matrix dimensions in commitment order.
    #[must_use]
    pub fn as_slice(&self) -> &[Dimensions] {
        &self.0
    }

    /// Number of matrices in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the batch is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Iterate over the matrix dimensions in commitment order.
    pub fn iter(&self) -> core::slice::Iter<'_, Dimensions> {
        self.0.iter()
    }

    /// Number of values in a complete batch opening row set.
    #[must_use]
    pub fn opened_values_len(&self) -> usize {
        self.iter().map(|dims| dims.width).sum()
    }
}

impl From<Vec<Dimensions>> for BatchDimensions {
    fn from(dimensions: Vec<Dimensions>) -> Self {
        Self::new(dimensions)
    }
}

impl<const N: usize> From<[Dimensions; N]> for BatchDimensions {
    fn from(dimensions: [Dimensions; N]) -> Self {
        Self::new(dimensions.into())
    }
}

impl AsRef<[Dimensions]> for BatchDimensions {
    fn as_ref(&self) -> &[Dimensions] {
        self.as_slice()
    }
}

impl Deref for BatchDimensions {
    type Target = [Dimensions];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// A "Mixed Matrix Commitment Scheme" (MMCS) is a generalization of a vector commitment scheme.
///
/// It supports committing to matrices and then opening rows. It is also batch-oriented; one can commit
/// to a batch of matrices at once even if their widths and heights differ.
///
/// When a particular row index is opened, it is interpreted directly as a row index for matrices
/// with the largest height. For matrices with smaller heights, some bits of the row index are
/// removed (from the least-significant side) to get the effective row index. These semantics are
/// useful in the FRI protocol. See the documentation for `open_batch` for more details.
pub trait Mmcs<T: Send + Sync + Clone>: Clone {
    type ProverData<M>;
    type Commitment: Clone + Serialize + DeserializeOwned;
    type Proof: Clone + Serialize + DeserializeOwned;
    type Error: Debug;

    /// Commits to a batch of matrices at once and returns both the commitment and associated prover data.
    ///
    /// Each matrix in the batch may have different dimensions.
    ///
    /// # Parameters
    /// - `inputs`: A vector of matrices to commit to.
    ///
    /// # Returns
    /// A tuple `(commitment, prover_data)` where:
    /// - `commitment` is a compact representation of all matrix elements and will be sent to the verifier.
    /// - `prover_data` is auxiliary data used by the prover open the commitment.
    fn commit<M: Matrix<T>>(&self, inputs: Vec<M>) -> (Self::Commitment, Self::ProverData<M>);

    /// Convenience method to commit to a single matrix.
    ///
    /// Internally wraps the matrix in a singleton vector and delegates to [`commit`].
    ///
    /// # Parameters
    /// - `input`: The matrix to commit to.
    ///
    /// # Returns
    /// A tuple `(commitment, prover_data)` as in [`commit`].
    fn commit_matrix<M: Matrix<T>>(&self, input: M) -> (Self::Commitment, Self::ProverData<M>) {
        self.commit(vec![input])
    }

    /// Convenience method to commit to a single column vector, treated as a column matrix.
    ///
    /// Automatically wraps the vector into a column matrix using [`RowMajorMatrix::new_col`].
    ///
    /// # Parameters
    /// - `input`: A vector of field elements representing a single column.
    ///
    /// # Returns
    /// A tuple `(commitment, prover_data)` for the resulting 1-column matrix.
    fn commit_vec(&self, input: Vec<T>) -> (Self::Commitment, Self::ProverData<RowMajorMatrix<T>>)
    where
        T: Clone + Send + Sync,
    {
        self.commit_matrix(RowMajorMatrix::new_col(input))
    }

    /// Opens a specific row (identified by `index`) from each matrix in the batch.
    ///
    /// This function is designed to support batch opening semantics where matrices may have different heights.
    /// The given index is interpreted relative to the maximum matrix height; smaller matrices apply a
    /// bit-shift to extract the corresponding row.
    ///
    /// # Parameters
    /// - `index`: The global row index (relative to max height).
    /// - `prover_data`: Prover data returned from [`commit`] or related methods.
    ///
    /// # Returns
    /// A [`BatchOpening`] containing the opened rows and the proof of their correctness.
    ///
    /// # Opening Index Semantics
    /// For each matrix `M[i]`, the row index used is:
    /// ```text
    /// j = index >> (log2_ceil(max_height) - log2_ceil(M[i].height))
    /// ```
    fn open_batch<M: Matrix<T>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> BatchOpening<T, Self>;

    /// Returns references to all matrices originally committed to in the batch.
    ///
    /// This allows access to the underlying data for inspection or additional logic.
    ///
    /// # Parameters
    /// - `prover_data`: The prover data returned by [`commit`].
    ///
    /// # Returns
    /// A vector of references to the committed matrices.
    fn get_matrices<'a, M: Matrix<T>>(&self, prover_data: &'a Self::ProverData<M>) -> Vec<&'a M>;

    /// Returns the height (number of rows) of each matrix in the batch.
    ///
    /// This is a utility method derived from [`get_matrices`].
    ///
    /// # Parameters
    /// - `prover_data`: The prover data returned by [`commit`].
    ///
    /// # Returns
    /// A vector containing the height of each committed matrix.
    fn get_matrix_heights<M: Matrix<T>>(&self, prover_data: &Self::ProverData<M>) -> Vec<usize> {
        self.get_matrices(prover_data)
            .iter()
            .map(|matrix| matrix.height())
            .collect()
    }

    /// Get the largest height of any committed matrix.
    ///
    /// # Panics
    /// This may panic if there are no committed matrices.
    fn get_max_height<M: Matrix<T>>(&self, prover_data: &Self::ProverData<M>) -> usize {
        self.get_matrix_heights(prover_data)
            .into_iter()
            .max()
            .unwrap_or_else(|| panic!("No committed matrices?"))
    }

    /// Verifies a batch opening at a specific row index against the original commitment.
    ///
    /// This is the verifier-side analogue of [`open_batch`]. The verifier receives:
    /// - The original commitment.
    /// - Dimensions of each matrix being opened (in the same order as originally committed).
    /// - The global index used for opening (interpreted as in [`open_batch`]).
    /// - A [`BatchOpeningRef`] containing the claimed opened values and the proof.
    ///
    /// # Parameters
    /// - `commit`: The original commitment.
    /// - `dimensions`: Dimensions of the committed matrices, in order.
    /// - `index`: The global row index that was opened.
    /// - `batch_opening`: A reference to the values and proof to verify.
    ///
    /// # Returns
    /// `Ok(())` if the opening is valid; otherwise returns a verification error.
    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, T, Self>,
    ) -> Result<(), Self::Error>;
}

/// Transcript I/O extension for MMCS implementations.
///
/// This is separate from [`Mmcs`] because commitments and opening proofs do not have
/// a universal canonical transcript representation.
pub trait MmcsTranscript<T: Send + Sync + Clone>: Mmcs<T> + Sized {
    /// Words used inside commitments and Merkle proofs.
    type Digest: Send + Sync + Clone;

    /// Append the transcript endpoint used to carry a commitment.
    fn append_commitment(
        &self,
        interactions: &mut Vec<Interaction>,
        label: Label,
        dimensions: &BatchDimensions,
    );

    /// Append the transcript endpoint used to carry only an opening proof hint.
    fn append_opening_proof_hint(
        &self,
        interactions: &mut Vec<Interaction>,
        label: Label,
        dimensions: &BatchDimensions,
    );
}

/// Prover-side transcript writing extension for MMCS implementations.
pub trait MmcsWriter<T: Send + Sync + Clone>: MmcsTranscript<T> {
    /// Write a commitment into the transcript.
    fn write_commitment<Ch>(
        &self,
        transcript: &mut ProverState<Ch>,
        label: Label,
        commitment: Self::Commitment,
    ) where
        Ch: DefaultCodec<Self::Digest>;

    /// Write only an opening proof as non-absorbed proof data.
    fn write_proof_hint<Ch>(
        &self,
        transcript: &mut ProverState<Ch>,
        opening_proof_label: Label,
        dimensions: &BatchDimensions,
        opening_proof: Self::Proof,
    ) where
        Ch: DefaultCodec<Self::Digest>;
}

/// Verifier-side transcript reading extension for MMCS implementations.
pub trait MmcsReader<T: Send + Sync + Clone>: MmcsTranscript<T> {
    /// Read a commitment from the transcript.
    fn read_commitment<'a, Ch>(
        &self,
        transcript: &mut VerifierState<'a, Ch>,
        label: Label,
        dimensions: &BatchDimensions,
    ) -> Result<TranscriptBound<Self::Commitment>, TranscriptError>
    where
        Ch: DefaultCodec<Self::Digest>;

    /// Read only a non-absorbed opening proof.
    fn read_opening_proof<'a, Ch>(
        &self,
        transcript: &mut VerifierState<'a, Ch>,
        opening_proof_label: Label,
        dimensions: &BatchDimensions,
    ) -> Result<Self::Proof, TranscriptError>
    where
        Ch: DefaultCodec<Self::Digest>;
}

/// A Batched opening proof.
///
/// Contains a collection of opened values at a Merkle proof for those openings.
///
/// Primarily used by the prover.
#[derive(Serialize, Deserialize, Clone)]
// Enable Serialize/Deserialize whenever T supports it.
#[serde(bound(serialize = "T: Serialize"))]
#[serde(bound(deserialize = "T: DeserializeOwned"))]
pub struct BatchOpening<T: Send + Sync + Clone, InputMmcs: Mmcs<T>> {
    /// The opened row values from each matrix in the batch.
    /// Each inner vector corresponds to one matrix.
    pub opened_values: Vec<Vec<T>>,
    /// The proof showing the values are valid openings.
    pub opening_proof: InputMmcs::Proof,
}

impl<T: Send + Sync + Clone, InputMmcs: Mmcs<T>> BatchOpening<T, InputMmcs> {
    /// Creates a new batch opening proof.
    #[inline]
    pub const fn new(opened_values: Vec<Vec<T>>, opening_proof: InputMmcs::Proof) -> Self {
        Self {
            opened_values,
            opening_proof,
        }
    }

    /// Unpacks the batch opening proof into its components.
    #[inline]
    pub fn unpack(self) -> (Vec<Vec<T>>, InputMmcs::Proof) {
        (self.opened_values, self.opening_proof)
    }
}

/// A reference to a batched opening proof.
///
/// Contains references to a collection of claimed opening values and a Merkle proof for those values.
///
/// Primarily used by the verifier.
#[derive(Copy, Clone)]
pub struct BatchOpeningRef<'a, T: Send + Sync + Clone, InputMmcs: Mmcs<T>> {
    /// Reference to the opened row values, as slices of base elements.
    pub opened_values: &'a [Vec<T>],
    /// Reference to the proof object used for verification.
    pub opening_proof: &'a InputMmcs::Proof,
}

impl<'a, T: Send + Sync + Clone, InputMmcs: Mmcs<T>> BatchOpeningRef<'a, T, InputMmcs> {
    /// Creates a new batch opening proof.
    #[inline]
    pub const fn new(opened_values: &'a [Vec<T>], opening_proof: &'a InputMmcs::Proof) -> Self {
        Self {
            opened_values,
            opening_proof,
        }
    }

    /// Unpacks the batch opening proof into its components.
    #[inline]
    pub const fn unpack(&self) -> (&'a [Vec<T>], &'a InputMmcs::Proof) {
        (self.opened_values, self.opening_proof)
    }
}

impl<'a, T: Send + Sync + Clone, InputMmcs: Mmcs<T>> From<&'a BatchOpening<T, InputMmcs>>
    for BatchOpeningRef<'a, T, InputMmcs>
{
    #[inline]
    fn from(batch_opening: &'a BatchOpening<T, InputMmcs>) -> Self {
        Self::new(&batch_opening.opened_values, &batch_opening.opening_proof)
    }
}
