use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::Deref;

use p3_challenger::fs::{
    DefaultCodec, Interaction, Label, ProverState, TranscriptBound, TranscriptError, VerifierState,
};
use p3_field::{ExtensionField, Field};
use p3_matrix::extension::FlatMatrixView;
use p3_matrix::{Dimensions, Matrix};

use crate::{
    BatchDimensions, BatchOpening, BatchOpeningRef, Mmcs, MmcsReader, MmcsTranscript, MmcsWriter,
};

/// A wrapper to lift an MMCS from a base field `F` to an extension field `EF`.
///
/// `ExtensionMmcs` allows committing to and opening matrices over an extension field by internally
/// using an MMCS defined on the base field. It works by flattening each extension field element
/// into its base field coordinates for commitment, and then reconstructing them on opening.
#[derive(Clone, Debug)]
pub struct ExtensionMmcs<F, EF, InnerMmcs> {
    /// The inner MMCS instance used to handle commitments at the base field level.
    pub(crate) inner: InnerMmcs,

    pub(crate) _phantom: PhantomData<(F, EF)>,
}

impl<F, EF, InnerMmcs> MmcsTranscript<EF> for ExtensionMmcs<F, EF, InnerMmcs>
where
    F: Field,
    EF: ExtensionField<F>,
    InnerMmcs: MmcsTranscript<F>,
{
    type Digest = InnerMmcs::Digest;

    fn append_commitment(
        &self,
        interactions: &mut Vec<Interaction>,
        label: Label,
        dimensions: &BatchDimensions,
    ) {
        self.inner
            .append_commitment(interactions, label, dimensions);
    }

    fn append_opening_proof_hint(
        &self,
        interactions: &mut Vec<Interaction>,
        label: Label,
        dimensions: &BatchDimensions,
    ) {
        self.inner
            .append_opening_proof_hint(interactions, label, dimensions);
    }
}

impl<F, EF, InnerMmcs> MmcsWriter<EF> for ExtensionMmcs<F, EF, InnerMmcs>
where
    F: Field,
    EF: ExtensionField<F>,
    InnerMmcs: MmcsWriter<F>,
{
    fn write_commitment<Ch>(
        &self,
        transcript: &mut ProverState<Ch>,
        label: Label,
        commitment: Self::Commitment,
    ) where
        Ch: DefaultCodec<Self::Digest>,
    {
        self.inner.write_commitment(transcript, label, commitment);
    }

    fn write_proof_hint<Ch>(
        &self,
        transcript: &mut ProverState<Ch>,
        opening_proof_label: Label,
        dimensions: &BatchDimensions,
        opening_proof: Self::Proof,
    ) where
        Ch: DefaultCodec<Self::Digest>,
    {
        self.inner
            .write_proof_hint(transcript, opening_proof_label, dimensions, opening_proof);
    }
}

impl<F, EF, InnerMmcs> MmcsReader<EF> for ExtensionMmcs<F, EF, InnerMmcs>
where
    F: Field,
    EF: ExtensionField<F>,
    InnerMmcs: MmcsReader<F>,
{
    fn read_commitment<'a, Ch>(
        &self,
        transcript: &mut VerifierState<'a, Ch>,
        label: Label,
        dimensions: &BatchDimensions,
    ) -> Result<TranscriptBound<Self::Commitment>, TranscriptError>
    where
        Ch: DefaultCodec<Self::Digest>,
    {
        self.inner.read_commitment(transcript, label, dimensions)
    }

    fn read_opening_proof<'a, Ch>(
        &self,
        transcript: &mut VerifierState<'a, Ch>,
        opening_proof_label: Label,
        dimensions: &BatchDimensions,
    ) -> Result<Self::Proof, TranscriptError>
    where
        Ch: DefaultCodec<Self::Digest>,
    {
        self.inner
            .read_opening_proof(transcript, opening_proof_label, dimensions)
    }
}

impl<F, EF, InnerMmcs> ExtensionMmcs<F, EF, InnerMmcs> {
    pub const fn new(inner: InnerMmcs) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<F, EF, InnerMmcs> Mmcs<EF> for ExtensionMmcs<F, EF, InnerMmcs>
where
    F: Field,
    EF: ExtensionField<F>,
    InnerMmcs: Mmcs<F>,
{
    type ProverData<M> = InnerMmcs::ProverData<FlatMatrixView<F, EF, M>>;
    type Commitment = InnerMmcs::Commitment;
    type Proof = InnerMmcs::Proof;
    type Error = InnerMmcs::Error;

    fn commit<M: Matrix<EF>>(&self, inputs: Vec<M>) -> (Self::Commitment, Self::ProverData<M>) {
        self.inner
            .commit(inputs.into_iter().map(FlatMatrixView::new).collect())
    }

    fn open_batch<M: Matrix<EF>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> BatchOpening<EF, Self> {
        let (inner_opened_values, inner_proof) = self.inner.open_batch(index, prover_data).unpack();
        let opened_ext_values = inner_opened_values
            .into_iter()
            .map(EF::reconstitute_from_base)
            .collect();
        BatchOpening::new(opened_ext_values, inner_proof)
    }

    fn get_matrices<'a, M: Matrix<EF>>(&self, prover_data: &'a Self::ProverData<M>) -> Vec<&'a M> {
        self.inner
            .get_matrices(prover_data)
            .into_iter()
            .map(|mat| mat.deref())
            .collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, EF, Self>,
    ) -> Result<(), Self::Error> {
        let opened_base_values: Vec<Vec<F>> = batch_opening
            .opened_values
            .iter()
            .cloned()
            .map(EF::flatten_to_base)
            .collect();
        let base_dimensions = dimensions
            .iter()
            .map(|dim| Dimensions {
                width: dim.width * EF::DIMENSION,
                height: dim.height,
            })
            .collect::<Vec<_>>();
        self.inner.verify_batch(
            commit,
            &base_dimensions,
            index,
            BatchOpeningRef::new(&opened_base_values, batch_opening.opening_proof),
        )
    }
}
