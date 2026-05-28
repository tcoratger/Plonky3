//! Debug helpers for transcript inspection.

use alloc::collections::BTreeMap;
use core::any::type_name;

use p3_field::{BasedVectorSpace, PrimeField};

use crate::fs::codecs::Codec;
use crate::fs::codecs::decode_field::field_byte_size;
use crate::fs::domain_separator::DomainSeparator;
use crate::fs::pattern::{Hierarchy, Interaction, InteractionPattern, Kind, Label, Length};
use crate::fs::unit::Unit;

pub(crate) fn validate_unique_atomic_labels(interactions: &[Interaction]) {
    for (position, interaction) in interactions.iter().enumerate() {
        if interaction.hierarchy() != Hierarchy::Atomic {
            continue;
        }

        assert!(
            !interactions[..position].iter().any(|previous| {
                previous.hierarchy() == Hierarchy::Atomic && previous.label() == interaction.label()
            }),
            "duplicate atomic transcript label: {}",
            interaction.label()
        );
    }
}

pub(crate) fn build_label_to_index(interactions: &[Interaction]) -> BTreeMap<Label, usize> {
    validate_unique_atomic_labels(interactions);
    let mut label_to_index = BTreeMap::new();
    for (position, interaction) in interactions.iter().enumerate() {
        if interaction.hierarchy() == Hierarchy::Atomic {
            let previous = label_to_index.insert(interaction.label(), position);
            debug_assert!(
                previous.is_none(),
                "duplicate labels are checked before building the debug label index"
            );
        }
    }

    label_to_index
}

impl InteractionPattern {
    /// Return the interaction index for an atomic label.
    #[must_use]
    pub(crate) fn interaction_index_for(&self, label: Label) -> Option<usize> {
        self.label_to_index.get(label).copied()
    }
}

impl<U: Unit> DomainSeparator<U> {
    /// Debug helper: return the proof byte offset for a wire-carrying transcript label.
    ///
    /// This interprets the pattern using a single base-field codec family:
    /// base-field values use `Cdc`, extension-field values use one `Cdc` encoding per basis
    /// coefficient, and proof-of-work witnesses use the base-field canonical byte width.
    ///
    /// Returns `None` for non-wire labels, dynamic lengths, or types outside this layout.
    #[must_use]
    pub fn byte_offset_for_label<Val, Challenge, Cdc, C>(&self, label: Label) -> Option<usize>
    where
        Val: PrimeField,
        Challenge: BasedVectorSpace<Val>,
        Cdc: Codec<C, Val>,
    {
        let target_index = self.pattern().interaction_index_for(label)?;
        let target = self.pattern().interactions()[target_index];
        if !matches!(
            target.kind(),
            Kind::Message | Kind::Hint | Kind::Pow | Kind::Salt
        ) {
            return None;
        }
        wire_byte_len::<Val, Challenge, Cdc, C>(&target)?;

        self.pattern().interactions()[..target_index]
            .iter()
            .try_fold(0usize, |offset, interaction| {
                wire_byte_len::<Val, Challenge, Cdc, C>(interaction).map(|len| offset + len)
            })
    }
}

fn wire_byte_len<Val, Challenge, Cdc, C>(interaction: &Interaction) -> Option<usize>
where
    Val: PrimeField,
    Challenge: BasedVectorSpace<Val>,
    Cdc: Codec<C, Val>,
{
    if interaction.hierarchy() != Hierarchy::Atomic {
        return Some(0);
    }

    match interaction.kind() {
        Kind::Challenge | Kind::Protocol | Kind::Public => Some(0),
        Kind::Pow => Some(field_byte_size::<Val>()),
        Kind::Salt => match interaction.length() {
            Length::Fixed(n) => Some(n),
            _ => None,
        },
        Kind::Message | Kind::Hint => {
            let count = match interaction.length() {
                Length::Scalar => 1,
                Length::Fixed(n) => n,
                Length::None => 0,
                Length::Bits(_) | Length::Dynamic => return None,
            };

            if interaction.type_name() == type_name::<Val>() {
                Some(count * Cdc::byte_len())
            } else if interaction.type_name() == type_name::<Challenge>() {
                Some(count * Challenge::DIMENSION * Cdc::byte_len())
            } else if interaction.type_name() == type_name::<u8>() {
                Some(count)
            } else {
                None
            }
        }
    }
}
