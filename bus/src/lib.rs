#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

mod argument;
mod builder;
mod evaluation;
mod leaf;
mod multilinear;
mod plan;
mod product;
mod security;

pub use argument::{BusArgumentError, BusChallenges, BusProof, BusReductionOutput};
pub use builder::{
    BusActivation, BusInteractionBuilder, BusInteractionRecorder, BusSymbolicBuilder, RecordToken,
    SymbolicBusInteraction,
};
pub use evaluation::{BusEvaluation, BusEvaluationError};
pub use leaf::{BusDirection, BusLeafDeclaration, BusLeafError, BusLeaves, BusSelector};
pub use plan::{
    BusBlock, BusBlockOwner, BusDomain, BusExpressionLocation, BusPlan, BusPlanError, BusPlanInput,
    BusSecurityGeometry, BusTerminalShare, BusTupleSlot, UnsupportedBusAccess,
};
pub use product::{
    ProductGkrError, ProductGkrLayerProof, ProductGkrOutput, ProductGkrProof, ProductGkrRootShape,
    ProductGkrShape, ProductGkrShapeError,
};
