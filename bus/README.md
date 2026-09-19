# Plonky3 bus

Field-generic primitives for a direction-aware multiset bus.

The crate plans mixed-height bus layouts, materializes fingerprint factors, and reduces their
products with GKR.

The standalone reduction returns unauthenticated terminal claims. `p3-multi-stark` binds them to
committed trace columns with a composition sumcheck and prescribed-point openings.
