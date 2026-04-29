#!/usr/bin/env python3
"""WHIR parameter optimizer with 3D Pareto frontier and matplotlib plots.

Explores the WHIR polynomial commitment scheme parameter space to find
optimal tradeoffs between prover cost, proof size, and verifier cost.

# Background

WHIR (Arnon, Chiesa, Fenzi, Yogev 2024) is an IOP of proximity for
constrained Reed-Solomon codes. It reduces testing proximity to a
large code into testing proximity to a smaller code over multiple
rounds. Each round folds the polynomial (eliminating k variables),
shrinks the evaluation domain, and performs STIR proximity queries.

The protocol has many tunable parameters that create a rich tradeoff
space. This script analytically estimates the three cost metrics for
every valid parameter combination and computes the Pareto frontier.

# Soundness model

The soundness functions mirror the Rust implementation exactly:
  - whir/src/parameters/soundness.rs
  - whir/src/parameters/whir.rs

Three security assumptions are supported:
  - UniqueDecoding:  delta = (1 - rho) / 2.  No conjectures needed.
  - JohnsonBound:    delta = 1 - sqrt(rho) - eta.  Uses [BCSS25]
                     improved proximity gaps (O(n / eta^5) vs old O(n^2 / eta^7)).
  - CapacityBound:   delta = 1 - rho - eta.  Requires unproven conjectures
                     on mutual correlated agreement up to capacity.

# Knobs explored

Per Giacomo Fenzi's suggestion, the sweep covers (per round):
  1) Starting code rate (log_2(1/rho) in {1, 2, 3, 4})
  2) Folding parameter (variables eliminated per round, in {3..8})
  3) RS domain reduction schedule (how much the domain shrinks)
  4) Proof-of-work bits (global cap, in {10..32})
  5) Merkle cap height (0, 4, or 8 — wider caps = shorter auth paths)
  6) Proximity parameter (controlled via the soundness assumption choice)

# Output

  - Ranked tables: Pareto frontier, fastest prover, smallest proof, fastest verifier
  - Recommended configs with ready-to-paste cargo commands
  - Optional CSV/JSON export
  - Optional matplotlib plots showing all configs color-coded by rate
    with the Pareto frontier overlaid

# Usage

  python3 whir_optimizer.py -d 26 -l 128 --soundness JohnsonBound --field-bits 155 --plot
  python3 whir_optimizer.py -d 20 -l 128 --soundness CapacityBound --field-bits 155 --csv out.csv

# Field size requirements for 128-bit security

  - JohnsonBound requires degree-5 extension of KoalaBear (155 bits).
    Degree-4 (124 bits) only reaches ~100-bit JB security.
  - CapacityBound works with degree-5 at 128-bit, or degree-4 at ~115-bit.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Optional

# log_2(10) appears in the proximity parameter eta = rho/20 or sqrt(rho)/20.
# The factor 20 contributes log_2(10) + 1 bits to the eta formula.
LOG2_10 = math.log2(10)


# Soundness model
# ===============
#
# These functions replicate the Rust code in soundness.rs line-by-line.
# Each computes a quantity from the WHIR security proof (Theorem 5.2):
#
#   - log_eta:             gap between proximity parameter and bound
#   - list_size_bits:      log_2 of the list-decoding radius
#   - prox_gaps_error:     bits of security from proximity gaps
#   - log_1_delta:         log_2(1 - delta) for query error computation
#   - queries:             number of STIR queries to match target security
#   - ood_error:           out-of-domain sampling error
#   - fold_sumcheck_error: sumcheck verification error during folding
#   - folding_pow_bits:    proof-of-work needed to cover the folding gap


def log_eta(soundness_type: str, log_inv_rate: int) -> float:
    """Compute log_2(eta), the distance between the proximity parameter and the bound.

    # Overview

    In the WHIR soundness proof, delta must be strictly less than the
    proximity bound. The gap eta controls how tight we operate:
      - Larger eta = farther from the bound = more security margin per query
      - Smaller eta = closer to the bound = fewer queries needed but weaker gaps

    # Returns

    log_2(eta) where eta is set as:
      - UniqueDecoding: not used (returns 0)
      - JohnsonBound:   eta = sqrt(rho) / 20
      - CapacityBound:  eta = rho / 20
    """
    if soundness_type == "UniqueDecoding":
        # Unique decoding doesn't use eta — the distance is fixed at (1-rho)/2.
        return 0.0
    elif soundness_type == "JohnsonBound":
        # eta = sqrt(rho) / 20
        # log_2(eta) = log_2(sqrt(rho)) - log_2(20)
        #            = -0.5 * log_inv_rate - (log_2(10) + 1)
        return -(0.5 * log_inv_rate + LOG2_10 + 1.0)
    elif soundness_type == "CapacityBound":
        # eta = rho / 20
        # log_2(eta) = -log_inv_rate - (log_2(10) + 1)
        return -(log_inv_rate + LOG2_10 + 1.0)
    raise ValueError(f"Unknown: {soundness_type}")


def list_size_bits(soundness_type: str, log_degree: int, log_inv_rate: int) -> float:
    """Compute log_2 of the list-decoding radius at the chosen proximity parameter.

    # Overview

    The number of codewords within distance delta of any received word
    is bounded by the Johnson bound or conjectured capacity bound.
    This list size appears in union bounds throughout the security proof.

    # Returns

    log_2(L) where L is the maximum list size:
      - UniqueDecoding: L = 1 (always, by definition)
      - JohnsonBound:   L = 1 / (2 * eta * sqrt(rho))  [Theorem 4.3]
      - CapacityBound:  L = degree / (rho * eta)        [Conjecture]
    """
    if soundness_type == "UniqueDecoding":
        # Unique decoding: at most one codeword within distance (1-rho)/2.
        return 0.0
    elif soundness_type == "JohnsonBound":
        # L = (2 * eta * sqrt(rho))^{-1}
        # log_2(L) = log_inv_rate/2 - 1 - log_2(eta)
        return log_inv_rate / 2.0 - (1.0 + log_eta(soundness_type, log_inv_rate))
    elif soundness_type == "CapacityBound":
        # L = degree / (rho * eta)
        # log_2(L) = log_degree + log_inv_rate - log_2(eta)
        return (log_degree + log_inv_rate) - log_eta(soundness_type, log_inv_rate)
    raise ValueError


def prox_gaps_error(
    soundness_type: str,
    log_degree: int,
    log_inv_rate: int,
    field_size_bits: int,
    num_functions: int = 2,
) -> float:
    """Compute bits of security from the proximity gaps theorem.

    # Overview

    Proximity gaps bound the probability that a random linear combination
    of functions close to a code stays close to the code. This is the core
    technical tool enabling the WHIR folding step.

    The error is: (num_exceptional_z) * (num_functions - 1) / |F|

    # JohnsonBound improvement [BCSS25]

    The [BCSS25] paper ("On Proximity Gaps for RS Codes", eprint 2025/2055)
    improves the bound from O(n^2 / eta^7) to O(n / eta^5).
    This gives ~log_2(n) extra bits of security, making 128-bit provable
    security feasible with degree-5 extensions of small primes.

    The dominant term with eta = sqrt(rho)/20 and m = 10 is:
      a ~ (2 * 10.5^5 / 3) * n / rho^{3/2}

    # Returns

    Bits of security = field_size_bits - log_2(error_probability).
    """
    assert num_functions >= 2
    if soundness_type == "UniqueDecoding":
        # In unique decoding the error is |L| / |F| = degree / (rho * |F|).
        # log_2(error) = log_degree + log_inv_rate
        error = log_degree + log_inv_rate
    elif soundness_type == "JohnsonBound":
        # Theorem 1.5 from [BCSS25]: exceptional z count is O(n / eta^5).
        # With eta = sqrt(rho)/20 and m = max(ceil(sqrt(rho)/(2*eta)), 3) = 10:
        #   a ~ (2 * 10.5^5 / 3) * n * rho^{-3/2}
        #
        # In log form:
        #   log_2(a) = log_2(n) + log_2(2 * 10.5^5 / 3) + 1.5 * log_inv_rate
        log_n = log_degree + log_inv_rate
        # 2 * 10.5^5 / 3 ~ 85085.44, log_2 ~ 16.38
        constant = math.log2(2.0 * (10.5**5) / 3.0)
        error = log_n + constant + 1.5 * log_inv_rate
    elif soundness_type == "CapacityBound":
        # Conjectured: error is degree / (eta * rho^2)
        le = log_eta(soundness_type, log_inv_rate)
        error = (log_degree + 2 * log_inv_rate) - le
    else:
        raise ValueError
    # Union bound over (num_functions - 1) choices, then divide by |F|.
    return field_size_bits - (error + math.log2(max(1, num_functions - 1)))


def log_1_delta(soundness_type: str, log_inv_rate: int) -> float:
    """Compute log_2(1 - delta) where delta is the proximity parameter.

    # Overview

    Each STIR query catches a disagreement with probability delta.
    After t independent queries, the probability of missing all
    disagreements is (1 - delta)^t. This function returns log_2(1 - delta)
    so the query count can be computed as t = -lambda / log_2(1 - delta).

    # Proximity parameter delta

    ```
    UniqueDecoding:  delta = (1 - rho) / 2
    JohnsonBound:    delta = 1 - sqrt(rho) - eta
    CapacityBound:   delta = 1 - rho - eta
    ```

    Higher rate (smaller rho) gives larger delta, meaning fewer queries
    are needed per round. This is why the code rate increases across
    WHIR rounds — each round needs fewer queries than the last.
    """
    le = log_eta(soundness_type, log_inv_rate)
    # Convert log-space eta back to a real number.
    eta = 2.0**le
    # rho = 2^{-log_inv_rate}
    rate = 1.0 / (1 << log_inv_rate)
    if soundness_type == "UniqueDecoding":
        delta = 0.5 * (1.0 - rate)
    elif soundness_type == "JohnsonBound":
        delta = 1.0 - math.sqrt(rate) - eta
    elif soundness_type == "CapacityBound":
        delta = 1.0 - rate - eta
    else:
        raise ValueError
    # Guard against degenerate cases where delta leaves [0, 1).
    if delta <= 0 or delta >= 1:
        return -1e-10
    return math.log2(1.0 - delta)


def queries(soundness_type: str, protocol_sec: int, log_inv_rate: int) -> int:
    """Compute the number of STIR proximity queries to reach the target security.

    # Overview

    Solves (1 - delta)^t <= 2^{-lambda} for t, where lambda is the
    protocol security level (total security minus proof-of-work bits).

    # Returns

    Minimum number of queries (at least 1).
    """
    l1d = log_1_delta(soundness_type, log_inv_rate)
    # If log_2(1-delta) >= 0, each query provides zero bits of security.
    # Return a sentinel large value so this config is rejected.
    if l1d >= 0:
        return 10000
    # t = ceil(-lambda / log_2(1-delta))
    return max(1, math.ceil(-protocol_sec / l1d))


def queries_error(soundness_type: str, log_inv_rate: int, nq: int) -> float:
    """Compute bits of security from a given number of proximity queries.

    # Returns

    -nq * log_2(1 - delta), i.e. how many bits of security the queries provide.
    """
    return -nq * log_1_delta(soundness_type, log_inv_rate)


def ood_error(
    soundness_type: str,
    log_degree: int,
    log_inv_rate: int,
    field_size_bits: int,
    ood_samples: int,
) -> float:
    """Compute bits of security from out-of-domain (OOD) sampling.

    # Overview

    OOD sampling (Lemma 4.25, adapted from STIR Lemma 4.5) reduces
    the list-decoding regime to a unique-decoding-like setting.
    The verifier samples random points outside the evaluation domain
    and checks that the prover's claimed evaluations are consistent.

    Two distinct codewords in the list can only agree at a random
    OOD point with probability degree / |F|. After s independent
    samples, the collision probability drops to (degree / |F|)^s.

    # Returns

    Bits of security = s * field_size_bits - 2 * list_size_bits - s * log_degree + 1.
    """
    if soundness_type == "UniqueDecoding":
        # No OOD needed — unique decoding already guarantees at most one codeword.
        return 0.0
    ls = list_size_bits(soundness_type, log_degree, log_inv_rate)
    # Error probability: list_size^2 * (degree / |F|)^s
    # In bits: 2 * ls + s * log_degree - s * field_size_bits - 1
    error = 2.0 * ls + (log_degree * ood_samples)
    return (ood_samples * field_size_bits) + 1.0 - error


def determine_ood_samples(
    soundness_type: str,
    security_level: int,
    log_degree: int,
    log_inv_rate: int,
    field_size_bits: int,
) -> int:
    """Find the minimum number of OOD samples to reach the target security.

    # Algorithm

    Binary search starting from 1 sample, incrementing until the
    OOD error bound meets or exceeds the required security level.

    # Returns

    Number of OOD samples needed (0 for unique decoding).
    """
    if soundness_type == "UniqueDecoding":
        return 0
    # Linear scan — the number is always small (typically 1-3).
    for s in range(1, 64):
        if (
            ood_error(soundness_type, log_degree, log_inv_rate, field_size_bits, s)
            >= security_level
        ):
            return s
    # Fallback — this means the field is too small for the target security.
    return 64


def fold_sumcheck_error(
    soundness_type: str, field_size_bits: int, num_variables: int, log_inv_rate: int
) -> float:
    """Compute bits of security from the folding sumcheck step.

    # Overview

    During folding, the verifier checks a degree-2 sumcheck identity
    with a random challenge. An adversary controlling L codewords
    in the list can bias the check with probability L / |F|.

    # Returns

    Bits of security = field_size_bits - list_size_bits - 1.
    The -1 accounts for the union bound over the list.
    """
    ls = list_size_bits(soundness_type, num_variables, log_inv_rate)
    return field_size_bits - (ls + 1.0)


def queries_combination_error(
    soundness_type: str,
    field_size_bits: int,
    num_variables: int,
    log_inv_rate: int,
    ood_samples: int,
    nq: int,
) -> float:
    """Compute bits of security from the query-combination step.

    # Overview

    After STIR queries and OOD samples, the verifier takes a random
    linear combination. An adversary must fool this combination for
    every codeword in the list, giving error:

      (ood_samples + num_queries) * list_size / |F|

    # Returns

    Bits of security = field_size_bits - log_2(ood + queries) - list_size_bits - 1.
    """
    ls = list_size_bits(soundness_type, num_variables, log_inv_rate)
    # Total evaluation points available for combination.
    log_comb = math.log2(max(1, ood_samples + nq))
    return field_size_bits - (log_comb + ls + 1.0)


def folding_pow_bits(
    soundness_type: str,
    security_level: int,
    field_size_bits: int,
    num_variables: int,
    log_inv_rate: int,
) -> float:
    """Compute the proof-of-work difficulty needed for the folding step.

    # Overview

    The folding step has two independent error sources:
      1. Proximity gaps:  far-from-RS functions surviving the fold
      2. Sumcheck:        wrong claim accepted by the sumcheck verifier

    The overall security is limited by the weaker bound. Proof-of-work
    must bridge the gap between the weakest bound and the target:

      pow_bits = max(0, security_level - min(prox_gaps, sumcheck))

    # Returns

    Minimum proof-of-work bits needed (0 if algebraic bounds suffice).
    """
    pg = prox_gaps_error(
        soundness_type, num_variables, log_inv_rate, field_size_bits, 2
    )
    sc = fold_sumcheck_error(
        soundness_type, field_size_bits, num_variables, log_inv_rate
    )
    # The chain is only as strong as its weakest link.
    return max(0.0, security_level - min(pg, sc))


# Config derivation
# =================
#
# These structures and functions mirror WhirConfig::new in whir.rs.
# They derive the full per-round protocol parameters from the user-facing
# knobs (rate, fold factor, RS reduction, PoW, Merkle cap).

# When fewer than this many variables remain, the prover sends the
# polynomial coefficients directly instead of folding further.
# This avoids the overhead of a Merkle commitment for a tiny polynomial.
MAX_VARS_DIRECT = 6


@dataclass
class RoundConfig:
    """Derived parameters for a single intermediate WHIR round.

    Each round folds the polynomial, commits the folded oracle,
    runs OOD sampling, performs STIR queries, and does sumcheck.
    """

    # Proof-of-work bits for the STIR query phase of this round.
    pow_bits: int
    # Proof-of-work bits for the folding sumcheck phase.
    folding_pow_bits: int
    # Number of STIR proximity queries sent by the verifier.
    num_queries: int
    # Number of out-of-domain evaluation samples.
    ood_samples: int
    # Multilinear variables remaining after folding in this round.
    num_variables: int
    # Variables eliminated by folding in this round.
    folding_factor: int
    # log_2 of the evaluation domain size before folding.
    log_domain_size: int
    # By how many bits the RS domain shrinks in this round.
    rs_reduction: int


@dataclass
class WhirConfig:
    """Fully derived WHIR protocol configuration with cost estimates.

    Built from user-facing parameters (rate, fold schedule, PoW, etc.)
    plus the polynomial size. Contains all per-round derived values
    and the three estimated cost metrics.
    """

    # Number of variables in the multilinear polynomial (degree = 2^num_variables).
    num_variables: int
    # Target security level in bits.
    security_level: int
    # Maximum allowed proof-of-work difficulty across all rounds.
    max_pow_bits: int
    # Which proximity assumption is used for the soundness analysis.
    soundness_type: str
    # Number of bits in the extension field.
    field_size_bits: int
    # log_2(1/rho) for the initial Reed-Solomon code.
    starting_log_inv_rate: int
    # By how many bits the RS domain shrinks at the first round.
    rs_domain_initial_reduction: int
    # Per-round folding factors. Repeats the last entry for remaining rounds.
    fold_schedule: list
    # OOD samples during the commitment phase (before any folding).
    commitment_ood_samples: int
    # PoW bits for the very first folding sumcheck.
    starting_folding_pow_bits: int
    # Per-round derived configurations.
    rounds: list
    # Number of STIR queries in the final proximity test.
    final_queries: int
    # PoW bits for the final query phase.
    final_pow_bits: int
    # Number of sumcheck rounds in the final direct-send phase.
    final_sumcheck_rounds: int
    # PoW bits for the final folding sumcheck.
    final_folding_pow_bits: int
    # Whether all required PoW fits within the configured maximum.
    pow_ok: bool = True
    # Merkle tree cap height (0 = single root hash).
    merkle_cap_height: int = 0
    # Estimated proof size in bytes.
    est_proof_size_bytes: int = 0
    # Raw prover cost in weighted operation counts (unitless).
    est_prover_cost: float = 0.0
    # Raw verifier cost in weighted operation counts (unitless).
    est_verifier_cost: float = 0.0
    # Estimated prover time in milliseconds (calibrated).
    est_prover_ms: float = 0.0
    # Estimated verifier time in microseconds (calibrated).
    est_verifier_us: float = 0.0


def compute_schedule(num_variables: int, fold_schedule: list):
    """Expand a fold schedule into the full list of per-round folding factors.

    # Overview

    The user provides a (possibly short) list of folding factors.
    If the polynomial has more variables than the schedule covers,
    the last entry is repeated until the remaining variables drop
    to the direct-send threshold.

    # Example

    ```
    num_variables = 26, fold_schedule = [7, 4]
      Round 0: fold 7 → 19 remaining
      Round 1: fold 4 → 15 remaining
      Round 2: fold 4 → 11 remaining  (repeating last entry)
      Round 3: fold 4 →  7 remaining
      Round 4: fold 4 →  3 remaining  (below threshold, stop)
      → schedule = [7, 4, 4, 4, 4], final_vars = 3
    ```

    # Returns

    Tuple of (num_stir_rounds, final_sumcheck_vars, expanded_schedule).
    """
    remaining = num_variables
    schedule = []
    # Repeat the last entry for any rounds beyond the explicit schedule.
    last_k = fold_schedule[-1]
    # First, consume the explicit entries.
    for k in fold_schedule:
        schedule.append(k)
        remaining -= k
        # Stop as soon as the polynomial is small enough to send directly.
        if remaining <= MAX_VARS_DIRECT:
            break
    # Extend with the last factor if more folding is needed.
    while remaining > MAX_VARS_DIRECT:
        schedule.append(last_k)
        remaining -= last_k
    final_vars = max(0, remaining)
    # The first fold happens at commitment time, not during STIR rounds.
    # So the number of STIR rounds is one less than the number of folds.
    num_stir = max(0, len(schedule) - 1)
    return num_stir, final_vars, schedule


def derive_config(
    num_variables: int,
    starting_log_inv_rate: int,
    fold_schedule: list,
    rs_reduction_schedule: list,
    pow_bits: int,
    security_level: int,
    soundness_type: str,
    field_size_bits: int,
    merkle_cap_height: int = 0,
    two_adicity: int = 24,
) -> Optional[WhirConfig]:
    """Derive the full protocol configuration from user-facing parameters.

    # Overview

    Mirrors WhirConfig::new from whir.rs. Validates constraints, computes
    per-round query counts and PoW requirements, estimates all three cost
    metrics, and checks whether the configuration is feasible.

    # Algorithm

    Phase 1: Validate inputs (fold <= num_vars, rs_red <= fold, two-adicity).
    Phase 2: Determine round structure from the fold schedule.
    Phase 3: Compute commitment-phase OOD samples and starting PoW.
    Phase 4: For each STIR round, derive query count, OOD, and PoW.
    Phase 5: Derive final-round parameters and check PoW feasibility.

    # Returns

    A fully populated config, or None if the parameters are invalid.
    """
    # Phase 1: Input validation.
    if not fold_schedule or fold_schedule[0] == 0 or fold_schedule[0] > num_variables:
        return None
    rs_red_0 = rs_reduction_schedule[0] if rs_reduction_schedule else 1
    # RS reduction must not exceed fold factor (would increase code rate).
    if rs_red_0 > fold_schedule[0]:
        return None
    # After the first fold, the domain size must fit within the base field's
    # two-adic subgroup so that FFT twiddle factors stay in the base field.
    log_domain = num_variables + starting_log_inv_rate
    if log_domain - fold_schedule[0] > two_adicity:
        return None

    # PoW provides an independent additive security contribution.
    # The algebraic protocol only needs to achieve the remainder.
    protocol_sec = max(0, security_level - pow_bits)

    # Phase 2: Expand the fold schedule and determine round structure.
    num_stir, final_sc_rounds, schedule = compute_schedule(num_variables, fold_schedule)

    # Pad the RS reduction schedule: round 0 uses the explicit value,
    # all subsequent rounds halve the domain (reduction factor = 1).
    full_rs_red = list(rs_reduction_schedule)
    while len(full_rs_red) < num_stir:
        full_rs_red.append(1)

    # Mutable state that evolves as we step through rounds.
    log_inv_rate = starting_log_inv_rate
    nv = num_variables

    # Phase 3: Commitment-phase parameters (before any STIR rounds).
    commitment_ood = determine_ood_samples(
        soundness_type, security_level, nv, log_inv_rate, field_size_bits
    )
    start_fold_pow = folding_pow_bits(
        soundness_type, security_level, field_size_bits, nv, log_inv_rate
    )

    # The initial fold (at commitment time) consumes schedule[0] variables.
    nv -= schedule[0]
    log_ds = num_variables + starting_log_inv_rate

    # Phase 4: Per-round derivation.
    rounds = []
    for i in range(num_stir):
        rs_red = full_rs_red[i]
        fold_k = schedule[i]
        next_fold_k = schedule[i + 1] if i + 1 < len(schedule) else schedule[-1]
        if rs_red > fold_k:
            return None

        # Code rate increases by (fold - rs_reduction) bits each round.
        # Queries use the OLD rate; OOD and folding use the NEW rate.
        next_rate = log_inv_rate + (fold_k - rs_red)
        # Number of queries at the current (pre-fold) rate.
        nq = queries(soundness_type, protocol_sec, log_inv_rate)
        # OOD samples at the post-fold rate.
        ood = determine_ood_samples(
            soundness_type, security_level, nv, next_rate, field_size_bits
        )
        # Two independent error bounds determine the PoW requirement:
        #   query_error:       (1 - delta)^nq
        #   combination_error: union bound over OOD + queries
        q_err = queries_error(soundness_type, log_inv_rate, nq)
        c_err = queries_combination_error(
            soundness_type, field_size_bits, nv, next_rate, ood, nq
        )
        # PoW bridges the gap between the target and the weaker bound.
        rnd_pow = max(0.0, security_level - min(q_err, c_err))
        f_pow = folding_pow_bits(
            soundness_type, security_level, field_size_bits, nv, next_rate
        )

        rounds.append(
            RoundConfig(int(rnd_pow), int(f_pow), nq, ood, nv, fold_k, log_ds, rs_red)
        )

        # Advance state for the next round.
        nv -= next_fold_k
        log_inv_rate = next_rate
        log_ds -= rs_red

    # Phase 5: Final round.
    final_q = queries(soundness_type, protocol_sec, log_inv_rate)
    final_pow = max(
        0.0, security_level - queries_error(soundness_type, log_inv_rate, final_q)
    )
    # The final sumcheck error is bounded by 1/|F|.
    final_fold_pow = max(0.0, security_level - (field_size_bits - 1))

    # Check that all required PoW values fit within the global cap.
    all_pow = [int(start_fold_pow), int(final_pow), int(final_fold_pow)]
    for r in rounds:
        all_pow.extend([r.pow_bits, r.folding_pow_bits])
    pow_ok = all(p <= pow_bits for p in all_pow)

    cfg = WhirConfig(
        num_variables=num_variables,
        security_level=security_level,
        max_pow_bits=pow_bits,
        soundness_type=soundness_type,
        field_size_bits=field_size_bits,
        starting_log_inv_rate=starting_log_inv_rate,
        rs_domain_initial_reduction=rs_red_0,
        fold_schedule=schedule,
        commitment_ood_samples=commitment_ood,
        starting_folding_pow_bits=int(start_fold_pow),
        rounds=rounds,
        final_queries=final_q,
        final_pow_bits=int(final_pow),
        final_sumcheck_rounds=final_sc_rounds,
        final_folding_pow_bits=int(final_fold_pow),
        pow_ok=pow_ok,
        merkle_cap_height=merkle_cap_height,
    )
    # Fill in proof size (real bytes) and cost estimates.
    estimate_proof_size(cfg)
    estimate_prover_cost(cfg)
    estimate_verifier_cost(cfg)
    # Convert raw costs to calibrated wall-clock estimates.
    cfg.est_prover_ms = cfg.est_prover_cost * PROVER_MS_PER_UNIT
    cfg.est_verifier_us = cfg.est_verifier_cost * VERIFIER_US_PER_UNIT
    return cfg


# Cost estimation
# ===============
#
# Proof size is computed in real bytes by counting all transmitted data.
#
# Prover and verifier costs are estimated as weighted operation counts,
# then converted to approximate wall-clock time using calibration
# constants fitted against real measurements from the Rust implementation
# (examples/whir.rs on Apple M-series, d=26, single-threaded).
#
# Calibration data points (12 configs, sec=90, CapacityBound, KoalaBear):
#   Prover:   median ratio = 8.87e-7 ms per cost unit
#   Verifier: median ratio = 0.128 us per cost unit
#
# The estimates are approximate (within ~30% of real timings) and useful
# for ranking configurations. For precise numbers, validate the top
# Pareto candidates by running the actual Rust implementation.

# Calibration constants: convert raw cost units to wall-clock time.
# Fitted against 12 real measurements on Apple M-series (single-threaded).
PROVER_MS_PER_UNIT = 8.87e-7
VERIFIER_US_PER_UNIT = 0.128


def estimate_proof_size(cfg: WhirConfig):
    """Estimate proof size in bytes by counting all transmitted data.

    # Proof components

    ```
    Component                   Size per unit
    ─────────────────────────   ──────────────
    Merkle commitment           cap_hashes * 32 B     (hash digest)
    OOD answer                  16 B                  (extension field element)
    Sumcheck round polynomial   2 * 16 B              (two EF evaluations)
    PoW witness                 4 B                   (base field element)
    Query opening (base)        2^k * 4 B + path * 32 B
    Query opening (extension)   2^k * 16 B + path * 32 B
    Final polynomial            2^{final_vars} * 16 B
    ```

    The Merkle authentication path length depends on the tree height
    minus the cap height: path = log_domain - fold_factor - cap_height.
    """
    # Size of one base field element (KoalaBear = 31-bit prime ≈ 4 bytes).
    F = 4
    # Size of one extension field element (degree-4 or degree-5 extension).
    EF = 16
    # Size of one Merkle hash digest (8 x u32 = 32 bytes).
    H = 32

    total = 0
    # Number of hash digests in the Merkle tree cap.
    # cap_height=0 means a single root hash, cap_height=k means 2^k hashes.
    cap_hashes = max(1, 1 << cfg.merkle_cap_height)

    # Initial commitment: Merkle cap for the encoded polynomial.
    total += cap_hashes * H
    # Initial OOD answers: one EF element per sample.
    total += cfg.commitment_ood_samples * EF
    # Initial sumcheck: fold_schedule[0] rounds, each sends two EF evaluations.
    # Plus one base field PoW witness.
    total += cfg.fold_schedule[0] * 2 * EF + F

    log_dom = cfg.num_variables + cfg.starting_log_inv_rate
    for i, r in enumerate(cfg.rounds):
        # Round commitment: Merkle cap for the folded oracle.
        total += cap_hashes * H
        # OOD answers plus PoW witness.
        total += r.ood_samples * EF + F
        # Query openings: each query reveals 2^fold_factor leaf values
        # plus a Merkle authentication path of (tree_height - cap_height) hashes.
        vals = 1 << r.folding_factor
        path = max(0, r.log_domain_size - r.folding_factor - cfg.merkle_cap_height)
        # First round queries use base field; subsequent use extension field.
        vbytes = F if i == 0 else EF
        total += r.num_queries * (vals * vbytes + path * H)
        # Next round's sumcheck polynomial evaluations plus PoW witness.
        nf = (
            cfg.fold_schedule[i + 1]
            if i + 1 < len(cfg.fold_schedule)
            else cfg.fold_schedule[-1]
        )
        total += nf * 2 * EF + F
        log_dom -= r.rs_reduction

    # Final polynomial: sent in the clear as 2^{final_vars} EF coefficients.
    total += (1 << cfg.final_sumcheck_rounds) * EF + F
    # Final query openings (extension field values + Merkle paths).
    lf = cfg.fold_schedule[-1]
    fp = max(0, log_dom - lf - cfg.merkle_cap_height)
    total += cfg.final_queries * ((1 << lf) * EF + fp * H)
    # Final sumcheck (if there are remaining variables).
    if cfg.final_sumcheck_rounds > 0:
        total += cfg.final_sumcheck_rounds * 2 * EF + F

    cfg.est_proof_size_bytes = total


def estimate_prover_cost(cfg: WhirConfig):
    """Estimate relative prover computational cost.

    # Cost model

    The prover's dominant operations per round are:

    ```
    Operation               Cost model               Notes
    ───────────────────     ────────────────────      ──────────────
    DFT (RS encoding)       N * log(N)                N = domain size
    Merkle tree build       N * 5                     one hash per leaf
    Sumcheck proving        N * fold_factor           linear scan
    PoW grinding            2^{pow_bits}              brute-force search
    OOD evaluation          ood_samples * 2^{nv}      polynomial eval
    ```

    Extension field operations cost ~4x base field operations.
    The initial commitment operates in the base field;
    subsequent rounds operate in the extension field.
    """
    cost = 0.0
    # Initial domain size: 2^{num_variables + starting_rate}.
    N0 = 1 << (cfg.num_variables + cfg.starting_log_inv_rate)
    logN0 = cfg.num_variables + cfg.starting_log_inv_rate
    # Initial DFT (base field, so no 4x multiplier).
    cost += N0 * logN0
    # Initial Merkle tree construction.
    cost += N0 * 5
    # Initial sumcheck for the first fold.
    cost += N0 * cfg.fold_schedule[0]
    # Starting PoW grinding (capped at 2^28 to prevent overflow in estimates).
    cost += 2 ** min(cfg.starting_folding_pow_bits, 28)
    # OOD polynomial evaluations during commitment.
    cost += cfg.commitment_ood_samples * (1 << cfg.num_variables)

    log_dom = cfg.num_variables + cfg.starting_log_inv_rate
    for i, r in enumerate(cfg.rounds):
        # Domain after RS reduction.
        new_log = log_dom - r.rs_reduction
        new_dom = 1 << new_log
        # DFT for the new folded oracle (extension field, 4x cost).
        cost += new_dom * new_log * 4
        # Merkle tree for the new oracle (extension field hashing).
        cost += new_dom * 20
        # Sumcheck for the next fold.
        nf = (
            cfg.fold_schedule[i + 1]
            if i + 1 < len(cfg.fold_schedule)
            else cfg.fold_schedule[-1]
        )
        cost += new_dom * nf * 4
        # PoW grinding for both query and folding phases.
        cost += 2 ** min(r.pow_bits, 28) + 2 ** min(r.folding_pow_bits, 28)
        # OOD polynomial evaluations.
        cost += r.ood_samples * (1 << r.num_variables) * 4
        log_dom = new_log

    # Final round PoW.
    cost += 2 ** min(cfg.final_pow_bits, 28) + 2 ** min(cfg.final_folding_pow_bits, 28)
    cfg.est_prover_cost = cost


def estimate_verifier_cost(cfg: WhirConfig):
    """Estimate relative verifier computational cost.

    # Cost model

    The verifier's operations per round are:

    ```
    Operation                   Cost model                    Notes
    ─────────────────────       ──────────────────────        ──────────────
    Merkle path verification    num_queries * path_len * 10   10 = hash cost
    Fold evaluation             num_queries * 2^fold_factor   one EF eval per leaf
    Sumcheck verification       fold_factor * 5               check polynomials
    OOD evaluation              ood_samples * num_variables   equality polynomial
    PoW check                   10                            single hash
    ```

    The hash cost weight (10) reflects that one Poseidon2 permutation
    is roughly 10x the cost of one field multiplication.
    """
    cost = 0.0
    # Relative cost of one hash invocation vs one field multiplication.
    HC = 10

    # Initial sumcheck verification.
    cost += cfg.fold_schedule[0] * 5
    # Commitment-phase OOD evaluation (equality polynomial over num_variables).
    cost += cfg.commitment_ood_samples * cfg.num_variables

    log_dom = cfg.num_variables + cfg.starting_log_inv_rate
    for i, r in enumerate(cfg.rounds):
        # Merkle path length after subtracting the cap height.
        path = max(0, r.log_domain_size - r.folding_factor - cfg.merkle_cap_height)
        # Per query: verify Merkle path + evaluate fold + evaluate equality polynomial.
        cost += r.num_queries * (path * HC + (1 << r.folding_factor) + r.num_variables)
        # OOD evaluation in extension field (4x base cost).
        cost += r.ood_samples * r.num_variables * 4
        # Sumcheck verification for the next fold.
        nf = (
            cfg.fold_schedule[i + 1]
            if i + 1 < len(cfg.fold_schedule)
            else cfg.fold_schedule[-1]
        )
        cost += nf * 5 + HC
        log_dom -= r.rs_reduction

    # Final proximity queries.
    lf = cfg.fold_schedule[-1]
    fp = max(0, log_dom - lf - cfg.merkle_cap_height)
    nv_f = cfg.final_sumcheck_rounds
    cost += cfg.final_queries * (fp * HC + (1 << lf) + nv_f)
    # Final sumcheck + polynomial evaluation.
    cost += cfg.final_sumcheck_rounds * 5 + (1 << cfg.final_sumcheck_rounds) * 4
    cfg.est_verifier_cost = cost


# Pareto frontier
# ===============


def pareto_frontier_3d(points: list) -> list:
    """Compute the 3-dimensional Pareto frontier.

    # Overview

    A point p is on the Pareto frontier if no other point q exists
    that is better or equal on all three metrics AND strictly better
    on at least one. Points on the frontier represent the best
    possible tradeoffs — improving one metric requires worsening another.

    # Arguments

    - points: list of (metric_1, metric_2, metric_3, config) tuples.
      All three metrics are minimized (lower = better).

    # Returns

    Sublist of non-dominated points.

    # Performance

    O(n^2) pairwise comparison. Acceptable for n < 10000 configs.
    """
    frontier = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            # q dominates p if q is <= on all metrics and < on at least one.
            if (
                q[0] <= p[0]
                and q[1] <= p[1]
                and q[2] <= p[2]
                and (q[0] < p[0] or q[1] < p[1] or q[2] < p[2])
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(p)
    return frontier


# Parameter sweep
# ===============


def generate_configs(
    nv: int, sec: int, snd: str, fsb: int, tad: int = 24
) -> list:
    """Generate all valid parameter combinations and return feasible configs.

    # Sweep space

    ```
    Parameter              Values
    ─────────────────      ──────────────────────────────
    Starting rate          {1, 2, 3, 4}
    Fold factor (constant) {3, 4, 5, 6, 7, 8}
    RS reduction           {1, ..., fold_factor}
    PoW bits               {10, 12, 14, 16, 18, 20, 22, 24, 28, 32}
    Merkle cap height      {0, 4, 8}
    First-round fold       {5, 6, 7, 8} with subsequent {3, 4, 5, 6}
    ```

    Configs that fail validation or require more PoW than allowed
    are silently discarded.

    # Returns

    List of feasible configs (pow_ok = True).
    """
    configs = []
    for rate in [1, 2, 3, 4]:
        # Constant folding factor (same every round).
        for fold in [3, 4, 5, 6, 7, 8]:
            for rs_red in range(1, fold + 1):
                # Two-adicity check: folded domain must fit in the base field.
                if nv + rate - fold > tad:
                    continue
                for pw in [10, 12, 14, 16, 18, 20, 22, 24, 28, 32]:
                    for cap in [0, 4, 8]:
                        c = derive_config(
                            nv, rate, [fold], [rs_red], pw, sec, snd, fsb, cap, tad
                        )
                        if c and c.pow_ok:
                            configs.append(c)
        # Variable first round: large initial fold, smaller subsequent folds.
        # This captures the pattern Giacomo found effective (e.g. fold 7,4,5,4).
        for ff in [5, 6, 7, 8]:
            for f2 in [3, 4, 5, 6]:
                # Skip if both factors are equal (already covered above).
                if f2 == ff:
                    continue
                for rs_red in range(1, ff + 1):
                    if nv + rate - ff > tad:
                        continue
                    for pw in [14, 16, 18, 20, 22, 24, 28, 32]:
                        for cap in [0, 4, 8]:
                            c = derive_config(
                                nv,
                                rate,
                                [ff, f2],
                                [rs_red],
                                pw,
                                sec,
                                snd,
                                fsb,
                                cap,
                                tad,
                            )
                            if c and c.pow_ok:
                                configs.append(c)
    return configs


def label(c: WhirConfig) -> str:
    """Generate a short human-readable label for a configuration."""
    # Collapse the fold schedule if all entries are identical.
    s = ",".join(str(k) for k in c.fold_schedule)
    if len(set(c.fold_schedule)) == 1:
        s = str(c.fold_schedule[0])
    return (
        f"r={c.starting_log_inv_rate} k=[{s}] "
        f"red={c.rs_domain_initial_reduction} "
        f"pow={c.max_pow_bits} cap={c.merkle_cap_height}"
    )


def cargo_cmd(c: WhirConfig) -> str:
    """Generate a cargo command to run this configuration in the Rust example.

    # Overview

    The generated command invokes `examples/whir.rs` with the matching
    CLI flags. Note that the Rust CLI only supports constant folding
    factors, so configs with variable schedules use the first entry.
    """
    return (
        f"cargo run --release --example whir -- -d {c.num_variables} "
        f"-r {c.starting_log_inv_rate} -k {c.fold_schedule[0]} "
        f"--initial-rs-reduction {c.rs_domain_initial_reduction} "
        f"-p {c.max_pow_bits} -l {c.security_level} --sec {c.soundness_type}"
    )


# Plotting
# ========


def plot_results(configs: list, frontier: list, args):
    """Generate matplotlib plots showing the parameter space and Pareto frontier.

    # Output files

    Two PNG files are saved to the current directory:

    1. Three-panel overview: prover x size, prover x verifier, size x verifier.
       All configs shown as colored dots, frontier as a black line with stars.

    2. Zoomed single-panel: prover x size in the practical region,
       with annotated recommended picks.

    # Color coding

    Dots are colored by the starting rate parameter:
      - Blue  = rate 1 (rho = 1/2, smallest initial domain, fastest prover)
      - Orange = rate 2 (rho = 1/4)
      - Green = rate 3 (rho = 1/8)
      - Red   = rate 4 (rho = 1/16, largest domain, smallest proofs)

    # Pareto frontier line

    The black line connects non-dominated points from the 3D Pareto frontier,
    projected onto each 2D plane. Some points appear non-dominated in 2D
    because they excel on the hidden third dimension.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(
        f"WHIR d={args.num_variables}  {args.security}-bit {args.soundness}  "
        f"field={args.field_bits}b\n"
        f"{len(configs)} configs explored, {len(frontier)} on Pareto frontier",
        fontsize=14,
        fontweight="bold",
    )

    # Map each rate to a distinct color from the Tableau palette.
    rate_colors = {1: "#4e79a7", 2: "#f28e2b", 3: "#59a14f", 4: "#e15759"}
    rates = [c.starting_log_inv_rate for c in configs]
    colors = [rate_colors.get(r, "#999") for r in rates]

    # Use calibrated units for readable axis labels.
    prover_ms = [c.est_prover_ms for c in configs]
    proof_kb = [c.est_proof_size_bytes / 1024 for c in configs]
    verifier_us = [c.est_verifier_us for c in configs]

    # Panel 1: Prover time vs Proof size.
    ax = axes[0]
    ax.scatter(prover_ms, proof_kb, c=colors, alpha=0.4, s=15, edgecolors="none")
    # Overlay the Pareto frontier as a connected line.
    fp = [
        (c.est_prover_ms, c.est_proof_size_bytes / 1024, c)
        for _, _, _, c in frontier
    ]
    fp.sort()
    if fp:
        ax.plot(
            [x[0] for x in fp],
            [x[1] for x in fp],
            "k-",
            linewidth=2,
            label="Pareto frontier",
        )
        ax.plot([x[0] for x in fp], [x[1] for x in fp], "k*", markersize=6)
    ax.set_xlabel("Est. Prover Time (ms)", fontsize=11)
    ax.set_ylabel("Est. Proof Size (KiB)", fontsize=11)
    ax.set_title("Prover Time vs Proof Size", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Panel 2: Prover time vs Verifier time.
    ax = axes[1]
    ax.scatter(prover_ms, verifier_us, c=colors, alpha=0.4, s=15, edgecolors="none")
    fp2 = [
        (c.est_prover_ms, c.est_verifier_us, c)
        for _, _, _, c in frontier
    ]
    fp2.sort()
    if fp2:
        ax.plot([x[0] for x in fp2], [x[1] for x in fp2], "k-", linewidth=2)
        ax.plot([x[0] for x in fp2], [x[1] for x in fp2], "k*", markersize=6)
    ax.set_xlabel("Est. Prover Time (ms)", fontsize=11)
    ax.set_ylabel("Est. Verifier Time (µs)", fontsize=11)
    ax.set_title("Prover Time vs Verifier Time", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Panel 3: Proof size vs Verifier time.
    ax = axes[2]
    ax.scatter(proof_kb, verifier_us, c=colors, alpha=0.4, s=15, edgecolors="none")
    fp3 = [
        (c.est_proof_size_bytes / 1024, c.est_verifier_us, c)
        for _, _, _, c in frontier
    ]
    fp3.sort()
    if fp3:
        ax.plot([x[0] for x in fp3], [x[1] for x in fp3], "k-", linewidth=2)
        ax.plot([x[0] for x in fp3], [x[1] for x in fp3], "k*", markersize=6)
    ax.set_xlabel("Est. Proof Size (KiB)", fontsize=11)
    ax.set_ylabel("Est. Verifier Time (µs)", fontsize=11)
    ax.set_title("Proof Size vs Verifier Time", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Shared legend for all panels.
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=c,
            markersize=8,
            label=f"rate={r}",
        )
        for r, c in sorted(rate_colors.items())
    ]
    handles.append(
        Line2D(
            [0], [0], color="k", linewidth=2, marker="*", markersize=6, label="Pareto"
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )

    # Annotate three recommended picks on the prover-vs-size panel.
    by_prov = sorted(configs, key=lambda c: c.est_prover_ms)
    by_size = sorted(configs, key=lambda c: c.est_proof_size_bytes)
    by_bal = sorted(
        configs,
        key=lambda c: (c.est_prover_ms * c.est_proof_size_bytes * c.est_verifier_us)
        ** (1 / 3),
    )

    annotations = [
        (by_prov[0], "fastest\nprover", "#d62728"),
        (by_size[0], "smallest\nproof", "#2ca02c"),
        (by_bal[0], "best\nbalanced", "#9467bd"),
    ]
    # Offset directions to avoid label overlap.
    offsets = [(10, 15), (-80, 15), (10, -20)]
    for (c, txt, col), ofs in zip(annotations, offsets):
        x = c.est_prover_ms
        y = c.est_proof_size_bytes / 1024
        axes[0].annotate(
            f"{txt}\n{x:.0f} ms, {y:.0f} KiB",
            (x, y),
            fontsize=8,
            fontweight="bold",
            color=col,
            textcoords="offset points",
            xytext=ofs,
            arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.8),
        )
        axes[0].plot(x, y, "*", color=col, markersize=12, zorder=5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    outfile = f"whir_pareto_d{args.num_variables}_{args.soundness}_{args.security}b.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {outfile}")

    # Zoomed plot: prover vs size in the practical region.
    #
    # Unlike the three-panel overview (which uses the 3D Pareto frontier),
    # this plot shows the **2D Pareto frontier** over prover time and proof
    # size only. This avoids confusing spikes that appear when a 3D frontier
    # point has excellent verifier cost but poor prover×size tradeoff.
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
    fig2.suptitle(
        f"WHIR d={args.num_variables}  {args.security}-bit {args.soundness}  "
        f"Zoom: practical tradeoff region",
        fontsize=13,
        fontweight="bold",
    )

    # Include ALL annotated points in the zoom range so none are cut off.
    annot_p = [c.est_prover_ms for c, _, _ in annotations]
    annot_s = [c.est_proof_size_bytes / 1024 for c, _, _ in annotations]

    # Zoom range: cover up to 2x median, but also include all annotated points.
    med_p = sorted(prover_ms)[len(prover_ms) // 2]
    med_s = sorted(proof_kb)[len(proof_kb) // 2]
    max_p = max(med_p * 2, max(annot_p) * 1.15)
    max_s = max(med_s * 2, max(annot_s) * 1.15)

    mask = [(p <= max_p and s <= max_s) for p, s in zip(prover_ms, proof_kb)]
    zp = [p for p, m in zip(prover_ms, mask) if m]
    zs = [s for s, m in zip(proof_kb, mask) if m]
    zc = [c for c, m in zip(colors, mask) if m]

    ax2.scatter(zp, zs, c=zc, alpha=0.5, s=20, edgecolors="none")

    # Compute the 2D Pareto frontier (prover time vs proof size only).
    # A point is non-dominated in 2D if no other point is both faster
    # AND smaller. This produces a clean monotonically decreasing staircase.
    pts_2d = [
        (c.est_prover_ms, c.est_proof_size_bytes / 1024, c)
        for c in configs
    ]
    pts_2d.sort(key=lambda x: x[0])
    frontier_2d = []
    best_size = float("inf")
    for p, s, c in pts_2d:
        if s < best_size:
            frontier_2d.append((p, s, c))
            best_size = s

    # Clip to the zoomed region.
    zfp = [
        (p, s, c)
        for p, s, c in frontier_2d
        if p <= max(zp, default=1) * 1.1
        and s <= max(zs, default=1) * 1.1
    ]
    if zfp:
        ax2.plot(
            [x[0] for x in zfp],
            [x[1] for x in zfp],
            "k-",
            linewidth=2.5,
            label="Pareto frontier",
        )
        ax2.plot([x[0] for x in zfp], [x[1] for x in zfp], "k*", markersize=8)

    # Annotate ALL three recommended picks on the zoomed plot.
    zoom_offsets = [(15, 15), (-100, -20), (15, -25)]
    for (c, txt, col), ofs in zip(annotations, zoom_offsets):
        x = c.est_prover_ms
        y = c.est_proof_size_bytes / 1024
        ax2.annotate(
            f"{txt}\n{x:.0f} ms, {y:.0f} KiB",
            (x, y),
            fontsize=9,
            fontweight="bold",
            color=col,
            textcoords="offset points",
            xytext=ofs,
            arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.8),
        )
        ax2.plot(x, y, "*", color=col, markersize=14, zorder=5)

    ax2.set_xlabel("Est. Prover Time (ms)", fontsize=12)
    ax2.set_ylabel("Est. Proof Size (KiB)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(handles=handles, fontsize=10)

    outfile2 = (
        f"whir_pareto_d{args.num_variables}_{args.soundness}_{args.security}b_zoom.png"
    )
    plt.savefig(outfile2, dpi=150, bbox_inches="tight")
    print(f"Zoomed plot saved to {outfile2}")


# Entry point
# ===========


def main():
    """Parse CLI arguments, sweep the parameter space, and output results."""
    p = argparse.ArgumentParser(description="WHIR Parameter Optimizer")
    p.add_argument("-d", "--num-variables", type=int, default=26)
    p.add_argument("-l", "--security", type=int, default=128)
    p.add_argument(
        "--soundness",
        default="JohnsonBound",
        choices=["UniqueDecoding", "JohnsonBound", "CapacityBound"],
    )
    p.add_argument(
        "--field-bits",
        type=int,
        default=155,
        help="Extension field bits (KoalaBear deg-4=124, deg-5=155)",
    )
    p.add_argument("--two-adicity", type=int, default=24)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--plot", action="store_true", help="Generate matplotlib plots")
    args = p.parse_args()

    print("WHIR Parameter Optimizer")
    print("========================")
    print(
        f"  d={args.num_variables}  sec={args.security}  {args.soundness}  field={args.field_bits}b"
    )
    print()

    # Phase 1: Sweep the parameter space.
    print("Sweeping...", end=" ", flush=True)
    configs = generate_configs(
        args.num_variables,
        args.security,
        args.soundness,
        args.field_bits,
        args.two_adicity,
    )
    print(f"{len(configs)} valid configs.")

    if not configs:
        print("ERROR: No valid configs. Try larger --field-bits or lower --security.")
        sys.exit(1)

    # Phase 2: Compute the 3D Pareto frontier over calibrated metrics.
    print("Computing Pareto frontier...", end=" ", flush=True)
    pts = [
        (c.est_prover_ms, c.est_proof_size_bytes, c.est_verifier_us, c)
        for c in configs
    ]
    frontier = pareto_frontier_3d(pts)
    # Sort by prover cost for readable output.
    frontier.sort(key=lambda x: x[0])
    print(f"{len(frontier)} non-dominated points.")

    # Phase 3: Print ranked tables with calibrated units.
    def show_table(title, items, n):
        """Print a ranked table of configs with real-world units."""
        print(f"\n{'=' * 105}")
        print(f"  {title}")
        print(f"{'=' * 105}")
        print(
            f"{'Config':<52} {'Prover(ms)':>11} {'Proof(KiB)':>11} {'Verif(us)':>10} {'Rnds':>5}"
        )
        print(f"{'-' * 105}")
        for c in items[:n]:
            print(
                f"{label(c):<52} {c.est_prover_ms:>11.0f} "
                f"{c.est_proof_size_bytes / 1024:>11.1f} "
                f"{c.est_verifier_us:>10.0f} {len(c.rounds):>5}"
            )

    show_table("PARETO FRONTIER", [c for _, _, _, c in frontier], args.top * 2)
    show_table(
        "TOP FASTEST PROVER",
        sorted(configs, key=lambda c: c.est_prover_cost),
        args.top,
    )
    show_table(
        "TOP SMALLEST PROOF",
        sorted(configs, key=lambda c: c.est_proof_size_bytes),
        args.top,
    )
    show_table(
        "TOP FASTEST VERIFIER",
        sorted(configs, key=lambda c: c.est_verifier_cost),
        args.top,
    )

    # Phase 4: Select recommended configs using calibrated metrics.
    by_p = sorted(configs, key=lambda c: c.est_prover_ms)[0]
    by_s = sorted(configs, key=lambda c: c.est_proof_size_bytes)[0]
    by_v = sorted(configs, key=lambda c: c.est_verifier_us)[0]
    # "Best balanced" minimizes the geometric mean of all three metrics.
    by_b = sorted(
        configs,
        key=lambda c: (c.est_prover_ms * c.est_proof_size_bytes * c.est_verifier_us)
        ** (1 / 3),
    )[0]
    # "Knee" is the fastest prover among configs with below-median proof size.
    med = sorted(c.est_proof_size_bytes for c in configs)[len(configs) // 2]
    small = [c for c in configs if c.est_proof_size_bytes <= med]
    knee = min(small, key=lambda c: c.est_prover_ms) if small else by_b

    print(f"\n{'=' * 105}")
    print("  RECOMMENDED CONFIGS (with cargo commands)")
    print(f"{'=' * 105}")
    for name, c in [
        ("Fastest Prover", by_p),
        ("Smallest Proof", by_s),
        ("Fastest Verifier", by_v),
        ("Best Balanced", by_b),
        ("Knee (fast+small)", knee),
    ]:
        # Find the highest PoW requirement across all phases.
        max_needed = (
            max(
                c.starting_folding_pow_bits,
                c.final_pow_bits,
                c.final_folding_pow_bits,
                *(r.pow_bits for r in c.rounds),
                *(r.folding_pow_bits for r in c.rounds),
            )
            if c.rounds
            else max(
                c.starting_folding_pow_bits, c.final_pow_bits, c.final_folding_pow_bits
            )
        )
        print(f"\n  [{name}]")
        print(f"    {label(c)}")
        print(
            f"    prover ~{c.est_prover_ms:.0f} ms  "
            f"proof ~{c.est_proof_size_bytes / 1024:.1f} KiB  "
            f"verifier ~{c.est_verifier_us:.0f} us"
        )
        print(
            f"    rounds={len(c.rounds)}  final_vars={c.final_sumcheck_rounds}  "
            f"max_pow_needed={max_needed}"
        )
        print(f"    $ {cargo_cmd(c)}")

    # Phase 5: Optional CSV export (all valid configs).
    if args.csv:
        with open(args.csv, "w") as f:
            f.write(
                "rate,fold_schedule,rs_red,pow,cap,"
                "prover_cost,proof_bytes,verifier_cost,rounds,final_vars\n"
            )
            for c in configs:
                s = "|".join(str(k) for k in c.fold_schedule)
                f.write(
                    f"{c.starting_log_inv_rate},{s},{c.rs_domain_initial_reduction},"
                    f"{c.max_pow_bits},{c.merkle_cap_height},"
                    f"{c.est_prover_cost:.0f},{c.est_proof_size_bytes},"
                    f"{c.est_verifier_cost:.0f},"
                    f"{len(c.rounds)},{c.final_sumcheck_rounds}\n"
                )
        print(f"\nCSV: {args.csv}")

    # Phase 6: Optional JSON export (Pareto frontier only).
    if args.json:
        data = [
            {
                "label": label(c),
                "prover_ms": round(c.est_prover_ms),
                "proof_bytes": c.est_proof_size_bytes,
                "proof_kib": round(c.est_proof_size_bytes / 1024, 1),
                "verifier_us": round(c.est_verifier_us),
                "cmd": cargo_cmd(c),
            }
            for _, _, _, c in frontier
        ]
        with open(args.json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"JSON: {args.json}")

    # Phase 7: Optional matplotlib plots.
    if args.plot:
        plot_results(configs, frontier, args)

    print(f"\nDone. {len(configs)} explored, {len(frontier)} on Pareto frontier.")


if __name__ == "__main__":
    main()
