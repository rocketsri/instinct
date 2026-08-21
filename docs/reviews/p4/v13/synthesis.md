# P4 synthesis v13

- **Unchanged:** the frozen recovery quantile and prefix-hinge training target.
- **Clarified:** exact inertial recovery and whole-layout estimator coverage are
  instrument gates, not policy results.
- **Amended branch:** `p4-1.1` permits an inference-time verifier test without
  substituting it for slack training.
- **Resolved:** v11 blocker B2 (missing recovery-2 evidence) is resolved by the
  preserved v10 and v12 artifacts.
- **Still blocking future training claims:** v11 blocker B1; the current stage2
  expected hinge is not the frozen hinge of a lower disturbance quantile.
- **Final bounded outcome:** P4.2 STOP after baseline dominance and failure to
  transfer the verifier effect under exact cost matching.

P4 is archived at the bounded policy stage. Reopening it requires a new theory
version, the exact quantile objective, a learned environment, and the matched
resource/frontier controls listed in v11.
