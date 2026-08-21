# P2.2 recovery-2 extended train-only pilot result

Artifact: `/private/tmp/instinct-p2-r2-pilot2.json`  
Runtime: Colab Tesla T4, Torch `2.11.0+cu128`, TorchVision `0.26.0+cu128`

The extended pilot repaired the short-smoke degeneracy without opening any
official test image. The source head reached `0.832031` clean accuracy and the
24 horizon-eligible candidates included six positive-utility writes
(`0.25` positive fraction; utility range `[-0.713896, 0.255358]`). All required
baselines, exact admitted-write counts, candidate-before-top-m ordering,
selection locking, and matched deployable-compute checks passed.

The mechanism readiness gate did not pass. The offline oracle's best accuracy
frontier gain over matched controls was `-0.035156`. Although its selected
writes had positive offline utility, downstream accuracy was worse than
never-write by `0.035156`, `0.140625`, and `0.195312` at admitted fractions
`0.10`, `0.25`, and `0.50`. Dense adaptation was worse by `0.441406`,
`0.433594`, and `0.425781`; no selector rescued the accuracy frontier.

Verdict: **pilot mechanism warning, not a held-out recovery verdict**. Per the
v5 preregistration, the reserved seed-241 test partition is not opened. This
looks like offline-utility/downstream misalignment rather than missing useful
and harmful candidates. The full recovery remains unconsumed, and P7 remains
locked. A future recovery version must repair the causal validation target from
pilot-only evidence without silently replacing the original multi-horizon
utility or downstream-accuracy gate.
