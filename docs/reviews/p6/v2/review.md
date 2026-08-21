# P6.2 independent review synthesis

## Theory review

The original P6 target is unchanged: execute the longest prefix whose simultaneous upper certificate is at most the episode risk allocation. P6.2 adds the required model-mismatch split from frozen section 9.5. If `r_M(p)` is model risk and `r_T(p)` is environment risk, then for paired binary failures `Y_M,Y_T`,

`r_T(p) - r_M(p) <= P(Y_T(p)=1, Y_M(p)=0)`.

Consequently a model-risk upper process plus a simultaneous upper bound on positive discordance is an environment-risk upper certificate, subject to the paired calibration/evaluation exchangeability assumption. Confidence error is split between components. Optional stopping is handled only by the model process; the shift envelope is calibrated once on an independent fixed-size sample. Prefix selection is handled in both components. Repeated episode-level risk spending is outside this single-decision P6.2 cell and remains established only by P6.1.

BLOCKER resolved: exact environment risk never enters fitting, calibration, stopping, prefix selection, or certification. It is used only to score selected prefixes. MAJOR limitation: the shift guarantee is conditional on the paired-discordance law carrying from calibration to evaluation. Unseen shift is not covered. MINOR limitation: finite smoke repetitions make absence of observed false certification weak evidence, so the result remains E1/NARROW.

## Heuristics review

Adversarial cases include upward environment shift with unchanged model samples, a boundary prefix at risk 0.20, early-look peeking, and an always-execute control. Model-relative validity can coexist with true false certification under adverse shift; the reporting schema prevents that from being silently collapsed into one number. The robust method can game validity by abstaining, so every condition reports selected-prefix length, abstention, and nonvacuity. Adding arbitrarily wide shift radii remains detectable as vacuity.

## Literature review and novelty boundary

- [Anytime-Valid Conformal Risk Control](https://arxiv.org/abs/2602.04364) develops anytime-valid risk control across growing calibration sets and includes distribution-shift/importance-weighting extensions. The local CP-spending comparator is a bounded Bernoulli analogue, not that general method.
- [Conformal Selective Acting](https://arxiv.org/abs/2605.20270) uses threshold-indexed e-processes and Ville-style anytime selection. The local implementation tests one declared Bernoulli risk threshold with a finite prefix Bonferroni correction; it is not the full conformal acting procedure.
- [Conformal Risk Control](https://arxiv.org/abs/2208.02814) establishes the broader bounded monotone-loss risk-control framework and shift extensions. P6.2 does not claim its general loss or distributional scope.
- [Active, anytime-valid risk-controlling prediction sets](https://openreview.net/pdf?id=4ZH48aGD60) is an additional primary anytime-valid boundary. Active labeling is outside the present benchmark.

No novelty claim is supported by implementing these bounded analogues. The contribution under test is strictly the auditable separation of learned-model validity from true-environment validity and a paired-discordance robustness mechanism.

## Implementation review

Scientific choices trace to frozen sections 9.1–9.6 and the P6 theory map. P6.0/1 files and result paths remain intact; stage dispatch selects P6.2 only when explicitly configured. The learned model uses pilot-only rollouts. Evaluation uses a disjoint seed. Robust inputs are sampled binary observations rather than oracle risks. Metric rows carry proposal/theory/amendment identifiers through the shared writer.

## Synthesis

Unchanged claims: optional-stopping, prefix-selection, episode-spending, and nonvacuity remain separate; exact truth remains evaluation-only. Clarified claim: a model-valid certificate is not environment-valid under shift. No amendment is required because section 9.5 explicitly requires this analysis. Added controls are fixed-time, union, abstain/execute, AVCRC-style, CSA-style, and upward-shift cells. Falsification occurs if the robust method exceeds the locked true false-certification tolerance, becomes vacuous, or loses validity when the paired-discordance law changes on fresh units.
