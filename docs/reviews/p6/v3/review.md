# P6.3 theory and implementation review

## Theory review

The frozen P6 estimand is unchanged. P6.3 changes only the scale and candidate certificate. Let `q_p = alpha - epsilon_p`, where `epsilon_p` is an independent simultaneous upper bound on true/model positive failure discordance. For Bernoulli failure `X_t` and a fixed predictable alternative `theta_p < q_p`,

`E_p[(theta_p/q_p)^X ((1-theta_p)/(1-q_p))^(1-X)] <= 1` for every `p >= q_p`.

The likelihood-ratio product is therefore an e-process for the composite unsafe null. Ville's inequality handles all 8,192 computation exits. Bonferroni handles the finite prefix grid. Conditioning on the independent shift-calibration sample makes each adjusted threshold fixed for the model-rollout filtration; combining the two half-delta events yields the required true-environment certificate. Exact model and true risks are used only after selection.

The AVCRC comparator uses the primary paper's Theorem 4.1 bounded-loss stitched correction with `B=1`. Subtracting the largest simultaneous shift envelope preserves the monotone loss family. This composition is valid but intentionally conservative. It is not Theorem 4.7: that result assumes supplied importance weights/density ratios, which the paired-discordance benchmark does not have.

No theoretical blocker remains inside the IID nested-Bernoulli scope. MAJOR limitation: exchangeability of paired calibration and evaluation shifts is still necessary. MAJOR limitation: the fixed CSA bet margin can lose power when the true model-risk gap differs from the pilot family. MINOR limitation: latency is CPU/Python-specific.

## Heuristics review

The frontier includes stable, diffuse, late-cliff, and early-shift families. The late-cliff family detects global-radius gaming: AVCRC's common shift adjustment became vacuous although safe early prefixes existed. Always abstain detects validity-through-refusal; always execute and the model-only process expose unsafe shift. The model-only process remained model-valid while its true false-certification rate reached 0.615, so the two estimands cannot be conflated.

The gain threshold was set using pilot seed 632. The evaluation seed remained untouched. Shared shift calibration dominates end-to-end latency at large scale, so certificate-only latency and calibration latency are separate metrics; the result does not claim an 87x end-to-end speedup.

## Primary-literature boundary

- [Anytime-Valid Conformal Risk Control](https://arxiv.org/abs/2602.04364) supplies the bounded monotone-loss stitched correction and an importance-weighted distribution-shift theorem. Only the former's Bernoulli specialization is used here; the latter's density-ratio assumptions do not hold.
- [Conformal Selective Acting](https://arxiv.org/abs/2605.20270) specifies a Ville-type e-process per threshold on a Bonferroni grid with max-certified selection. P6.3 faithfully specializes that statistical cell to IID Bernoulli prefix failures and a fixed predictable bet, not to RLVR/LLM deployment.
- [Conformal Risk Control](https://arxiv.org/abs/2208.02814) establishes the bounded monotone-loss family and warns that monotonicity is necessary. The nested prefix failures meet that assumption.

No general-method novelty or reproduction claim is made.

## Implementation review

P6.3 is stage-dispatched and leaves P6.0--P6.2 unchanged. Pilot and evaluation seeds are disjoint. The configuration locks 16 prefixes, four shift families, five reported scales, 8,192 declared exits, the fixed CSA bet margin, and both gain criteria. Metrics include model/true false certification, selected model/true risk, prefix length, abstention, nonvacuity, trajectory-prefix evaluations, certificate latency, and shift-calibration latency, with theory identifiers added by the shared writer.

## Synthesis

Unchanged: longest safe-prefix target, optional-stopping validity, prefix-selection validity, separate risk/confidence spending, and evaluation-only truth. Clarified: a faithful statistical specialization is not a reproduction of the full application method. Added controls: faithful bounded AVCRC and CSA specializations where assumptions permit, dense-exit robust union, four shift shapes, scale and latency curves. Falsifiers are true false certification above tolerance, vacuity, failure to beat the robust union, or breakdown under nonexchangeable fresh shifts.
