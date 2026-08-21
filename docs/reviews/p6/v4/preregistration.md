# P6.4 learned-dynamics preregistration

P6.3 remains frozen. P6.4 fits four action-conditioned transition matrices for a three-state safe/warning/failure model using pilot seed 651 and 20,000 complete trajectories. Pilot seed 652 establishes variance only. Evaluation seed 661 is untouched when this document is frozen.

The action schedule, nominal matrices, four structural-shift families, 24 candidate prefixes, five trajectory scales through 8,192, 8,192 declared exits, risk/confidence budgets, paired calibration, fixed CSA bet margin, 96 evaluation repetitions, and verdict thresholds are locked in `configs/p6/p6_4/cpu.yaml`.

The primary method is shift-robust CSA. At the largest scale it must have a one-sided 95% upper bound on true false certification at most 0.05 and positive nonvacuity in every shift family. GO additionally requires a mean selected-prefix gain of at least 0.2 actions over the shift-robust finite union. Valid, nonvacuous evidence below that gain is NARROW. Invalid evidence is STOP.

The certificate sees model-rollout failure counts and a disjoint paired-discordance envelope. Fitted-model exact risks and true transition-system exact risks are computed only after selection. Model fitting, model rollout, shift calibration, certificate latency, transition evaluations, and array working sets are separate measurements.

Pilot seed 652 found nominal transition L1 error near 0.001 and model-only true false certification in all shifted largest-scale cells. Robust CSA was valid and nonvacuous, with an estimated 0.148-action gain over union. The 0.2 threshold was not lowered after seeing that pilot result.
