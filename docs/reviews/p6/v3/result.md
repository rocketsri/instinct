# P6.3 frozen CPU frontier result

Config: `configs/p6/p6_3/cpu.yaml`  
Config hash: `000093f01f7e06018f4226310b3c9a10`  
Pilot seeds: 631/632  
Evaluation seed: 641  
Repetitions per family: 96  
Verdict: **GO** for the bounded CPU frontier

At 8,192 trajectories, the shift-robust CSA specialization observed zero true false certifications in every shift family. The worst one-sided 95% Clopper--Pearson upper bound was 0.0307, below 0.05. Minimum family nonvacuity was 0.5853.

| Family | Robust union prefix | Robust CSA prefix | CSA true false cert. | CSA nonvacuity |
|---|---:|---:|---:|---:|
| stable | 11.8229 | 12.3646 | 0.0000 | 0.7728 |
| diffuse shift | 10.5417 | 11.0833 | 0.0000 | 0.6927 |
| late cliff | 10.9688 | 11.0625 | 0.0000 | 0.6914 |
| early shift | 8.7917 | 9.3646 | 0.0000 | 0.5853 |

The average largest-scale prefix gain was 0.4375 actions, exceeding the locked 0.3 target. Certificate-only latency averaged 5.45 microseconds for CSA and 476.90 microseconds for the finite union, an 87.45x speedup over the locked 10x target. The common shift-calibration cost averaged 742.87 microseconds and dominates end-to-end latency; no 87x end-to-end claim is supported.

## Scaling curve averaged over shift families

| Trajectories | Union prefix | CSA prefix | Union cert. us | CSA cert. us | Shared calibration us |
|---:|---:|---:|---:|---:|---:|
| 512 | 5.4375 | 4.5885 | 460.23 | 3.42 | 510.39 |
| 1,024 | 7.0964 | 7.4089 | 461.15 | 4.29 | 503.47 |
| 2,048 | 8.5417 | 9.2500 | 460.15 | 4.87 | 536.66 |
| 4,096 | 9.7083 | 10.3620 | 459.33 | 5.30 | 601.62 |
| 8,192 | 10.5313 | 10.9688 | 476.90 | 5.45 | 742.87 |

Both robust methods had zero observed true false certifications throughout the curve. CSA was less efficient at 512 trajectories, crossed the union between 512 and 1,024, and retained a positive prefix advantage thereafter. This is the expected finite-sample cost of the e-process before its horizon-independent exit guarantee amortizes.

The model-only anytime comparator had zero model false certifications but true false-certification rates of 0.0208, 0.4583, and 0.6146 in the diffuse, late-cliff, and early-shift largest-scale cells. AVCRC's common-radius composition was valid but vacuous in the late-cliff cell, demonstrating the cost of preserving a single monotone family under highly localized shift.

Conclusion: the preregistered bounded frontier reached GO. This authorizes a fresh realistic learned-dynamics integration, not deployment and not a claim that the repository reproduces general AVCRC or CSA.
