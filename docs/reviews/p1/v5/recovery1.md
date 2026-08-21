# P1.1 CPU recovery and realism review v5

## Preserved evidence and fixed diagnosis

The earlier 81-context smoke remains preserved. It had exact Sigma identity but
zero natural continuous-control failures, M4 regret UCB `0.321329` versus M5
`0.099624`, no official Finding-the-Time-to-Think equivalence, and no T4 timing.
This recovery uses fresh seeds and does not revise or overwrite those units.

The original identity remains

`Sigma = G_plan + R_intermediate - L_arrival - L_wait - C_hw + L_base_delay + epsilon_id`.

Recurring policy return, matched one-handoff decomposition, and matched-time
failure remain separate outputs under amendment P1 1.1.

## Fresh CPU recovery

- Six oracle and eight regret seeds are new and disjoint from prior P1.1 seeds.
- The full 3x3 speed/runtime grid is retained; whole held-out cells remain
  `(speed_index, runtime_index) = (1,2)` and `(2,1)`.
- The horizon increases from 8 to 24, with CPU work scales `[2,8,24]` and event
  rates `[8000,30000,120000]`.
- Continuous initial states are sampled from a preregistered distribution over
  recovery margin `[-0.12,0.24]` before clipping to legal positions. This gives
  both recoverable and dynamically unrecoverable legal states without selecting
  outcomes after execution. A valid recovery requires observed failures and
  survivals on recurring trajectories.
- M5 remains the deployable nearest-training-context lookup. A distinct
  `LOOKUP_ORACLE_EVAL_ONLY` uses the held-out oracle argmax, has zero regret by
  construction, and is never presented as transfer.

## Official FTT boundary

The official [Finding the Time to Think code](https://github.com/Aneeshers/realtime-rl-code)
uses a PPO gate over frozen AlphaZero-style MCTS, with released JAX/Jumanji/PGX
environments and checkpoints. Its README states CPU-only execution is untested
and paper experiments used H100/A100/A40 GPUs. P1 now exposes a compatible
finite-budget adapter taking state features, planner-policy features, and a
planner value, but `FTT_LITE` remains explicitly non-equivalent.

Official equivalence requires a verified checkout containing `committed_action`,
`clock`, `checkpoints/MANIFEST.md`, plus a configured released gating checkpoint.
Those artifacts are absent locally, so the comparative control fails closed.

## T4 timing boundary

A T4 artifact must contain at least nine measured cells with budget, event rate,
positive latency, occupancy, and device=`T4`. No such artifact is configured.
CPU timing cells therefore cannot support hardware transfer, asynchronous
interference, or deployment claims.

## Verdict logic

The CPU recovery can establish failure-rich instrumentation and within-CPU
transfer evidence. GO additionally requires M4 to beat M0/M2/M5/FTT, official
FTT equivalence, and measured T4 held-out cells. Missing external controls force
INCONCLUSIVE. Product collapse is neither imposed nor inferred universally.

## Executed recovery result

The frozen recovery completed 81 contexts and 18 held-out family/reflex/cell
combinations. All CPU instrumentation controls passed:

- natural continuous trajectories: 1,545 failed and 1,479 survived;
- initial recovery margins ranged from `-0.119139` to `0.239835`;
- maximum episodewise Sigma identity residual: `9.30e-16`;
- median informative target range: `0.389729`.

The transfer hypothesis failed honestly:

- M4 context-regret UCB95: `0.124188`;
- deployable M5 lookup UCB95: `0.040792`;
- FTT-lite UCB95: `0.188191`;
- M5 improvement-over-M4 LCB95: `-0.126633`.

M4 also exceeds the frozen 20%-of-range bound (`0.077946`). The result is
**INCONCLUSIVE/F4 baseline dominance**. It repairs the failure-occupancy
instrumentation gap but does not repair conditional transfer. The separate
held-out lookup oracle is recorded at zero regret only as an evaluation ceiling.
Official FTT equivalence and T4 transfer remain unmeasured.
