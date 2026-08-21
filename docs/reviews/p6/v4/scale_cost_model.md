# P6.4 industry-scale cost and extrapolation model

This note defines the extrapolation before held-out evaluation. It is not a deployment estimate.

Let `N` be trajectories, `H` the certified action horizon, `K` candidate computation exits, `S` states, and `A` action-conditioned transition matrices.

- Pilot fitting consumes `N_fit * H` observed transitions, stores `A*S*S` counters, and can stream trajectories. Its persistent memory is `O(A*S^2)`.
- One model rollout frontier consumes `N*H` categorical transitions. Materializing Boolean prefix failures costs `N*H` bytes; streaming prefix counts reduces this to `O(H)` counters.
- Paired shift calibration consumes `2*N*H` categorical transitions and, in this prototype, `2*N*H` Boolean bytes. It can also stream discordance counts in `O(H)` memory.
- Finite union certification costs `O(H)` exact-binomial inversions at each reported exit and pays confidence for `K*H` hypotheses. CSA costs `O(H)` likelihood-ratio updates and its Ville guarantee does not grow with `K`.
- Dynamic-programming evaluation truth costs `O(H*S^2)` and is evaluation-only.

For an illustrative industrial horizon `H=1,024` and `N=1,000,000`, materialized model plus paired-calibration Boolean buffers require about 3.07 GB (`3*N*H` bytes). Streaming reduces this to tens of kilobytes of counters, excluding simulator state. The transition workload is approximately 3.07 billion transition steps per shift cell. Core-hours must be extrapolated as `steps / measured_steps_per_second / 3600`; no dollar estimate is justified without a named hardware, utilization, and current price contract.

The held-out report will insert measured CPU rates and identify whether transition generation, paired calibration, or certificate evaluation dominates. Extrapolation assumes IID trajectories, vectorized categorical sampling, fixed `S` and `A`, no simulator synchronization, and no cost for obtaining representative paired structural-shift observations. Violating any of those assumptions can dominate the projected cost.
