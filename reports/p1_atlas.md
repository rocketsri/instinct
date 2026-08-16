## P1 kill conditions

**KILL — transfer across environment speed, transfer across hardware latency, surface survives a reflex change**

| Condition | Verdict | Detail |
| --- | --- | --- |
| transfer across environment speed | **KILL** | speed: 0.5 -> 1.0 | regret +0.9672 (excess +0.9642), tau +0.17, argmax hit 0%; excess regret +0.9642 vs limit 0.2719 (25% of a 1.0874 effect) |
| transfer across hardware latency | **KILL** | hardware: 0.25 -> 0.5 | regret +0.9672 (excess +0.9642), tau +0.17, argmax hit 0%; excess regret +0.9642 vs limit 0.2719 (25% of a 1.0874 effect) |
| transfer across environments | PASS | environment: chase_chain -> corridor_with_pit | regret +0.0907 (excess +0.0907), tau +0.08, argmax hit 0%; excess regret +0.0907 vs limit 0.2719 (25% of a 1.0874 effect) |
| surface survives a reflex change | **KILL** | reflex: greedy -> hold | regret +1.7399 (excess +1.7399), tau -0.47, argmax hit 0%; excess regret +1.7399 vs limit 0.2719 (25% of a 1.0874 effect) |
| beats a lookup table | PASS | 71% of cells beat the table (need 50%); median margin +172.5378 nats |
| decomposition resolves the effect | PASS | worst |eps_cross| 4.441e-16 against a 1.0874 budget effect |
| structure is not just delay rounding | PASS | 21% of budget points share a delay with another point |

A tripped condition is a result. The remaining figures describe a surface that failed its own stopping rule and should be read as diagnostics, not as findings.

## What was measured

- 2592 cells: 2 environments x 2 reflexes x 3 speeds x 3 latencies x 12 budgets x 10 start states
- Exact closed-form solves, no sampling. Wall clock 18.2s.
- Worst |eps_cross - L_base_delay| residual: identity holds to 7.38 in identified terms.

## Regime taxonomy

| Regime | Cells | Share |
| --- | --- | --- |
| irreversible | 102 | 47% |
| flat | 63 | 29% |
| smooth-decay | 44 | 20% |
| discretization-artifact | 7 | 3% |

## Transfer

| Transfer | Budget regret | Excess | Kendall tau | Argmax hit |
| --- | --- | --- | --- | --- |
| speed (0.5 → 1.0) | +0.9672 | +0.9642 | +0.17 | 0% |
| hardware (0.25 → 0.5) | +0.9672 | +0.9642 | +0.17 | 0% |
| reflex (greedy → hold) | +1.7399 | +1.7399 | -0.47 | 0% |
| environment (chase_chain → corridor_with_pit) | +0.0907 | +0.0907 | +0.08 | 0% |
| collapse ((0.5, 0.5) → (1.0, 0.25)) | +0.2395 | +0.2395 | +nan | 88% |

## Decomposition, averaged over cells

| Term | Mean | Note |
| --- | --- | --- |
| G_plan | +0.0644 | benefit of the better decision |
| R_intermediate | +1.1929 | banked by the reflex while waiting |
| L_arrival | +0.5369 | arrival regret: the decision went stale |
| L_wait | +2.6347 | whole cost of waiting, incl. discounting |
| L_irreversible | +0.0014 | the part no later planning recovers |
| sigma | -1.1417 | net gain over the base budget |

`L_irreversible` by environment, which is the term that should separate an absorbing environment from a recoverable one:

| Environment | mean | min | one-signed? |
| --- | --- | --- | --- |
| chase_chain | -0.1016 | -1.2805 | no |
| corridor_with_pit | +0.1043 | -0.1356 | no |
