## P1 kill conditions

**KILL — transfer across environment speed, transfer across hardware latency, surface survives a reflex change**

| Condition | Verdict | Detail |
| --- | --- | --- |
| transfer across environment speed | **KILL** | speed: 0.5 -> 1.0 | regret +1.2766 (excess +1.2736), tau +0.74, argmax hit 58%; excess regret +1.2736 vs limit 0.2719 (25% of a 1.0874 effect) |
| transfer across hardware latency | **KILL** | hardware: 0.25 -> 0.5 | regret +1.2766 (excess +1.2736), tau +0.74, argmax hit 58%; excess regret +1.2736 vs limit 0.2719 (25% of a 1.0874 effect) |
| transfer across environments | _inconclusive_ | environment: chase_chain -> corridor_with_pit | regret +0.0000 (excess +0.0000), tau +0.70, argmax hit 39%; the target frontier is flat (effect 0.0705 <= 0.2719), so budget buys nothing there and no prediction can be wrong |
| surface survives a reflex change | **KILL** | reflex: greedy -> hold | regret +1.6148 (excess +1.6148), tau -0.48, argmax hit 39%; excess regret +1.6148 vs limit 0.2719 (25% of a 1.0874 effect) |
| beats a lookup table | PASS | 71% of cells beat the table (need 50%); median margin +172.5378 nats |
| decomposition resolves the effect | PASS | worst |eps_cross| 4.441e-16 against a 1.0874 budget effect |
| structure is not just delay rounding | PASS | 21% of budget points share a delay with another point |

A tripped condition is a result. The remaining figures describe a surface that failed its own stopping rule and should be read as diagnostics, not as findings.

## What was measured

- 2592 cells: 2 environments x 2 reflexes x 3 speeds x 3 latencies x 12 budgets x 10 start states
- Exact closed-form solves, no sampling. Wall clock 6.9s.
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
| speed (0.5 → 1.0) | +1.2766 | +1.2736 | +0.74 | 58% |
| hardware (0.25 → 0.5) | +1.2766 | +1.2736 | +0.74 | 58% |
| reflex (greedy → hold) | +1.6148 | +1.6148 | -0.48 | 39% |
| environment (chase_chain → corridor_with_pit) | +0.0000 | +0.0000 | +0.70 | 39% |
| collapse ((0.5, 0.5) → (1.0, 0.25)) | +0.2395 | +0.2395 | +nan | 88% |

## Decomposition, averaged over cells

| Term | Mean | Note |
| --- | --- | --- |
| G_plan | +0.0644 | benefit of the better decision |
| R_intermediate | +1.1929 | banked by the reflex |
| L_arrival | +0.5369 | arrival regret: the decision went stale |
| L_wait | +2.6347 | whole cost of waiting, incl. discounting |
| L_irreversible | +0.0014 | what planning cannot undo |
| sigma | -1.1417 | net gain over the base budget |

`L_irreversible` by environment, which is the term that should separate an absorbing environment from a recoverable one:

| Environment | mean | min | one-signed? |
| --- | --- | --- | --- |
| chase_chain | -0.1016 | -1.2805 | no |
| corridor_with_pit | +0.1043 | -0.1356 | no |
