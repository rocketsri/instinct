# P7 simulator implementation review v2

- **Resolved:** audited rejected items are propensity-logged and cannot enter
  update content; accepted real tokens are unique across padded flushes.
- **Resolved:** continuous trickle, burst, multi-owner, and overload traces are
  separate controls with offered load and score-queue delay reporting.
- **Preserved blocker:** no realistic qualifying P2.2 artifact exists, so P7.0
  cannot execute and simulator verdicts remain INCONCLUSIVE.
- **Open:** finite-capacity overflow policy, concurrent arrivals during service,
  measured T4 kernels, rejected-item delayed utility, and task-quality evidence.
