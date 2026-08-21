# P2.2 bounded natural-image pilot review v2

- **Theory — unchanged:** the section 5.2 multi-horizon utility remains the
  estimand. Benefit, retention damage, and write cost are stored separately;
  candidate restriction precedes top-m; the future oracle is an offline ceiling.
- **Heuristics — controls added:** always-write, never-write, eight paired random
  selectors, validation-only top-m, and oracle-ceiling arms expose inactivity,
  write-rate, lucky-random, reusable-validation, and future-leakage failure modes.
  Random and validation-only arms compute every candidate update and are charged
  the same validation-forward budget.
- **Identifiability — fail closed:** aged-probe damage must correlate with damage
  on the disjoint downstream band at preregistered Spearman >= 0.25. Reused
  oracle probes are capped at four uses. Failure makes the run INCONCLUSIVE.
- **Realism — BLOCKER remains:** the source is a real photograph under a changing
  corruption stream, but one image and a 3x4 affine RGB head do not satisfy the
  frozen small-pretrained-ResNet/ViT CIFAR-C benchmark. The realism prerequisite
  is deliberately false, irrespective of observed metrics.
- **P7 dependency — BLOCKER remains:** no qualification JSON is emitted. P7 stays
  locked unless a later whole-stream realistic run passes matched count, matched
  deployable compute, identifiability, and positive useful sparsity.
- **Falsifiers:** oracle utility fails to beat validation/random at the same `m`;
  never-write has lower downstream MSE; aged/downstream damage correlation falls
  below threshold; any stream IDs overlap; or any selected timestep lies outside
  the preregistered eligible window.
