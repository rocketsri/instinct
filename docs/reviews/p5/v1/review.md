# P5 pre-implementation review v1

- **Theory — MAJOR:** distinguish relabeling invariance, generator equivariance, and coordinate sensitivity; declare fan-in scaling and count all serialized state.
- **Heuristics — MAJOR:** reject all-zero symmetry, test local-update taint, width duplication, fixed bytes, and Cartesian holdouts.
- **Literature — BLOCKER for P5.1+ claims:** add Differentiable Plasticity, learned local rules, CPPN/indirect encoding, muP, Meta-SGD, and size-conditioned generators.
- **Synthesis:** P5.0/P5.1 instrumentation only; no learned superiority claim.
- **Falsifier:** fixed ordering is required, scaling fails, or gains vanish under parameter/code accounting.
