# P5 theory map

- Question/hypothesis: frozen section 8.1.
- Target: one fixed serialized coordinate program emits initialization and a local update rule across widths/depths.
- Assumptions/audit: distinguish function relabeling, generator equivariance, and coordinate-order sensitivity; count all code/task/architecture/broadcast state; declare broadcast error dimensionality.
- Controls: nonzero/rank, neuron permutations, duplicated widths/fan-in scaling, strict-locality taint, Cartesian task–architecture holdouts; initialization/plasticity factorial later.
- Baselines required before claims: orthogonal/standard, muP, Meta-SGD, CPPN/indirect encoding, Differentiable Plasticity/shared local-rule, backprop, size-conditioned generator.
- Verdict: P5.0/P5.1 can validate instrumentation/transfer only. Learned joint-program claims remain blocked.
- P5.2 plasticity-only estimand: normalized area under the zero-to-few-shot
  loss curve on composition tasks and width/depth combinations excluded from
  local-rule selection. Every arm starts from the identical conventional
  fan-in initialization for its topology.
- P5.2 local information: each synapse may use its own weight, pre- and
  postsynaptic activities, a declared scalar broadcast output error, step/type,
  and the postsynaptic coordinate. It cannot access target weights, remote
  neuron activity, a full gradient, task tokens, or topology-sized learned state.
- P5.2 accounting: the plasticity program owns four coefficients and one step
  size (40 float64 bytes). Conventional instantiated weights are reported
  separately and are never attributed to the program.
- P5.2 verdict: at most NARROW. P5.3 stays blocked without a positive held-out
  plasticity gate and independent review.
- P5.2 final recovery outcome: v3's sub-percent NARROW result did not survive
  either pilot-derived meaningful-effect gate. A distinct coordinate/type/time
  feedback mechanism was worse than the sealed recovery-1 frontier. P5.2 is
  STOP/F3 after two valid recoveries; P5.3 remains blocked.
- Source: proposal version 2.1; portfolio amendment 2.1 changes scheduling only.
