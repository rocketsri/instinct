# P5.0/P5.1 implementation review v2

- **Resolved:** weights, functions, and local updates commute with declared
  neuron relabeling.
- **Resolved:** remote-neuron taint cannot change a selected local update;
  duplicate-width normalization and coordinate-order negative controls pass.
- **Resolved:** serialization counts all persistent fields and remains fixed
  while instantiated parameter count changes.
- **Open before a P5.1 scientific comparison:** Cartesian learned tasks and
  CPPN, muP, Meta-SGD, Differentiable Plasticity, shared-rule, and backprop
  baselines at matched examples/steps/compute.
- **Verdict boundary:** E0 generator instrumentation only.
