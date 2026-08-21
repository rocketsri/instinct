# P5.2 bounded plasticity-only review v3

- **Theory:** initialization is conventional and identical across arms; only a
  four-coefficient shared local rule and scalar step size are selected.
- **Locality:** deployment uses endpoint activities, the selected synapse's
  weight, coordinate/type, and a declared scalar broadcast output error.
  Remote-neuron taint and neuron-permutation tests are mandatory controls.
- **Holdout:** sinusoid/polynomial pilot tasks at widths 4/6 and depth 1 are
  separated from composition tasks at width 8 and depth 2.
- **Baselines:** frozen, scale-matched random, no-plasticity, and same-step
  direct backprop arms share examples and initialization.
- **Boundary:** P5.2 can only be NARROW. NNiT and a joint P5.3 generator are not
  reproduced. The observed `0.2207%` gain cleared a preregistered zero threshold
  but has no uncertainty-based meaningful-effect gate, while direct backprop is
  substantially stronger. P5.3 therefore remains blocked pending independent
  review and a fresh, nonzero threshold; this held-out result cannot set it.
