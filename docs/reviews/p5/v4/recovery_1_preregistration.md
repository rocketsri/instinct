# P5.2 recovery cycle 1 — preregistration and diagnosis

## Preserved initial result

P5.2 v3 remains immutable: evaluation seeds 591/592, composition tasks at
width 8/depth 2, learned normalized AUC 0.984585, best non-backprop AUC
0.986764, relative gain 0.00220727, verdict NARROW. Those seeds are sealed and
cannot appear in recovery selection, calibration, or evaluation.

The initial result cannot distinguish a real effect from one lucky random-rule
comparison because it used one random rule and a zero minimum-effect threshold.
The four-term rule family may also be too weak, while its 0.22% margin is far
smaller than the gap to same-step backprop.

## Frozen recovery design

- Selection seeds: 621/622. Pilot-threshold calibration seed: 623 only.
- Fresh evaluation seeds: 691/692/693; the evaluation family is `warped`, not
  used by rule selection or threshold calibration.
- Pilot topologies: widths 5/7 and depths 1/2. Evaluation: widths 9/11 and
  depth 3.
- Program: eight shared coefficients plus one scalar step size. Total serialized
  state is 72 float64 bytes; initialization and task/topology tokens remain zero.
- Threshold: 1.96 times the standard error of per-cell relative gains on pilot
  calibration seed 623. It is computed before fresh evaluation and must be
  strictly positive.
- Controls: frozen four-term rule, eight fixed random eight-term rules,
  no-plasticity, and same-example/same-step direct backprop.
- Pass: fresh gain over the strongest non-backprop control must exceed the
  pilot threshold, beat the random-rule median, beat no plasticity, and pass
  locality, permutation, initialization-separation, and accounting controls.
- Verdict is at most NARROW. P5.3 is not implemented in this recovery.

## Current primary-literature boundary

- Differentiable Plasticity already meta-learns plastic connection strengths
  and provides an official implementation. Its recurrent/episodic setup is not
  reproduced here: <https://proceedings.mlr.press/v80/miconi18a.html> and
  <https://github.com/uber-research/differentiable-plasticity>.
- Learning to Learn with Feedback and Local Plasticity already meta-learns
  feedback, initialization, and local plasticity rates for multilayer online
  learning. This is a required future functional baseline, but it entangles
  initialization and feedback machinery that P5.2 intentionally excludes:
  <https://papers.nips.cc/paper/2020/hash/f291e10ec3263bd7724556d62e70e25d-Abstract.html>
  and <https://github.com/jlindsey15/FeedbackAndLocalPlasticity>.
- Meta-Learning Biologically Plausible Plasticity Rules with Random Feedback
  Pathways directly occupies interpretable shared-term rule discovery under
  random feedback. P5 cannot claim novelty for an eight-term rule search:
  <https://www.nature.com/articles/s41467-023-37562-1>.
- Confavreux et al. likewise meta-learn low-dimensional plausible rules and
  release code: <https://github.com/basile6/MetaLearnBiologicallyPlausibleRules>.
- NNiT is the direct width/depth-generalizing initialization boundary. Its ICML
  2026 project page currently marks code as TBD, so it cannot be honestly
  reproduced in this recovery: <https://indigoyeoma.github.io/projects/nnit/>.
- Recent three-factor and richer polynomial/MLP plasticity-rule work further
  narrows any generic learned-rule claim:
  <https://arxiv.org/abs/2512.09366> and
  <https://doi.org/10.1371/journal.pcbi.1012998>.

Consequently this recovery tests only whether the local mechanism survives
better controls. It makes no novelty claim. A future P5.3 comparison would need
Differentiable Plasticity, Feedback and Local Plasticity, random-feedback rule
meta-learning, Meta-SGD/MAML, a size-conditioned generator, and NNiT when code
or a comparable checkpoint becomes available.
