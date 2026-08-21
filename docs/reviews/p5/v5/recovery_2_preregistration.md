# P5.2 recovery cycle 2 — final preregistration

## Preserved evidence and sealed units

- Initial v3: NARROW, 0.220727% relative gain, eval seeds 591/592.
- Recovery 1: INCONCLUSIVE, 0.475202% gain versus a 1.665423%
  pilot-derived threshold, eval seeds 691/692/693.
- All five earlier evaluation seeds are sealed. Recovery 2 uses selection seeds
  721/722, threshold-only seed 723, and evaluation seeds 791–794.

## Scientifically distinct mechanism

Recovery 2 does not enlarge the earlier polynomial term pool. It generates a
layer-local feedback signal from a fixed Fourier basis over postsynaptic
coordinate and normalized layer coordinate, distinguishes output versus hidden
type, and modulates that feedback and learning rate with normalized adaptation
time. The resulting signal multiplies the scalar broadcast output error and the
presynaptic activity. This follows the local-feedback mechanism boundary of
Learning to Learn with Feedback and Local Plasticity while retaining a much
smaller fixed program and no learned feedback matrix:
<https://github.com/jlindsey15/FeedbackAndLocalPlasticity>.

Each synapse sees only its pre/post activity, own weight, scalar broadcast
error, coordinate, layer coordinate/type, and step. The program has twelve
coefficients plus a step size: 104 bytes. It has no topology-sized persistent
state, generated initialization, task token, or architecture embedding.

## Frozen gate

- Pilot tasks: new polynomial/composition/warped instances at widths 6/8 and
  depths 1/2. Evaluation: unseen `chirp` family, widths 10/12, depths 3/4.
- Threshold: 1.96 times the standard error of gains on calibration seed 723,
  computed before final evaluation and strictly positive.
- Controls: frozen feedback basis, sealed recovery-1 program, eight random
  feedback-basis programs, no plasticity, and same-step direct backprop.
- Pass: fresh gain over the strongest non-backprop comparator exceeds the
  threshold, beats random median, no plasticity, and recovery-1 frontier, with
  locality/equivariance/time/accounting controls passing.
- Pass is at most NARROW. Failure is STOP/F3 and archives P5.2 after its second
  allowed mechanism recovery. Criteria will not be weakened and P5.3 will not
  start on failure.
