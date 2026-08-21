# P4.2 final recovery preregistration v10

Recovery 1 improved same-call slack chunks by only `0.00096`, below the frozen
`0.01` threshold. The higher-call per-step verifier remained much stronger.
The final recovery therefore tests the frozen falsifier directly: can adaptive
verification obtain the gain at the same average replanning budget?

A separate pilot layout selects one fixed recovery-slack threshold targeting
four total calls: two long-chunk planning calls plus approximately two expected
verifier interventions. The threshold is then frozen. A new layout and new
magnitude-two axial impulses are evaluated once. Task-only and slack chunks use
four deterministic calls; shorter chunks use six and remain an exposed
higher-cost control.

If the matched-call verifier wins, P4 is narrowed to an inference-time recovery
gate and slack-policy training is archived. If it fails, both allowed P4.2
recovery cycles are consumed and the bounded P4.2 mechanism is archived.
