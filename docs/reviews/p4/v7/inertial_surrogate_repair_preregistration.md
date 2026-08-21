# P4.1 inertial estimator F2 repair v7

The v1 held-out layout remains sealed. It achieved `0.914` aggregate coverage
but only `6/9` conditions attained 90% state coverage, so its result remains
INCONCLUSIVE/F2.

The repair adds only explicit obstacle geometry: sorted obstacle coordinates,
relative offsets, and Manhattan distances. It retains the same quadratic basis,
exact terminal constraints, and condition-balanced calibration. New train,
calibration, and test layouts are disjoint from every v1 layout. Thresholds are
unchanged: 90% aggregate coverage, 80% condition success, and full width at
most 75% of the exact-value range.

No v1 test state or label enters fitting, calibration, threshold selection, or
the fresh test layout.
