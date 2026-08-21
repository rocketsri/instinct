# P2.2 recovery-2 extended train-only pilot preregistration

The first guarded T4 smoke used 12 stream steps and produced only four
horizon-eligible candidates. Consequently the 0.10 and 0.25 write fractions
both rounded to one write. All four offline utilities were negative, useful and
harmful events were not both present, and the oracle had no frontier advantage.
The artifact is preserved at `/private/tmp/instinct-p2-r2-pilot.json`.

This second pilot remains entirely inside disjoint bands of the official
CIFAR-10 **train** split and can never qualify P7. It expands source fitting to
4,096 images/two epochs and uses 32 stream steps with 256 images per stream.
All adaptation parameters, learning rate, horizons, utility weights, selectors,
baselines, write fractions, thresholds, and the reserved seed-241 test contract
remain unchanged.

The full recovery unit will not be opened unless the extended pilot contains
both useful and harmful candidate events and gives the offline oracle a
nonzero frontier advantage over matched deployable selectors. Failure is a
pilot mechanism warning, not a held-out recovery verdict; it leaves P7 locked.
