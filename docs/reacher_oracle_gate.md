# Reacher Specialist Oracle Gate

## Conclusion: neither passes

This conclusion is reached at the mandatory pre-training partition gate. No low/high
specialist was trained, because aggregate-metadata torque magnitude is not a validated
motion-effect regime axis. Consequently neither hypothetical teacher is eligible for
distillation.

On the complete 1,000-episode feasibility subset, median magnitude splitting is
balanced at 500/500, but high-torque episodes have only `1.0705×` the visual motion
proxy of low-torque episodes. The episode-level action/motion correlation is only
`r=0.0546`. Non-degenerate action magnitude is necessary but not sufficient.

The machine-readable decision is
`results/alrd_oracle_gate/reacher/oracle_summary.json`. It explicitly records that
this is a partition-precondition failure, not a measured specialist-versus-general
model comparison.

Because neither teacher passes, the fixed pipeline forbids prediction KD, response KD,
and leave-one-counterfactual-out training on this partition. Creating teachers merely
to finish the experiment table would bypass the oracle gate.

The result supports conclusion **C: specialist hypothesis fails; redesign partition**.
The next route is singular: move to an action-semantics-aware non-magnitude regime axis
in Robot Arm, after completing Reacher's independent official-baseline diagnosis.
