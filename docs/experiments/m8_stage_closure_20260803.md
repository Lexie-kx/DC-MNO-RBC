# M8 Stage Closure

Date: 2026-08-03

## Completed formal experiments

- M8-A-O5-FullStatic-H4
- M8-B-O5-ParamConditionedCoupling-H4
- M8-Gate-GateOnly

## Formal conclusion

Under the tested static fusion, parameter-conditioned generic coupling,
and module-level gate designs, no stable synergistic improvement was
observed.

M8 remained competitive in some short-range settings, but it did not
consistently preserve both:

- the unseen-Pr advantage of M7-ParamTokenOnly;
- the unseen-Ra advantage of M7-FieldCoupling.

Therefore, the M8 stage is formally closed.

## M8-C status

M8-C Physics-Inspired Structured Coupling was implemented as an
exploratory code prototype.

It contains five physics-inspired branch families:

- buoyancy;
- advection;
- pressure;
- viscous;
- thermal diffusion.

It retains the independent ParameterToken and provides static and
limited parameter-conditioned coupling modes.

M8-C was not completed as a formal O5 training/evaluation experiment.
It is not included in the formal M8 result tables or conclusions.

Status:

- code preserved;
- development stopped;
- no further training;
- no Gate-Joint;
- no additional M8 tuning.

## Numbering

The existing M9-Codomain-Attention-H4 branch remains paused and keeps
the M9 number. Its number must not be overwritten.
