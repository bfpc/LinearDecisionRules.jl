# [Design decisions](@id developer_design)

This page records why the package is built the way it is: the aims of the rewrite of version 0.4,
the choices that shaped it, and the measurements behind them. The behaviour itself is in the
docstrings, and the organization of the code in the other developer pages. Before you change one of
these choices, read its reason here, and discuss the change before you code it.

The plan agreed after the prototype of October 2026 is archived, frozen, in
[`spike/DESIGN.md`](https://github.com/bfpc/LinearDecisionRules.jl/blob/main/spike/DESIGN.md),
with the prototypes and their audit.

## The aims of the rewrite

Version 0.4 keeps the mathematics of the previous versions: primal and dual linear decision rules
(Kuhn, Wiesemann and Georghiou, 2011), axis-aligned piecewise liftings (a special case of
Georghiou, Wiesemann and Kuhn, 2015), a sampled formulation, and stages. It rebuilds everything
around them, with three aims.

- **Flexibility and extensibility.** One typed canonical form sits between the JuMP model and the
  formulations, so that a new law, lifting or formulation is a small addition to one layer. The
  `Experimental` submodule is where such additions can ship before their interface is settled.
- **Performance, where it was possible.** Smaller inner problems and a better structure, each tried
  and measured: the closed form of box supports in the primal, the structured dual, the re-solves
  that keep the inner problems. The benchmarks became systematic only once the interface settled;
  their READMEs record what was measured, and when.
- **Better-organized code.** The package is now much larger than the previous one, so it needs a
  structure that a reader can follow: one folder per layer, one concern per file, and a core that
  does not know JuMP's modelling layer (see [Architecture](@ref developer_architecture)).

What version 0.3 showed, and the rewrite set out to fix:

- building the model cost more than solving it: on the inventory problem with T = 128, the primal
  LP had 223k variables, against 58k in a model written by hand (the package now matches it,
  57,792);
- the dual exploded in multistage models, since `S M W'ᵀ` was dense: 300 million nonzeros on the
  hydrothermal problem with 12 stages and 3 breakpoints. The structured dual has 13 to 40 times
  fewer on the same instances, with the same values;
- implicit syntax was error-prone: a constraint without decisions silently conditioned the law;
- multistage models were not tested in CI.

## The user layer: an optimizer under a plain JuMP model

The user writes a plain `JuMP.Model`, and `LinearDecisionRules.Optimizer` reads it at `optimize!`.
The prototype compared three user layers on the same core
([`spike/audit/AUDIT.md`](https://github.com/bfpc/LinearDecisionRules.jl/blob/main/spike/audit/AUDIT.md)):

| | A wrapper `AbstractModel` | An MOI optimizer (chosen) | An optimize hook |
|---|---|---|---|
| Code lines, foreign methods, forwarders | 575, 114, about 42 | 310, 59, 0 | 261, 20, 0 |
| Generic JuMP functions | `MethodError`, unless forwarded | all work | all work, through a test mock |
| Builders typed `::JuMP.Model` | do not compose | work | work |
| Errors | mostly at declaration | at declaration for unsupported types, else at `optimize!` | at `optimize!` |

The wrapper gives the earliest errors and full control of the results, but every JuMP function a
user calls needs a method of its own, and every new JuMP function is a gap. The optimizer gets all
of JuMP for free. Its costs, accepted:

- **No bridges.** A bridge adds auxiliary variables that have no stage, so the optimizer supports
  natively exactly what it reads, and models are built with `add_bridges = false` (`LDRModel` does
  it).
- **Some checks wait for `optimize!`.** A bound, `fix` or integrality set on an uncertainty after
  its declaration reaches the model through JuMP's own methods, where catching it would be type
  piracy. `optimize!` reports these, with messages that name the variables and constraints.
- **Metadata are part of the model.** Stages are `VariableIndex`-in-`Stage(t)` constraints and
  laws are MOI sets, so they show in `print(model)`, follow `delete` and `copy_model`, and need no
  side tables.

The core does not depend on this choice: replacing `src/moi/` and `src/jump/` by another front end
leaves the core and its tests unchanged. A wrapper front end on the same core remains a research
question. To try it: start from the prototype's `spike/wrapper/LDRWrapper.jl`, keep the public names
so that the tutorials and the end-to-end tests of `test/jump/` and `test/moi/` run unchanged, and
measure it with the audit's `spike/audit/metrics.jl` and adversarial probes, so that the comparison
is like for like.

## Results

- **Result 1 is the primal rule and result 2 the sampled rule; the dual takes no result.** The rule
  of the dual is not a policy, so its value is `objective_bound`, and `relative_gap` is the
  certificate of the primal rule. The bound needs no primal point: an inner MIP can prove it
  without an incumbent.
- **`value(x)` is the expected decision `E[x(ξ)]`**, the one number that summarizes a rule. The
  rule itself is `decision_rule(x)`, and the matrices are public too (`rule_matrix`), since users
  post-process rules.
- **A certificate rather than a warning.** Each value carries `certified`, with the reason in its
  raw status when it is false (moments estimated by sampling, a singular second-moment matrix, a
  law across stages without linear conditional expectations, a nonconvex objective). A flag that
  results carry is harder to miss than a warning.
- **The canonical layer is documented but not exported** (the Internals page), so that it can still
  change; it becomes public with the `Experimental` submodule.

## Modelling conventions

- **One `Stage(t)` tag** for decisions and uncertainties. An untagged variable is in the last stage,
  with a warning when there are more than two.
- **Constraints over the uncertainties alone must be tagged:** `ValidInequality()` only tightens the
  support, `Truncate()` conditions the law. Untagged, they are an error, since 0.3 silently
  conditioned the law.
- **Constraints have no stage.** The support blocks make the masking of the slacks automatic; a
  `Stage` tag on a constraint is an error, and an expectation constraint at time `t`, if it comes,
  will have a tag of its own.
- **Exact laws before sampled ones.** A truncated univariate law is truncated exactly, a conditioned
  product of discrete laws is enumerated, and only the other cases are sampled by rejection, with a
  seed per law, a cache, and a share of the time limit.
- **A parameter is a variable fixed to its value**, not a JuMP `Parameter`. HiGHS bridges a
  `Parameter` and cannot change it without a rebuild, while `fix` changes a bound in place and keeps
  the basis: on the inventory with T = 64, a re-solve took 0 simplex iterations instead of 4999.

## Formulations

- **Inner problems are JuMP models built from sparse blocks**, with structural nonzeros only.
- **The primal uses the closed form of box supports**: one auxiliary per coefficient instead of two
  multipliers and an equality. It brought the inventory primal with T = 128 to the size of the
  hand-written LP.
- **The dual keeps the auxiliary `u = S μ̄`.** Substituting it out was faster on the hydrothermal
  problem but 20 times slower on the inventory, whose rows become dense: smaller is not always
  faster (`docs/math`, "Smaller is not always faster"). A rule by density would need more
  instances.
- **The mean of a decision in `E[x'Px]` is a variable**, with one equality, so that the term has
  `nnz(P)` entries instead of one per pair of columns: on a coupled inventory with T = 48, 749,700
  terms became 20,776.
- **The dual's condition across stages is linear conditional expectations, law by law**, not zero
  correlation: correlation flags laws that satisfy it and misses dependent laws that do not
  (`docs/math`, `rem:lce`).
- **`ConfidenceMvNormal(…; support = :auto)` takes the ellipsoid when the inner solver supports
  second-order cones and no decision is integer** (a mixed-integer cone program is much harder than
  a MILP). With the same solver (Clarabel), the cone program took 0.9 s against 1.3 s for the LP of
  the boxes, with a better primal value. Its draws stay by rejection, whose acceptance rate is
  exactly `α` in any dimension.

## Lessons about extending JuMP

These cost time in every design of the prototype, and will again in any new front end.

- `build_variable(error_fn, info, tag)` is the hook of `@variable(model, x, Tag(…))`, and it
  composes: for `x in Set, Tag`, JuMP calls it on the result of the set, and the vector form gets
  the tagged result of each component. Return a type of your own, with a method of `add_variable`
  for it, and there is no piracy.
- Tags play the role of an `AbstractVariable`, not of an `AbstractVariableRef`: the confusion broke
  `@variable(model, x, Int, Stage(2))` in 0.3.
- `@constraint(model, expr, Tag())` accepts one extra positional argument only.
- Every custom attribute needs `MOI.supports`, and `copy_model` needs
  `MOI.Utilities.map_indices` for attribute values that hold indices.
- JuMP discards the results after any change of the model, including `set_silent`,
  `set_time_limit_sec` and raw attributes; the package does the same, which is the least
  surprising choice.

## The phases

The package was built in phases, each closed by a git tag:

| Phase | Content | Tag |
|---|---|---|
| 0 | The prototype of the user layers, archived in `spike/`; the repository, CI and the golden values of 0.3 | `phase0-complete` |
| 1 | The canonical form, the information structure, the support blocks, the moments, the block primal, the structured dual, the sampled formulation, the optimizer and the results | `phase1-complete` |
| 2 | Breakpoints (`AxisPiecewise`), their moments, `evaluate` and `simulate` | `phase2-complete` |
| 3 | The roles of the rows of the support, rejection sampling, conic supports, the access to the rules, the mathematical notes and the tutorials | `phase3-complete` |
| 4 | The closed form of box supports, the re-solves with parameters, and the solver settings per formulation | `phase4-complete` |

The comparison of solve times with 0.3 is the tag `study-0.3-comparison`, summarized in
"Past studies" of the
[benchmark README](https://github.com/bfpc/LinearDecisionRules.jl/tree/main/benchmark).
