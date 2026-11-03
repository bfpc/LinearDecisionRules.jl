# Release notes

## Version 0.4.0

Version 0.4 is a rewrite. It keeps the mathematics of the previous versions — primal and dual
linear decision rules (Kuhn, Wiesemann and Georghiou, 2011), piecewise linear liftings (Georghiou,
Wiesemann and Kuhn, 2015), sampled rules, stages — and rebuilds everything around them: a plain
JuMP model, a typed canonical form, and much smaller reformulations. Models written for 0.2 or 0.3
need changes; the mapping is below. The mathematical notes (a PDF built from `docs/math/`) are
linked from the developer documentation.

Versions 0.1 and 0.2 were registered; their history is in the `legacy` branch. Version 0.3 was
never registered: the tests compare 0.4 with values computed by 0.3.1.

### The new way to write a model

```julia
using JuMP, HiGHS, Distributions, LinearDecisionRules
model = LDRModel(HiGHS.Optimizer)
@variable(model, buy >= 0, Stage(1))
@variable(model, d in Uncertainty(Uniform(80, 120)))   # unmarked: last stage
@variable(model, sell >= 0)
@variable(model, ret >= 0)
@constraint(model, sell + ret <= buy)
@constraint(model, sell <= d)
@objective(model, Max, -10buy + 8ret + 15sell)
optimize!(model)
objective_value(model)    # 460: expected value of the primal rule, a feasible policy
objective_bound(model)    # 1460/3: the dual bound on the optimal value
relative_gap(model)       # the primal–dual certificate
decision_rule(sell)       # sell = d, callable on a scenario
```

`LDRModel(optimizer)` returns a plain `JuMP.Model` whose optimizer reformulates the model at
`optimize!` and solves the reformulations with `optimizer`. Every JuMP function works on it:
`delete`, `set_normalized_rhs`, `fix`, parameters, `copy_model`, `print`, `solution_summary`,
`set_silent`, `set_time_limit_sec`, and functions from other packages that take a `JuMP.Model`.

### Breaking changes: from 0.2 / 0.3 to 0.4

| 0.2 / 0.3 | 0.4 |
|---|---|
| `LinearDecisionRules.LDRModel(opt)`, a wrapper with its own methods | `LDRModel(opt)`, a plain `JuMP.Model`; names are now exported |
| `@variable(m, x, LinearDecisionRules.FirstStage)` | `@variable(m, x, Stage(1))` (`FirstStage` remains as an alias of `Stage(1)`) |
| `set_attribute(x, Stage(t))` after the declaration | `@variable(m, x[t = 1:T], Stage(t))` at the declaration, or `set_stage(x, t)` |
| `@variable(m, d in Uncertainty(distribution = law, stage = t))` | `@variable(m, d in Uncertainty(law), Stage(t))`; per component with containers: `@variable(m, w[t = 2:5] in Uncertainty(law), Stage(t))` |
| An unmarked variable was in stage 2 | An unmarked variable is in the last stage T = max(2, largest declared stage), with a warning when T > 2 |
| A constraint over uncertainties only silently conditioned the law (rejection sampling) | It must be tagged: `@constraint(m, expr, Truncate())` conditions the law; `@constraint(m, expr, ValidInequality())` only tightens the support; untagged, it is an error |
| `set_attribute(m, SolvePrimal(), b)`, `SolveDual()`, `SolveSampled()`, `NumScenarios()` | `set_attribute(m, Formulations(), [PrimalLDR(), DualLDR(), SampledLDR(N; seed)])` |
| `objective_value(m; dual = true)`, `objective_value(m; sampled = true)` | `objective_bound(m)` for the dual (an error when the dual has no result: check `termination_status(m, DualLDR())` first); `objective_value(m; result = 2)` for the sampled rule; any formulation: `objective_value(m, DualLDR())`, `termination_status(m, f)` |
| `get_decision(m, x, η; dual, piece)` | `decision_rule(x; formulation)` (a callable rule), `rule_coefficient(x, η; formulation)` (one coefficient per piece for a lifted η), `rule_matrix(m; formulation)`; `value(x)` for a first-stage decision (an error when the primal rule has no solution, where 0.3 returned values: check `has_values(m)` first) |
| `set_optimizer(m, opt)` changed the inner solver | `set_attribute(m, InnerOptimizer(), opt)` (`set_optimizer` would replace the reformulating optimizer) |
| `SlackAsExpr()` | Removed: slacks are always expressions |
| `RejectionSamplingTimeLimit()`, `RejectionSamplingSeed()`, `RejectionSamplingMaxIterations()`, `RejectionSamplingWarnAttempts()` | `RejectionSamplingTimeLimit()` (default: half of the remaining `TimeLimitSec`, else 10 s per law), `RejectionSamplingSeed()`, `RejectionSamplingMaxSamples()` (default 10,000); the warning attempts are gone |
| `set_parametric_objective!` | Not in 0.4; the value-function recursion returns in an `Experimental` submodule |
| Expectations.jl dependency | Replaced by QuadGK.jl |

### New features

- **Results through JuMP:** `value(x)` is the expected decision `E[x(ξ)]` (exact for affine
  expressions; `expected_value(expr)` is exact for quadratic ones); the primal rule is result 1, the
  sampled rule result 2; `termination_status` is `INFEASIBLE` only on the dual's evidence, and
  `LOCALLY_INFEASIBLE` when only the primal rule is infeasible.
- **Certification:** `is_certified(model)` and the `Certified()` attribute say whether a value is
  exact for the law — not when moments are estimated by rejection sampling, when a law across
  stages does not have linear conditional expectations, when the second-moment matrix is
  singular, or, for the bound, when the quadratic objective is not convex.
- **Where the time goes:** `time_breakdown(model)` and the `TimeBreakdown()` attribute split the
  time of `optimize!` into reading the model, the moments, the checks, and the build, solve and
  results of the inner problems; `time_breakdown(model, formulation)` gives the last three for one
  formulation.
- **Simulation:** `evaluate(model, scenario)` applies a rule to one scenario (decisions, cost,
  violated constraints by name); `simulate(model, n)` draws from the declared laws and
  `simulate(model, scenarios)` takes scenarios from elsewhere (for example SDDP simulations).
- **Supports:** second-order cone constraints tagged `ValidInequality()` or `Truncate()`; the
  `support_constraints(law)` hook; vector constraints in `Nonnegatives`, `Nonpositives`, `Zeros`.
- **Laws:** `ConfidenceMvNormal(μ, Σ, α; support = :auto)` with an exact ellipsoid (SOC) or an LP
  outer approximation by boxes (`:auto` picks the ellipsoid when the inner solver supports cones
  and no decision is integer); exact truncation of univariate laws and of discrete laws (atom
  filtering) under `Truncate()`; rejection sampling with a seed per law and a cache.
- **Breakpoints** (`set_attribute(d, BreakPoints(), points)`) work in multistage models and on
  uncertainties in the objective; `AxisPiecewise` liftings are public, with `lift` and
  `retraction`. As before, `points` is a count `n` (equally spaced) or a vector of positions;
  `0` or `nothing` mean no breakpoints.
- **Breakpoints on combinations (experimental):** `Experimental.set_breakpoints(model, d[1] +
  d[2], points)` makes the rules piecewise affine in a combination of uncertainties, such as the
  total demand served from one stock (the `CombinationPiecewise` lifting). The primal rule works
  with any combination; the dual bound needs the uncertainties of the combination to be in one
  stage. It works best with few uncertainties per combination: see the Experimental page.
- **Re-solves:** parameters in right-hand sides and the objective stay symbolic; when only their
  values change, `optimize!` updates the kept inner problems instead of rebuilding them (45× faster
  for the primal, 148× for the dual on the inventory benchmark at T = 64).
- **Solver settings per formulation:** `DualLDR(; attributes = ["solver" => "ipm",
  "run_crossover" => "off"])`, for example; `PrimalLDR(; box_closed_form = false)` turns off the
  closed form of box supports; `lifted_box` chooses whether the support of an uncertainty with
  breakpoints keeps the bounds that its convex-hull rows imply (by default, the primal leaves them
  out and the dual keeps them: see "Past studies" in `benchmark/README.md`).
- **Low-level access to the rules:** `rule_coefficient(x, η)` and `rule_matrix(model)`, whose
  `RuleMatrix` carries the lifting map, so that the rules are `R.matrix * lift(R.lifting, ξ)`.
  `inner_model(model, f)` returns the inner problem of a formulation, to inspect it or save it with
  `write_to_file`.
- **Internals:** the typed canonical form, `canonicalize`, `moments` and `solve_formulation` are
  documented on the Internals page, but not exported: their contract may change until the
  `Experimental` submodule makes it stable. `LinearDecisionRules.model_info(backend(model))` reads
  the tags of a model for `canonicalize`.

### Public names

Exported:
- **Modelling:** `LDRModel`, `Stage`, `FirstStage`, `stage`, `set_stage`, `unset_stage`,
  `Uncertainty`, `ValidInequality`, `Truncate`, `BreakPoints`, `Formulations`, `InnerOptimizer`,
  `RejectionSamplingSeed`, `RejectionSamplingTimeLimit`, `RejectionSamplingMaxSamples`.
- **Results and rules:** `AbstractFormulation`, `PrimalLDR`, `DualLDR`, `SampledLDR`,
  `FormulationAttribute`, `Certified`, `is_certified`, `TimeBreakdown`, `time_breakdown`,
  `inner_model`, `decision_rule`,
  `DecisionRule`, `LiftedTerm`, `expected_value`, `rule_coefficient`, `rule_matrix`, `RuleMatrix`;
  and methods of `termination_status`, `primal_status`, `objective_value`, `objective_bound`,
  `raw_status` and `solve_time` that take a formulation.
- **Simulation:** `evaluate`, `Evaluation`, `simulate`, `Simulation`.
- **Distributions and liftings:** `ConditionedDistribution`, `ConfidenceMvNormal`,
  `MvDiscreteNonParametric`, `LiftedDistribution`, `support_constraints`, `AbstractLifting`,
  `AxisPiecewise`, `lift`, `retraction`, `LiftingMap`, `LiftingBlock`.

Public, not exported: `LinearDecisionRules.Optimizer` and the submodule
`LinearDecisionRules.Experimental`, with `Experimental.set_breakpoints` (declared `public` on
Julia 1.11 and later).

### Performance

- The primal LDR uses multipliers per support block, and a closed form for box supports: on the
  inventory benchmark at T = 128 it has 57,792 variables, as many as a hand-written model, against
  223,168 in 0.3; the time of `optimize!` outside the inner solver went from about 1.4 s to 0.04 s.
- The dual LDR keeps the structure of the second-moment matrix: on the 12-stage hydrothermal model
  it has 1.06 million nonzeros against 14.2 million in 0.3, with the same value; the design
  measurements with 3 breakpoints per inflow give 15.5 million against 301 million at 12 stages,
  and 64.9 million against 2.6 billion at 24.
- Smaller is not always faster for the solver: see `benchmark/README.md`. For large duals, HiGHS's
  interior point without crossover is often much faster (`optimize!` on the hydrothermal model
  with 8 stages: 5.3 s against 116 s, in `benchmark/hydro/README.md`), but it fails on some
  problems; it is opt-in.

### Fixed since 0.3

- Interval constraints on decisions were silently dropped.
- With breakpoints, uncertainties in the objective were not lifted.
- Clearing breakpoints after a solve left stale state.
- A failed rejection sampler counted a zero vector as a sample; every group reused the same seed.
- An integer first-stage decision could depend on first-stage uncertainty.
- The dual dropped every inequality of a model with no bounded uncertainty (bound 0 for `Max 2x, x ≤ 3`).
- Values of an infeasible or unbounded inner problem (rays, certificates) were reported as results;
  the bound of an inner MIP used its incumbent instead of its bound; an inner solve ending in an
  error could report an invalid bound.

### Requirements and known limitations

- Julia 1.10 or later; JuMP 1.25, MathOptInterface 1.37, Distributions 0.25.
- Fixed recourse: an uncertainty may multiply a decision in the objective, not in a constraint.
- Breakpoints apply to uncertainties whose law is univariate and not coupled to others by
  `Truncate()` rows.
- No expected-value constraints yet; the value-function recursion (LFDR) and nonlinear liftings
  are planned for the `Experimental` submodule.
- Breakpoints on a combination of uncertainties of several stages have no dual bound. The moments
  of a combination are exact and fast for discrete and uniform laws, but integrate over every
  uncertainty of the combination with another continuous law (slow beyond about 3).
