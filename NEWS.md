# Release notes

## Version 0.4.0 (draft)

Version 0.4 is a rewrite. It keeps the mathematics of the previous versions — primal and dual
linear decision rules (Kuhn, Wiesemann and Georghiou, 2011), piecewise linear liftings (Georghiou,
Wiesemann and Kuhn, 2015), sampled rules, stages — and rebuilds everything around them: a plain
JuMP model, a typed canonical form, and much smaller reformulations. Models written for 0.2 or 0.3
need changes; the mapping is below. `DESIGN.md` explains the decisions and `docs/math/` the
mathematics.

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
| `Uncertainty(distribution = law, stage = t)` | `@variable(m, d in Uncertainty(law), Stage(t))`; per component with containers: `@variable(m, w[t = 2:5] in Uncertainty(law), Stage(t))` |
| An unmarked variable was in stage 2 | An unmarked variable is in the last stage T = max(2, largest declared stage), with a warning when T > 2 |
| A constraint over uncertainties only silently conditioned the law (rejection sampling) | It must be tagged: `@constraint(m, expr, Truncate())` conditions the law; `@constraint(m, expr, ValidInequality())` only tightens the support; untagged, it is an error |
| `set_attribute(m, SolvePrimal(), b)`, `SolveDual()`, `SolveSampled()`, `NumScenarios()` | `set_attribute(m, Formulations(), [PrimalLDR(), DualLDR(), SampledLDR(N; seed)])` |
| `objective_value(m; dual = true)`, `objective_value(m; sampled = true)` | `objective_bound(m)` for the dual; `objective_value(m; result = 2)` for the sampled rule; any formulation: `objective_value(m, DualLDR())`, `termination_status(m, f)` |
| `get_decision(m, x, η; dual, piece)` | `decision_rule(x; formulation)` (a callable rule), `rule_coefficient(x, η; formulation)` (one coefficient per piece for a lifted η), `rule_matrix(m; formulation)` |
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
  stages does not have linear conditional expectations, or when the second-moment matrix is
  singular.
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
  `retraction`.
- **Re-solves:** parameters in right-hand sides and the objective stay symbolic; when only their
  values change, `optimize!` updates the kept inner problems instead of rebuilding them (45× faster
  for the primal, 148× for the dual on the inventory benchmark at T = 64).
- **Solver settings per formulation:** `DualLDR(; attributes = ["solver" => "ipm",
  "run_crossover" => "off"])`, for example; `PrimalLDR(; box_closed_form = false)` turns off the
  closed form of box supports.
- **Low-level access:** `canonicalize`, `moments`, `support_blocks`, `solve_formulation` and the
  typed canonical form are public, for users who build their own formulations.

### Performance

- The primal LDR uses multipliers per support block, and a closed form for box supports: on the
  inventory benchmark at T = 128 it has 57,792 variables, as many as a hand-written model, against
  223,168 in 0.3; time outside the solver went from about 1.4 s to under 0.5 ms.
- The dual LDR keeps the structure of the second-moment matrix: on the 12-stage hydrothermal model
  it has 1.06 million nonzeros against 14.2 million in 0.3, with the same value; the design
  measurements with 3 breakpoints per inflow give 15.5 million against 301 million at 12 stages,
  and 64.9 million against 2.6 billion at 24.
- Smaller is not always faster for the solver: see `benchmark/README.md` and
  `benchmark/compare/README.md`. For large duals, HiGHS's interior point without crossover is often
  much faster (hydro, 8 stages: 5.4 s against 117 s), but it fails on some problems; it is opt-in.

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
- No expected-value constraints yet; the value-function recursion (LFDR) and new liftings are
  planned for an `Experimental` submodule.
