# [Call flows](@id developer_flows)

This page follows a model through the package: what runs, in order, from your call down to the
package's functions. File names are relative to `src/`, and the line of every definition is in
[Structure of `src/`](structure_current.md).

## Declaration time

Nothing of the core runs while the model is written. Our methods of JuMP's extension points
(`build_variable`, `add_variable`, `build_constraint`, `add_constraint`) turn each tag into MOI data,
and JuMP stores it in its cache. Before storing a constraint or an attribute, JuMP asks whether the
`Optimizer` supports it (`MOI.supports_constraint` and `MOI.supports` in `moi/optimizer.jl`), so an
unsupported constraint type is an error when it is added.

| User code | Our methods | Stored in JuMP's cache |
|---|---|---|
| `@variable(model, x >= 0, Stage(t))`, or `FirstStage` | `build_variable(error_fn, info, ::Stage)` returns a `_StagedVariable`; `add_variable(model, ::_StagedVariable)` adds the variable and then its stage (`jump/variables.jl`). A second `Stage` is an error (`jump/checks.jl`). | the variable, and a `VariableIndex`-in-`Stage` constraint |
| `@variable(model, d in Uncertainty(law))` | `Uncertainty(law)` returns a `UnivariateUncertainty` or a `MultivariateUncertainty` (`moi/sets.jl`). `build_variable(error_fn, ::ScalarVariable, ::UnivariateUncertainty)` rejects bounds and integrality and an unbounded support (`_check_uncertainty_declaration`, `jump/checks.jl`). | a variable constrained on creation, in `UnivariateUncertainty` (scalar) or as a `VectorOfVariables` in `MultivariateUncertainty` |
| `@variable(model, d[1:n] in Uncertainty(mvlaw))` over a container | `build_variable(error_fn, ::AbstractArray, ::MultivariateUncertainty)` wraps the components in an `_ArrayOfVariables`, and `add_variable` reshapes them (`jump/variables.jl`). | one `VectorOfVariables`-in-`MultivariateUncertainty` constraint |
| `@variable(model, d in Uncertainty(law), Stage(t))` | `_StagedVariable` wraps the uncertainty builder: `build_variable(error_fn, ::_StagedVariable, set)` (`jump/variables.jl`). | both constraints |
| `@constraint(model, a'η <= b, ValidInequality())`, or `Truncate()` | `build_constraint(error_fn, f, set, tag)` checks that the row is affine (`_check_tagged_row`) and returns a `_TaggedConstraint`; `add_constraint` adds the row and sets its role (`jump/constraints.jl`). | the row, and the constraint attribute `XiRole()` set to the tag |
| `@constraint(model, [r; η .- μ] in SecondOrderCone(), Truncate())` | The vector method of `build_constraint`. | the cone row, and `XiRole()` |
| `@constraint(model, …, Stage(t))` | An error from `build_constraint`: constraints have no stage. | nothing |
| `set_stage`, `unset_stage`, `stage` | They add, change or delete the `Stage` constraint of a variable (`jump/variables.jl`). | as above |
| `set_attribute(d, BreakPoints(), points)` | none: JuMP's `MOI.set`, accepted by `MOI.supports` | the variable attribute `BreakPoints()` |
| `@variable(model, p in Parameter(v))`, `set_parameter_value(p, w)` | none: JuMP's own | a `VariableIndex`-in-`MOI.Parameter` constraint |
| `set_attribute(model, Formulations(), […])`, `InnerOptimizer()`, `RejectionSampling*()`, `set_time_limit_sec`, `set_silent`, raw attributes | `MOI.set(::Optimizer, …)` (`moi/optimizer.jl`). `Formulations()` rejects two formulations of one type. | on the `Optimizer` itself, not in the cache |

The uncertainty-only rows, the stages of the decisions, and products of uncertainties and
decisions are checked at `optimize!`, not at declaration, because their meaning depends on the
whole model.

## Solve time: `optimize!`

`JuMP.optimize!(model)` calls the caching optimizer, which calls `MOI.optimize!(dest::Optimizer,
src)` with its cache as `src` (`moi/optimizer.jl`). Everything below runs at each `optimize!`. Only
the caches of the moments and of the inner problems carry over from the last one.

```mermaid
sequenceDiagram
    autonumber
    participant J as JuMP
    participant O as moi/optimizer.jl
    participant C as canonical/
    participant P as probability/
    participant K as moi/inner_cache.jl
    participant F as formulations/
    J->>O: MOI.optimize!(dest, src = JuMP's cache)
    O->>O: MOI.empty!(dest), _inner_factory(dest)
    O->>O: model_info(src) → ModelInfo (moi/model_info.jl)
    O->>O: _check_bounded_support(src, info)
    O->>C: canonicalize(src, info; conic_support = _inner_supports_cones(factory, info))
    C-->>O: CanonicalForm
    O->>P: _moments(form, rejection_sampling, moments_cache; budget)
    P-->>O: Moments
    O->>O: _check_convex_objective(form), _warn_singular_moments(form, mom) (moi/checks.jl)
    O->>F: if RescaleUncertainties(), r = _rescaling(form, mom), then _rescale(form, r) and _rescale(mom, r)
    O->>K: _update!(inner_cache, _Fingerprint(form, mom, factory, silent, raw))
    loop each f in Formulations()
        O->>K: _cached_problem(inner_cache, f)
        O->>F: _solve!(build, f, form, mom, start, TimeLimitSec; inner)
        Note over F: time left? otherwise _time_limit_result(f, form, mom)
        alt not cached
            F->>F: build() = _build_inner(form, f, mom, factory), kept in the inner cache
        end
        F->>F: _solve_inner!(inner, f, form, mom; time_limit = what remains)
        F-->>O: FormulationResult, then _unscale(result, r) if rescaled
    end
    O->>O: keep form, moments, results, decision_rows, uncertainty_columns, parameters
    O-->>J: (identity index map, false)
```

### `canonicalize(src, info; conic_support)` (`canonical/canonicalize.jl`)

1. `_parameter_list(src)`: the parameters and their values.
2. `_check_declarations`: the declarations of `info` against the model (dimensions, duplicates,
   parameters declared as uncertainties, stages ≥ 1).
3. `_layout`: the columns of ξ, the uncertainties in declaration order, stably sorted by stage;
   the decisions, the other variables in model order. The number of stages `T` is the largest stage
   declared, and at least 2. A variable without a stage goes to stage `T`; when `T > 2`,
   `_warn_unmarked` warns about such variables.
4. `_read_constraints`: one `_read_constraints!` per constraint type, skipping the metadata
   constraints (`_is_metadata_constraint`: the `Stage` and `Uncertainty` sets) and the parameters.
   Rows go to the blocks `(A_e, B_e)`, `(A_u, B_u)` and `(A_l, B_l)`. Uncertainty-only rows go to
   `W_valid`, `h_valid` and `conic_rows`, with their role. This step raises the errors about the
   model class: untagged uncertainty-only rows, random recourse, quadratic constraints, integer
   decisions after stage 1.
5. `_read_objective`, then `_canonical_form`: the `CanonicalForm`, with `lifting = nothing` and the
   `ParameterTerms`.
6. `_resolve_representations(form, conic_support)`: each law with `support = :auto`
   (`_has_auto_representation`) gets its representation from `_resolve_representation`:
   second-order cones if `conic_support` holds and no decision is integer, linear rows otherwise
   (`canonical/support.jl`).
7. `_condition(src, form)` (`canonical/conditioning.jl`) conditions the laws by the rows tagged
   `Truncate()`. A univariate law whose linear rows involve it alone is truncated exactly. Laws
   coupled by such rows, or by conic rows, are merged by union–find into a `ConditionedDistribution`.
   When they all have finitely many atoms, they become the exact law of the atoms instead.
8. `_check_valid_rows_at_mean` and `_check_observable_equalities`.
9. `_breakpoint_liftings(src, form, info.breakpoints)`: an `AxisPiecewise` per uncertainty with
   breakpoints. If there are any, `lift(form, liftings)` (`lifting/lifted_form.jl`) returns the lifted
   form, with its `LiftingMap`.

### `_moments(form, sampling, cache; budget)` (`probability/moments.jl`)

For each law (`UncertaintyBlock`) of the form:
- a `ConditionedDistribution` is estimated by rejection sampling (`_estimate`, in
  `probability/rejection_sampling.jl`). The estimate is cached in `moments_cache` under (law,
  seed, number of draws). The laws still to sample share the time budget: half of what remains of
  `TimeLimitSec()`, unless `RejectionSamplingTimeLimit()` is set.
- any other law gives `_law_mean` and `_law_covariance` (`probability/laws.jl`). A truncated
  univariate law is integrated with QuadGK.

The result is `Moments(mean, blocks)`, one `CovarianceBlock` per law.

### The rescaling (`formulations/rescaling.jl`)

With `RescaleUncertainties()` (the default), `_rescaling(form, mom)` takes the center `c` and the
half-width `s` of the box of each column of the form, the lifted coordinates when there are
breakpoints (or the mean and the standard deviation of a law without a bounded box). `_rescale`
returns the form and the moments in `ζ = (η - c) ./ s`: the laws become `_RescaledLaw`s, and
`_rescale_rows` divides each rescaled row of the support by its largest coefficient. From here on,
the inner cache and the formulations see only the rescaled form and moments, which `optimize!`
does not keep; `_unscale` maps each rule back to ξ. The optimizer keeps the form and the moments in
the uncertainties of the model, for the results. The mathematics is the subsection "Rescaling the
uncertainties" of the [math notes](../assets/ldr-math.pdf).

### `_build_inner` and `_build!` (`formulations/`)

`_build_inner(form, f, mom, factory)` (`inner.jl`):
1. creates the inner `JuMP.Model`, then sets silence, the raw attributes and the formulation's own
   `attributes`.
2. `_with_parameter_decisions(form)` turns each parameter in use into a decision of stage 0. It
   observes only the constant, and its column of `A` is `-Bp`.
3. calls `_build!(model, f, extended, mom)`, then records which inner variable holds each parameter.

The three methods of `_build!` share one skeleton:

| Step | `PrimalLDR` (`primal.jl`) | `DualLDR` (`dual.jl`) | `SampledLDR` (`sampled.jl`) |
|---|---|---|---|
| before | | `_warn_stage_dependence(form)` | |
| information | `InformationStructure(form)` | same | same |
| rule variables | `_rule_variables!` (the stage mask) | same | same |
| equalities | `_add_equalities!` (coefficient matching) | same | same |
| Ξ | `support_blocks(form; f.lifted_box)`; the box bounds for the closed form | `_dual_cones(support_blocks(form; f.lifted_box), mom)` | `_sample_outcomes(form, f)` |
| each inequality | deterministic: `_add_deterministic!`; otherwise `_add_robust_counterpart!` (box closed form, or multipliers per block) | deterministic, or `_add_dual_cone!` | deterministic, or one row per outcome |
| objective | `_set_objective!(model, form, mom, X)` | same | with `_empirical_moments(outcomes)` |

### `_solve!`, `_solve_inner!` and the result (`formulations/inner.jl`, `result.jl`)

`_solve!` returns `_time_limit_result` when no time is left, builds the inner problem unless the
inner cache has it, and gives `_solve_inner!` the time that remains. `_solve_inner!` fixes the parameter variables whose value changed and sets or clears the time
limit. Then it runs `JuMP.optimize!` on the inner model and calls `_formulation_result`, which
collects:
- the statuses (`NO_SOLUTION` after a stop without a solution);
- the rule matrix `X` and the objective value;
- `_objective_bound`: the inner solver's bound for a MIP, the dual objective for an optimal LP, the
  objective value for another optimal status, the dual objective after a limit if the dual point is
  feasible, and `NaN` otherwise;
- `_certification`: whether the dual bound holds, and the reason if not.

## Re-solve with new parameter values

The second `optimize!` runs the same steps. `canonicalize` and `_moments` run again: the form is
cheap to read, and the conditioned laws come from `moments_cache`. The difference is in
`_update!` (`moi/inner_cache.jl`), which compares the new `_Fingerprint` with the last one using
`_same_problem`:
- `_same_form` compares every field of the two forms except the values of the parameters. In
  `B_*`, it skips the constant column (`_same_except_constant`). It ignores `f`, and the values in
  `ParameterTerms`.
- the moments, the inner factory, silence and the raw attributes must be equal (`_same_value`,
  structural).

If they are the same, the inner problems are kept: `_cached_problem` returns each formulation's
problem, and `_solve_inner!` fixes the parameter variables to the new values and solves the same
inner model again. The inner solver can then reuse its last solution, if it supports that. Any other
change rebuilds the problems. So does a parameter multiplied by a decision,
since the form sees it as a different coefficient.

## Reading results

All results come from the `FormulationResult`s that `optimize!` stored. Reading them solves nothing.

| Call | Path |
|---|---|
| `termination_status(model)` | `MOI.TerminationStatus` → `_termination_status` (`moi/results.jl`). It is `INFEASIBLE` if a bound formulation is infeasible. Otherwise it is the status of the principal formulation: the first with a rule, `PrimalLDR` first. |
| `objective_value(model)`, `value(x)`, `primal_status(model; result)` | The *result slots* `_RESULT_SLOTS = (PrimalLDR, SampledLDR)`: result 1 is the primal rule, result 2 the sampled rule. `value(x)` is `E[x(ξ)]`, the rule's coefficients (`_rule_coefficients`) times the mean. |
| `objective_bound(model)` | The first bound formulation with a bound, that is, `DualLDR`. |
| `termination_status(model, f)`, `objective_value(model, f)`, `objective_bound(model, f)`, `raw_status(model, f)`, `solve_time(model, f)` | `jump/results.jl` → `MOI.get(model, FormulationAttribute(f, attr))` → `moi/results.jl` |
| `is_certified(model, f = DualLDR())` | `FormulationAttribute(f, Certified())` |
| `inner_model(model, f)` | `FormulationAttribute(f, InnerModel())`, the `model` field of the `FormulationResult` of `f` |
| `decision_rule(x; formulation)` | The internal attributes `RuleCoefficients(f)`, `UncertaintyVariables()` and `UncertaintyLifting()` (`jump/rules.jl`), which give a `DecisionRule` with `LiftedTerm`s for lifted uncertainties. |
| `expected_value(expr; formulation)` | `RuleCoefficients(f)` and `UncertaintyMoments()`, then `_product_moment` for products. |
| `rule_coefficient(x, η)`, `rule_matrix(model)` | `RuleCoefficients(f)`, `DecisionVariables()`, `UncertaintyLifting()`. The lifting map is the identity when nothing is lifted (`_lifting_map`), and `rule_matrix` keeps it in the field `lifting` of the `RuleMatrix`. |
| `evaluate(model, scenario)`, `simulate(model, n or scenarios)` | `_simulator` → `MOI.get(o, SimulationData(f))` (in `jump/simulate.jl`) → `_Simulator`. Then `_outcome` (the scenario as ξ, lifted), `_assess` (the decisions, objective and violated rows per scenario) and `_support_reasons` (outside Ξ). |

The internal attributes (`moi/attributes.jl`) are the only channel between `jump/` and the state
of the `Optimizer`. The JuMP layer never reads the optimizer's fields: it calls `MOI.get`, through
JuMP's caching optimizer.

## Without the optimizer

The low-level path runs the same steps by hand. `model_info`, `canonicalize`, `moments` and
`solve_formulation` are documented on the Internals page, but not exported:

```julia
import LinearDecisionRules as LDR
info = LDR.model_info(backend(model))    # or a ModelInfo built by hand, for a plain Model()
form = LDR.canonicalize(backend(model), info)
mom = LDR.moments(form)
result = LDR.solve_formulation(form, PrimalLDR(), mom, HiGHS.Optimizer)
```

`solve_formulation` calls `_solve!`, like `optimize!`, so a time limit is handled alike. It does not
rescale the uncertainties, and does not use the inner cache or the result slots. Unlike `optimize!`,
`canonicalize` does not ask the inner optimizer whether it supports second-order cones: its
keyword `conic_support` is `true` unless given.
