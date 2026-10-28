# [Public and internal API](@id developer_api)

This page explains what users can rely on and what may still change. The package exports 45
names (`src/LinearDecisionRules.jl`), grouped by the page of the manual that documents them, and
`Optimizer` is public without being exported. The canonical layer, the typed description of the
problem that the formulations work on, is documented on the Internals page but not exported: its
design may still change, and it will become stable with a planned `Experimental` submodule.

The API pages list their names in explicit `@docs` blocks, and the documentation build fails if a
docstring of the package is on no page (`checkdocs = :all`). So when you add a docstring, place it
on a page, among the public names or on the Internals page.

## Exported names, by page

| Page | Group | Names |
|---|---|---|
| Modelling | Model | `LDRModel` |
| | Stages | `Stage`, `FirstStage`, `stage`, `set_stage`, `unset_stage` |
| | Uncertainties and their support | `Uncertainty`, `ValidInequality`, `Truncate`, `BreakPoints` |
| | Settings | `Formulations`, `InnerOptimizer`, `RejectionSamplingSeed`, `RejectionSamplingTimeLimit`, `RejectionSamplingMaxSamples` |
| Results and rules | Formulations | `AbstractFormulation`, `PrimalLDR`, `DualLDR`, `SampledLDR` |
| | Results of one formulation | `FormulationAttribute`, `Certified`, `is_certified`, `inner_model`, and the methods of JuMP functions with a formulation argument, such as `objective_value(model, f)` |
| | Decision rules | `decision_rule`, `DecisionRule`, `LiftedTerm`, `expected_value`, `rule_coefficient`, `rule_matrix`, `RuleMatrix` |
| Simulation | | `evaluate`, `Evaluation`, `simulate`, `Simulation` |
| Distributions and liftings | Laws | `ConditionedDistribution`, `ConfidenceMvNormal`, `MvDiscreteNonParametric`, `LiftedDistribution` |
| | The support of a law | `support_constraints`, the hook for a law whose support is not its box |
| | Liftings | `AbstractLifting`, `AxisPiecewise`, `lift`, `retraction`, `LiftingMap`, `LiftingBlock` |

Two of them give access to what is otherwise internal:

- `inner_model(model, f)` returns the inner `JuMP.Model` of a formulation: the `model` field of its
  `FormulationResult`. It is to be read, not modified, since a re-solve reuses it.
- `RuleMatrix` has a field `lifting`, the `LiftingMap` of the model, the identity when nothing is
  lifted, so that the rules are `R.matrix * lift(R.lifting, ξ)`.

## Public, not exported

`LinearDecisionRules.Optimizer` is declared `public` (Julia 1.11 and later), for
`Model(() -> LinearDecisionRules.Optimizer(HiGHS.Optimizer))`. Its docstring, and the one of
`MOI.optimize!(::Optimizer, src)`, are on the Modelling page.

## Internal, documented on the Internals page

| Group | Names |
|---|---|
| Reading a model | `model_info`, `ModelInfo`, `UncertaintyDeclaration`, `canonicalize`; the MOI form of the tags, `UnivariateUncertainty`, `MultivariateUncertainty`, `AbstractXiRole`, `XiRole`, `ConstraintStage`; the hook `_is_metadata_constraint` |
| The canonical form | `CanonicalForm`, `ConicRow`, `UncertaintyBlock`, `ParameterTerms`, `lift(form, liftings)` |
| The information structure | `InformationStructure`, `observes`, `visible_columns` |
| The support | `SupportBlock`, `support_blocks` |
| The moments | `CovarianceBlock`, `Moments`, `moments`, `second_moment` |
| Solving one formulation | `FormulationCapabilities`, `capabilities`, `FormulationResult`, `solve_formulation` |

Write them qualified, `LinearDecisionRules.canonicalize`, or import them by name; the tests do the
latter. [Call flows](@ref developer_flows) shows the low-level path that they make up.

## The channel between `jump/` and `moi/`

The JuMP layer reads the state of the last `optimize!` only through `MOI.get` on attributes. Some
are MOI's own (`ObjectiveValue`, `VariablePrimal`, …), some are the public `FormulationAttribute`
and `Certified`, and some are internal ones defined in `moi/attributes.jl`:

| Internal attribute | Answered in | Read by |
|---|---|---|
| `RuleCoefficients(f)` (variable) | `moi/results.jl` | `decision_rule`, `expected_value`, `rule_coefficient`, `rule_matrix` |
| `UncertaintyVariables()`, `DecisionVariables()` | `moi/results.jl` | the same |
| `UncertaintyLifting()` | `moi/results.jl` | `decision_rule`, `rule_coefficient`, `rule_matrix` |
| `UncertaintyMoments()` | `moi/results.jl` | `expected_value` |
| `InnerModel()`, through `FormulationAttribute(f, InnerModel())` | `moi/results.jl` | `inner_model` |
| `SimulationData(f)` | `jump/simulate.jl`, the only `::Optimizer` method outside `moi/` | `evaluate`, `simulate` |

Some tests still read `unsafe_backend(model).results`, an undocumented field, for the solve time or
the inner model of the only formulation. Prefer `solve_time(model, f)` and `inner_model(model, f)`,
as the benchmarks do.

## Extensions of JuMP and MOI

Each method that the package adds to a function of JuMP, MOI, Distributions or Base involves one of
the package's own types, as a tag, a wrapper or a set. There is no type piracy, and Aqua checks
this. The extension points used:

- **JuMP:** `build_variable`, `add_variable`, `build_constraint`, `add_constraint`, and the
  result functions with a formulation argument (`termination_status`, `primal_status`,
  `objective_value`, `objective_bound`, `raw_status`, `solve_time`), plus `value` on an
  `Evaluation`.
- **MOI:** the optimizer interface (`supports`, `supports_constraint`, `get`, `set`, `empty!`,
  `is_empty`, `optimize!(dest, src)`, `copy_to`, which raises an error), `dimension` of the
  multivariate set, and `head_name` for MOF files.
- **Distributions:** the interface of each new law (`length`, `eltype`, `_rand!`, `mean`, `cov`,
  `insupport`, …).
