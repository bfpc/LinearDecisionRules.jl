# Results and rules

```@meta
CurrentModule = LinearDecisionRules
```

The formulations, their results, and the decision rules read from a solved model.

## Formulations

```@docs
AbstractFormulation
PrimalLDR
DualLDR
SampledLDR
```

## Results of one formulation

```@docs
FormulationAttribute
Certified
is_certified
JuMP.termination_status(::JuMP.GenericModel, ::AbstractFormulation)
JuMP.primal_status(::JuMP.GenericModel, ::AbstractFormulation)
JuMP.objective_value(::JuMP.GenericModel, ::AbstractFormulation)
JuMP.objective_bound(::JuMP.GenericModel, ::AbstractFormulation)
JuMP.raw_status(::JuMP.GenericModel, ::AbstractFormulation)
JuMP.solve_time(::JuMP.GenericModel, ::AbstractFormulation)
inner_model
```

## Where the time goes

`solve_time(model)` is the time of the whole `optimize!`. To see where it went, for instance
whether the moments of a law with breakpoints take longer than the inner solver, ask for its
breakdown: reading the model, the moments, the checks, and the build, solve and results of the
inner problems.

```julia
optimize!(model)
time_breakdown(model)               # all the formulations
time_breakdown(model, PrimalLDR())  # (build = …, solve = …, results = …)
```

```@docs
TimeBreakdown
time_breakdown
```

## Decision rules

```@docs
decision_rule
DecisionRule
LiftedTerm
expected_value
rule_coefficient
rule_matrix
RuleMatrix
```
