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
