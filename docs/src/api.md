```@meta
CurrentModule = LinearDecisionRules
```

# [API Reference](@id api_reference)

This page documents the public API of LinearDecisionRules.jl.

## Model

```@docs
LDRModel
```

## Variable Attributes

```@docs
FirstStage
Uncertainty
BreakPoints
```

## Model Attributes

```@docs
SolvePrimal
SolveDual
```

## Solution Queries

```@docs
get_decision
```

## Distributions

Distributions are used to model uncertainty in the `Uncertainty` variable attribute. Many distributions from the Distributions.jl package are supported, as well as some custom ones:

```@docs
MvDiscreteNonParametric
```

## JuMP Extensions

LinearDecisionRules extends standard JuMP functions:

| Function | Description |
|----------|-------------|
| `optimize!(model)` | Solve the LDR model |
| `termination_status(model)` | Get solver termination status |
| `primal_status(model)` | Get primal solution status |
| `solution_summary(model)` | Print solution summary |
| `set_silent(model)` | Silence solver output |
| `unset_silent(model)` | Enable solver output |
| `set_optimizer(model, optimizer)` | Set/change the optimizer |

All these functions accept an optional `dual=false` keyword to query the dual problem.
