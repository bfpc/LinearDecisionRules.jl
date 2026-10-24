```@meta
CurrentModule = LinearDecisionRules
```

# Getting started

The user writes a plain JuMP model with [`LDRModel`](@ref). Decisions are tagged with a stage
([`Stage`](@ref), or [`FirstStage`](@ref)), and random variables are declared with
[`Uncertainty`](@ref).

## The newsvendor

The newsvendor buys `buy` newspapers at 10 before the demand `d`, uniform on `[80, 120]`, is
known. Afterwards she sells `sell ≤ d` at 15 and returns the rest at 8.

```@example newsvendor
using JuMP, HiGHS, Distributions, LinearDecisionRules

model = LDRModel(HiGHS.Optimizer)
set_silent(model)
@variable(model, buy >= 0, FirstStage)
@variable(model, d in Uncertainty(Uniform(80, 120)))
@variable(model, sell >= 0)
@variable(model, ret >= 0)
@constraint(model, sell + ret <= buy)
@constraint(model, sell <= d)
@objective(model, Max, -10buy + 8ret + 15sell)
optimize!(model)
nothing # hide
```

## Results

The primal linear decision rule is a feasible policy for every demand in the support, so its value
is the objective value of the model. The dual rule gives a bound on the optimal value.

```@example newsvendor
objective_value(model)
```

```@example newsvendor
objective_bound(model)
```

The policy is read with [`decision_rule`](@ref); `value` is the expected value of a decision, and
[`expected_value`](@ref) is exact for affine and quadratic expressions.

```@example newsvendor
decision_rule(buy)
```

```@example newsvendor
decision_rule(sell)
```

```@example newsvendor
value(sell)
```

```@example newsvendor
expected_value(d * sell)
```
