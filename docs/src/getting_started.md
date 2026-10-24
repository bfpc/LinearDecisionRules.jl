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

## Decision rules

For post-processing, the coefficients of the rules are available directly. `rule_coefficient(x, η)`
is the coefficient of the uncertainty `η` in the rule of the decision `x`, and
`rule_coefficient(x)` is its constant. For an uncertainty with [`BreakPoints`](@ref), it is the
vector of the coefficients on its pieces.

```@example newsvendor
rule_coefficient(buy), rule_coefficient(sell, d)
```

[`rule_matrix`](@ref) is the whole matrix `X`, with the decisions as rows and the constant and the
uncertainties as columns; its entries are zero where a decision does not observe a column. The
columns are in lifted coordinates when there are breakpoints.

```@example newsvendor
rule_matrix(model)
```

## Simulation

[`evaluate`](@ref) applies the policy to one scenario: a `Dict`, or pairs, from each uncertainty to
a value. The result has the value of every decision, the objective at the scenario, and the
constraints that it violates. A scenario outside the support is an error, since the rule is
only guaranteed on the support (`outside_support = :warn` or `:ignore` evaluate it anyway).

```@example newsvendor
r = evaluate(model, d => 90.0)
```

```@example newsvendor
value(r, sell), value(r, ret), objective_value(r)
```

[`simulate`](@ref) applies it to many scenarios, drawn from the laws of the uncertainties or given
(for example, simulated by another method). The sample mean of the objective of the primal rule
estimates the objective value of the model, and the rule has no violation on the support.

```@example newsvendor
using Random
s = simulate(model, 10_000; rng = Random.Xoshiro(1))
```

```@example newsvendor
(objective_value(model), s.objective_mean, s.objective_stderr)
```

```@example newsvendor
simulate(model, [Dict(d => 90.0), Dict(d => 110.0)])
```

The rule of the [`SampledLDR`](@ref) formulation is feasible on the scenarios that it was solved
on, and only there: `simulate` reports its violations out of sample, how often and by how much.

```@example newsvendor
sampled = SampledLDR(5; seed = 1)
set_attribute(model, Formulations(), [PrimalLDR(), sampled])
optimize!(model)
simulate(model, 10_000; formulation = sampled, rng = Random.Xoshiro(1))
```

```@example newsvendor
s = simulate(model, 10_000; formulation = sampled, rng = Random.Xoshiro(1))
s.violation_counts
```
