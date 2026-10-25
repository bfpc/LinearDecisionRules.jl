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

## Re-solving with parameters

In model predictive control, the same model is solved many times, with a new initial state or a new
right-hand side each time. Declare such data as a JuMP parameter: when nothing else changes, the
next `optimize!` updates the parameter in the inner problems it kept, with `set_parameter_value`,
and the inner solver starts from its previous solution instead of a new model being built.

```@example resolve
using JuMP, HiGHS, Distributions, LinearDecisionRules

model = LDRModel(HiGHS.Optimizer)
set_silent(model)
@variable(model, stock in Parameter(0.0))
@variable(model, d[t = 1:3] in Uncertainty(Uniform(60, 100)), Stage(t + 1))
@variable(model, 0 <= x[t = 1:3] <= 150, Stage(t))
@variable(model, I[t = 1:3], Stage(t + 1))
@variable(model, y[t = 1:3], Stage(t + 1))
@constraint(model, I[1] == stock + x[1] - d[1])
@constraint(model, [t in 2:3], I[t] == I[t-1] + x[t] - d[t])
@constraint(model, [t in 1:3], y[t] >= 0.5 * I[t])
@constraint(model, [t in 1:3], y[t] >= -3 * I[t])
@objective(model, Min, sum(x[t] + y[t] for t in 1:3))
optimize!(model)
map([0.0, 30.0, -20.0]) do value
    set_parameter_value(stock, value)
    optimize!(model)
    return value => (objective_value(model), decision_rule(x[1]).constant)
end
```

The inner problems are kept only while the parameters are all that changes. A parameter is kept
symbolic where it appears alone, in a constraint or in the objective (see
[`ParameterTerms`](@ref)); multiplied by a variable, it is replaced by its value, so a change of its
value builds the inner problems again, as does any other change of the model, of its laws, or of
the attributes of the optimizer or of the formulations. Each `optimize!` compares the whole
canonical form of the model with the one of the previous solve, so the results never depend on
whether the inner problems were kept. Distributions are treated as immutable values: to change a
law, set a new one rather than modifying it in place.

## Correlated demands: an ellipsoidal support

[`ConfidenceMvNormal`](@ref) is a normal law conditioned on its confidence ellipsoid. Two products
share a stock bought before their correlated demands are known:

```@example conic
using JuMP, HiGHS, Clarabel, LinearDecisionRules

function shared_stock(optimizer; support = :auto)
    model = LDRModel(optimizer)
    set_silent(model)
    law = ConfidenceMvNormal([100.0, 80.0], [100.0 20.0; 20.0 64.0], 0.95; support)
    @variable(model, d[1:2] in Uncertainty(law))
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell[1:2] >= 0)
    @constraint(model, sum(sell) <= buy)
    @constraint(model, sell .<= d)
    @objective(model, Max, -10buy + 15sell[1] + 12sell[2])
    optimize!(model)
    return objective_value(model), objective_bound(model)
end
nothing # hide
```

The formulations see the ellipsoid through [`support_constraints`](@ref): with
`support = :ellipsoid`, as a second-order cone, so the inner problems are second-order cone
programs; with `support = :boxes`, as its box along the principal axes, a polyhedron around it,
so they stay linear programs with a feasible primal rule and a valid, looser dual bound. The
default, `:auto`, takes the ellipsoid when the inner optimizer supports second-order cones and no
decision is integer. HiGHS solves linear programs only, so it gets the boxes:

```@example conic
shared_stock(HiGHS.Optimizer)
```

and Clarabel, a conic solver, the ellipsoid, with a better rule and a tighter bound:

```@example conic
shared_stock(Clarabel.Optimizer)
```

A constraint in `SecondOrderCone` over the uncertainties, tagged [`ValidInequality`](@ref) or
[`Truncate`](@ref), describes the support in the same way, such as the disc
`@constraint(model, [20; d .- 100] in SecondOrderCone(), Truncate())`.
