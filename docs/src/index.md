```@meta
CurrentModule = LinearDecisionRules
```

# LinearDecisionRules.jl

Linear decision rules for multistage stochastic programs written in [JuMP](https://jump.dev).

You write a plain JuMP model: the decisions, each with its stage, the uncertain data with their
probability laws, the constraints and the objective. At `optimize!`, the package reformulates the
model into the primal and dual linear decision rule problems of Kuhn, Wiesemann and Georghiou
(2011), and solves them with the solver of your choice.

* The **primal** rule is a policy: every decision is an affine function of the uncertainties
  observed before it, feasible for every outcome, and its expected value is the objective value;
* The **dual** rule gives a bound on the best value that any policy can reach.

The gap between the two bounds measures how much is lost by restricting the policy to linear rules.
Piecewise linear rules, with breakpoints on the uncertainties, close it:
see the [piecewise linear](@ref piecewise_linear_tutorial) tutorial.

## Installation

```julia
import Pkg
Pkg.add("LinearDecisionRules")
Pkg.add(["HiGHS", "Distributions"])  # a solver, and the laws of the uncertainties
```

!!! note
    These pages document version 0.4, which is not released yet. Until it is, install it from
    the `rewrite` branch of the repository:
    `Pkg.add(url = "https://github.com/bfpc/LinearDecisionRules.jl", rev = "rewrite")`.

## A first example

A newsvendor buys newspapers at 10 before the demand, uniform on `[80, 120]`, is known; she then
sells them at 15, and returns the rest at 8.

```@example home
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
objective_value(model), objective_bound(model)
```

The primal rule earns 460 on average, and no policy can earn more than 1460/3 ≈ 486.67. The rules
themselves are read with [`decision_rule`](@ref): buy 120, and sell the demand.

```@example home
decision_rule(buy), decision_rule(sell)
```

## Problem class

Multistage problems with linear constraints and a linear or quadratic objective.
The uncertainties can appear in right-hand sides of constraints and costs,
and their laws must have bounded support.
Currently, we accept laws from [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl),
discrete multivariate laws [`MvDiscreteNonParametric`](@ref),
or a normal law on its confidence ellipsoid [`ConfidenceMvNormal`](@ref).
More complex laws can be obtained by imposing constraints on the support of a law,
which amounts to conditioning it; see the tutorial on [advanced distributions](@ref advanced_distributions_tutorial).

Integer decisions can appear in the first stage,
and JuMP parameters can be used to solve the same model again with new data.

[What can be modelled](@ref) in Getting started has the list.

## Reading path

1. [Getting started](getting_started.md) walks through the example above: the results, the rules,
   simulation, re-solving with parameters, and an ellipsoidal support.
2. The tutorials add one feature at a time, in two groups: the rules (piecewise linear, sampled and
   multistage), then the modelling of the uncertainty (laws, constraints on the support, the
   confidence ellipsoid of a normal law).
3. [Performance](@ref performance) explains the settings that change the time, not the values: the
   rescaling of the uncertainties, the rows of the pieces, and the inner solver and its modes.
4. The API pages, from [Modelling](@ref), document every function.
5. The [Developer documentation](@ref developer_overview) explains how the package works inside,
   for those who change it, and the mathematics is in [the notes](assets/ldr-math.pdf) (PDF).

## References

* D. Kuhn, W. Wiesemann and A. Georghiou. Primal and dual linear decision rules in stochastic and
  robust optimization. *Mathematical Programming* 130, 177–209 (2011).
* A. Georghiou, W. Wiesemann and D. Kuhn. Generalized decision rule approximations for stochastic
  programming via liftings. *Mathematical Programming* 152, 301–338 (2015).
