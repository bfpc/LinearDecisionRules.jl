# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# # [Working with distributions](@id distributions_tutorial)

# ```@meta
# CurrentModule = LinearDecisionRules
# ```

# This tutorial shows how to model uncertainty with the laws of
# [Distributions.jl](https://github.com/JuliaStats/Distributions.jl), univariate and
# multivariate. An uncertainty is declared with [`Uncertainty`](@ref), a set that holds the law:
#
# ```julia
# @variable(model, d in Uncertainty(Uniform(80, 120)))
# @variable(model, e[1:2] in Uncertainty(product_distribution([Uniform(0, 1), Uniform(0, 2)])))
# ```
#
# Separate declarations are independent; the coordinates of a multivariate law have its joint
# law.

# ## Setup

using JuMP
using HiGHS
using Distributions
using LinearDecisionRules

# ## A simple inventory problem

# Throughout this tutorial, we use variations of an inventory problem:
# * **Buy cost**: 10 per unit (first-stage decision)
# * **Sell price**: 15 per unit
# * **Salvage value**: 8 per unsold unit
# * **Demand**: uncertain, with a law that varies

# The helper function solves it with a given law of the demand, and returns the model. We ask for
# the primal rule (a policy, whose value is the `objective_value`) and for the dual rule, which
# gives the bound `objective_bound`.

function solve_inventory(demand_law)
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, salvage >= 0)
    @variable(model, demand in Uncertainty(demand_law))
    @constraint(model, sell + salvage <= buy)
    @constraint(model, sell <= demand)
    @objective(model, Max, -10buy + 15sell + 8salvage)
    set_attribute(model, Formulations(), [PrimalLDR(), DualLDR()])
    optimize!(model)
    return model
end

# We print one row for each law: its mean and standard deviation, the first-stage decision of the
# primal rule (`rule_coefficient` of a first-stage decision is its constant), and the primal and
# dual values.

function print_row(name, demand_law)
    model = solve_inventory(demand_law)
    buy = rule_coefficient(model[:buy])
    values =
        (mean(demand_law), std(demand_law), buy, objective_value(model), objective_bound(model))
    println(rpad(name, 32), join([rpad(round(v; digits = 2), 9) for v in values]))
    return
end

function print_header()
    println(rpad("law", 32), "mean     std      buy      primal   dual")
    return
end

# ## Part 1: univariate distributions

# ### Uniform distribution

# The simplest case: demand is uniformly distributed between a minimum and a maximum.

print_header()
print_row("Uniform(80, 120)", Uniform(80.0, 120.0))

# ### Truncated Normal distribution

# When demand follows a bell curve but is bounded (for example, it cannot be negative):

print_row("truncated Normal(100, 15)", truncated(Normal(100.0, 15.0), 60.0, 140.0))

# !!! note
#     All the laws in LinearDecisionRules must have a **bounded support**, since the support of the
#     uncertainties is what the robust formulations are built on. The check is made when the
#     variable is declared, with a message that tells what to do:

try
    solve_inventory(Normal(100.0, 15.0))
catch err
    ## The message has the location of the declaration, then the explanation.
    println(last(split(
        sprint(showerror, err),
        "

",
    )))
end

# Use `truncated(law, lower, upper)` to bound an unbounded law, as above.

# ### Triangular distribution

# The triangular distribution is useful when you have estimates for the minimum, the maximum and
# the mode (the most likely value):

print_row("Triangular(70, 130, mode 110)", TriangularDist(70.0, 130.0, 110.0))

# ### Discrete non-parametric distribution

# When the demand takes specific values with known probabilities (for example, from historical
# scenarios):

law = DiscreteNonParametric([80.0, 90.0, 100.0, 110.0, 120.0], [0.1, 0.2, 0.4, 0.2, 0.1])
print_row("discrete, 5 scenarios", law)

# ## Part 2: comparing distributions

# All the four laws below have the same mean (100), but different supports and shapes.

print_header()
for (name, law) in [
    ("Uniform(80, 120)", Uniform(80.0, 120.0)),
    ("Triangular(70, 130, mode 100)", TriangularDist(70.0, 130.0, 100.0)),
    ("truncated Normal(100, 12)", truncated(Normal(100.0, 12.0), 70.0, 130.0)),
    ("discrete, {80, 100, 120}", DiscreteNonParametric([80.0, 100.0, 120.0], [0.25, 0.5, 0.25])),
]
    print_row(name, law)
end

# The linear rule sells exactly the demand, and for that it buys the largest possible one: `buy`
# is the maximum of the support, 120 or 130 here. Its value is then
# ``-10\,\max + 15\,\text{mean} + 8\,(\max - \text{mean}) = 7\,\text{mean} - 2\,\max``, so that
# the primal value depends on the law only through its mean and the maximum of its support: 460
# for the maximum 120, and 440 for the maximum 130. The dual bound depends on the moments of the
# law, and it differs for each of them. A piecewise linear rule (see the
# [piecewise linear](@ref piecewise_linear_tutorial) tutorial) uses more of the law.

# ## Part 3: multivariate distributions

# When the model has several uncertainties, a multivariate law captures their joint behavior.

# ### Product of independent distributions

# The simplest multivariate case: independent uncertainties with `product_distribution`.
# Consider a two-product inventory problem, with one law for the two demands:

function solve_two_products(demand_law)
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, buy[1:2] >= 0, FirstStage)
    @variable(model, sell[1:2] >= 0)
    @variable(model, salvage[1:2] >= 0)
    @variable(model, demand[1:2] in Uncertainty(demand_law))
    @constraint(model, [i in 1:2], sell[i] + salvage[i] <= buy[i])
    @constraint(model, [i in 1:2], sell[i] <= demand[i])
    @objective(model, Max, sum(-10buy[i] + 15sell[i] + 8salvage[i] for i in 1:2))
    set_attribute(model, Formulations(), [PrimalLDR(), DualLDR()])
    optimize!(model)
    return model
end

function print_row_two(name, demand_law)
    model = solve_two_products(demand_law)
    buy = [rule_coefficient(model[:buy][i]) for i in 1:2]
    println(
        rpad(name, 32),
        rpad(round.(buy; digits = 2), 20),
        rpad(round(objective_value(model); digits = 2), 9),
        round(objective_bound(model); digits = 2),
    )
    return
end

println(rpad("law", 32), rpad("buy", 20), "primal   dual")
print_row_two(
    "Uniform x Uniform",
    product_distribution([Uniform(80.0, 120.0), Uniform(60.0, 100.0)]),
)

# Different laws can be mixed:

print_row_two(
    "Uniform x Triangular",
    product_distribution([Uniform(80.0, 120.0), TriangularDist(60.0, 100.0, 85.0)]),
)

# and so can truncated normals, which are independent (uncorrelated) here:

print_row_two(
    "truncated Normals",
    product_distribution([
        truncated(Normal(100.0, 10.0), 70.0, 130.0),
        truncated(Normal(80.0, 15.0), 40.0, 120.0),
    ]),
)

# ### Multivariate discrete non-parametric distribution

# For scenario-based uncertainty with joint realizations, use [`MvDiscreteNonParametric`](@ref).
# Each scenario specifies the values of **all** the uncertainties at once, so that the demands of
# the products can be correlated:

scenarios = [
    [80.0, 60.0],   # low demand for both products
    [100.0, 80.0],  # medium
    [120.0, 100.0], # high for both
    [80.0, 100.0],  # low product 1, high product 2
    [120.0, 60.0],  # high product 1, low product 2
]
probabilities = [0.2, 0.3, 0.2, 0.15, 0.15]
print_row_two("MvDiscreteNonParametric", MvDiscreteNonParametric(scenarios, probabilities))

# !!! tip
#     Use `MvDiscreteNonParametric` when:
#     * you have historical data with joint observations;
#     * the demands are correlated (for example, both are high or both are low together);
#     * you want an exact scenario-based optimization.
#     A discrete law with at most as many atoms as dimensions makes the matrix of second moments
#     singular: the dual bound is then reported as not certified, with a warning.

# ## Key takeaways

# | Law | Use case |
# |-----|----------|
# | `Uniform(a, b)` | equal likelihood across a range |
# | `truncated(Normal(μ, σ), a, b)` | bell-shaped, with a known mean and standard deviation |
# | `TriangularDist(a, b, c)` | known minimum, maximum and most likely value |
# | `DiscreteNonParametric(values, probabilities)` | specific values with probabilities |
# | `product_distribution([...])` | independent multivariate uncertainties |
# | `MvDiscreteNonParametric(scenarios, probabilities)` | correlated scenarios (joint realizations) |

# ## What's next?

# * Learn about [advanced distribution modeling](@ref advanced_distributions_tutorial), with
#   constraints that cut the support or condition the law.
# * Explore [piecewise linear decision rules](@ref piecewise_linear_tutorial) for better
#   approximations.
