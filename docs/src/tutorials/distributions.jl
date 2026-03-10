# # [Working with distributions](@id distributions_tutorial)

# This tutorial demonstrates how to model uncertainty using various probability
# distributions in LinearDecisionRules.jl. We cover both univariate and
# multivariate distributions.

# LinearDecisionRules integrates with Julia's
# [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) package,
# giving you access to a wide variety of probability distributions.

# ## Setup

using JuMP
import LinearDecisionRules
import HiGHS
import Distributions

# ## A simple inventory problem

# Throughout this tutorial, we'll use variations of an inventory problem:
# - **Buy cost**: \$10 per unit (first-stage decision)
# - **Sell price**: \$15 per unit
# - **Salvage value**: \$5 per unsold unit
# - **Demand**: uncertain, varies by distribution

buy_cost = 10
sell_price = 15
salvage_value = 5

# We'll create a helper function to solve the problem with different demand distributions:

function solve_inventory(demand_distribution; name = "")
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell >= 0)
    @variable(ldr, salvage >= 0)
    @variable(ldr, demand in LinearDecisionRules.Uncertainty(
        distribution = demand_distribution,
    ))

    @constraint(ldr, sell + salvage <= buy)
    @constraint(ldr, sell <= demand)

    @objective(ldr, Max, -buy_cost * buy + sell_price * sell + salvage_value * salvage)

    optimize!(ldr)

    println("Distribution: $name")
    println("  Expected demand: ", round(Distributions.mean(demand_distribution), digits=2))
    println("  Demand std dev:  ", round(Distributions.std(demand_distribution), digits=2))
    println("  Optimal buy:     ", round(LinearDecisionRules.get_decision(ldr, buy), digits=2))
    println("  Primal bound:    ", round(objective_value(ldr), digits=2))
    println("  Dual bound:      ", round(objective_value(ldr; dual=true), digits=2))
    println()

    return ldr
end

# ## Part 1: Univariate distributions

# ### Uniform distribution

# The simplest case: demand is uniformly distributed between a minimum and maximum.

uniform_dist = Distributions.Uniform(80.0, 120.0)
solve_inventory(uniform_dist; name = "Uniform(80, 120)");

# ### Truncated Normal distribution

# When demand follows a bell curve but is bounded (e.g., cannot be negative):

truncated_normal_dist = Distributions.truncated(Distributions.Normal(100.0, 15.0), 60.0, 140.0)
solve_inventory(truncated_normal_dist; name = "Truncated Normal(μ=100, σ=15)");

# !!! note
#     All distributions in LinearDecisionRules must have **finite bounds**.
#     Use `Distributions.truncated(dist, lower, upper)` to bound unbounded distributions.

# ### Triangular distribution

# The triangular distribution is useful when you have estimates for minimum,
# mode (most likely), and maximum values:

triangular_dist = Distributions.TriangularDist(70.0, 130.0, 110.0)  # min, max, mode
solve_inventory(triangular_dist; name = "Triangular(70, 130, mode=110)");

# ### Discrete Non-Parametric distribution

# When demand takes specific values with known probabilities (e.g., from
# historical scenarios):

scenarios = [80.0, 90.0, 100.0, 110.0, 120.0]
probabilities = [0.1, 0.2, 0.4, 0.2, 0.1]
discrete_dist = Distributions.DiscreteNonParametric(scenarios, probabilities)
solve_inventory(discrete_dist; name = "Discrete (5 scenarios)");

# ## Part 2: Comparing distributions

# Let's compare how different distribution assumptions affect the optimal decision.
# All four distributions below have the same mean (100) but different shapes:

println("="^60)
println("Comparison of distribution choices")
println("="^60)
println()

distributions = [
    ("Uniform(80, 120)", Distributions.Uniform(80.0, 120.0)),
    ("Triangular(70, 130, 100)", Distributions.TriangularDist(70.0, 130.0, 100.0)),
    ("Truncated Normal", Distributions.truncated(Distributions.Normal(100.0, 12.0), 70.0, 130.0)),
    ("Discrete (symmetric)", Distributions.DiscreteNonParametric([80.0, 100.0, 120.0], [0.25, 0.5, 0.25])),
]

for (name, dist) in distributions
    solve_inventory(dist; name = name)
end

# ## Part 3: Multivariate distributions

# When your model has multiple uncertain parameters, you can use multivariate
# distributions to capture correlations or define joint distributions.

# ### Product of independent distributions

# The simplest multivariate case: independent uncertainties using `product_distribution`.

# Consider a two-product inventory problem:

function solve_two_product_inventory(demand_distribution; name = "")
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, buy[1:2] >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell[1:2] >= 0)
    @variable(ldr, salvage[1:2] >= 0)
    @variable(ldr, demand[1:2] in LinearDecisionRules.Uncertainty(
        distribution = demand_distribution,
    ))

    for i in 1:2
        @constraint(ldr, sell[i] + salvage[i] <= buy[i])
        @constraint(ldr, sell[i] <= demand[i])
    end

    @objective(ldr, Max,
        sum(-buy_cost * buy[i] + sell_price * sell[i] + salvage_value * salvage[i] for i in 1:2)
    )

    optimize!(ldr)

    println("Distribution: $name")
    println("  Optimal buy:  ", [round(LinearDecisionRules.get_decision(ldr, buy[i]), digits=2) for i in 1:2])
    println("  Primal bound: ", round(objective_value(ldr), digits=2))
    println()

    return ldr
end

# #### Product of Uniform distributions

product_uniform = Distributions.product_distribution([
    Distributions.Uniform(80.0, 120.0),
    Distributions.Uniform(60.0, 100.0),
])
solve_two_product_inventory(product_uniform; name = "Product of Uniforms");

# #### Mixed product: Uniform and Triangular

product_mixed = Distributions.product_distribution([
    Distributions.Uniform(80.0, 120.0),
    Distributions.TriangularDist(60.0, 100.0, 85.0),
])
solve_two_product_inventory(product_mixed; name = "Uniform × Triangular");

# #### Product of Truncated Normals

# This creates independent (uncorrelated) normal demands:

product_truncated_normal = Distributions.product_distribution([
    Distributions.truncated(Distributions.Normal(100.0, 10.0), 70.0, 130.0),
    Distributions.truncated(Distributions.Normal(80.0, 15.0), 40.0, 120.0),
])
solve_two_product_inventory(product_truncated_normal; name = "Product of Truncated Normals");

# ### Multivariate Discrete Non-Parametric

# For scenario-based uncertainty with joint realizations, use
# `MvDiscreteNonParametric`. Each scenario specifies values for ALL uncertain
# parameters simultaneously:

scenarios_mv = [
    [80.0, 60.0],   # Low demand for both products
    [100.0, 80.0],  # Medium demand
    [120.0, 100.0], # High demand for both
    [80.0, 100.0],  # Low product 1, high product 2
    [120.0, 60.0],  # High product 1, low product 2
]
probs_mv = [0.2, 0.3, 0.2, 0.15, 0.15]

mv_discrete = LinearDecisionRules.MvDiscreteNonParametric(scenarios_mv, probs_mv)
solve_two_product_inventory(mv_discrete; name = "MvDiscreteNonParametric (5 scenarios)");

# !!! tip
#     Use `MvDiscreteNonParametric` when:
#     - You have historical data with joint observations
#     - Demands are correlated (e.g., both high or both low together)
#     - You want exact scenario-based optimization

# ## Key takeaways

# | Distribution | Use case |
# |--------------|----------|
# | `Uniform(a, b)` | Equal likelihood across a range |
# | `truncated(Normal(μ, σ), a, b)` | Bell-shaped with known mean/std |
# | `TriangularDist(a, b, c)` | Known min, max, and most likely value |
# | `DiscreteNonParametric(vals, probs)` | Specific scenarios with probabilities |
# | `product_distribution([...])` | Independent multivariate uncertainties |
# | `MvDiscreteNonParametric` | Correlated scenarios (joint realizations) |

# ## What's next?

# - Learn about [advanced distribution modeling](@ref advanced_distributions_tutorial)
#   with constraints and rejection sampling
# - Explore [piecewise linear decision rules](@ref piecewise_linear_tutorial) for better approximations
