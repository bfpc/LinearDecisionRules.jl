# # [Advanced distribution modeling](@id advanced_distributions_tutorial)

# This tutorial covers advanced techniques for modeling uncertainty, including:
# - Constraining the support of distributions
# - Handling rejection sampling automatically
# - Modeling correlated uncertainties with linear constraints

# ## Setup

using JuMP
import LinearDecisionRules
import HiGHS
import Distributions

# ## The challenge: constrained uncertainty regions

# In many real-world problems, uncertainties are not independent. They may be
# subject to constraints such as:
# - **Budget constraints**: total demand across products is limited
# - **Logical constraints**: if demand for A is high, demand for B is low

# LinearDecisionRules handles these cases through **rejection sampling**:
# constraints on uncertainty variables define the feasible region, and the
# package automatically adjusts the optimization accordingly.

# ## Part 1: Simple bounds on uncertainty

# ### Narrowing the support with constraints

# Consider an inventory problem where we know demand will be in a narrower
# range than the underlying distribution suggests:

function solve_with_bounds(; lower_bound = nothing, upper_bound = nothing)
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    buy_cost, sell_price, salvage = 10, 15, 5

    @variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell >= 0)
    @variable(ldr, leftover >= 0)
    @variable(
        ldr,
        demand in LinearDecisionRules.Uncertainty(;
            distribution = Distributions.Uniform(50.0, 150.0),
        )
    )

    ## Add constraints that narrow the uncertainty support
    if lower_bound !== nothing
        @constraint(ldr, demand >= lower_bound)
    end
    if upper_bound !== nothing
        @constraint(ldr, demand <= upper_bound)
    end

    @constraint(ldr, sell + leftover <= buy)
    @constraint(ldr, sell <= demand)

    @objective(
        ldr,
        Max,
        -buy_cost * buy + sell_price * sell + salvage * leftover
    )

    optimize!(ldr)

    return (
        buy = LinearDecisionRules.get_decision(ldr, buy),
        primal = objective_value(ldr),
        dual = objective_value(ldr; dual = true),
    )
end

# Without constraints (full Uniform(50, 150) support):
result_full = solve_with_bounds()
println("Full support [50, 150]:")
println("  Optimal buy: ", round(result_full.buy; digits = 2))
println("  Primal: ", round(result_full.primal; digits = 2))
println()

# With tighter bounds (rejection sampling kicks in):
result_narrow = solve_with_bounds(; lower_bound = 80.0, upper_bound = 120.0)
println("Narrowed support [80, 120] via constraints:")
println("  Optimal buy: ", round(result_narrow.buy; digits = 2))
println("  Primal: ", round(result_narrow.primal; digits = 2))
println()

# !!! note
#     When you add constraints on uncertainty variables, LinearDecisionRules
#     automatically uses rejection sampling to compute expectations over the
#     constrained region.

# ## Part 2: Linear inequality constraints on multivariate uncertainty

# The real power of constrained uncertainty comes with multivariate
# distributions where linear inequalities define complex feasible regions.

# ### Example: Correlated demands with a total budget

# Consider two products where total market demand is constrained:

function solve_budget_constrained(; total_demand_max = nothing)
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    buy_cost, sell_price, salvage = 10, 15, 5

    @variable(ldr, buy[1:2] >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell[1:2] >= 0)
    @variable(ldr, leftover[1:2] >= 0)
    @variable(
        ldr,
        demand[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = Distributions.product_distribution([
                Distributions.Uniform(40.0, 80.0),
                Distributions.Uniform(40.0, 80.0),
            ]),
        )
    )

    ## Add a constraint that total demand is limited
    if total_demand_max !== nothing
        @constraint(ldr, demand[1] + demand[2] <= total_demand_max)
    end

    for i in 1:2
        @constraint(ldr, sell[i] + leftover[i] <= buy[i])
        @constraint(ldr, sell[i] <= demand[i])
    end

    @objective(
        ldr,
        Max,
        sum(
            -buy_cost * buy[i] + sell_price * sell[i] + salvage * leftover[i]
            for i in 1:2
        )
    )

    optimize!(ldr)

    return (
        buy = [LinearDecisionRules.get_decision(ldr, buy[i]) for i in 1:2],
        primal = objective_value(ldr),
    )
end

# Without budget constraint:
result_no_budget = solve_budget_constrained()
println("No budget constraint (demands independent):")
println("  Optimal buy: ", round.(result_no_budget.buy, digits = 2))
println("  Primal: ", round(result_no_budget.primal; digits = 2))
println()

# With budget constraint (creates negative correlation):
result_with_budget = solve_budget_constrained(; total_demand_max = 120.0)
println("With budget constraint (demand[1] + demand[2] <= 120):")
println("  Optimal buy: ", round.(result_with_budget.buy, digits = 2))
println("  Primal: ", round(result_with_budget.primal; digits = 2))
println()

# !!! tip
#     The constraint `demand[1] + demand[2] <= 120` creates a triangular
#     feasible region within the original rectangle. This induces negative
#     correlation: when one demand is high, the other tends to be lower.

# ## Part 3: Multiple linear constraints

# You can add multiple constraints to define more complex uncertainty regions.

# ### Example: Reservoir inflows with balance constraints

# Consider a water system with three reservoirs where:
# - Each has uncertain inflow from its own catchment
# - Total system inflow is bounded
# - Adjacent reservoirs have correlated inflows

function solve_reservoir_system()
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    n_reservoirs = 3
    capacity = [100.0, 80.0, 120.0]
    release_benefit = [5.0, 4.0, 6.0]
    spill_penalty = [1.0, 1.0, 1.0]

    @variable(
        ldr,
        0 <= storage[i = 1:n_reservoirs] <= capacity[i],
        LinearDecisionRules.FirstStage
    )
    @variable(ldr, release[1:n_reservoirs] >= 0)
    @variable(ldr, spill[1:n_reservoirs] >= 0)
    @variable(
        ldr,
        inflow[1:n_reservoirs] in LinearDecisionRules.Uncertainty(;
            distribution = Distributions.product_distribution([
                Distributions.Uniform(10.0, 50.0),
                Distributions.Uniform(10.0, 40.0),
                Distributions.Uniform(10.0, 60.0),
            ]),
        )
    )

    ## Constraint 1: Total system inflow is bounded (weather system constraint)
    @constraint(ldr, inflow[1] + inflow[2] + inflow[3] <= 120.0)

    ## Constraint 2: Adjacent reservoirs have similar inflows (spatial correlation)
    @constraint(ldr, inflow[1] - inflow[2] <= 20.0)
    @constraint(ldr, inflow[2] - inflow[1] <= 20.0)
    @constraint(ldr, inflow[2] - inflow[3] <= 25.0)
    @constraint(ldr, inflow[3] - inflow[2] <= 25.0)

    ## Water balance constraints
    for i in 1:n_reservoirs
        @constraint(ldr, release[i] + spill[i] <= storage[i] + inflow[i])
    end

    @objective(
        ldr,
        Max,
        sum(
            release_benefit[i] * release[i] - spill_penalty[i] * spill[i] for
            i in 1:n_reservoirs
        )
    )

    optimize!(ldr)

    println("Reservoir system with constrained inflows:")
    println(
        "  Initial storage: ",
        [
            round(
                LinearDecisionRules.get_decision(ldr, storage[i]);
                digits = 2,
            ) for i in 1:n_reservoirs
        ],
    )
    println("  Primal bound: ", round(objective_value(ldr); digits = 2))
    println(
        "  Dual bound:   ",
        round(objective_value(ldr; dual = true); digits = 2),
    )

    return ldr
end

ldr_reservoir = solve_reservoir_system();

# ## Part 4: Combining discrete scenarios with constraints

# Constraints can also be applied to discrete distributions:

function solve_discrete_constrained()
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    ## Define scenarios for two correlated demands
    scenarios = [
        [60.0, 60.0],   # Both low
        [60.0, 80.0],   # Low-medium
        [80.0, 60.0],   # Medium-low
        [80.0, 80.0],   # Both medium
        [100.0, 60.0],  # High-low
        [60.0, 100.0],  # Low-high
        [100.0, 100.0], # Both high (will be partially excluded)
    ]
    probs = fill(1.0 / 7, 7)

    buy_cost, sell_price, salvage = 10, 15, 5

    @variable(ldr, buy[1:2] >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell[1:2] >= 0)
    @variable(ldr, leftover[1:2] >= 0)
    @variable(
        ldr,
        demand[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = LinearDecisionRules.MvDiscreteNonParametric(
                scenarios,
                probs,
            ),
        )
    )

    ## Exclude the "both high" scenario via constraint
    @constraint(ldr, demand[1] + demand[2] <= 160.0)

    for i in 1:2
        @constraint(ldr, sell[i] + leftover[i] <= buy[i])
        @constraint(ldr, sell[i] <= demand[i])
    end

    @objective(
        ldr,
        Max,
        sum(
            -buy_cost * buy[i] + sell_price * sell[i] + salvage * leftover[i]
            for i in 1:2
        )
    )

    optimize!(ldr)

    println(
        "\nDiscrete distribution with constraint (excludes 'both high' scenario):",
    )
    println(
        "  Optimal buy: ",
        [
            round(LinearDecisionRules.get_decision(ldr, buy[i]); digits = 2) for
            i in 1:2
        ],
    )
    println("  Primal bound: ", round(objective_value(ldr); digits = 2))

    return ldr
end

solve_discrete_constrained();

# ## How constrained uncertainty works

# When you add constraints on uncertainty variables:

# 1. **Detection**: LinearDecisionRules identifies which uncertainty variables
#    have additional constraints beyond their distribution bounds.

# 2. **Polytope construction**: The constraints define a polytope within the
#    original distribution support.

# 3. **Expectation computation**: For primal bounds, expectations are computed
#    over the constrained region using rejection sampling or analytical methods
#    when available.

# 4. **Dual bounds**: The robust optimization formulation naturally handles
#    the constrained uncertainty set.

# !!! warning
#     Rejection sampling can be computationally expensive for highly constrained
#     regions. If the feasible region is a very small fraction of the original
#     support, consider reparametrizing your uncertainty model.

# ## Summary

# | Feature | Description |
# |---------|-------------|
# | Simple bounds | `@constraint(model, ξ >= a)` and `@constraint(model, ξ <= b)` |
# | Linear inequalities | `@constraint(model, a₁ξ₁ + a₂ξ₂ <= b)` |
# | Multiple constraints | Combine to define complex polytopes |
# | Automatic handling | Package detects and handles rejection sampling |

# ## What's next?

# - Explore [piecewise linear decision rules](@ref piecewise_linear_tutorial) for better approximations
# - See the [mathematical formulation](@ref math_formulation) for theory details
