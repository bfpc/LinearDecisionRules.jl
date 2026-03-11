# # [Ellipsoidal uncertainty with `ConfidenceMvNormal`](@id confidence_mvnormal_tutorial)

# This tutorial introduces the `ConfidenceMvNormal` distribution, which models
# uncertainty as a multivariate normal truncated to an ellipsoid containing a
# chosen fraction of probability mass. It is especially useful when you want to
# control how *conservative* the uncertainty set is.

# ## Setup

using JuMP
import LinearDecisionRules
import HiGHS
import Distributions

# ## What is `ConfidenceMvNormal`?

# Suppose demand follows a multivariate normal distribution
# ``d \sim \mathcal{N}(\mu, \Sigma)``.
# In practice we may not want to plan for the entire (unbounded) support.
# `ConfidenceMvNormal(μ, Σ, α)` restricts the uncertainty to the
# smallest **ellipsoid** centred at ``\mu`` that contains fraction ``\alpha``
# of the probability mass:
# ```
# E_{α} = { x :  (x - μ)' Σ⁻¹ (x - μ) ≤ ρ²(α) }
# ```
# where ``\rho^2(\alpha) = \text{quantile}(\chi^2_d, \alpha)``.

# The distribution provides the LDR framework with:
# * **Mean** – equal to ``\mu`` (the ellipsoid is symmetric)
# * **Covariance** – a scalar multiple of ``\Sigma``, smaller than ``\Sigma``
#   because the tails are cut
# * **Finite bounds** – required by the framework, given by the
#   axis-aligned box that contains the ellipsoid

# ## Inspecting the distribution

μ = [100.0, 80.0]
Σ = [100.0 40.0
     40.0  64.0]

dist_95 = LinearDecisionRules.ConfidenceMvNormal(μ, Σ, 0.95)

println("α = 0.95")
println("  Ellipsoid radius  ρ = ", round(dist_95.ρ; digits = 4))
println("  Mean              = ", Distributions.mean(dist_95))
println("  Covariance        = ", round.(Distributions.cov(dist_95); digits = 2))
println("  Component bounds:")
for i in 1:2
    lo = round(Distributions.minimum(dist_95)[i]; digits = 2)
    hi = round(Distributions.maximum(dist_95)[i]; digits = 2)
    println("    demand[$i] ∈ [$lo, $hi]")
end

# As ``\alpha`` grows the ellipsoid expands, so the covariance approaches
# ``\Sigma`` and the bounds widen:

for α in [0.50, 0.80, 0.95, 0.99]
    d = LinearDecisionRules.ConfidenceMvNormal(μ, Σ, α)
    s = round(Distributions.cov(d)[1, 1] / Σ[1, 1]; digits = 3)
    println("α = $α  →  cov scaling = $s  (ρ = $(round(d.ρ; digits=2)))")
end

# ## An inventory problem with ellipsoidal demand

# Two products have correlated demand.  Before demand is revealed a retailer
# decides how many units to stock (`buy`); afterwards they sell as much as
# possible (`sell`).
#
# | Parameter | Value |
# |-----------|-------|
# | Buy cost  | \$10  |
# | Sell price | \$15 |
# | Salvage value | \$5 |

buy_cost    = 10.0
sell_price  = 15.0
salvage_val =  5.0

function solve_inventory(α)
    dist = LinearDecisionRules.ConfidenceMvNormal(μ, Σ, α)

    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)
    set_attribute(ldr, LinearDecisionRules.SolveDual(), true)

    @variable(ldr, buy[1:2] >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell[1:2]    >= 0)
    @variable(ldr, salvage[1:2] >= 0)
    @variable(
        ldr,
        demand[1:2] in LinearDecisionRules.Uncertainty(distribution = dist),
    )

    for i in 1:2
        @constraint(ldr, sell[i] + salvage[i] <= buy[i])
        @constraint(ldr, sell[i] <= demand[i])
    end

    @objective(
        ldr,
        Max,
        sum(
            -buy_cost * buy[i] + sell_price * sell[i] + salvage_val * salvage[i]
            for i in 1:2
        ),
    )

    optimize!(ldr)
    return (
        buy    = [LinearDecisionRules.get_decision(ldr, buy[i]) for i in 1:2],
        primal = objective_value(ldr),
        dual   = objective_value(ldr; dual = true),
    )
end

# ## Effect of the confidence level ``\alpha``

# A smaller ``\alpha`` means a tighter uncertainty set: the planner expects
# demand to stay close to the mean and buys less speculatively.

for α in [0.50, 0.80, 0.90, 0.95, 0.99]
    r = solve_inventory(α)
    println(
        "α = $α  buy = $(round.(r.buy; digits=1))  " *
        "primal = $(round(r.primal; digits=2))  " *
        "dual = $(round(r.dual; digits=2))",
    )
end

# !!! note
#     Higher ``\alpha`` → wider ellipsoid → demand can be further from the mean
#     → optimal buy quantities grow to cover extreme scenarios.

# ## Bounds: box approximation of the ellipsoid

# The LDR framework requires finite `minimum` and `maximum` for each
# uncertainty variable.  `ConfidenceMvNormal` provides the **tightest
# axis-aligned box** that contains the ellipsoid:
#
# ```math
# \mu_k - \rho\sqrt{\Sigma_{kk}}
# \;\leq\; d_k \;\leq\;
# \mu_k + \rho\sqrt{\Sigma_{kk}}
# ```
#
# These are outer (conservative) bounds on the actual ellipsoid.
# A future extension could use *rotated box* bounds aligned with the
# principal axes of ``\Sigma``, which would be tighter for highly correlated
# distributions.

# ## What's next?

# - See [`ConfidenceMvNormal`](@ref confidence_mvnormal) for the mathematical
#   details
# - Explore [piecewise linear decision rules](@ref piecewise_linear_tutorial)
#   for tighter primal–dual gaps
# - Use [advanced constraint-based uncertainty](@ref advanced_distributions_tutorial)
#   for polytope uncertainty sets

