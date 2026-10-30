# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# # [Advanced distribution modeling](@id advanced_distributions_tutorial)

# ```@meta
# CurrentModule = LinearDecisionRules
# ```

# This tutorial covers techniques to model uncertainty beyond a product of independent laws:
# * constraining the support of a law with [`Truncate`](@ref) and [`ValidInequality`](@ref);
# * the difference between these two tags, which is the most important point;
# * correlated uncertainties, defined by linear constraints, and the rejection sampling that
#   estimates their moments.

# ## Setup

using JuMP
using HiGHS
using Distributions
using LinearDecisionRules
using Random

# ## The challenge: constrained uncertainty regions

# In many problems, the uncertainties are not independent. They may satisfy constraints such as:
# * **budget constraints**: the total demand over the products is limited;
# * **logical constraints**: if the demand for A is high, the demand for B is low.

# A constraint over the uncertainties only is written as any other JuMP constraint. But a
# constraint without decisions could mean two different things, and the package does not guess:
# it must say which one, with a tag. An untagged constraint over uncertainties only is an error,
# whose message names both tags:

untagged = LDRModel(HiGHS.Optimizer)
@variable(untagged, x in Uncertainty(Uniform(50, 150)))
@constraint(untagged, 80 <= x <= 120)
try
    optimize!(untagged)
catch err
    println(sprint(showerror, err))
end

# The two meanings are:
# * `Truncate()`: the constraint **conditions the law**. The uncertainties follow their declared
#   laws *given* that the constraints hold; the set `C` where they hold is the support.
# * `ValidInequality()`: the constraint **describes the support** without changing the law. It is a
#   statement that the law already satisfies the constraint (with probability one), and it makes
#   the support tighter than the box of the law, for the robust formulations.

# ## Part 1: a small example of the difference

# Consider a newsvendor whose demand has the law `Uniform(50, 150)`, and a constraint
# `80 <= d <= 120`, with each tag. The function builds the model with the tag as an argument.

function newsvendor(tag)
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, buy >= 0, FirstStage)
    @variable(model, d in Uncertainty(Uniform(50, 150)))
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @constraint(model, sell + ret <= buy)
    @constraint(model, sell <= d)
    @objective(model, Max, -10buy + 8ret + 15sell)
    if tag == :truncate
        @constraint(model, 80 <= d <= 120, Truncate())
    elseif tag == :valid
        @constraint(model, 80 <= d <= 120, ValidInequality())
    end
    optimize!(model)
    return model
end

# Let us compare, for each case, the first two moments of the demand (as the optimizer sees
# them), the primal value and the dual bound.

println("constraint        E[d]   E[d^2]      primal   dual")
for (name, tag) in [("none", :none), ("Truncate", :truncate), ("ValidInequality", :valid)]
    m = newsvendor(tag)
    moments = round.([value(m[:d]), expected_value(m[:d]^2)]; digits = 2)
    values = round.([objective_value(m), objective_bound(m)]; digits = 2)
    println(
        rpad(name, 18),
        rpad(moments[1], 7),
        rpad(moments[2], 12),
        rpad(values[1], 9),
        values[2],
    )
end

# * With no constraint, the support is `[50, 150]`.
# * With `Truncate()`, the law of `d` is `Uniform(80, 120)`: its second moment is 10133.3, and
#   the values are exactly those of the newsvendor with this law: 460 and 1460/3. A constraint on
#   a single law that is a box is a truncation of this law in closed form (here,
#   `Distributions.truncated`), so that no sampling is needed.
# * With `ValidInequality()`, the law is still `Uniform(50, 150)`, with the moments of the first
#   line, but the support is `[80, 120]`. The constraint is a **claim** that no demand is outside
#   `[80, 120]`, and the claim is false for 60% of the law. The results are inconsistent: the
#   "bound" of the dual is *below* the primal value, which is impossible for a bound. And the
#   simulation, which draws from the law, finds that the rule (which returns `120 - d`) violates
#   `ret >= 0` in the 30% of the draws with a demand above 120:

model = newsvendor(:valid)
simulate(model, 10_000; rng = Xoshiro(1)).violation_counts

# !!! warning
#     Use `ValidInequality()` only for a constraint that the law satisfies with probability one.
#     When a constraint cuts the law, use `Truncate()`.

# ## Part 2: a valid inequality that helps

# A valid inequality is useful when the support of the law is smaller than its box. Two products
# share a stock bought in the first stage. Their demands have joint scenarios, whose sum is at
# most 100, while the box of the law is `[0, 100]²`, whose corner `(100, 100)` has a total demand
# of 200:

function shared_stock(tag)
    scenarios = [[100.0, 0.0], [0.0, 100.0], [40.0, 40.0], [60.0, 20.0], [20.0, 60.0]]
    law = MvDiscreteNonParametric(scenarios, fill(0.2, 5))
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell[1:2] >= 0)
    @variable(model, d[1:2] in Uncertainty(law))
    @constraint(model, sum(sell) <= buy)
    @constraint(model, sell .<= d)
    @objective(model, Max, -10buy + 15sell[1] + 12sell[2])
    if tag == :truncate
        @constraint(model, d[1] + d[2] <= 100, Truncate())
    elseif tag == :valid
        @constraint(model, d[1] + d[2] <= 100, ValidInequality())
    end
    optimize!(model)
    return model
end

println("constraint        buy     primal    dual")
for (name, tag) in [("none", :none), ("ValidInequality", :valid), ("Truncate", :truncate)]
    m = shared_stock(tag)
    row = [rule_coefficient(m[:buy]), objective_value(m), objective_bound(m)]
    println(
        rpad(name, 18),
        rpad(round(row[1]; digits = 1), 8),
        rpad(round(row[2]; digits = 2), 10),
        round(row[3]; digits = 2),
    )
end

# Without the constraint, the rule has to be feasible on the whole box. Selling the demand would
# then need a stock of 200, which costs more than it earns, so the best rule does nothing and the
# value is zero. The valid inequality `d[1] + d[2] <= 100`
# holds for every scenario, so that the law is not changed, and the support is the polytope under
# it: the rule buys 100 and the value is 188.

# `Truncate()` of the same constraint conditions the law on the set where it holds, which contains
# all of the scenarios, so the law is the same. A discrete law is conditioned **exactly**: the
# scenarios that satisfy the constraint are kept, with their probabilities renormalized. So the
# values are those of `ValidInequality()`, and they are exact, as [`is_certified`](@ref) says of
# the primal value and of the dual bound:

model = shared_stock(:truncate)
is_certified(model, PrimalLDR()), is_certified(model, DualLDR())

# Prefer `ValidInequality()` for a constraint that is already satisfied by the law, and
# `Truncate()` for one that cuts it: for a continuous law, `Truncate()` of a constraint on several
# uncertainties is done by rejection sampling, as the next part shows.

# ## Part 3: truncation of a multivariate law

# For a constraint that cuts a multivariate law, `Truncate()` is what we want. Two products with
# independent demands, uniform on `[40, 80]`, and a limit on the total demand:

function two_products(; total = nothing)
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, buy[1:2] >= 0, FirstStage)
    @variable(model, sell[1:2] >= 0)
    @variable(model, ret[1:2] >= 0)
    law = product_distribution([Uniform(40.0, 80.0), Uniform(40.0, 80.0)])
    @variable(model, d[1:2] in Uncertainty(law))
    @constraint(model, [i in 1:2], sell[i] + ret[i] <= buy[i])
    @constraint(model, [i in 1:2], sell[i] <= d[i])
    @objective(model, Max, sum(-10buy[i] + 8ret[i] + 15sell[i] for i in 1:2))
    if total !== nothing
        @constraint(model, d[1] + d[2] <= total, Truncate())
    end
    optimize!(model)
    return model
end

# Without the constraint, the demands are independent:

model = two_products()
objective_value(model), objective_bound(model)

# With `d[1] + d[2] <= 120`, the demands live in a triangle of the square, and they are negatively
# correlated: when one is high, the other tends to be low.

model = two_products(; total = 120)
objective_value(model), objective_bound(model)

# The moments of this law (the mean, and the covariance, which is negative off the diagonal) are
# estimated by rejection sampling: the law of the demands is drawn, and the draws that satisfy the
# constraint are kept. The estimates are cached, and they are controlled by the optimizer
# attributes [`RejectionSamplingMaxSamples`](@ref) (the number of draws kept),
# [`RejectionSamplingSeed`](@ref) and [`RejectionSamplingTimeLimit`](@ref), whose docstrings give
# their defaults. The values are estimates, so the primal value is not exactly the expected profit
# of the rule and the dual is not exactly a bound: they are not certified,

is_certified(model, PrimalLDR()), is_certified(model, DualLDR())

# and the status of each formulation ends with the reason:

raw_status(model, PrimalLDR())

# A larger sample gives a more precise estimate. Here is the effect of the number of draws on the
# primal and dual values, for the same seed:

println("draws    primal      dual")
for n in [100, 1_000, 10_000, 100_000]
    m = two_products(; total = 120)
    set_attribute(m, RejectionSamplingMaxSamples(), n)
    optimize!(m)
    println(
        rpad(n, 9),
        rpad(round(objective_value(m); digits = 3), 12),
        round(objective_bound(m); digits = 3),
    )
end

# `simulate` draws from the same conditioned law, in the original coordinates, so that the mean of
# the objective estimates the value of the rule without relying on the moments that were estimated:

simulate(model, 10_000; rng = Xoshiro(1))

# ## Part 4: several constraints: reservoirs

# Several `Truncate()` constraints define more complex regions. Three reservoirs have uncertain
# inflows, whose total is bounded (a weather system), and that cannot be too different for
# adjacent reservoirs (spatial correlation).

function reservoirs()
    capacity = [100.0, 80.0, 120.0]
    benefit = [5.0, 4.0, 6.0]
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= storage[i = 1:3] <= capacity[i], FirstStage)
    @variable(model, release[1:3] >= 0)
    @variable(model, spill[1:3] >= 0)
    law = product_distribution([Uniform(10.0, 50.0), Uniform(10.0, 40.0), Uniform(10.0, 60.0)])
    @variable(model, inflow[1:3] in Uncertainty(law))
    ## The total inflow is bounded.
    @constraint(model, inflow[1] + inflow[2] + inflow[3] <= 120, Truncate())
    ## Adjacent reservoirs have similar inflows.
    @constraint(model, inflow[1] - inflow[2] <= 20, Truncate())
    @constraint(model, inflow[2] - inflow[1] <= 20, Truncate())
    @constraint(model, inflow[2] - inflow[3] <= 25, Truncate())
    @constraint(model, inflow[3] - inflow[2] <= 25, Truncate())
    ## The water balance.
    @constraint(model, [i in 1:3], release[i] + spill[i] <= storage[i] + inflow[i])
    @objective(model, Max, sum(benefit[i] * release[i] - spill[i] for i in 1:3))
    optimize!(model)
    return model
end

model = reservoirs()
[rule_coefficient(model[:storage][i]) for i in 1:3], objective_value(model), objective_bound(model)

# ## Part 5: discrete laws with constraints

# `Truncate()` also conditions a discrete law: the scenarios that violate the constraint are
# excluded and the others are renormalized, exactly (no sampling is needed). Seven scenarios of
# two demands, equally likely, one of which has both demands high:

function discrete_with_truncate(total)
    scenarios = [[60.0, 60.0], [60.0, 80.0], [80.0, 60.0], [80.0, 80.0]]
    append!(scenarios, [[100.0, 60.0], [60.0, 100.0], [100.0, 100.0]])
    law = MvDiscreteNonParametric(scenarios, fill(1 / 7, 7))
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, buy[1:2] >= 0, FirstStage)
    @variable(model, sell[1:2] >= 0)
    @variable(model, ret[1:2] >= 0)
    @variable(model, d[1:2] in Uncertainty(law))
    @constraint(model, [i in 1:2], sell[i] + ret[i] <= buy[i])
    @constraint(model, [i in 1:2], sell[i] <= d[i])
    @objective(model, Max, sum(-10buy[i] + 8ret[i] + 15sell[i] for i in 1:2))
    @constraint(model, d[1] + d[2] <= total, Truncate())
    optimize!(model)
    return model
end

# With `d[1] + d[2] <= 160`, the scenario `(100, 100)` is excluded:

model = discrete_with_truncate(160)
[rule_coefficient(model[:buy][i]) for i in 1:2], objective_value(model), objective_bound(model)

# ## How constrained uncertainty works

# 1. **Tags**: each constraint over the uncertainties only has a role, `Truncate()` or
#    `ValidInequality()`. The first conditions the law, the second only describes the support.
# 2. **Support**: the constraints and the bounds of the laws define a polytope, the support of the
#    uncertainties. The primal rule is feasible on all of it, and the dual bound is for the
#    uncertainties on it.
# 3. **Moments**: a univariate law with `Truncate()` constraints that are a box on it is truncated in
#    closed form. The uncertainties that are coupled by `Truncate()` constraints share a
#    conditioned law: exact when their laws are discrete with finitely many atoms, and otherwise
#    with moments estimated by rejection sampling.
# 4. **Certificates**: the values that use estimated moments are marked as not certified.

# !!! warning
#     Rejection sampling can be expensive for highly constrained regions. If the feasible region
#     is a very small fraction of the support of the law, there may be too few draws within the
#     time limit, which is an error that names the constraints and the acceptance rate. Consider
#     reparametrizing the uncertainty.

# ## Summary

# | Feature | Description |
# |---------|-------------|
# | `@constraint(model, a <= d <= b, Truncate())` | the law of `d` truncated to `[a, b]`, exactly |
# | `@constraint(model, a'ξ <= b, Truncate())` | the law conditioned on the constraint (exact for discrete laws, else rejection sampling) |
# | `@constraint(model, a'ξ <= b, ValidInequality())` | a constraint that the law satisfies, which tightens the support |
# | untagged constraint on uncertainties | an error that asks for one of the tags |
# | `RejectionSamplingMaxSamples()`, `RejectionSamplingSeed()` | the accuracy and the seed of the estimates |

# ## What's next?

# The [`ConfidenceMvNormal`](@ref confidence_mvnormal_tutorial) tutorial uses an ellipsoid as the
# support of a normal law.
