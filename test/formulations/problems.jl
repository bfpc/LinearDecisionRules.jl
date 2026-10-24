# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The problem library of the formulation tests: plain JuMP models with a hand-made `ModelInfo`.
# `primal` and `dual` are the known values (NaN when unknown), `legacy` the key in
# `test/reference/legacy_values.toml` (empty when absent).

module Problems

using JuMP

import Distributions
import LinearDecisionRules:
    CanonicalForm,
    ModelInfo,
    Moments,
    MvDiscreteNonParametric,
    UncertaintyDeclaration,
    canonicalize,
    moments

struct Problem
    name::String
    legacy::String
    model::Model
    info::ModelInfo
    primal::Float64
    dual::Float64
    atol::Float64 # 0 for the default tolerance of `isapprox`
end

function Problem(name, legacy, model, info, primal, dual; atol = 0.0)
    return Problem(name, legacy, model, info, primal, dual, atol)
end

"""
    matches(value, expected, p::Problem) -> Bool

Whether `value` is the known value `expected` of `p`, with the tolerance of `p`.
"""
function matches(value, expected, p::Problem)
    return p.atol > 0 ? isapprox(value, expected; atol = p.atol) : isapprox(value, expected)
end

form(p::Problem) = canonicalize(backend(p.model), p.info)

function declare(dist, vars...; stages = fill(nothing, length(vars)))
    return UncertaintyDeclaration(dist, collect(index.(vars)), stages)
end

stages(pairs...) = Dict(index(v) => t for (v, t) in pairs)

function newsvendor(; integer = false)
    model = Model()
    @variable(model, buy >= 0, integer = integer)
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @variable(model, demand)
    @constraint(model, sell + ret <= buy)
    @constraint(model, sell <= demand)
    @objective(model, Max, -10buy + 8ret + 15sell)
    info = ModelInfo(;
        uncertainties = [declare(Distributions.Uniform(80, 120), demand)],
        decision_stages = stages(buy => 1),
    )
    name = integer ? "newsvendor_integer" : "newsvendor"
    return Problem(name, name, model, info, 460, integer ? 486 : 1460 / 3)
end

function newsvendor_random_price()
    model = Model()
    @variable(model, buy >= 0)
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @variable(model, demand)
    @variable(model, price)
    @constraint(model, sell + ret <= buy)
    @constraint(model, sell <= demand)
    @objective(model, Max, -10buy + 8ret + price * sell)
    info = ModelInfo(;
        uncertainties = [
            declare(Distributions.Uniform(80, 120), demand),
            declare(Distributions.Uniform(13, 17), price),
        ],
        decision_stages = stages(buy => 1),
    )
    name = "newsvendor_random_price"
    return Problem(name, name, model, info, 460, 1460 / 3)
end

function newsvendor_two_products()
    model = Model()
    @variable(model, buy[1:2] >= 0)
    @variable(model, sell[1:2] >= 0)
    @variable(model, ret[1:2] >= 0)
    @variable(model, demand[1:2])
    @constraint(model, [i in 1:2], sell[i] + ret[i] <= buy[i])
    @constraint(model, [i in 1:2], sell[i] <= demand[i])
    @objective(model, Max, sum(-10buy[i] + 8ret[i] + 15sell[i] for i in 1:2))
    dist = Distributions.product_distribution([Distributions.Uniform(80, 120) for _ in 1:2])
    info = ModelInfo(;
        uncertainties = [declare(dist, demand...)],
        decision_stages = stages((b => 1 for b in buy)...),
    )
    name = "newsvendor_two_products"
    return Problem(name, name, model, info, 920, 2920 / 3)
end

function newsvendor_three_stages()
    model = Model()
    @variable(model, buy >= 0)
    @variable(model, sell2 >= 0)
    @variable(model, ret2 >= 0)
    @variable(model, demand2)
    @variable(model, buy2 >= 0)
    @variable(model, buy_cost2)
    @variable(model, sell3 >= 0)
    @variable(model, ret3 >= 0)
    @variable(model, demand3)
    @constraint(model, sell2 + ret2 <= buy + buy2)
    @constraint(model, sell2 <= demand2)
    @constraint(model, sell2 + ret2 + sell3 + ret3 <= buy + buy2)
    @constraint(model, sell3 <= demand3)
    @objective(model, Max, -10buy - buy_cost2 * buy2 + 8ret2 + 15sell2 + 8ret3 + 15sell3)
    info = ModelInfo(;
        uncertainties = [
            declare(Distributions.Uniform(80, 120), demand2; stages = [2]),
            declare(Distributions.Uniform(8, 14), buy_cost2; stages = [2]),
            declare(Distributions.Uniform(80, 120), demand3; stages = [3]),
        ],
        decision_stages = stages(buy => 1, sell2 => 2, ret2 => 2, buy2 => 2, sell3 => 3, ret3 => 3),
    )
    name = "newsvendor_three_stages"
    return Problem(name, name, model, info, 940, 1000)
end

# The inventory with backlog of `benchmark/timing.jl` in the previous version.
function inventory(T::Int; dual = NaN)
    model = Model()
    @variable(model, d[1:T])
    @variable(model, 0 <= x[1:T] <= 150)
    @variable(model, I[1:T])
    @variable(model, y[1:T])
    @constraint(model, I[1] == x[1] - d[1])
    @constraint(model, [t in 2:T], I[t] == I[t-1] + x[t] - d[t])
    @constraint(model, [t in 1:T], y[t] >= 0.5 * I[t])
    @constraint(model, [t in 1:T], y[t] >= -3 * I[t])
    @objective(model, Min, sum(x[t] + y[t] for t in 1:T))
    info = ModelInfo(;
        uncertainties = [
            declare(Distributions.Uniform(60, 100), d[t]; stages = [t + 1]) for t in 1:T
        ],
        decision_stages = merge(
            stages((x[t] => t for t in 1:T)...),
            stages((I[t] => t + 1 for t in 1:T)...),
            stages((y[t] => t + 1 for t in 1:T)...),
        ),
    )
    name = "inventory_T$T"
    return Problem(name, name, model, info, 90T + 20, dual)
end

function hydro_toy()
    model = Model()
    @variable(model, vi == 0.5)
    @variable(model, 0 <= vf <= 1)
    @variable(model, gh >= 0)
    @variable(model, gt >= 0)
    @variable(model, inflow[1:3])
    @constraint(model, vf == vi - gh + sum(inflow))
    @constraint(model, gt + gh == 0.3)
    @objective(model, Min, gt^2 + vf^2 / 2 - vf)
    info = ModelInfo(;
        uncertainties = [declare(Distributions.Uniform(0, 0.1i), inflow[i]) for i in 1:3],
    )
    name = "hydro_toy_fixed_volume"
    return Problem(name, name, model, info, -0.412778, -0.412778; atol = 1e-6)
end

# Two products that share the stock bought in stage 1; `d[1] + d[2] <= 230` is a valid inequality.
function shared_stock(; valid::Bool = true)
    model = Model()
    @variable(model, buy >= 0)
    @variable(model, sell[1:2] >= 0)
    @variable(model, d[1:2])
    @constraint(model, sum(sell) <= buy)
    @constraint(model, sell .<= d)
    @objective(model, Max, -10buy + 15sum(sell))
    rows = Set{MOI.ConstraintIndex}()
    if valid
        push!(rows, index(@constraint(model, d[1] + d[2] <= 230)))
    end
    info = ModelInfo(;
        uncertainties = [declare(Distributions.Uniform(80, 120), d[i]) for i in 1:2],
        decision_stages = stages(buy => 1),
        valid_rows = rows,
    )
    return Problem(valid ? "shared_stock" : "shared_stock_box", "", model, info, NaN, NaN)
end

# Spike audit, bug 1: the dual must keep the inequalities of a model without bounded uncertainty.
function deterministic(; bound::Bool = false)
    model = Model()
    if bound
        @variable(model, x <= 3)
    else
        @variable(model, x)
        @constraint(model, x <= 3)
    end
    @objective(model, Max, 2x)
    name = bound ? "deterministic_bound" : "deterministic"
    return Problem(name, "", model, ModelInfo(), 6, 6)
end

# An unbounded support: `x(η) = 3 + η` is feasible, so both values are 6.
function unbounded_support()
    model = Model()
    @variable(model, x)
    @variable(model, η)
    @constraint(model, x <= 3 + η)
    @objective(model, Max, 2x)
    info = ModelInfo(; uncertainties = [declare(Distributions.Normal(0, 1), η)])
    return Problem("unbounded_support", "", model, info, 6, 6)
end

function infeasible()
    model = Model()
    @variable(model, buy >= 0)
    @variable(model, sell >= 0)
    @variable(model, demand)
    @constraint(model, sell <= demand)
    @constraint(model, sell + buy <= -1)
    @objective(model, Max, sell - buy)
    info = ModelInfo(;
        uncertainties = [declare(Distributions.Uniform(80, 120), demand)],
        decision_stages = stages(buy => 1),
    )
    return Problem("infeasible", "", model, info, NaN, NaN)
end

function unbounded()
    model = Model()
    @variable(model, buy >= 0)
    @variable(model, sell >= 0)
    @variable(model, demand)
    @constraint(model, sell <= demand)
    @objective(model, Max, sell + buy)
    info = ModelInfo(;
        uncertainties = [declare(Distributions.Uniform(80, 120), demand)],
        decision_stages = stages(buy => 1),
    )
    return Problem("unbounded", "", model, info, NaN, NaN)
end

# Without rows, HiGHS reports DUAL_INFEASIBLE with a FEASIBLE_POINT (the origin of the ray).
function unbounded_without_rows()
    model = Model()
    @variable(model, buy >= 0)
    @variable(model, demand)
    @objective(model, Max, buy)
    info = ModelInfo(;
        uncertainties = [declare(Distributions.Uniform(80, 120), demand)],
        decision_stages = stages(buy => 1),
    )
    return Problem("unbounded_without_rows", "", model, info, NaN, NaN)
end

# The tutorial newsvendor (buy 10, sell 15, salvage 5) with a discrete demand.
function newsvendor_discrete(dist, name, primal, dual)
    model = Model()
    k = length(dist)
    @variable(model, buy[1:k] >= 0)
    @variable(model, sell[1:k] >= 0)
    @variable(model, salvage[1:k] >= 0)
    @variable(model, demand[1:k])
    @constraint(model, [i in 1:k], sell[i] + salvage[i] <= buy[i])
    @constraint(model, [i in 1:k], sell[i] <= demand[i])
    @objective(model, Max, sum(-10buy[i] + 15sell[i] + 5salvage[i] for i in 1:k))
    info = ModelInfo(;
        uncertainties = [declare(dist, demand...)],
        decision_stages = stages((b => 1 for b in buy)...),
    )
    return Problem(name, name, model, info, primal, dual)
end

function newsvendor_mv_discrete()
    scenarios = [[80.0, 60.0], [100.0, 80.0], [120.0, 100.0], [80.0, 100.0], [120.0, 60.0]]
    dist = MvDiscreteNonParametric(scenarios, [0.2, 0.3, 0.2, 0.15, 0.15])
    return newsvendor_discrete(dist, "newsvendor_mv_discrete", 700, 760)
end

function newsvendor_discrete_demand()
    dist = Distributions.DiscreteNonParametric([80.0, 90, 100, 110, 120], [0.1, 0.2, 0.4, 0.2, 0.1])
    return newsvendor_discrete(dist, "newsvendor_discrete_demand", 400, 470)
end

const CORRELATED = MvDiscreteNonParametric(
    [[80.0, 60.0], [100.0, 80.0], [120.0, 100.0], [80.0, 100.0], [120.0, 60.0]],
    [0.2, 0.3, 0.2, 0.15, 0.15],
)

# One correlated law over stages 2 and 3, a row touching both of its blocks, and an interval row.
function law_spanning_stages()
    model = Model()
    @variable(model, buy >= 0)
    @variable(model, s2 >= 0)
    @variable(model, s3 >= 0)
    @variable(model, d[1:2])
    @constraint(model, s2 + s3 <= buy)
    @constraint(model, s2 <= d[1])
    @constraint(model, 10 <= s3 + d[1] - d[2] <= 100)
    @objective(model, Max, -10buy + 15s2 + 12s3)
    info = ModelInfo(;
        uncertainties = [declare(CORRELATED, d...; stages = [2, 3])],
        decision_stages = stages(buy => 1, s2 => 2, s3 => 3),
    )
    return Problem("law_spanning_stages", "", model, info, NaN, NaN)
end

# A Min QP: `q` appears in the objective only (linear, square and products), and `z` in no
# constraint.
function objective_only_uncertainty()
    model = Model()
    @variable(model, x1)
    @variable(model, x2)
    @variable(model, z)
    @variable(model, e[1:2])
    @variable(model, q)
    @constraint(model, x1 + x2 >= e[1])
    @constraint(model, x2 <= 5 + e[2])
    @objective(model, Min, x1^2 + x2^2 + z^2 - z * q + 3q^2 + 2q + q * e[1] + x2 * e[2])
    law = MvDiscreteNonParametric([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]], [0.3, 0.3, 0.4])
    info = ModelInfo(;
        uncertainties = [
            declare(law, e...; stages = [2, 2]),
            declare(Distributions.Uniform(0, 1), q; stages = [2]),
        ],
        decision_stages = stages(x1 => 1, x2 => 2, z => 3),
    )
    return Problem("objective_only_uncertainty", "", model, info, NaN, NaN; atol = 1e-6)
end

# A valid row couples a bivariate and a univariate declaration.
function valid_row_across_declarations()
    model = Model()
    @variable(model, buy >= 0)
    @variable(model, sell[1:3] >= 0)
    @variable(model, d[1:3])
    @constraint(model, sum(sell) <= buy)
    @constraint(model, sell .<= d)
    valid = @constraint(model, d[1] + d[3] <= 210)
    @objective(model, Max, -10buy + 15sum(sell))
    info = ModelInfo(;
        uncertainties = [
            declare(CORRELATED, d[1], d[2]),
            declare(Distributions.Uniform(80, 120), d[3]),
        ],
        decision_stages = stages(buy => 1),
        valid_rows = Set{MOI.ConstraintIndex}([index(valid)]),
    )
    return Problem("valid_row_across_declarations", "", model, info, NaN, NaN)
end

# A stage-1 uncertainty, observed by the stage-1 decision, and bounds on decisions that observe
# uncertainties.
function stage_one_uncertainty()
    model = Model()
    @variable(model, 0 <= x <= 50)
    @variable(model, 0 <= y <= 200)
    @variable(model, a)
    @variable(model, b)
    @constraint(model, x + y >= a + b)
    @objective(model, Min, 2x + 3y)
    info = ModelInfo(;
        uncertainties = [
            declare(Distributions.Uniform(10, 30), a; stages = [1]),
            declare(Distributions.Uniform(0, 40), b; stages = [2]),
        ],
        decision_stages = stages(x => 1, y => 2),
    )
    return Problem("stage_one_uncertainty", "", model, info, NaN, NaN)
end

"""
    with_breakpoints(p::Problem, name, legacy, breakpoints::Pair...; primal = NaN, dual = NaN)

The problem `p` with the `uncertainty => breakpoints` pairs (uncertainties by name).
"""
function with_breakpoints(p::Problem, name, legacy, breakpoints::Pair...; primal = NaN, dual = NaN)
    points = Dict{MOI.VariableIndex,Union{Int,Vector{Float64}}}(
        index(variable_by_name(p.model, u)) => b for (u, b) in breakpoints
    )
    info = ModelInfo(;
        uncertainties = p.info.uncertainties,
        decision_stages = p.info.decision_stages,
        valid_rows = p.info.valid_rows,
        breakpoints = points,
    )
    return Problem(name, legacy, p.model, info, primal, dual, p.atol)
end

# The nested breakpoints of the previous version's `test_newsvendor_piecewise`, with the values of
# `legacy_values.toml`: the primal increases and the dual decreases.
const NEWSVENDOR_BREAKPOINTS = [
    ([100.0], 465, 1430 / 3),
    ([90.0, 100.0, 110.0], 471.25, 2845 / 6),
    (collect(85.0:5.0:115.0), 471.25, 22655 / 48),
    (collect(82.5:2.5:117.5), 471.328125, 45265 / 96),
]

function newsvendor_breakpoints(k::Int)
    points, primal, dual = NEWSVENDOR_BREAKPOINTS[k]
    name = "newsvendor_breakpoints_$k"
    return with_breakpoints(newsvendor(), name, name, "demand" => points; primal, dual)
end

# A single breakpoint at the critical quantile 760/7 of the optimal `buy`: the primal LDR is
# optimal, 3300/7.
function newsvendor_breakpoint_critical()
    name = "newsvendor_breakpoint_critical"
    p = newsvendor()
    critical = "demand" => [760 / 7]
    return with_breakpoints(p, name, name, critical; primal = 3300 / 7, dual = 10100 / 21)
end

# Breakpoints on the stage-2 demand of the three-stage newsvendor.
function newsvendor_three_stages_breakpoints()
    p = newsvendor_three_stages()
    name = "newsvendor_three_stages_breakpoints"
    return with_breakpoints(p, name, "", "demand2" => [90.0, 100.0, 110.0])
end

# Breakpoints on an uncertainty of the objective, so that the lifting applies to `C` too, and a
# number of equally spaced breakpoints.
function newsvendor_random_price_breakpoints()
    p = newsvendor_random_price()
    name = "newsvendor_random_price_breakpoints"
    return with_breakpoints(p, name, "", "price" => [14.0, 16.0], "demand" => 3)
end

# Breakpoints on an uncertainty of a quadratic objective (`q`) and of a valid row (`d[3]`).
function lifted_edge_cases()
    return [
        with_breakpoints(objective_only_uncertainty(), "objective_only_breakpoints", "", "q" => 2),
        with_breakpoints(
            valid_row_across_declarations(),
            "valid_row_breakpoints",
            "",
            "d[3]" => [95.0, 105.0],
        ),
    ]
end

"""
    edge_cases() -> Vector{Problem}

The edge models of the Phase 1 review, with no known value: the dense oracles are the reference.
The dual of `law_spanning_stages` warns that its bound is not certified.
"""
function edge_cases()
    return [
        law_spanning_stages(),
        objective_only_uncertainty(),
        valid_row_across_declarations(),
        stage_one_uncertainty(),
        lifted_edge_cases()...,
    ]
end

"""
    library() -> Vector{Problem}

Every problem with an optimal primal and dual LDR.
"""
function library()
    return [
        newsvendor(),
        newsvendor(; integer = true),
        newsvendor_random_price(),
        newsvendor_two_products(),
        newsvendor_three_stages(),
        inventory(4; dual = 340),
        inventory(8; dual = 2020 / 3),
        hydro_toy(),
        shared_stock(),
        shared_stock(; valid = false),
        deterministic(),
        deterministic(; bound = true),
        unbounded_support(),
        newsvendor_mv_discrete(),
        newsvendor_discrete_demand(),
        (newsvendor_breakpoints(k) for k in eachindex(NEWSVENDOR_BREAKPOINTS))...,
        newsvendor_breakpoint_critical(),
        newsvendor_three_stages_breakpoints(),
        newsvendor_random_price_breakpoints(),
    ]
end

# The shared stock with the disc `‖d - (100, 100)‖ ≤ radius` as a second-order cone row. Tagged
# `ValidInequality()` with a radius of 30, the disc contains the box [80, 120]², so Ξ and the values
# are those of `shared_stock(; valid = false)`; tagged `Truncate()`, it conditions the law. With
# `cut`, the valid row `d[1] + d[2] <= 215` joins the block of the disc.
function shared_stock_disc(radius; truncate::Bool = false, cut::Bool = false)
    model = Model()
    @variable(model, buy >= 0)
    @variable(model, sell[1:2] >= 0)
    @variable(model, d[1:2])
    @constraint(model, sum(sell) <= buy)
    @constraint(model, sell .<= d)
    @objective(model, Max, -10buy + 15sum(sell))
    disc = index(@constraint(model, [radius; d .- 100] in SecondOrderCone()))
    valid = Set{MOI.ConstraintIndex}(truncate ? [] : [disc])
    if cut
        push!(valid, index(@constraint(model, d[1] + d[2] <= 215)))
    end
    info = ModelInfo(;
        uncertainties = [declare(Distributions.Uniform(80, 120), d[i]) for i in 1:2],
        decision_stages = stages(buy => 1),
        valid_rows = valid,
        truncate_rows = Set{MOI.ConstraintIndex}(truncate ? [disc] : []),
    )
    name = "shared_stock_disc_$(radius)$(truncate ? "_truncate" : "")$(cut ? "_cut" : "")"
    return Problem(name, "", model, info, NaN, NaN)
end

# The three-stage newsvendor with a valid disc of radius 30 around the means of the demands of
# stages 2 and 3, which contains their box: one support block across stages, and the same values.
function newsvendor_three_stages_disc()
    p = newsvendor_three_stages()
    demand2, demand3 = variable_by_name(p.model, "demand2"), variable_by_name(p.model, "demand3")
    disc = @constraint(p.model, [30, demand2 - 100, demand3 - 100] in SecondOrderCone())
    info = ModelInfo(;
        uncertainties = p.info.uncertainties,
        decision_stages = p.info.decision_stages,
        valid_rows = Set{MOI.ConstraintIndex}([index(disc)]),
    )
    return Problem("newsvendor_three_stages_disc", "", p.model, info, 940, 1000)
end

"""
    conic_library() -> Vector{Problem}

The problems whose support has second-order cone rows, for a conic solver.
"""
function conic_library()
    p = newsvendor_three_stages_disc()
    name = "newsvendor_three_stages_disc_breakpoints"
    return [
        shared_stock_disc(30),
        shared_stock_disc(20; truncate = true),
        shared_stock_disc(20; truncate = true, cut = true),
        p,
        with_breakpoints(p, name, "", "demand2" => [90.0, 100.0, 110.0]),
    ]
end

end # module Problems
