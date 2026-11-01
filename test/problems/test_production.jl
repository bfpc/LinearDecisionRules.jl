# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestProduction

using JuMP
using Test

import HiGHS
import LinearDecisionRules as LDR

# The production planning of Georghiou, Kuhn and Wiesemann (2019), section 6.1: see
# `production.jl`. The paper reports its values only in plots (Figure 8), which this
# implementation does not match closely, so the tests check the exact optimum at T = 2 and
# values measured with this implementation instead. The model is badly scaled: HiGHS fails on the
# dual of some instances as small as T = 4, so the tests stay at T ≤ 3.
include(joinpath(@__DIR__, "production.jl"))

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$name" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function _solve(T; kwargs...)
    model = production_model(T; formulations = [LDR.PrimalLDR(), LDR.DualLDR()], kwargs...)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test LDR.is_certified(model)
    return objective_value(model), objective_bound(model)
end

# At T = 2 the production p of week 1 is the only decision that matters: the best sales of week 2
# are s = min(p, b + ξ) for each product, with b the backlog of week 1. With ξ uniform on
# [lo, hi], the expected sales are p up to b + lo, then concave quadratic up to b + hi, then
# constant: with p = y + z + w, y ≤ b + lo, z ≤ hi - lo, they are y + z - z² / 2(hi - lo). The
# exact optimum is then a small concave QP.
function _two_weeks_optimum()
    data = PRODUCTION_DATA
    products = eachindex(data.average_demand)
    b = [production_nominal(1, i) for i in products]
    lo = [0.8 * production_nominal(2, i) for i in products]
    hi = [1.2 * production_nominal(2, i) for i in products]
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= y[i = products] <= b[i] + lo[i])
    @variable(model, 0 <= z[i = products] <= hi[i] - lo[i])
    @variable(model, w[products] >= 0)
    @expression(model, p[i = products], y[i] + z[i] + w[i])
    @expression(model, sales[i = products], y[i] + z[i] - z[i]^2 / (2 * (hi[i] - lo[i])))
    @constraint(model, sum(p[i] / data.rate[i] for i in products) <= data.uptime)
    # Week 1 sells nothing and backlogs b; week 2 backlogs b + E[ξ] - sales and holds p - sales.
    @objective(
        model,
        Max,
        sum(
            data.price[i] * sales[i] -
            data.backlog_cost[i] * (2 * b[i] + (lo[i] + hi[i]) / 2 - sales[i]) -
            data.holding_cost * (p[i] - sales[i]) for i in products
        )
    )
    optimize!(model)
    @test is_solved_and_feasible(model)
    return objective_value(model)
end

function test_two_weeks()
    optimum = _two_weeks_optimum()
    @test optimum ≈ 78091.8 atol = 0.1
    value, bound = _solve(2)
    @test value < optimum < bound
    @test value ≈ 76327.768881 rtol = 1e-7
    @test bound ≈ 79351.768881 rtol = 1e-7
    value1, bound1 = _solve(2; breakpoints = 1)
    @test value < value1 < optimum < bound1 < bound
    @test value1 ≈ 77893.631181 rtol = 1e-7
    @test bound1 ≈ 78883.677081 rtol = 1e-7
    return
end

# Breakpoints tighten both sides; the values were measured with this implementation.
function test_three_weeks()
    value, bound = _solve(3)
    value1, bound1 = _solve(3; breakpoints = 1)
    @test value < value1 < bound1 < bound
    @test value ≈ 150599.154825 rtol = 1e-7
    @test bound ≈ 157538.984905 rtol = 1e-7
    @test value1 ≈ 154603.707679 rtol = 1e-7
    @test bound1 ≈ 157113.201756 rtol = 1e-7
    return
end

end # module TestProduction

TestProduction.runtests()
