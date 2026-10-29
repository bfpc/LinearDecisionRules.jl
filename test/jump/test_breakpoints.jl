# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Models with `BreakPoints`, written as a user would and solved end to end.

module TestBreakpoints

using JuMP
using Test

import HiGHS
import LinearDecisionRules:
    AxisPiecewise,
    BreakPoints,
    DualLDR,
    Formulations,
    LDRModel,
    PrimalLDR,
    SampledLDR,
    decision_rule,
    expected_value,
    moments,
    solve_formulation

include(joinpath(@__DIR__, "..", "problems", "library.jl"))

include(joinpath(@__DIR__, "..", "common.jl"))
import .TestCommon: error_message

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

include(joinpath(@__DIR__, "..", "problems", "tagged.jl"))

ldr_model() = TaggedProblems.ldr_model(; time_limit = 60.0)

newsvendor() = TaggedProblems.newsvendor(ldr_model())
newsvendor_random_price() = TaggedProblems.newsvendor_random_price(ldr_model())
newsvendor_three_stages() = TaggedProblems.newsvendor_three_stages(ldr_model())

# The value of the canonical problem of the library, solved directly.
function library_values(p)
    form = Problems.form(p)
    mom = moments(form)
    primal = solve_formulation(form, PrimalLDR(), mom, HiGHS.Optimizer; time_limit = 60.0)
    dual = solve_formulation(form, DualLDR(), mom, HiGHS.Optimizer; time_limit = 60.0)
    return primal.objective_value, dual.objective_value
end

# At the critical quantile 760/7 of `buy`, the primal LDR is the optimal policy, 3300/7.
function test_critical_breakpoint()
    model = newsvendor()
    buy, sell, ret, demand = model[:buy], model[:sell], model[:ret], model[:demand]
    set_attribute(demand, BreakPoints(), [760 / 7])
    optimize!(model)
    @test objective_value(model) ≈ 3300 / 7
    # The rules are piecewise affine in the demand: sell = min(demand, 760/7).
    rule = decision_rule(sell)
    @test isempty(rule.coefficients)
    term = rule.lifted[demand]
    @test term.lifting == AxisPiecewise([760 / 7])
    @test term.coefficients ≈ [1, 0] atol = 1e-9
    @test rule(demand => 90.0) ≈ 90
    @test rule(Dict(demand => 115.0)) ≈ 760 / 7
    @test term(115.0) ≈ 760 / 7
    @test decision_rule(ret)(demand => 90.0) ≈ 760 / 7 - 90
    @test occursin("⋅ lift(AxisPiecewise([108.57142857142857]), demand)", sprint(show, rule))
    # The rule of an uncertainty is itself, even lifted.
    @test decision_rule(demand).coefficients == Dict(demand => 1.0)
    @test isempty(decision_rule(demand).lifted)
    # `value` and `expected_value` use the moments of the pieces, which are exact.
    b = 760 / 7
    E_sell = ((b^2 - 80^2) / 2 + b * (120 - b)) / 40
    @test value(sell) ≈ E_sell
    @test value(demand) ≈ 100
    @test value(buy) ≈ b
    @test expected_value(sell) ≈ E_sell
    @test expected_value(demand * sell) ≈ ((b^3 - 80^3) / 3 + b * (120^2 - b^2) / 2) / 40
    @test expected_value(-10buy + 8ret + 15sell) ≈ 3300 / 7
    return
end

# Breakpoints as a number of equally spaced breakpoints, or as a vector of integers.
function test_breakpoint_types()
    for points in (3, [110, 90, 100], 90:10:110, [90.0, 100.0, 110.0])
        model = newsvendor()
        set_attribute(model[:demand], BreakPoints(), points)
        optimize!(model)
        @test objective_value(model) ≈ 471.25
        @test objective_bound(model) ≈ 2845 / 6
        term = decision_rule(model[:sell]).lifted[model[:demand]]
        @test term.lifting == AxisPiecewise([90, 100, 110])
        @test length(term.coefficients) == 4
    end
    for points in (0, Float64[], nothing)
        model = newsvendor()
        set_attribute(model[:demand], BreakPoints(), points)
        optimize!(model)
        @test objective_value(model) ≈ 460
        @test isempty(decision_rule(model[:sell]).lifted)
    end
    return
end

# Clearing or changing the breakpoints after a solve: the next solve reads them again.
function test_change_breakpoints_after_solve()
    model = newsvendor()
    demand = model[:demand]
    set_attribute(demand, BreakPoints(), [760 / 7])
    optimize!(model)
    @test objective_value(model) ≈ 3300 / 7
    set_attribute(demand, BreakPoints(), nothing)
    @test termination_status(model) == MOI.OPTIMIZE_NOT_CALLED
    optimize!(model)
    @test objective_value(model) ≈ 460
    @test objective_bound(model) ≈ 1460 / 3
    @test isempty(decision_rule(model[:sell]).lifted)
    @test decision_rule(model[:sell]).coefficients[demand] ≈ 1
    set_attribute(demand, BreakPoints(), [100.0])
    optimize!(model)
    @test objective_value(model) ≈ 465
    set_attribute(demand, BreakPoints(), Float64[])
    optimize!(model)
    @test objective_value(model) ≈ 460
    return
end

# The lifting applies to the uncertainties of the objective: the price multiplies `sell`.
function test_random_price()
    model = newsvendor_random_price()
    buy, sell, ret, price = model[:buy], model[:sell], model[:ret], model[:price]
    set_attribute(price, BreakPoints(), [14.0, 16.0])
    set_attribute(model[:demand], BreakPoints(), 3)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    primal, dual = library_values(Problems.newsvendor_random_price_breakpoints())
    @test objective_value(model) ≈ primal
    @test objective_bound(model) ≈ dual
    # The objective value is the expected profit of the rules, with the exact moments of the
    # pieces of the price.
    @test expected_value(-10buy + 8ret + price * sell) ≈ objective_value(model)
    @test 460 - 1e-6 <= objective_value(model) <= objective_bound(model) + 1e-6
    @test objective_bound(model) <= 1460 / 3 + 1e-6
    return
end

# Breakpoints on the stage-2 demand of three stages: the rules of stage 2 do not see the stage-3
# demand, the pieces are seen from stage 2 on, and the bounds are ordered: the primal improves on
# 940 (to 946.25) and the dual is at most 1000.
function test_three_stages()
    model = newsvendor_three_stages()
    demand2, demand3 = model[:demand2], model[:demand3]
    set_attribute(demand2, BreakPoints(), [90.0, 100.0, 110.0])
    # No warning: the pieces of `demand2` are in its stage, so the dual bound is certified.
    @test_logs optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    primal, dual = objective_value(model), objective_bound(model)
    @test 940 - 1e-6 <= primal <= dual + 1e-6
    @test primal ≈ 946.25
    @test dual <= 1000 + 1e-6
    library = library_values(Problems.newsvendor_three_stages_breakpoints())
    @test primal ≈ library[1] && dual ≈ library[2]
    for name in (:buy, :sell2, :ret2, :buy2)
        rule = decision_rule(model[name])
        @test !haskey(rule.coefficients, demand3)
        name == :buy && @test isempty(rule.lifted) && isempty(rule.coefficients)
    end
    @test haskey(decision_rule(model[:sell2]).lifted, demand2)
    for name in (:sell3, :ret3), f in (PrimalLDR(), DualLDR())
        @test !haskey(decision_rule(model[name]; formulation = f).coefficients, demand2)
    end
    # The values are stable: a second solve gives the same rules.
    rule = decision_rule(model[:sell2])
    optimize!(model)
    @test objective_value(model) ≈ primal
    scenario = Dict(demand2 => 95.0, demand3 => 100.0, model[:buy_cost2] => 10.0)
    @test decision_rule(model[:sell2])(scenario) ≈ rule(scenario)
    return
end

function test_sampled()
    model = newsvendor()
    set_attribute(model[:demand], BreakPoints(), [760 / 7])
    set_attribute(model, Formulations(), [PrimalLDR(), SampledLDR(200)])
    optimize!(model)
    @test result_count(model) == 2
    @test primal_status(model; result = 2) == MOI.UNKNOWN_RESULT_STATUS
    # The outcomes are drawn from the law of the pieces, so the sampled rule is near 3300/7.
    @test objective_value(model; result = 2) ≈ 3300 / 7 rtol = 0.02
    rule = decision_rule(model[:sell]; formulation = SampledLDR(200))
    @test haskey(rule.lifted, model[:demand])
    return
end

function test_errors()
    model = newsvendor()
    set_attribute(model[:buy], BreakPoints(), [100.0])
    msg = error_message(() -> optimize!(model))
    @test occursin("buy has breakpoints, but it is not an uncertainty", msg)
    model = newsvendor()
    set_attribute(model[:demand], BreakPoints(), [70.0, 100.0])
    msg = error_message(() -> optimize!(model))
    @test occursin("The lifting of the uncertainty demand is invalid", msg)
    @test occursin("strictly inside the support [80.0, 120.0]", msg)
    model = newsvendor()
    set_attribute(model[:demand], BreakPoints(), "100")
    msg = error_message(() -> optimize!(model))
    @test occursin("must be an integer (their number) or a vector of numbers", msg)
    return
end

end # module TestBreakpoints

TestBreakpoints.runtests()
