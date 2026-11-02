# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestExperimental

using JuMP
using Test

import Distributions: Uniform
import HiGHS
import Random
import LinearDecisionRules:
    BreakPoints,
    CombinationBreakPoints,
    CombinationPiecewise,
    CombinationTerm,
    DualLDR,
    Experimental,
    Formulations,
    LDRModel,
    PrimalLDR,
    SampledLDR,
    Stage,
    Uncertainty,
    decision_rule,
    is_certified,
    rule_matrix,
    simulate

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

# Two demands served from one stock (see `test_combination_piecewise.jl`), whose optimal stock is
# `POOLED_STOCK`, with breakpoints on the total demand.
const POOLED_STOCK = 160 + sqrt(3200 / 3)

const POOLED_OPTIMUM = -10POOLED_STOCK + 15 * (POOLED_STOCK - (POOLED_STOCK - 160)^3 / 9600)

function pooled(points; formulations = [PrimalLDR(), DualLDR()])
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    set_attribute(model, Formulations(), formulations)
    @variable(model, d[1:2] in Uncertainty(Uniform(80, 120)), Stage(2))
    @variable(model, buy >= 0, Stage(1))
    @variable(model, sell[1:2] >= 0, Stage(2))
    @constraint(model, sell .<= d)
    @constraint(model, stock, sum(sell) <= buy)
    @objective(model, Max, -10buy + 15sum(sell))
    Experimental.set_breakpoints(model, d[1] + d[2], points)
    return model
end

function entries(model)
    return MOI.get(backend(model), CombinationBreakPoints())
end

function test_set_breakpoints()
    model = pooled(3)
    d = model[:d]
    @test length(entries(model)) == 1
    @test last(only(entries(model))) == 3
    # The same combination, in another order, replaces the breakpoints.
    Experimental.set_breakpoints(model, d[2] + d[1], [190.0, 210.0])
    @test last(only(entries(model))) == [190.0, 210.0]
    Experimental.set_breakpoints(model, 2d[1], 1)
    @test length(entries(model)) == 2
    for none in (nothing, 0, Float64[])
        Experimental.set_breakpoints(model, d[1] + d[2], 2)
        Experimental.set_breakpoints(model, d[1] + d[2], none)
        @test length(entries(model)) == 1
    end
    @test_throws ArgumentError Experimental.set_breakpoints(model, d[1], "three")
    return
end

function test_values()
    model = pooled([POOLED_STOCK])
    optimize!(model)
    @test objective_value(model) ≈ POOLED_OPTIMUM
    @test objective_bound(model) ≈ 954.5213797275389
    @test is_certified(model)
    @test value(model[:buy]) ≈ POOLED_STOCK
    return
end

# With the breakpoint at the optimal stock, the total sales of the optimal rule are
# `min(stock, d₁ + d₂)` at every scenario.
function test_decision_rule()
    model = pooled([POOLED_STOCK])
    optimize!(model)
    d, sell = model[:d], model[:sell]
    rules = decision_rule.(sell)
    # The split of the sales between the products is not unique: one rule at least uses the piece.
    with_piece = [r for r in rules if !isempty(r.combinations)]
    @test !isempty(with_piece)
    term = only(first(with_piece).combinations)
    @test term isa CombinationTerm
    @test term.lifting == CombinationPiecewise([1.0, 1.0], [POOLED_STOCK])
    @test term.inputs == [d[1], d[2]]
    @test occursin("CombinationPiecewise", sprint(show, first(with_piece)))
    for x in 80.0:8.0:120.0, y in 80.0:10.0:120.0
        scenario = Dict(d[1] => x, d[2] => y)
        @test sum(r(scenario) for r in rules) ≈ min(POOLED_STOCK, x + y) atol = 1e-6
    end
    m = rule_matrix(model)
    @test ((d[1], d[2]), 2) in m.columns
    return
end

function test_simulate()
    model = pooled([POOLED_STOCK])
    optimize!(model)
    s = simulate(model, 20_000; rng = Random.Xoshiro(1))
    @test s.violation_fraction == 0
    @test abs(s.objective_mean - POOLED_OPTIMUM) <= 4 * s.objective_stderr
    return
end

# A combination of uncertainties of two stages: the primal and the sampled rules, but no dual.
function across_stages(formulations)
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    set_attribute(model, Formulations(), formulations)
    @variable(model, d[t = 2:3] in Uncertainty(Uniform(80, 120)), Stage(t))
    @variable(model, buy >= 0, Stage(1))
    @variable(model, sell[t = 2:3] >= 0, Stage(t))
    @constraint(model, [t = 2:3], sell[t] <= d[t])
    @constraint(model, sell[2] + sell[3] <= buy)
    @objective(model, Max, -10buy + 15sum(sell))
    Experimental.set_breakpoints(model, d[2] + d[3], [200.0])
    return model
end

function test_across_stages()
    model = across_stages([PrimalLDR(), DualLDR()])
    message = error_message(() -> optimize!(model))
    @test occursin("DualLDR() does not support breakpoints on a combination", message)
    @test occursin("d[2], d[3] (stages 2, 3)", message)
    model = across_stages([PrimalLDR(), SampledLDR(50)])
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    # The pieces are in stage 3: `sell[2]` does not observe them.
    m = rule_matrix(model)
    piece = findfirst(==(((model[:d][2], model[:d][3]), 2)), m.columns)
    row = findfirst(==(model[:sell][2]), m.decisions)
    @test m.matrix[row, piece] == 0
    # The rule improves on the linear one.
    linear = across_stages([PrimalLDR()])
    Experimental.set_breakpoints(linear, linear[:d][2] + linear[:d][3], nothing)
    optimize!(linear)
    @test objective_value(model) > objective_value(linear) + 1
    return
end

# Breakpoints on one uncertainty and on a combination of others in the same model.
function test_with_breakpoints()
    model = pooled([POOLED_STOCK])
    @variable(model, price in Uncertainty(Uniform(13, 17)), Stage(2))
    set_attribute(price, BreakPoints(), [15.0])
    @variable(model, extra >= 0, Stage(2))
    @constraint(model, extra <= price - 13)
    set_objective_function(model, objective_function(model) + extra)
    optimize!(model)
    @test objective_value(model) ≈ POOLED_OPTIMUM + 2
    return
end

end # module TestExperimental

TestExperimental.runtests()
