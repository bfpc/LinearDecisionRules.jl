# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestExperimental

using JuMP
using Test

import Distributions
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
    MvDiscreteNonParametric,
    PrimalLDR,
    RescaleUncertainties,
    SampledLDR,
    Stage,
    Uncertainty,
    decision_rule,
    is_certified,
    rule_matrix,
    simulate

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
    @test occursin("(d[1], d[2])[2]", sprint(show, m))
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

# A combination of uncertainties of two stages. Its pieces are not affine in the earlier
# uncertainty, so the dual relaxes its information structure on their law (`prop:relaxed-mask`).
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
    # The dual bound is certified, with no warning, and above the value of the primal rule.
    model = across_stages([PrimalLDR(), DualLDR()])
    @test_logs optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test termination_status(model, DualLDR()) == MOI.OPTIMAL
    @test is_certified(model)
    @test objective_value(model) <= objective_bound(model) + 1e-6
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

# A joint law of `d[1], d[2]` (stage 2) and `d[3] = d[1] + d[2] ± 10` (stage 3), whose conditional
# expectations are linear, and a combination of the uncertainties `combination` of one stage.
function joint_law_over_stages(combination, stage_of)
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    atoms = [[a, b, a + b + e] for a in (40.0, 60.0) for b in (40.0, 60.0) for e in (-10.0, 10.0)]
    law = MvDiscreteNonParametric(atoms, fill(1 / 8, 8))
    @variable(model, d[i = 1:3] in Uncertainty(law), Stage(stage_of[i]))
    @variable(model, buy >= 0, Stage(1))
    @variable(model, sell[i = 1:3] >= 0, Stage(stage_of[i]))
    @constraint(model, sell .<= d)
    @constraint(model, sum(sell) <= buy)
    @objective(model, Max, -10buy + 15sum(sell))
    Experimental.set_breakpoints(model, combination(d), 1)
    return model
end

# The law of the inputs of a combination of one stage may span several stages: the bound is
# certified when no coordinate of that law comes before the combination.
function test_joint_law_over_stages()
    model = joint_law_over_stages(d -> d[1] + d[2], [2, 2, 3])
    @test_logs optimize!(model)
    @test termination_status(model, DualLDR()) == MOI.OPTIMAL
    @test is_certified(model)
    @test objective_value(model) <= objective_bound(model) + 1e-6
    # `d[1]` comes before the combination `d[2] + d[3]`: the expectation of its pieces given `d[1]`
    # is not known to be linear, so the dual relaxes its information structure, and the bound is
    # still certified.
    model = joint_law_over_stages(d -> d[2] + d[3], [2, 3, 3])
    @test_logs optimize!(model)
    @test termination_status(model, DualLDR()) == MOI.OPTIMAL
    @test is_certified(model)
    @test objective_value(model) <= objective_bound(model) + 1e-6
    return
end

# The counterexample of `prop:relaxed-mask`: `x ≥ |e2|` with `e2` uniform on {-1, 0, 1} (stage 2)
# and breakpoints on `e2 + e3`, `e3 = ±1` (stage 3). The optimum is `E|e2| = 2/3`. The usual mask
# would give 12/17, above it, since the hull of the pieces acts as a nonlinear function of `e2`;
# the relaxed dual gives 2/3.
function test_relaxed_dual_counterexample()
    for rescale in (true, false)
        model = LDRModel(HiGHS.Optimizer)
        set_silent(model)
        set_attribute(model, RescaleUncertainties(), rescale)
        law2 = Distributions.DiscreteNonParametric([-1.0, 0.0, 1.0], fill(1 / 3, 3))
        law3 = Distributions.DiscreteNonParametric([-1.0, 1.0], [0.5, 0.5])
        @variable(model, e2 in Uncertainty(law2), Stage(2))
        @variable(model, e3 in Uncertainty(law3), Stage(3))
        @variable(model, x, Stage(2))
        @variable(model, y >= 0, Stage(3))
        @constraint(model, x >= e2)
        @constraint(model, x >= -e2)
        @objective(model, Min, x + y)
        Experimental.set_breakpoints(model, e2 + e3, [0.5])
        optimize!(model)
        @test objective_value(model) ≈ 1.0
        @test objective_bound(model) ≈ 2 / 3
        @test is_certified(model)
    end
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

include(joinpath(@__DIR__, "..", "problems", "tagged.jl"))

# The inventory of the benchmarks over T = 4 periods, whose linear rules cost 380, with values
# measured with this implementation. The demands of two consecutive periods improve the primal rule;
# the cumulative demand, the natural state of the problem, does not: the outer approximation of the
# hull of a combination of many uncertainties is loose. The breakpoints of each demand
# (`BreakPoints`) do better than both.
function test_inventory()
    function primal(; kwargs...)
        model = TaggedProblems.ldr_model(; formulations = [PrimalLDR()])
        TaggedProblems.inventory(4, model; kwargs...)
        optimize!(model)
        return objective_value(model)
    end
    @test primal() ≈ 380
    @test primal(; combinations = (:pair, 1)) ≈ 1120 / 3
    @test primal(; combinations = (:cumulative, 1)) ≈ 380
    @test primal(; points = 3) ≈ 365.625
    return
end

end # module TestExperimental

TestExperimental.runtests()
