# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestHydro

using JuMP
using Test

import LinearDecisionRules as LDR

include(joinpath(@__DIR__, "..", "common.jl"))

# The hydrothermal problem of `hydro.jl` (also the benchmark of benchmark/hydro): 4 reservoirs,
# 5 subsystems, T = 3 stages (2 uncertain vectors of 4 inflows, 82 joint scenarios each).
include(joinpath(@__DIR__, "hydro.jl"))

import .TestCommon: model_size

const T = 3

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

function _solve(data, formulation)
    model = ldr_model(data, T; formulations = [formulation])
    optimize!(model)
    return model, LDR.inner_model(model, formulation)
end

# Values and sizes were measured with this implementation at T = 3 (benchmark/results/hydro.csv
# has T = 4, 8 and 12 only): the primal LDR is an upper bound on the optimum, the dual LDR and the
# average-scenario model are lower bounds (the inflows enter only the right-hand sides, so the value
# is convex in them), and here the dual bound is the average-scenario value.
function test_ldr_values_and_sizes()
    data = hydro_thermal_data()
    primal, primal_inner = _solve(data, LDR.PrimalLDR())
    dual, dual_inner = _solve(data, LDR.DualLDR())
    average = average_scenario_model(data, T)
    optimize!(average)
    @test termination_status(primal) == MOI.OPTIMAL
    @test termination_status(dual) == MOI.OPTIMAL
    @test objective_value(primal) ≈ 1.888358847336 rtol = 1e-6
    @test objective_bound(dual) ≈ 0.7352371632 rtol = 1e-6
    @test objective_value(average) ≈ 0.7352371632 rtol = 1e-6
    @test objective_bound(dual) <= objective_value(primal)
    @test objective_value(average) <= objective_value(primal)
    @test model_size(primal_inner) == (variables = 5724, constraints = 4223, nonzeros = 17129)
    @test model_size(dual_inner) == (variables = 6892, constraints = 11815, nonzeros = 50417)
    return
end

# The inflows are boxes: the closed form of the primal has one auxiliary per (row, inflow) instead
# of two multipliers, with the same value.
function test_box_closed_form()
    data = hydro_thermal_data()
    closed, _ = _solve(data, LDR.PrimalLDR())
    general, general_inner = _solve(data, LDR.PrimalLDR(; box_closed_form = false))
    @test termination_status(general) == MOI.OPTIMAL
    @test objective_value(general) ≈ objective_value(closed) rtol = 1e-7
    @test model_size(general_inner) == (variables = 9228, constraints = 4223, nonzeros = 20633)
    return
end

function test_average_scenario_and_tree_sizes()
    data = hydro_thermal_data()
    average = average_scenario_model(data, T)
    @test model_size(average) == (variables = 444, constraints = 27, nonzeros = 509)
    # 13 nodes (1 + 3 + 9), each with the 148 variables of a stage.
    tree = scenario_tree_model(data, T, 3)
    @test num_variables(tree) == 13 * 148
    optimize!(tree)
    @test termination_status(tree) == MOI.OPTIMAL
    @test objective_value(tree) ≈ 0.741309440590 rtol = 1e-6
    return
end

end # module TestHydro

TestHydro.runtests()
