# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestResults

using JuMP
using Test

import Distributions
import Distributions: Uniform
import HiGHS
import LinearDecisionRules
import LinearDecisionRules:
    DualLDR,
    FirstStage,
    Formulations,
    FormulationAttribute,
    LDRModel,
    PrimalLDR,
    SampledLDR,
    Uncertainty

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

function error_of(f)
    try
        f()
    catch err
        return err
    end
    return nothing
end

function newsvendor(; formulations = nothing)
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    if formulations !== nothing
        set_attribute(model, Formulations(), formulations)
    end
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @variable(model, d in Uncertainty(Uniform(80, 120)))
    @constraint(model, sell + ret <= buy)
    @constraint(model, sell <= d)
    @objective(model, Max, -10buy + 8ret + 15sell)
    return model
end

function infeasible_model(; formulations = nothing)
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    if formulations !== nothing
        set_attribute(model, Formulations(), formulations)
    end
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, d in Uncertainty(Uniform(80, 120)))
    @constraint(model, sell <= d)
    @constraint(model, sell + buy <= -1)
    @objective(model, Max, sell - buy)
    return model
end

function unbounded_model()
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, d in Uncertainty(Uniform(80, 120)))
    @constraint(model, sell <= d)
    @objective(model, Max, sell + buy)
    return model
end

function test_before_optimize()
    model = newsvendor()
    status = termination_status(model)
    @test status == MOI.OPTIMIZE_NOT_CALLED
    @test_logs (:warn, r"has been modified since the last call") begin
        @test_throws OptimizeNotCalled termination_status(model, DualLDR())
    end
    return
end

function test_newsvendor_results()
    model = newsvendor()
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test primal_status(model) == MOI.FEASIBLE_POINT
    @test dual_status(model) == MOI.NO_SOLUTION
    @test result_count(model) == 1
    @test has_values(model)
    @test objective_value(model) ≈ 460
    @test objective_bound(model) ≈ 1460 / 3
    @test relative_gap(model) ≈ (1460 / 3 - 460) / 460
    @test solve_time(model) > 0
    @test occursin("PrimalLDR(): kHighsModelStatusOptimal", raw_status(model))
    @test occursin("DualLDR(): kHighsModelStatusOptimal", raw_status(model))
    # E[x(ξ)] for the decisions, E[d] for the uncertainty.
    @test value(model[:buy]) ≈ 120
    @test value(model[:sell]) ≈ 100
    @test value(model[:d]) ≈ 100
    @test value(model[:ret]) ≈ 20
    @test value(-10 * model[:buy] + 15 * model[:sell]) ≈ 300
    return
end

function test_results_of_one_formulation()
    model = newsvendor()
    optimize!(model)
    @test termination_status(model, PrimalLDR()) == MOI.OPTIMAL
    @test termination_status(model, DualLDR()) == MOI.OPTIMAL
    @test primal_status(model, PrimalLDR()) == MOI.FEASIBLE_POINT
    @test primal_status(model, DualLDR()) == MOI.NO_SOLUTION
    @test objective_value(model, PrimalLDR()) ≈ 460
    @test objective_value(model, DualLDR()) ≈ 1460 / 3
    @test objective_bound(model, DualLDR()) ≈ 1460 / 3
    @test raw_status(model, DualLDR()) == "kHighsModelStatusOptimal"
    @test solve_time(model, PrimalLDR()) >= 0
    # A formulation that was not solved.
    @test termination_status(model, SampledLDR(10)) == MOI.OPTIMIZE_NOT_CALLED
    err = error_of(() -> objective_value(model, SampledLDR(10)))
    @test err isa MOI.GetAttributeNotAllowed
    @test occursin("add it to `Formulations()`", sprint(showerror, err))
    attr = FormulationAttribute(PrimalLDR(), MOI.ObjectiveValue())
    @test MOI.get(model, attr) ≈ 460
    return
end

function test_sampled_is_result_two()
    model = newsvendor(; formulations = [DualLDR(), SampledLDR(200), PrimalLDR()])
    optimize!(model)
    @test result_count(model) == 2
    @test primal_status(model; result = 1) == MOI.FEASIBLE_POINT
    @test primal_status(model; result = 2) == MOI.UNKNOWN_RESULT_STATUS
    @test primal_status(model; result = 3) == MOI.NO_SOLUTION
    @test objective_value(model; result = 1) ≈ 460
    # The rule of the sample is feasible on the sample only: it can do better.
    @test objective_value(model; result = 2) >= 460 - 1e-6
    @test objective_value(model, SampledLDR(200)) == objective_value(model; result = 2)
    @test value(model[:buy]; result = 1) ≈ 120
    @test isfinite(value(model[:buy]; result = 2))
    @test_throws MOI.ResultIndexBoundsError value(model[:buy]; result = 3)
    return
end

function test_dual_only()
    model = newsvendor(; formulations = [DualLDR()])
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test result_count(model) == 0
    @test primal_status(model) == MOI.NO_SOLUTION
    @test objective_bound(model) ≈ 1460 / 3
    @test objective_value(model, DualLDR()) ≈ 1460 / 3
    @test_throws MOI.ResultIndexBoundsError objective_value(model)
    @test_throws MOI.ResultIndexBoundsError value(model[:buy])
    return
end

function test_primal_only_has_no_bound()
    model = newsvendor(; formulations = [PrimalLDR()])
    optimize!(model)
    @test objective_value(model) ≈ 460
    err = error_of(() -> objective_bound(model))
    @test err isa MOI.GetAttributeNotAllowed
    @test occursin("DualLDR()", sprint(showerror, err))
    @test_throws MOI.GetAttributeNotAllowed relative_gap(model)
    return
end

function test_infeasible_model()
    model = infeasible_model()
    optimize!(model)
    # Only the dual proves it.
    @test termination_status(model) == MOI.INFEASIBLE
    @test result_count(model) == 0
    @test primal_status(model) == MOI.NO_SOLUTION
    @test termination_status(model, DualLDR()) == MOI.INFEASIBLE
    @test_throws MOI.ResultIndexBoundsError objective_value(model)
    @test_throws MOI.ResultIndexBoundsError value(model[:buy])
    @test_throws MOI.GetAttributeNotAllowed objective_bound(model)
    @test_throws MOI.GetAttributeNotAllowed objective_value(model, PrimalLDR())
    @test_throws MOI.GetAttributeNotAllowed objective_value(model, DualLDR())
    return
end

function test_infeasible_primal_alone_is_locally_infeasible()
    for formulations in ([PrimalLDR()], [SampledLDR(10)], [PrimalLDR(), SampledLDR(10)])
        model = infeasible_model(; formulations)
        optimize!(model)
        @test termination_status(model) == MOI.LOCALLY_INFEASIBLE
        @test result_count(model) == 0
    end
    return
end

function test_unbounded_model()
    model = unbounded_model()
    optimize!(model)
    @test termination_status(model) in (MOI.DUAL_INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
    @test result_count(model) == 0
    @test primal_status(model) == MOI.NO_SOLUTION
    # The ray of the inner problem is not a result.
    @test_throws MOI.ResultIndexBoundsError objective_value(model)
    @test_throws MOI.ResultIndexBoundsError value(model[:buy])
    @test_throws MOI.GetAttributeNotAllowed objective_bound(model)
    @test_throws MOI.GetAttributeNotAllowed objective_value(model, DualLDR())
    return
end

function test_solution_summary()
    for model in (newsvendor(), infeasible_model(), unbounded_model())
        optimize!(model)
        text = sprint(show, solution_summary(model))
        @test occursin("termination_status", text)
        text = sprint(show, solution_summary(model; verbose = true))
        @test occursin("termination_status", text)
    end
    model = newsvendor()
    optimize!(model)
    summary = solution_summary(model; verbose = true)
    @test summary.objective_value ≈ 460
    @test summary.objective_bound ≈ 1460 / 3
    @test summary.result_count == 1
    @test summary.primal_status == MOI.FEASIBLE_POINT
    @test summary.has_values
    text = sprint(show, summary)
    @test occursin("buy", text) && occursin("sell", text)
    return
end

function test_minimization()
    values = Float64[]
    for sense in (MOI.MAX_SENSE, MOI.MIN_SENSE)
        model = LDRModel(HiGHS.Optimizer)
        set_silent(model)
        @variable(model, buy >= 0, FirstStage)
        @variable(model, sell >= 0)
        @variable(model, d in Uncertainty(Uniform(80, 120)))
        @constraint(model, sell <= d)
        @constraint(model, sell <= buy)
        sign = sense == MOI.MAX_SENSE ? 1 : -1
        set_objective(model, sense, sign * (15sell - 10buy))
        optimize!(model)
        push!(values, objective_value(model), objective_bound(model))
        # The primal is on the right side of the bound, in each sense.
        @test (objective_bound(model) - objective_value(model)) * sign >= -1e-9
    end
    @test values[3] ≈ -values[1]
    @test values[4] ≈ -values[2]
    return
end

function test_feasibility_problem()
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x <= 3)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) == 0
    @test value(x) <= 3
    return
end

end # module TestResults

TestResults.runtests()
