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
    Certified,
    DualLDR,
    FirstStage,
    Formulations,
    FormulationAttribute,
    LDRModel,
    MvDiscreteNonParametric,
    PrimalLDR,
    RejectionSamplingMaxSamples,
    SampledLDR,
    Stage,
    Truncate,
    Uncertainty,
    inner_model,
    is_certified,
    set_stage

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

include(joinpath(@__DIR__, "..", "problems", "tagged.jl"))

function newsvendor(; formulations = nothing)
    return TaggedProblems.newsvendor(TaggedProblems.ldr_model(; formulations))
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

# As in JuMP, statuses do not throw nor warn before `optimize!` or after a change to the model.
function test_before_optimize()
    model = newsvendor()
    for m in (model, LDRModel(HiGHS.Optimizer))
        @test termination_status(m) == MOI.OPTIMIZE_NOT_CALLED
        for f in (PrimalLDR(), DualLDR(), SampledLDR(10))
            @test (@test_logs termination_status(m, f)) == MOI.OPTIMIZE_NOT_CALLED
            @test (@test_logs primal_status(m, f)) == MOI.NO_SOLUTION
            @test (@test_logs raw_status(m, f)) == "optimize not called"
        end
    end
    optimize!(model)
    @test termination_status(model, DualLDR()) == MOI.OPTIMAL
    @variable(model, extra >= 0)
    @test termination_status(model, DualLDR()) == MOI.OPTIMIZE_NOT_CALLED
    @test primal_status(model, PrimalLDR()) == MOI.NO_SOLUTION
    # Values throw, as `objective_value(model)` does.
    @test_logs (:warn,) @test_throws OptimizeNotCalled objective_value(model, DualLDR())
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
    @test value(model[:demand]) ≈ 100
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
    @test primal_status(model, SampledLDR(10)) == MOI.NO_SOLUTION
    @test raw_status(model, SampledLDR(10)) == "optimize not called"
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
    @test result_count(model) == 1
    @test primal_status(model) == MOI.NO_SOLUTION
    @test termination_status(model, DualLDR()) == MOI.INFEASIBLE
    @test_throws MOI.GetAttributeNotAllowed objective_value(model)
    @test_throws MOI.GetAttributeNotAllowed value(model[:buy])
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
        @test result_count(model) == (any(f -> f isa SampledLDR, formulations) ? 2 : 1)
        @test primal_status(model; result = 1) == MOI.NO_SOLUTION
        @test primal_status(model; result = 2) == MOI.NO_SOLUTION
        @test !has_values(model)
    end
    return
end

# Result 1 is the primal rule even when it has no solution, so that the gap and `has_values` are
# about the primal rule, never about the sampled one.
function test_result_slots_are_fixed()
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    set_attribute(model, Formulations(), [SampledLDR(5), DualLDR(), PrimalLDR()])
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, d in Uncertainty(Uniform(80, 120)))
    # Robustly infeasible (buy ≥ 119), but feasible on the 5 outcomes, all below 118.
    @constraint(model, buy >= d - 1)
    @constraint(model, buy <= 118)
    @constraint(model, sell <= d)
    @objective(model, Max, -10buy + 15sell)
    optimize!(model)
    @test termination_status(model, PrimalLDR()) == MOI.INFEASIBLE
    @test termination_status(model, SampledLDR(5)) == MOI.OPTIMAL
    @test termination_status(model) == MOI.LOCALLY_INFEASIBLE
    @test result_count(model) == 2
    @test primal_status(model) == MOI.NO_SOLUTION
    @test primal_status(model; result = 2) == MOI.UNKNOWN_RESULT_STATUS
    @test !has_values(model)
    @test has_values(model; result = 2)
    err = error_of(() -> objective_value(model))
    @test err isa MOI.GetAttributeNotAllowed
    @test occursin("Result 1 is the rule of PrimalLDR, which has no", sprint(showerror, err))
    @test_throws MOI.GetAttributeNotAllowed value(buy)
    @test_throws MOI.GetAttributeNotAllowed relative_gap(model)
    @test objective_value(model; result = 2) == objective_value(model, SampledLDR(5))
    @test isfinite(objective_bound(model))
    # Without the primal, result 1 is still the empty primal slot.
    set_attribute(model, Formulations(), [DualLDR(), SampledLDR(5)])
    optimize!(model)
    @test result_count(model) == 2
    @test primal_status(model) == MOI.NO_SOLUTION
    err = error_of(() -> value(buy))
    @test occursin("PrimalLDR, which was not solved", sprint(showerror, err))
    @test isfinite(value(buy; result = 2))
    return
end

function test_duplicate_formulations()
    model = newsvendor()
    for formulations in ([PrimalLDR(), PrimalLDR()], [SampledLDR(5), DualLDR(), SampledLDR(9)])
        err = error_of(() -> set_attribute(model, Formulations(), formulations))
        @test err isa ArgumentError
        @test occursin("two formulations of type", err.msg)
    end
    @test get_attribute(model, Formulations()) == [PrimalLDR(), DualLDR()]
    return
end

function test_unbounded_model()
    model = unbounded_model()
    optimize!(model)
    @test termination_status(model) in (MOI.DUAL_INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
    @test result_count(model) == 1
    @test primal_status(model) == MOI.NO_SOLUTION
    # The ray of the inner problem is not a result.
    @test_throws MOI.GetAttributeNotAllowed objective_value(model)
    @test_throws MOI.GetAttributeNotAllowed value(model[:buy])
    # An unbounded dual LDR has no bound but the trivial one.
    if termination_status(model, DualLDR()) == MOI.DUAL_INFEASIBLE
        @test objective_bound(model) == Inf
        @test objective_value(model, DualLDR()) == Inf
    else
        @test_throws MOI.GetAttributeNotAllowed objective_bound(model)
    end
    return
end

# An inner MIP stopped before an incumbent still proves a bound (here, the trivial one).
function test_bound_without_incumbent()
    model = newsvendor(; formulations = [PrimalLDR(), DualLDR()])
    set_integer(model[:buy])
    for (name, value) in ("mip_max_nodes" => 0, "mip_heuristic_effort" => 0.0, "presolve" => "off")
        set_attribute(model, name, value)
    end
    optimize!(model)
    @test termination_status(model, DualLDR()) == MOI.SOLUTION_LIMIT
    @test primal_status(model, DualLDR()) == MOI.NO_SOLUTION
    @test objective_bound(model, DualLDR()) == Inf
    @test objective_value(model, DualLDR()) == Inf
    @test objective_bound(model) == Inf
    @test_throws MOI.GetAttributeNotAllowed objective_value(model)
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

# `is_certified` and `Certified()`: exact values are certified; values that use moments estimated by
# rejection sampling are not, nor is a bound whose law spans stages without linear conditional
# expectations, and the raw status gives the reason.
function test_is_certified()
    @test_logs (:warn,) @test_throws OptimizeNotCalled is_certified(newsvendor())
    model = newsvendor(; formulations = [PrimalLDR(), DualLDR(), SampledLDR(20)])
    optimize!(model)
    @test is_certified(model) === true
    @test is_certified(model, PrimalLDR()) === true
    @test get_attribute(model, FormulationAttribute(DualLDR(), Certified())) === true
    # Estimates by construction, with no reason.
    @test is_certified(model, SampledLDR(20)) === false
    @test raw_status(model, SampledLDR(20)) == "kHighsModelStatusOptimal"
    # The attribute is read for one formulation, which must have been solved.
    @test error_of(() -> get_attribute(model, Certified())) isa MOI.GetAttributeNotAllowed
    model = newsvendor()
    optimize!(model)
    err = error_of(() -> is_certified(model, SampledLDR(20)))
    @test err isa MOI.GetAttributeNotAllowed
    @test occursin("add it to `Formulations()`", sprint(showerror, err))
    # Two demands conditioned on their total: moments estimated by rejection sampling.
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    set_attribute(model, RejectionSamplingMaxSamples(), 1_000)
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell[1:2] >= 0)
    @variable(model, d[1:2] in Uncertainty(Uniform(80, 120)))
    @constraint(model, d[1] + d[2] <= 200, Truncate())
    @constraint(model, sum(sell) <= buy)
    @constraint(model, sell .<= d)
    @objective(model, Max, -10buy + 15sum(sell))
    optimize!(model)
    reason = "[not certified: moments of conditioned laws estimated by rejection sampling]"
    for f in (PrimalLDR(), DualLDR())
        @test is_certified(model, f) === false
        @test endswith(raw_status(model, f), reason)
    end
    # A law over stages 2 and 3, uncorrelated, whose second component is a function of the first.
    law = MvDiscreteNonParametric([[79.0, 61.0], [80.0, 60.0], [81.0, 61.0]], fill(1 / 3, 3))
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, buy >= 0, FirstStage)
    @variable(model, early >= 0, Stage(2))
    @variable(model, late >= 0, Stage(3))
    @variable(model, d[1:2] in Uncertainty(law))
    set_stage.(d, [2, 3])
    @constraint(model, early + late <= buy)
    @constraint(model, early <= d[1])
    @constraint(model, late <= d[2])
    @objective(model, Max, -10buy + 15early + 12late)
    @test_logs (:warn, r"not certified") optimize!(model)
    @test is_certified(model, PrimalLDR()) === true
    @test is_certified(model) === false
    reason = "[not certified: conditional expectations across stages are not linear]"
    @test endswith(raw_status(model, DualLDR()), reason)
    return
end

# `inner_model`: the inner problem of one formulation, which a re-solve with the same data reuses.
function test_inner_model()
    @test_logs (:warn,) @test_throws OptimizeNotCalled inner_model(newsvendor(), PrimalLDR())
    model = newsvendor(; formulations = [PrimalLDR(), DualLDR()])
    optimize!(model)
    primal, dual = inner_model(model, PrimalLDR()), inner_model(model, DualLDR())
    @test primal isa JuMP.Model && dual isa JuMP.Model && primal !== dual
    @test num_variables(primal) > 0
    @test termination_status(primal) == termination_status(model, PrimalLDR())
    @test objective_value(primal) ≈ objective_value(model, PrimalLDR())
    @test objective_value(dual) ≈ objective_value(model, DualLDR())
    optimize!(model)
    @test inner_model(model, PrimalLDR()) === primal
    err = error_of(() -> inner_model(model, SampledLDR(20)))
    @test err isa MOI.GetAttributeNotAllowed
    @test occursin("add it to `Formulations()`", sprint(showerror, err))
    return
end

end # module TestResults

TestResults.runtests()
