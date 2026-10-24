# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestOptimizer

using JuMP
using Test

import Distributions
import HiGHS
import LinearDecisionRules
import LinearDecisionRules:
    BreakPoints,
    FirstStage,
    DualLDR,
    Formulations,
    InnerOptimizer,
    LDRModel,
    PrimalLDR,
    SampledLDR,
    Stage,
    Uncertainty,
    ValidInequality,
    canonicalize,
    set_stage,
    stage

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

function newsvendor(model = LDRModel(HiGHS.Optimizer); silent = true)
    silent && set_silent(model)
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @variable(model, d in Uncertainty(Distributions.Uniform(80, 120)))
    @constraint(model, sell + ret <= buy)
    @constraint(model, sell <= d)
    @objective(model, Max, -10buy + 8ret + 15sell)
    return model
end

function test_ldr_model()
    model = LDRModel(HiGHS.Optimizer)
    @test model isa Model
    @test !JuMP.bridge_constraints(model)
    @test unsafe_backend(model) isa LinearDecisionRules.Optimizer
    @test solver_name(model) == "LinearDecisionRules"
    @test get_attribute(model, InnerOptimizer()) === HiGHS.Optimizer
    @test get_attribute(LDRModel(), InnerOptimizer()) === nothing
    @test get_attribute(LDRModel(), Formulations()) == [PrimalLDR(), DualLDR()]
    return
end

function test_newsvendor_model()
    model = newsvendor()
    @test num_variables(model) == 4
    @test stage(model[:buy]) == 1
    text = sprint(print, model)
    @test occursin(" buy in Stage(1)\n", text)
    @test occursin(" d in Uncertainty(Distributions.Uniform{Float64}(a=80.0, b=120.0))\n", text)
    @test occursin("Max -10 buy + 8 ret + 15 sell", text)
    # The model is read as the user wrote it.
    form = canonicalize(backend(model), LinearDecisionRules._model_info(backend(model)))
    @test form.decision_names == ["buy", "sell", "ret"]
    @test form.decision_stages == [1, 2, 2]
    @test form.uncertainty_names == ["d"]
    return
end

function test_no_bridge_variables()
    model = newsvendor()
    @variable(model, x in Parameter(1.0))
    @variable(model, y >= 0, Int, FirstStage)
    @variable(model, z in MOI.ZeroOne(), FirstStage)
    @constraint(model, 0 <= y + z <= 3)
    @constraint(model, y - z == 0)
    @constraint(model, y - z >= -1)
    @objective(model, Max, 2y + y^2 + model[:sell])
    @test num_variables(model) == 7
    @test num_constraints(model; count_variable_in_set_constraints = false) == 5
    return
end

function test_unsupported_constraints_are_rejected_at_declaration()
    model = newsvendor()
    sell, ret = model[:sell], model[:ret]
    rejected = (
        () -> @constraint(model, [sell, ret, 1] in SecondOrderCone()),
        () -> @constraint(model, [sell, ret] in MOI.Nonnegatives(2)),
        () -> @constraint(model, sell in MOI.Semicontinuous(1.0, 2.0)),
        () -> @constraint(model, sell^2 in MOI.Semicontinuous(1.0, 2.0)),
    )
    for f in rejected
        err = error_of(f)
        @test err isa ErrorException
        @test occursin("are not supported by the solver", err.msg)
    end
    msg = error_of(rejected[1]).msg
    @test occursin("VectorAffineFunction{Float64}-in-", msg) && occursin("SecondOrderCone", msg)
    # The rejected constraints leave no trace, and there are no bridge variables.
    @test num_variables(model) == 4
    @test num_constraints(model; count_variable_in_set_constraints = false) == 2
    return
end

# Quadratic rows are accepted, so that `canonicalize` names the constraint and the product in its
# error, and reads the products with a parameter.
function test_quadratic_rows_reach_canonicalize()
    for (row, message) in (
        (m -> @constraint(m, q, m[:d] * m[:sell] <= 100), "q multiplies the decision sell by the"),
        (m -> @constraint(m, q, m[:sell] * m[:ret] == 1), "q has the product of sell and ret"),
        (m -> @constraint(m, q, m[:sell]^2 <= 1), "q has the product of sell and sell"),
    )
        model = newsvendor()
        row(model)
        @test num_constraints(model, QuadExpr, MOI.LessThan{Float64}) +
              num_constraints(model, QuadExpr, MOI.EqualTo{Float64}) == 1
        @test occursin(message, sprint(showerror, error_of(() -> optimize!(model))))
    end
    return
end

# JuMP does not ask the optimizer whether it supports the set of a variable on creation, so
# `canonicalize` reports it at `optimize!`.
function test_unsupported_variable_sets_are_left_to_canonicalize()
    model = newsvendor()
    @variable(model, w in MOI.Semicontinuous(1.0, 2.0))
    msg = try
        canonicalize(backend(model), LinearDecisionRules._model_info(backend(model)))
    catch err
        sprint(showerror, err)
    end
    @test occursin("VariableIndex-in-MathOptInterface.Semicontinuous", msg)
    return
end

function test_silent_and_time_limit()
    model = newsvendor(; silent = false)
    @test !get_attribute(model, MOI.Silent())
    set_silent(model)
    @test get_attribute(model, MOI.Silent())
    unset_silent(model)
    @test !get_attribute(model, MOI.Silent())
    @test time_limit_sec(model) === nothing
    set_time_limit_sec(model, 30)
    @test time_limit_sec(model) == 30.0
    unset_time_limit_sec(model)
    @test time_limit_sec(model) === nothing
    return
end

function test_raw_attributes()
    model = newsvendor()
    set_attribute(model, "presolve", "off")
    set_attribute(model, "mip_rel_gap", 0.1)
    @test get_attribute(model, "presolve") == "off"
    @test get_attribute(model, "mip_rel_gap") == 0.1
    err = error_of(() -> get_attribute(model, "threads"))
    @test err isa MOI.GetAttributeNotAllowed
    @test occursin("\"threads\" was not set", sprint(showerror, err))
    set_attribute(model, "presolve", "on")
    @test get_attribute(model, "presolve") == "on"
    return
end

function test_settings_reach_the_inner_models()
    model = newsvendor()
    set_silent(model)
    set_time_limit_sec(model, 12.5)
    set_attribute(model, "presolve", "off")
    optimize!(model)
    for result in unsafe_backend(model).results
        @test get_attribute(result.model, MOI.Silent())
        @test 0 < time_limit_sec(result.model) <= 12.5
        @test get_attribute(result.model, "presolve") == "off"
    end
    # Without settings, the inner models keep their defaults.
    quiet = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
    other = newsvendor(LDRModel(quiet); silent = false)
    optimize!(other)
    for result in unsafe_backend(other).results
        @test time_limit_sec(result.model) === nothing
    end
    # The factory may carry its own attributes.
    model = newsvendor(LDRModel(optimizer_with_attributes(HiGHS.Optimizer, "presolve" => "off")))
    optimize!(model)
    @test get_attribute(unsafe_backend(model).results[1].model, "presolve") == "off"
    return
end

function test_no_inner_optimizer()
    model = newsvendor(LDRModel())
    msg = sprint(showerror, error_of(() -> optimize!(model)))
    @test occursin("no inner optimizer", msg) && occursin("LDRModel(HiGHS.Optimizer)", msg)
    set_attribute(model, InnerOptimizer(), HiGHS.Optimizer)
    optimize!(model)
    @test length(unsafe_backend(model).results) == 2
    return
end

function test_formulations_attribute()
    model = newsvendor()
    set_attribute(model, Formulations(), [DualLDR(), SampledLDR(10)])
    @test get_attribute(model, Formulations()) == [DualLDR(), SampledLDR(10)]
    # The attribute is a copy.
    get_attribute(model, Formulations())[1] = PrimalLDR()
    @test get_attribute(model, Formulations()) == [DualLDR(), SampledLDR(10)]
    @test_throws ArgumentError set_attribute(model, Formulations(), [])
    @test_throws ArgumentError set_attribute(model, Formulations(), [:a])
    return
end

function test_attributes_survive_changes_to_the_model()
    model = newsvendor()
    set_silent(model)
    set_attribute(model, "presolve", "off")
    set_attribute(model, Formulations(), [DualLDR()])
    @variable(model, extra >= 0)
    delete(model, extra)
    set_optimizer(model, () -> LinearDecisionRules.Optimizer(HiGHS.Optimizer); add_bridges = false)
    @test get_attribute(model, InnerOptimizer()) === HiGHS.Optimizer
    return
end

function test_optimize_keeps_the_results()
    model = newsvendor()
    optimizer = unsafe_backend(model)
    @test MOI.is_empty(optimizer)
    optimize!(model)
    @test !MOI.is_empty(optimizer)
    primal, dual = optimizer.results
    @test primal.formulation == PrimalLDR() && dual.formulation == DualLDR()
    @test primal.objective_value ≈ 460
    @test dual.objective_bound ≈ 1460 / 3
    @test optimizer.form isa LinearDecisionRules.CanonicalForm
    @test optimizer.solve_time > 0
    # Only the chosen formulations are solved.
    set_attribute(model, Formulations(), [DualLDR()])
    optimize!(model)
    @test [r.formulation for r in optimizer.results] == [DualLDR()]
    # The optimizer is not a copy of the model.
    @test error_of(() -> MOI.copy_to(optimizer, backend(model))) isa ErrorException
    return
end

# Uncertainties added through MOI bypass the checks of `@variable`.
function test_unbounded_support_added_through_moi()
    model = LDRModel(HiGHS.Optimizer)
    x, _ = MOI.add_constrained_variable(backend(model), Uncertainty(Distributions.Normal(0, 1)))
    MOI.set(backend(model), MOI.VariableName(), x, "eta")
    msg = sprint(showerror, error_of(() -> optimize!(model)))
    @test occursin("The uncertainty eta is not bounded", msg)
    @test occursin("unbounded support", msg)
    return
end

function test_delete()
    model = newsvendor()
    @constraint(model, c, model[:sell] + model[:ret] <= 100)
    @test is_valid(model, c)
    delete(model, c)
    @test !is_valid(model, c)
    @test num_constraints(model; count_variable_in_set_constraints = false) == 2
    # Deleting a staged variable deletes its `Stage` constraint.
    @test num_constraints(model, VariableRef, Stage) == 1
    delete(model, model[:buy])
    @test num_constraints(model, VariableRef, Stage) == 0
    # Deleting an uncertainty deletes its set.
    S = LinearDecisionRules.UnivariateUncertainty{Distributions.Uniform{Float64}}
    @test num_constraints(model, VariableRef, S) == 1
    delete(model, model[:d])
    @test num_constraints(model, VariableRef, S) == 0
    return
end

function test_parameters()
    model = newsvendor()
    @variable(model, p in Parameter(2.0))
    @constraint(model, model[:sell] <= p * 10)
    set_parameter_value(p, 3.0)
    @test parameter_value(p) == 3.0
    return
end

function test_constraint_attributes_are_kept()
    model = newsvendor()
    d = model[:d]
    cap = @constraint(model, d <= 110, ValidInequality())
    @test get_attribute(cap, LinearDecisionRules.XiRole()) == ValidInequality()
    set_attribute(d, BreakPoints(), [90.0, 100.0])
    @test get_attribute(d, BreakPoints()) == [90.0, 100.0]
    return
end

function test_copy_model_keeps_the_metadata()
    model = newsvendor()
    cap = @constraint(model, model[:d] <= 110, ValidInequality())
    set_attribute(model[:d], BreakPoints(), [90.0, 100.0])
    copied, map = copy_model(model)
    @test stage(map[model[:buy]]) == 1
    @test stage(map[model[:sell]]) === nothing
    @test get_attribute(map[model[:d]], BreakPoints()) == [90.0, 100.0]
    @test get_attribute(map[cap], LinearDecisionRules.XiRole()) == ValidInequality()
    info = LinearDecisionRules._model_info(backend(copied))
    @test length(info.uncertainties) == 1
    @test info.decision_stages == Dict(index(map[model[:buy]]) => 1)
    @test length(info.valid_rows) == 1
    return
end

end # module TestOptimizer

TestOptimizer.runtests()
