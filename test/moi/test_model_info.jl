# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestModelInfo

using JuMP
using Test

import Distributions
import Distributions: Uniform
import LinearDecisionRules
import LinearDecisionRules:
    BreakPoints, FirstStage, ModelInfo, Stage, Truncate, Uncertainty, ValidInequality, canonicalize

include(joinpath(@__DIR__, "..", "common.jl"))
import .TestCommon: error_message

include(joinpath(@__DIR__, "..", "problems", "library.jl"))
include(joinpath(@__DIR__, "..", "problems", "tagged.jl"))
import .Problems: same_form

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

info_of(model) = LinearDecisionRules.model_info(backend(model))

# The newsvendor written with the tags, on a model without an optimizer; its twin with a
# `ModelInfo` by hand is `Problems.newsvendor()`.
newsvendor_with_tags() = TaggedProblems.newsvendor(Model())

function test_newsvendor()
    model = newsvendor_with_tags()
    form = canonicalize(backend(model), info_of(model))
    @test same_form(form, Problems.form(Problems.newsvendor()))
    info = info_of(model)
    @test info isa ModelInfo
    @test length(info.uncertainties) == 1
    @test info.uncertainties[1].variables == [index(model[:demand])]
    @test info.uncertainties[1].stages == [nothing]
    @test info.uncertainties[1].distribution == Distributions.Uniform(80, 120)
    @test info.decision_stages == Dict(index(model[:buy]) => 1)
    @test isempty(info.valid_rows) && isempty(info.truncate_rows) && isempty(info.breakpoints)
    return
end

function test_integer_newsvendor()
    model = Model()
    @variable(model, buy >= 0, Int, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, d in Uncertainty(Distributions.Uniform(80, 120)))
    @constraint(model, sell <= buy)
    @constraint(model, sell <= d)
    @objective(model, Max, -10buy + 15sell)
    form = canonicalize(backend(model), info_of(model))
    @test form.integer == [true, false]
    @test form.decision_stages == [1, 2]
    return
end

function test_three_stages()
    model = TaggedProblems.newsvendor_three_stages(Model())
    form = @test_logs canonicalize(backend(model), info_of(model))
    @test same_form(form, Problems.form(Problems.newsvendor_three_stages()))
    @test form.decision_stages == [1, 2, 2, 2, 3, 3]
    @test form.uncertainty_stages == [2, 2, 3]
    info = info_of(model)
    @test [u.stages for u in info.uncertainties] == [[2], [2], [3]]
    @test length(info.decision_stages) == 6
    # The stages of the uncertainties are not stages of decisions.
    @test !haskey(info.decision_stages, index(model[:demand2]))
    return
end

function test_unmarked_variables_in_last_stage()
    model = Model()
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0, Stage(3))
    @variable(model, d in Uncertainty(Distributions.Uniform(0, 1)), Stage(3))
    @variable(model, e in Uncertainty(Distributions.Uniform(0, 1)))
    @variable(model, r)
    @constraint(model, sell <= buy + d + e)
    @constraint(model, r <= buy)
    form = @test_logs (:warn, r"the variables without a stage are in the last one") canonicalize(
        backend(model),
        info_of(model),
    )
    @test form.num_stages == 3
    @test form.uncertainty_stages == [3, 3]
    @test form.decision_stages == [1, 3, 3]
    return
end

function test_vector_uncertainty()
    model = Model()
    mv = Distributions.product_distribution(Uniform(0, 1), Uniform(0, 2))
    @variable(model, x >= 0, FirstStage)
    @variable(model, y >= 0)
    @variable(model, d[1:2] in Uncertainty(mv))
    @variable(model, e[t = 1:2] in Uncertainty(mv), Stage(t))
    @constraint(model, y <= x + d[1] + d[2] + e[1] + e[2])
    info = info_of(model)
    @test [u.variables for u in info.uncertainties] == [index.(d), index.(e)]
    @test [u.stages for u in info.uncertainties] == [[nothing, nothing], [1, 2]]
    form = canonicalize(backend(model), info)
    @test form.uncertainties == index.([d[1], d[2], e[1], e[2]])[sortperm([2, 2, 1, 2])]
    @test length(form.laws) == 2
    return
end

function test_parameters()
    model = Model()
    @variable(model, p in Parameter(3.0), FirstStage)
    @variable(model, x >= 0)
    @variable(model, d in Uncertainty(Distributions.Uniform(0, 1)))
    @constraint(model, x <= d + p)
    form = canonicalize(backend(model), info_of(model))
    @test form.decisions == [index(x)]
    @test form.num_stages == 2
    return
end

function test_valid_inequality()
    model = newsvendor_with_tags()
    cap = @constraint(model, 85 <= model[:demand] <= 115, ValidInequality())
    @constraint(model, model[:demand] >= 81, ValidInequality())
    info = info_of(model)
    @test length(info.valid_rows) == 2 && index(cap) in info.valid_rows
    @test isempty(info.truncate_rows)
    form = canonicalize(backend(model), info)
    # The order of the rows is not specified.
    rows = sort(collect(zip(form.W_valid[:, 1], form.h_valid)))
    @test rows == [(-1.0, -115.0), (1.0, 81.0), (1.0, 85.0)]
    return
end

function test_truncate()
    model = newsvendor_with_tags()
    cap = @constraint(model, cap, model[:demand] <= 115, Truncate())
    info = info_of(model)
    @test info.truncate_rows == Set{MOI.ConstraintIndex}([index(cap)])
    @test isempty(info.valid_rows)
    # A bound on a univariate law truncates it exactly.
    form = canonicalize(backend(model), info)
    @test only(form.laws).distribution == Distributions.Uniform(80, 115)
    @test isempty(form.h_valid)
    return
end

function test_breakpoints()
    model = newsvendor_with_tags()
    set_attribute(model[:demand], BreakPoints(), [90, 100])
    info = info_of(model)
    @test info.breakpoints == Dict(index(model[:demand]) => [90.0, 100.0])
    # `canonicalize` lifts the demand into its three pieces.
    form = canonicalize(backend(model), info)
    @test form.uncertainty_names == ["demand[piece 1]", "demand[piece 2]", "demand[piece 3]"]
    @test only(form.lifting.blocks).lifting == LinearDecisionRules.AxisPiecewise([90, 100])
    return
end

# A variable has one `Stage` constraint, whose index is that of the variable.
function test_set_stage_replaces_the_stage()
    model = newsvendor_with_tags()
    sell = model[:sell]
    LinearDecisionRules.set_stage(sell, 2)
    LinearDecisionRules.set_stage(sell, 3)
    @test info_of(model).decision_stages[index(sell)] == 3
    @test num_constraints(model, VariableRef, Stage) == 2
    ci = MOI.ConstraintIndex{MOI.VariableIndex,Stage}(index(sell).value)
    @test MOI.get(backend(model), MOI.ConstraintSet(), ci) == Stage(3)
    return
end

# What canonicalize checks at `optimize!`, since JuMP cannot intercept it at declaration.
function test_errors_left_to_canonicalize()
    model = newsvendor_with_tags()
    set_lower_bound(model[:demand], 0)
    msg = error_message(() -> canonicalize(backend(model), info_of(model)))
    @test occursin("The uncertainty demand has a bound or integrality constraint", msg)
    model = newsvendor_with_tags()
    @constraint(model, cap, model[:demand] <= 115)
    msg = error_message(() -> canonicalize(backend(model), info_of(model)))
    @test occursin("The constraint cap involves only uncertainties (demand)", msg)
    model = newsvendor_with_tags()
    fix(model[:demand], 100; force = true)
    msg = error_message(() -> canonicalize(backend(model), info_of(model)))
    @test occursin("The uncertainty demand", msg)
    return
end

function test_deleting_variables()
    model = newsvendor_with_tags()
    @variable(model, extra, Stage(2))
    @variable(model, other in Uncertainty(Distributions.Uniform(0, 1)), Stage(2))
    delete(model, extra)
    delete(model, other)
    form = canonicalize(backend(model), info_of(model))
    @test same_form(form, Problems.form(Problems.newsvendor()))
    return
end

end # module TestModelInfo

TestModelInfo.runtests()
