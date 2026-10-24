# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestModelInfo

using JuMP
using Test

import Distributions
import LinearDecisionRules
import LinearDecisionRules:
    BreakPoints,
    CanonicalForm,
    FirstStage,
    ModelInfo,
    Stage,
    Truncate,
    Uncertainty,
    UncertaintyDeclaration,
    ValidInequality,
    canonicalize

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

function error_message(f)
    try
        f()
    catch err
        return sprint(showerror, err)
    end
    return ""
end

_model_info(model) = LinearDecisionRules._model_info(backend(model))

# Equality of two canonical forms, field by field. The blocks of the laws are compared sorted by
# column, since their order is the order of declaration.
function same_form(a::CanonicalForm, b::CanonicalForm)
    for name in fieldnames(CanonicalForm)
        if name == :laws
            continue
        end
        if getfield(a, name) != getfield(b, name)
            return false
        end
    end
    key(law) = (law.columns, law.stages, law.distribution)
    return sort(key.(a.laws); by = first) == sort(key.(b.laws); by = first)
end

# The model of the canonical tests, written with a `ModelInfo` by hand...
function newsvendor_by_hand()
    model = Model()
    @variable(model, buy >= 0)
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @variable(model, d)
    @constraint(model, sell + ret <= buy)
    @constraint(model, sell <= d)
    @objective(model, Max, -10buy + 8ret + 15sell)
    info = ModelInfo(;
        uncertainties = [UncertaintyDeclaration(Distributions.Uniform(80, 120), [index(d)])],
        decision_stages = Dict(index(buy) => 1),
    )
    return model, info
end

# ... and written with the tags.
function newsvendor_with_tags()
    model = Model()
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @variable(model, d in Uncertainty(Distributions.Uniform(80, 120)))
    @constraint(model, sell + ret <= buy)
    @constraint(model, sell <= d)
    @objective(model, Max, -10buy + 8ret + 15sell)
    return model
end

function test_newsvendor()
    by_hand, info = newsvendor_by_hand()
    model = newsvendor_with_tags()
    @test same_form(
        canonicalize(backend(model), _model_info(model)),
        canonicalize(backend(by_hand), info),
    )
    info = _model_info(model)
    @test info isa ModelInfo
    @test length(info.uncertainties) == 1
    @test info.uncertainties[1].variables == [index(model[:d])]
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
    form = canonicalize(backend(model), _model_info(model))
    @test form.integer == [true, false]
    @test form.decision_stages == [1, 2]
    return
end

function three_stage_by_hand()
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
    stages = Dict(index(buy) => 1, index(sell3) => 3, index(ret3) => 3)
    for v in (sell2, ret2, buy2)
        stages[index(v)] = 2
    end
    uniform(a, b) = Distributions.Uniform(a, b)
    uncertainties = [
        UncertaintyDeclaration(uniform(80, 120), [index(demand2)], [2]),
        UncertaintyDeclaration(uniform(8, 14), [index(buy_cost2)], [2]),
        UncertaintyDeclaration(uniform(80, 120), [index(demand3)], [3]),
    ]
    return model, ModelInfo(; uncertainties, decision_stages = stages)
end

function test_three_stages()
    by_hand, info = three_stage_by_hand()
    model = Model()
    uniform(a, b) = Distributions.Uniform(a, b)
    @variable(model, buy >= 0, Stage(1))
    @variable(model, sell2 >= 0, Stage(2))
    @variable(model, ret2 >= 0, Stage(2))
    @variable(model, demand2 in Uncertainty(uniform(80, 120)), Stage(2))
    @variable(model, buy2 >= 0, Stage(2))
    @variable(model, buy_cost2 in Uncertainty(uniform(8, 14)), Stage(2))
    @variable(model, sell3 >= 0, Stage(3))
    @variable(model, ret3 >= 0, Stage(3))
    @variable(model, demand3 in Uncertainty(uniform(80, 120)), Stage(3))
    @constraint(model, sell2 + ret2 <= buy + buy2)
    @constraint(model, sell2 <= demand2)
    @constraint(model, sell2 + ret2 + sell3 + ret3 <= buy + buy2)
    @constraint(model, sell3 <= demand3)
    @objective(model, Max, -10buy - buy_cost2 * buy2 + 8ret2 + 15sell2 + 8ret3 + 15sell3)
    form = @test_logs canonicalize(backend(model), _model_info(model))
    @test same_form(form, canonicalize(backend(by_hand), info))
    @test form.decision_stages == [1, 2, 2, 2, 3, 3]
    @test form.uncertainty_stages == [2, 2, 3]
    info = _model_info(model)
    @test [u.stages for u in info.uncertainties] == [[2], [2], [3]]
    @test length(info.decision_stages) == 6
    # The stages of the uncertainties are not stages of decisions.
    @test !haskey(info.decision_stages, index(demand2))
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
        _model_info(model),
    )
    @test form.num_stages == 3
    @test form.uncertainty_stages == [3, 3]
    @test form.decision_stages == [1, 3, 3]
    return
end

function test_vector_uncertainty()
    model = Model()
    mv = Distributions.product_distribution(Distributions.Uniform(0, 1), Distributions.Uniform(0, 2))
    @variable(model, x >= 0, FirstStage)
    @variable(model, y >= 0)
    @variable(model, d[1:2] in Uncertainty(mv))
    @variable(model, e[t = 1:2] in Uncertainty(mv), Stage(t))
    @constraint(model, y <= x + d[1] + d[2] + e[1] + e[2])
    info = _model_info(model)
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
    form = canonicalize(backend(model), _model_info(model))
    @test form.decisions == [index(x)]
    @test form.num_stages == 2
    return
end

function test_valid_inequality()
    model = newsvendor_with_tags()
    cap = @constraint(model, 85 <= model[:d] <= 115, ValidInequality())
    @constraint(model, model[:d] >= 81, ValidInequality())
    info = _model_info(model)
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
    cap = @constraint(model, cap, model[:d] <= 115, Truncate())
    info = _model_info(model)
    @test info.truncate_rows == Set{MOI.ConstraintIndex}([index(cap)])
    @test isempty(info.valid_rows)
    msg = error_message(() -> canonicalize(backend(model), info))
    @test occursin("cap is tagged `Truncate()`", msg)
    return
end

function test_breakpoints()
    model = newsvendor_with_tags()
    set_attribute(model[:d], BreakPoints(), [90, 100])
    info = _model_info(model)
    @test info.breakpoints == Dict(index(model[:d]) => [90.0, 100.0])
    @test occursin(
        "not supported yet",
        error_message(() -> canonicalize(backend(model), info)),
    )
    return
end

function test_constraint_stage_is_ignored()
    model = newsvendor_with_tags()
    @constraint(model, model[:sell] <= 1000, Stage(2))
    by_hand, info = newsvendor_by_hand()
    @constraint(by_hand, by_hand[:sell] <= 1000)
    @test same_form(
        canonicalize(backend(model), _model_info(model)),
        canonicalize(backend(by_hand), info),
    )
    return
end

# MOI keeps one constraint per variable and set type, so there is no way to give two stages.
function test_a_second_stage_constraint_replaces_the_first()
    model = newsvendor_with_tags()
    @constraint(model, model[:sell] in Stage(2))
    @constraint(model, model[:sell] in Stage(3))
    @test _model_info(model).decision_stages[index(model[:sell])] == 3
    return
end

# What canonicalize checks at `optimize!`, since JuMP cannot intercept it at declaration.
function test_errors_left_to_canonicalize()
    model = newsvendor_with_tags()
    set_lower_bound(model[:d], 0)
    msg = error_message(() -> canonicalize(backend(model), _model_info(model)))
    @test occursin("The uncertainty d has a bound or integrality constraint", msg)
    model = newsvendor_with_tags()
    @constraint(model, cap, model[:d] <= 115)
    msg = error_message(() -> canonicalize(backend(model), _model_info(model)))
    @test occursin("The constraint cap involves only uncertainties (d)", msg)
    model = newsvendor_with_tags()
    fix(model[:d], 100; force = true)
    msg = error_message(() -> canonicalize(backend(model), _model_info(model)))
    @test occursin("The uncertainty d", msg)
    return
end

function test_deleting_variables()
    model = newsvendor_with_tags()
    @variable(model, extra, Stage(2))
    @variable(model, other in Uncertainty(Distributions.Uniform(0, 1)), Stage(2))
    delete(model, extra)
    delete(model, other)
    by_hand, info = newsvendor_by_hand()
    @test same_form(
        canonicalize(backend(model), _model_info(model)),
        canonicalize(backend(by_hand), info),
    )
    return
end

end # module TestModelInfo

TestModelInfo.runtests()
