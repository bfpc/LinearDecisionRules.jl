# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Models written as a user would, with `LDRModel` and the tags, solved end to end.

module TestEndToEnd

using JuMP
using Test

import Distributions
import Distributions: Uniform
import HiGHS
import LinearDecisionRules
import LinearDecisionRules:
    DualLDR,
    FirstStage,
    LDRModel,
    Stage,
    Truncate,
    Uncertainty,
    ValidInequality,
    decision_rule,
    expected_value,
    stage
import TOML

const LEGACY = TOML.parsefile(joinpath(@__DIR__, "..", "reference", "legacy_values.toml"))

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

# An exact MIP bound, so that the integer dual is the known value.
ldr_model() = TaggedProblems.ldr_model(; attributes = ("mip_rel_gap" => 0.0,))

import .TaggedProblems:
    hydro_toy,
    inventory,
    newsvendor_discrete_demand,
    newsvendor_mv_discrete,
    newsvendor_random_price,
    newsvendor_three_stages,
    newsvendor_two_products

newsvendor(; integer = false) = TaggedProblems.newsvendor(ldr_model(); integer)

# `x1` in the legacy file is `x[1]` here.
function variable_by_legacy_name(model, name)
    x = variable_by_name(model, name)
    if x === nothing
        m = match(r"^([a-z_]+)(\d+)$", name)
        x = variable_by_name(model, "$(m[1])[$(m[2])]")
    end
    return x
end

function check_rules(model, rules)
    for (name, rule) in rules
        r = decision_rule(variable_by_legacy_name(model, name))
        @test r.constant ≈ rule["constant"] atol = 1e-6
        for (u, coefficient) in rule["coefficients"]
            d = variable_by_legacy_name(model, u)
            @test get(r.coefficients, d, 0.0) ≈ coefficient atol = 1e-6
        end
    end
    return
end

const GOLDEN = [
    ("newsvendor", () -> newsvendor(), true),
    ("newsvendor_integer", () -> newsvendor(; integer = true), true),
    ("newsvendor_random_price", newsvendor_random_price, false),
    ("newsvendor_two_products", newsvendor_two_products, false),
    ("newsvendor_three_stages", newsvendor_three_stages, false),
    ("inventory_T4", () -> inventory(4), true),
    ("inventory_T8", () -> inventory(8), true),
    ("hydro_toy_fixed_volume", hydro_toy, false),
    ("newsvendor_mv_discrete", newsvendor_mv_discrete, false),
    ("newsvendor_discrete_demand", newsvendor_discrete_demand, false),
]

# The previous version's values, from models written with the tags.
function test_golden_values()
    for (name, build, with_rules) in GOLDEN
        @testset "$name" begin
            legacy = LEGACY[name]
            model = build()
            optimize!(model)
            @test termination_status(model) == MOI.OPTIMAL
            @test primal_status(model) == MOI.FEASIBLE_POINT
            tolerance = name == "hydro_toy_fixed_volume" ? 1e-6 : 1e-7
            @test objective_value(model) ≈ legacy["primal"] atol = tolerance rtol = 1e-7
            @test objective_bound(model) ≈ legacy["dual"] atol = tolerance rtol = 1e-7
            @test objective_value(model, DualLDR()) ≈ legacy["dual"] atol = tolerance rtol = 1e-7
            if with_rules
                check_rules(model, legacy["rules"]["primal"])
            end
        end
    end
    return
end

# A vector uncertainty over a named range has the same model as over `1:n`.
function test_uncertainty_in_named_range()
    law = Distributions.product_distribution(Uniform(0, 1), Uniform(0, 2))
    # `Base.OneTo(2)` gives a `Vector`, the others a `DenseAxisArray`.
    objectives = map((Base.OneTo(2), 1:2, [:a, :b])) do R
        model = ldr_model()
        @variable(model, x[R] >= 0)
        @variable(model, y >= 0, FirstStage)
        @variable(model, d[R] in Uncertainty(law), Stage(2))
        @constraint(model, [r in R], x[r] <= d[r] + y)
        @constraint(model, sum(x) <= 2)
        @objective(model, Max, sum(x) - 0.5y)
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        return objective_value(model)
    end
    @test objectives[1] ≈ objectives[2] ≈ objectives[3]
    return
end

function test_newsvendor()
    model = newsvendor()
    optimize!(model)
    @test objective_value(model) ≈ 460
    @test objective_bound(model) ≈ 1460 / 3
    buy, sell, ret, demand = model[:buy], model[:sell], model[:ret], model[:demand]
    @test decision_rule(buy).constant ≈ 120
    @test isempty(decision_rule(buy).coefficients)
    @test decision_rule(sell)(demand => 95.0) ≈ 95
    @test value(buy) ≈ 120
    @test value(sell) ≈ 100
    @test value(demand) ≈ 100
    @test value(ret) ≈ 20
    return
end

# The newsvendor with vector rows, which are read one output at a time.
function test_vector_rows()
    model = ldr_model()
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @variable(model, demand in Uncertainty(Uniform(80, 120)))
    @constraint(model, rows, [sell + ret - buy, sell - demand] in MOI.Nonpositives(2))
    @constraint(model, [sell, ret] >= 0)
    @objective(model, Max, -10buy + 8ret + 15sell)
    optimize!(model)
    @test objective_value(model) ≈ 460
    @test objective_bound(model) ≈ 1460 / 3
    @test name(rows) == "rows"
    return
end

function test_integer_first_stage()
    model = newsvendor(; integer = true)
    optimize!(model)
    @test objective_value(model) ≈ 460
    @test objective_bound(model) ≈ 486
    @test value(model[:buy]) ≈ 120
    @test is_integer(model[:buy])
    return
end

function test_two_products()
    model = newsvendor_two_products()
    optimize!(model)
    @test objective_value(model) ≈ 920
    @test objective_bound(model) ≈ 2920 / 3
    # The uncertainty of one product is not seen by the other.
    sell = decision_rule(model[:sell][1])
    @test get(sell.coefficients, model[:demand][2], 0.0) == 0
    return
end

function test_three_stages()
    model = newsvendor_three_stages()
    @test_logs optimize!(model)
    @test objective_value(model) ≈ 940
    @test objective_bound(model) ≈ 1000
    # A decision has no coefficient on an uncertainty of a later stage.
    demand3 = model[:demand3]
    for name in ("buy", "sell2", "ret2", "buy2")
        @test !haskey(decision_rule(model[Symbol(name)]).coefficients, demand3)
    end
    @test !haskey(decision_rule(model[:buy]).coefficients, model[:demand2])
    @test haskey(decision_rule(model[:sell3]).coefficients, demand3)
    @test stage(model[:sell2]) == 2 && stage(model[:demand3]) == 3
    return
end

function test_unmarked_variable_warns_once()
    model = newsvendor_three_stages(; unmarked_ret3 = true)
    @test_logs (:warn, r"variables without a stage are in the last one, Stage\(3\): ret3") begin
        optimize!(model)
    end
    @test objective_value(model) ≈ 940
    return
end

function test_random_price_depends_on_sell()
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @variable(model, d in Uncertainty(Uniform(80, 120)))
    @variable(model, p in Uncertainty(Uniform(13, 17)))
    @constraint(model, sell + ret <= buy)
    @constraint(model, sell <= d + 2p)
    @objective(model, Max, -10buy + 8ret + p * sell)
    optimize!(model)
    rule = decision_rule(model[:sell])
    @test haskey(rule.coefficients, model[:p])
    # The expectation of the objective uses E[p * sell], not E[p] * E[sell].
    expectation = expected_value(objective_function(model))
    @test expectation ≈ objective_value(model)
    @test value(objective_function(model)) != expectation
    @test expected_value(model[:p] * model[:sell]) != value(model[:p] * model[:sell])
    return
end

# The extra constraint binds, so that the objective is lower before it is deleted.
function test_delete_a_binding_constraint()
    model = newsvendor()
    @variable(model, extra >= 0)
    c = @constraint(model, model[:sell] + extra <= 90)
    optimize!(model)
    @test objective_value(model) < 460 - 1
    delete(model, c)
    delete(model, extra)
    optimize!(model)
    @test objective_value(model) ≈ 460
    @test objective_bound(model) ≈ 1460 / 3
    @test decision_rule(model[:buy]).constant ≈ 120
    return
end

function test_set_normalized_rhs()
    model = newsvendor()
    cap = @constraint(model, model[:buy] <= 150)
    optimize!(model)
    @test objective_value(model) ≈ 460
    set_normalized_rhs(cap, 110)
    optimize!(model)
    fresh = newsvendor()
    @constraint(fresh, fresh[:buy] <= 110)
    optimize!(fresh)
    @test objective_value(model) < 460 - 1
    @test objective_value(model) ≈ objective_value(fresh)
    @test objective_bound(model) ≈ objective_bound(fresh)
    @test decision_rule(model[:buy]).constant ≈ 110
    return
end

function test_copy_model()
    model = newsvendor_three_stages()
    copied, map = copy_model(model)
    @test stage(map[model[:sell3]]) == 3
    set_optimizer(copied, () -> LinearDecisionRules.Optimizer(HiGHS.Optimizer); add_bridges = false)
    set_silent(copied)
    optimize!(copied)
    @test objective_value(copied) ≈ 940
    @test objective_bound(copied) ≈ 1000
    @test decision_rule(map[model[:sell2]])(map[model[:demand2]] => 100.0) ≈ 100
    return
end

function test_valid_inequality_improves_the_primal()
    function build(valid)
        model = ldr_model()
        @variable(model, buy >= 0, FirstStage)
        @variable(model, sell[1:2] >= 0)
        @variable(model, d[i = 1:2] in Uncertainty(Uniform(80, 120)))
        @constraint(model, sum(sell) <= buy)
        @constraint(model, sell .<= d)
        @objective(model, Max, -10buy + 15sum(sell))
        if valid
            @constraint(model, d[1] + d[2] <= 230, ValidInequality())
        end
        optimize!(model)
        return model
    end
    with, without = build(true), build(false)
    @test objective_value(without) <= objective_value(with) + 1e-6
    @test objective_bound(with) < objective_bound(without)
    return
end

# A tagged variable-in-set constraint is a valid row, as the same affine row is.
function test_valid_inequality_on_a_variable()
    function build(row)
        model = newsvendor()
        row(model, model[:demand])
        optimize!(model)
        return model
    end
    affine = build((m, d) -> @constraint(m, 1.0 * d <= 110, ValidInequality()))
    for row in (
        (m, d) -> @constraint(m, d in MOI.LessThan(110.0), ValidInequality()),
        (m, d) -> @constraint(m, d in MOI.Interval(80.0, 110.0), ValidInequality()),
    )
        model = build(row)
        @test termination_status(model) == MOI.OPTIMAL
        @test objective_value(model) ≈ objective_value(affine)
        @test objective_bound(model) ≈ objective_bound(affine)
    end
    @test objective_value(affine) > 460 + 1
    # Tagged `Truncate()`, it truncates the law, as an affine row does: d ~ Uniform(80, 110).
    truncated = newsvendor()
    @constraint(truncated, truncated[:demand] in MOI.LessThan(110.0), Truncate())
    optimize!(truncated)
    narrow = ldr_model()
    @variable(narrow, buy >= 0, FirstStage)
    @variable(narrow, sell >= 0)
    @variable(narrow, ret >= 0)
    @variable(narrow, demand in Uncertainty(Uniform(80, 110)))
    @constraint(narrow, sell + ret <= buy)
    @constraint(narrow, sell <= demand)
    @objective(narrow, Max, -10buy + 8ret + 15sell)
    optimize!(narrow)
    @test objective_value(truncated) ≈ objective_value(narrow)
    @test objective_bound(truncated) ≈ objective_bound(narrow)
    @test value(truncated[:demand]) ≈ 95
    return
end

end # module TestEndToEnd

TestEndToEnd.runtests()
