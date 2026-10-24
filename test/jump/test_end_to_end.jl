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
    Formulations,
    LDRModel,
    MvDiscreteNonParametric,
    PrimalLDR,
    SampledLDR,
    Stage,
    Truncate,
    Uncertainty,
    ValidInequality,
    decision_rule,
    expected_value,
    solve_formulation,
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

# An exact MIP bound, so that the integer dual is the known value.
function ldr_model()
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    set_attribute(model, "mip_rel_gap", 0.0)
    return model
end

function newsvendor(; integer = false)
    model = ldr_model()
    @variable(model, buy >= 0, FirstStage, integer = integer)
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @variable(model, demand in Uncertainty(Uniform(80, 120)))
    @constraint(model, sell + ret <= buy)
    @constraint(model, sell <= demand)
    @objective(model, Max, -10buy + 8ret + 15sell)
    return model
end

function newsvendor_random_price()
    model = ldr_model()
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @variable(model, demand in Uncertainty(Uniform(80, 120)))
    @variable(model, price in Uncertainty(Uniform(13, 17)))
    @constraint(model, sell + ret <= buy)
    @constraint(model, sell <= demand)
    @objective(model, Max, -10buy + 8ret + price * sell)
    return model
end

function newsvendor_two_products()
    model = ldr_model()
    @variable(model, buy[1:2] >= 0, FirstStage)
    @variable(model, sell[1:2] >= 0)
    @variable(model, ret[1:2] >= 0)
    law = Distributions.product_distribution([Uniform(80, 120) for _ in 1:2])
    @variable(model, demand[1:2] in Uncertainty(law))
    @constraint(model, [i in 1:2], sell[i] + ret[i] <= buy[i])
    @constraint(model, [i in 1:2], sell[i] <= demand[i])
    @objective(model, Max, sum(-10buy[i] + 8ret[i] + 15sell[i] for i in 1:2))
    return model
end

function newsvendor_three_stages(; unmarked_ret3 = false)
    model = ldr_model()
    @variable(model, buy >= 0, Stage(1))
    @variable(model, sell2 >= 0, Stage(2))
    @variable(model, ret2 >= 0, Stage(2))
    @variable(model, demand2 in Uncertainty(Uniform(80, 120)), Stage(2))
    @variable(model, buy2 >= 0, Stage(2))
    @variable(model, buy_cost2 in Uncertainty(Uniform(8, 14)), Stage(2))
    @variable(model, sell3 >= 0, Stage(3))
    if unmarked_ret3
        @variable(model, ret3 >= 0)
    else
        @variable(model, ret3 >= 0, Stage(3))
    end
    @variable(model, demand3 in Uncertainty(Uniform(80, 120)), Stage(3))
    @constraint(model, sell2 + ret2 <= buy + buy2)
    @constraint(model, sell2 <= demand2)
    @constraint(model, sell2 + ret2 + sell3 + ret3 <= buy + buy2)
    @constraint(model, sell3 <= demand3)
    @objective(model, Max, -10buy - buy_cost2 * buy2 + 8ret2 + 15sell2 + 8ret3 + 15sell3)
    return model
end

function inventory(T::Int)
    model = ldr_model()
    @variable(model, d[t = 1:T] in Uncertainty(Uniform(60, 100)), Stage(t + 1))
    @variable(model, 0 <= x[t = 1:T] <= 150, Stage(t))
    @variable(model, I[t = 1:T], Stage(t + 1))
    @variable(model, y[t = 1:T], Stage(t + 1))
    @constraint(model, I[1] == x[1] - d[1])
    @constraint(model, [t in 2:T], I[t] == I[t-1] + x[t] - d[t])
    @constraint(model, [t in 1:T], y[t] >= 0.5 * I[t])
    @constraint(model, [t in 1:T], y[t] >= -3 * I[t])
    @objective(model, Min, sum(x[t] + y[t] for t in 1:T))
    return model
end

function hydro_toy()
    model = ldr_model()
    @variable(model, vi == 0.5)
    @variable(model, 0 <= vf <= 1)
    @variable(model, gh >= 0)
    @variable(model, gt >= 0)
    @variable(model, inflow[i = 1:3] in Uncertainty(Uniform(0, 0.1i)))
    @constraint(model, vf == vi - gh + sum(inflow))
    @constraint(model, gt + gh == 0.3)
    @objective(model, Min, gt^2 + vf^2 / 2 - vf)
    return model
end

# The tutorial newsvendor (buy 10, sell 15, salvage 5) with a discrete demand.
function newsvendor_discrete(law)
    model = ldr_model()
    k = length(law)
    @variable(model, buy[1:k] >= 0, FirstStage)
    @variable(model, sell[1:k] >= 0)
    @variable(model, salvage[1:k] >= 0)
    @variable(model, demand[1:k] in Uncertainty(law))
    @constraint(model, [i in 1:k], sell[i] + salvage[i] <= buy[i])
    @constraint(model, [i in 1:k], sell[i] <= demand[i])
    @objective(model, Max, sum(-10buy[i] + 15sell[i] + 5salvage[i] for i in 1:k))
    return model
end

function newsvendor_mv_discrete()
    scenarios = [[80.0, 60.0], [100.0, 80.0], [120.0, 100.0], [80.0, 100.0], [120.0, 60.0]]
    return newsvendor_discrete(MvDiscreteNonParametric(scenarios, [0.2, 0.3, 0.2, 0.15, 0.15]))
end

function newsvendor_discrete_demand()
    support = [80.0, 90, 100, 110, 120]
    law = Distributions.DiscreteNonParametric(support, [0.1, 0.2, 0.4, 0.2, 0.1])
    return newsvendor_discrete(law)
end

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

function test_parameters()
    function build(cost)
        model = ldr_model()
        @variable(model, c in Parameter(cost))
        @variable(model, buy >= 0, FirstStage)
        @variable(model, sell >= 0)
        @variable(model, ret >= 0)
        @variable(model, demand in Uncertainty(Uniform(80, 120)))
        @constraint(model, sell + ret <= buy)
        @constraint(model, sell <= demand)
        @objective(model, Max, -c * buy + 8ret + 15sell)
        return model, c
    end
    model, c = build(10.0)
    optimize!(model)
    @test objective_value(model) ≈ 460
    @test value(c) == 10
    set_parameter_value(c, 12.0)
    optimize!(model)
    fresh, _ = build(12.0)
    optimize!(fresh)
    @test objective_value(model) < 460 - 1
    @test objective_value(model) ≈ objective_value(fresh)
    @test objective_bound(model) ≈ objective_bound(fresh)
    @test value(c) == 12
    return
end

# A product of a parameter and a decision in a row is a quadratic row that `canonicalize` reads
# as affine.
function test_parameter_times_decision_in_a_row()
    function build(capacity)
        model = newsvendor()
        @variable(model, c in Parameter(capacity))
        @constraint(model, model[:sell] <= c * model[:buy])
        return model, c
    end
    model, c = build(0.8)
    @test num_constraints(model, QuadExpr, MOI.LessThan{Float64}) == 1
    optimize!(model)
    fresh = newsvendor()
    @constraint(fresh, fresh[:sell] <= 0.8 * fresh[:buy])
    optimize!(fresh)
    @test objective_value(model) < 460 - 1
    @test objective_value(model) ≈ objective_value(fresh)
    @test objective_bound(model) ≈ objective_bound(fresh)
    set_parameter_value(c, 2.0)
    optimize!(model)
    @test objective_value(model) ≈ 460
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

function test_sampled_formulation()
    model = newsvendor()
    set_attribute(model, Formulations(), [PrimalLDR(), DualLDR(), SampledLDR(1000)])
    optimize!(model)
    @test result_count(model) == 2
    @test primal_status(model; result = 2) == MOI.UNKNOWN_RESULT_STATUS
    @test objective_value(model; result = 2) ≈ 460 rtol = 0.05
    @test objective_value(model, SampledLDR(1000)) == objective_value(model; result = 2)
    @test objective_bound(model) ≈ 1460 / 3
    return
end

function test_infeasible_and_unbounded_models()
    model = ldr_model()
    @variable(model, buy >= 0, FirstStage)
    @variable(model, demand in Uncertainty(Uniform(80, 120)))
    @constraint(model, buy <= demand - 200)
    @objective(model, Max, buy)
    optimize!(model)
    @test termination_status(model) == MOI.INFEASIBLE
    @test result_count(model) == 1
    @test_throws MOI.GetAttributeNotAllowed value(buy)
    @test_throws MOI.GetAttributeNotAllowed objective_value(model)
    text = sprint(show, solution_summary(model))
    @test occursin("INFEASIBLE", text)
    model = ldr_model()
    @variable(model, buy >= 0, FirstStage)
    @variable(model, demand in Uncertainty(Uniform(80, 120)))
    @objective(model, Max, buy)
    optimize!(model)
    @test termination_status(model) in (MOI.DUAL_INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
    @test result_count(model) == 1
    @test_throws MOI.GetAttributeNotAllowed value(buy)
    return
end

function test_solution_summary()
    model = newsvendor()
    optimize!(model)
    summary = solution_summary(model; verbose = true)
    @test summary.termination_status == MOI.OPTIMAL
    @test summary.objective_value ≈ 460
    @test summary.objective_bound ≈ 1460 / 3
    @test summary.relative_gap ≈ (1460 / 3 - 460) / 460
    text = sprint(show, summary)
    @test occursin("buy", text) && occursin("demand", text)
    return
end

# `TimeLimitSec` limits the whole `optimize!`, so each formulation gets what remains.
function test_time_limit_is_shared()
    model = newsvendor()
    set_attribute(model, Formulations(), [PrimalLDR(), DualLDR(), SampledLDR(10)])
    set_time_limit_sec(model, 60)
    optimize!(model)
    @test objective_value(model) ≈ 460
    results = unsafe_backend(model).results
    for (k, r) in enumerate(results)
        # The time of the earlier solves (and of the builds) is taken from the limit.
        spent = sum(results[l].solve_time for l in 1:k-1; init = 0.0)
        @test 0 < time_limit_sec(r.model) < 60 - spent
    end
    return
end

# Reading the inventory with T = 64 takes far more than a microsecond, so no formulation is built.
function test_time_limit_before_the_build()
    model = inventory(64)
    set_attribute(model, Formulations(), [PrimalLDR(), DualLDR(), SampledLDR(10)])
    set_time_limit_sec(model, 1e-6)
    optimize!(model)
    @test termination_status(model) == MOI.TIME_LIMIT
    @test result_count(model) == 2
    @test !has_values(model)
    for f in (PrimalLDR(), DualLDR(), SampledLDR(10))
        @test termination_status(model, f) == MOI.TIME_LIMIT
        @test primal_status(model, f) == MOI.NO_SOLUTION
        @test occursin("time limit was reached before", raw_status(model, f))
    end
    @test_throws MOI.GetAttributeNotAllowed objective_bound(model)
    return
end

# With no time left after the build, the inner solver stops at once.
function test_time_limit_in_the_inner_solver()
    src = backend(inventory(16))
    form = LinearDecisionRules.canonicalize(src, LinearDecisionRules._model_info(src))
    mom = LinearDecisionRules.moments(form)
    result = solve_formulation(form, DualLDR(), mom, HiGHS.Optimizer; time_limit = 0.0)
    @test result.termination_status == MOI.TIME_LIMIT
    @test time_limit_sec(result.model) == 0
    @test isnan(result.objective_value)
    return
end

end # module TestEndToEnd

TestEndToEnd.runtests()
