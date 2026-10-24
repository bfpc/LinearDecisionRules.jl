# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# `evaluate`: the decision rules applied to scenarios.

module TestSimulate

using JuMP
using Test

import Distributions
import Distributions: DiscreteNonParametric, Uniform
import HiGHS
import LinearDecisionRules
import LinearDecisionRules:
    BreakPoints,
    DualLDR,
    Evaluation,
    FirstStage,
    Formulations,
    LDRModel,
    MvDiscreteNonParametric,
    PrimalLDR,
    SampledLDR,
    Stage,
    Uncertainty,
    ValidInequality,
    decision_rule,
    evaluate,
    expected_value

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

function ldr_model()
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    return model
end

function newsvendor(law = Uniform(80, 120))
    model = ldr_model()
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @variable(model, d in Uncertainty(law))
    @constraint(model, balance, sell + ret <= buy)
    @constraint(model, sell <= d)
    @objective(model, Max, -10buy + 8ret + 15sell)
    optimize!(model)
    return model
end

# The objective has the product `price * sell`.
function random_price(demand_law, price_law; breakpoints = nothing)
    model = ldr_model()
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @variable(model, d in Uncertainty(demand_law))
    @variable(model, price in Uncertainty(price_law))
    @constraint(model, sell + ret <= buy)
    @constraint(model, sell <= d)
    @objective(model, Max, -10buy + 8ret + price * sell)
    breakpoints === nothing || set_attribute(d, BreakPoints(), breakpoints)
    optimize!(model)
    return model
end

function error_message(f)
    try
        f()
    catch err
        return sprint(showerror, err)
    end
    return ""
end

function test_evaluate_newsvendor()
    model = newsvendor()
    d, buy, sell, ret = model[:d], model[:buy], model[:sell], model[:ret]
    r = evaluate(model, d => 90.0)
    @test r isa Evaluation
    @test value(r, buy) ≈ 120
    @test r[sell] ≈ 90
    @test r[ret] ≈ 30
    @test value(r, d) == 90
    @test value(r, 2sell + d - 1) ≈ 269
    # -10 * 120 + 8 * 30 + 15 * 90
    @test objective_value(r) ≈ 390
    @test r.objective_value ≈ 390
    @test r.max_violation <= 1e-6
    @test isempty(r.violations)
    @test r.in_support
    @test occursin("Evaluation of ", sprint(show, r)) && occursin("PrimalLDR()", sprint(show, r))
    # The same scenario as a `Dict`, with an integer value, and the rule of `decision_rule`.
    r2 = evaluate(model, Dict(d => 90))
    @test r2[sell] == r[sell] && r2.objective_value == r.objective_value
    @test r[sell] ≈ decision_rule(sell)(d => 90.0)
    @test r[ret] ≈ decision_rule(ret)(d => 90.0)
    # The scenario is stored, and the variables of the model are decisions or uncertainties.
    @test r.scenario == Dict(d => 90.0)
    @test Set(keys(r.decisions)) == Set([buy, sell, ret])
end

function test_evaluate_parameter()
    model = ldr_model()
    @variable(model, p in Parameter(10.0))
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, d in Uncertainty(Uniform(80, 120)))
    @constraint(model, sell <= buy)
    @constraint(model, sell <= d)
    @objective(model, Max, -p * buy + 15sell)
    optimize!(model)
    r = evaluate(model, d => 100.0)
    @test value(r, p) == 10
    @test value(r, 2p + buy) ≈ 20 + value(r, buy)
end

function test_evaluate_violations()
    model = newsvendor()
    d, sell, ret = model[:d], model[:sell], model[:ret]
    # Outside the support the rule need not be feasible: it returns `120 - 130 < 0`.
    r = evaluate(model, d => 130.0; outside_support = :ignore)
    @test !r.in_support
    @test r[sell] ≈ 130
    @test r[ret] ≈ -10
    @test r.max_violation ≈ 10
    @test first.(r.violations) == ["ret >= 0"]
    @test r.violations[1].second ≈ 10
    @test occursin("1 violated: ret >= 0", sprint(show, r))
    # A large tolerance forgives them.
    r = evaluate(model, d => 130.0; outside_support = :ignore, tolerance = 1.0)
    @test isempty(r.violations) && r.max_violation ≈ 10
end

function test_evaluate_errors()
    model = newsvendor()
    d, buy = model[:d], model[:buy]
    msg = error_message(() -> evaluate(model, d => 130.0))
    @test occursin("d = 130.0 is outside its support [80.0, 120.0]", msg)
    @test occursin("outside_support", msg)
    @test_logs (:warn, r"outside its support") evaluate(model, d => 130.0; outside_support = :warn)
    @test_logs evaluate(model, d => 130.0; outside_support = :ignore)
    msg = error_message(() -> evaluate(model, Dict{Any,Any}()))
    @test occursin("no value for the uncertainties d", msg)
    msg = error_message(() -> evaluate(model, buy => 1.0, d => 90.0))
    @test occursin("`buy` is not an uncertainty", msg)
    @test occursin("is NaN", error_message(() -> evaluate(model, d => NaN)))
    @test occursin("maps an uncertainty", error_message(() -> evaluate(model, "d" => 1.0)))
    msg = error_message(() -> evaluate(model, [d] => [1.0, 2.0]))
    @test occursin("2 values for 1 variables", msg)
    msg = error_message(() -> evaluate(model, d => 90; outside_support = :no))
    @test occursin("`outside_support`", msg)
    # The variable of another model.
    other = newsvendor()
    @test occursin("not an uncertainty", error_message(() -> evaluate(model, other[:d] => 90.0)))
    # No solution.
    model = ldr_model()
    @variable(model, x >= 0)
    @variable(model, z in Uncertainty(Uniform(0, 1)))
    @constraint(model, x <= z)
    @objective(model, Max, x)
    @test occursin("optimize!", error_message(() -> evaluate(model, z => 0.5)))
    optimize!(model)
    @test evaluate(model, z => 0.25)[x] ≈ 0.25
    # The dual and a formulation that was not solved have no rule to apply.
    model = newsvendor()
    d = model[:d]
    set_attribute(model, Formulations(), [PrimalLDR(), DualLDR()])
    optimize!(model)
    msg = error_message(() -> evaluate(model, d => 90; formulation = DualLDR()))
    @test occursin("value is a bound", msg)
    msg = error_message(() -> evaluate(model, d => 90; formulation = SampledLDR(10)))
    @test occursin("not solved", msg)
end

# Vectors and arrays of uncertainties, and a named range (a `DenseAxisArray`).
function test_evaluate_containers()
    model = ldr_model()
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell[1:2] >= 0)
    law = Distributions.product_distribution(Uniform(1, 2), Uniform(2, 4))
    @variable(model, d[1:2] in Uncertainty(law))
    @constraint(model, sell .<= d)
    @constraint(model, sum(sell) <= buy)
    @objective(model, Max, 3sum(sell) - 2buy)
    optimize!(model)
    scenario = Dict(d[1] => 1.5, d[2] => 3.0)
    r = evaluate(model, d => [1.5, 3.0])
    for x in (sell[1], sell[2], model[:buy])
        @test r[x] ≈ decision_rule(x)(scenario)
    end
    @test value(r, d[2]) == 3
    @test isempty(r.violations)
    @test evaluate(model, d[1] => 1.5, d[2] => 3.0).decisions == r.decisions
    @test evaluate(model, Dict(d => [1.5, 3.0])).decisions == r.decisions
    @test evaluate(model, [d[2] => 3.0, d[1] => 1.5]).decisions == r.decisions
end

# The rule is exact on the atoms of a discrete law: the weighted mean of the objective over them
# is the objective value of the model, for linear and quadratic objectives and with breakpoints.
function weighted_mean(model, scenarios, p)
    evaluations = [evaluate(model, scenario) for scenario in scenarios]
    return sum(p .* [e.objective_value for e in evaluations]), evaluations
end

function test_exact_mean_over_atoms()
    atoms, p = [82.0, 95.0, 101.0, 118.0], [0.1, 0.4, 0.3, 0.2]
    model = newsvendor(DiscreteNonParametric(atoms, p))
    mean, s = weighted_mean(model, [Dict(model[:d] => a) for a in atoms], p)
    @test mean ≈ objective_value(model)
    @test all(e -> isempty(e.violations), s)
    # Two independent uncertainties, a product in the objective, with and without breakpoints.
    demands, pd = [82.0, 95.0, 101.0, 118.0], [0.1, 0.4, 0.3, 0.2]
    prices, pp = [13.0, 15.5, 17.0], [0.3, 0.3, 0.4]
    for breakpoints in (nothing, [96.0], [90.0, 105.0])
        model = random_price(
            DiscreteNonParametric(demands, pd),
            DiscreteNonParametric(prices, pp);
            breakpoints,
        )
        d, price = model[:d], model[:price]
        scenarios = [Dict(d => a, price => b) for a in demands for b in prices]
        weights = [x * y for x in pd for y in pp]
        mean, s = weighted_mean(model, scenarios, weights)
        @test mean ≈ objective_value(model) atol = 1e-6
        @test all(e -> isempty(e.violations), s)
    end
end

function test_piecewise_rule()
    model = random_price(Uniform(80, 120), Uniform(13, 17); breakpoints = [100.0])
    d, price, sell = model[:d], model[:price], model[:sell]
    rule = decision_rule(sell)
    for a in (81.0, 100.0, 107.5, 119.0)
        r = evaluate(model, d => a, price => 15.0)
        @test r[sell] ≈ rule(d => a, price => 15.0) atol = 1e-8
        @test isempty(r.violations)
    end
    # The lifting is applied to the scenario, which is given in the original coordinates.
    @test !(evaluate(model, d => 90.0, price => 15.0)[sell] ≈ 100.0)
    msg = error_message(() -> evaluate(model, d => 90.0))
    @test occursin("no value for the uncertainties price", msg)
end

function test_evaluate_valid_inequality()
    model = ldr_model()
    @variable(model, x >= 0)
    law = Distributions.product_distribution(Uniform(0, 1), Uniform(0, 1))
    @variable(model, z[1:2] in Uncertainty(law))
    @constraint(model, z[1] + z[2] <= 1.5, ValidInequality())
    @constraint(model, x >= z[1] + z[2])
    @objective(model, Min, x)
    optimize!(model)
    @test objective_value(model) ≈ 1
    @test isempty(evaluate(model, z => [0.5, 0.5]).violations)
    msg = error_message(() -> evaluate(model, z => [0.9, 0.9]))
    @test occursin("valid inequality", msg) && occursin("z[1] + z[2] <= 1.5", msg)
    r = evaluate(model, z => [0.9, 0.9]; outside_support = :ignore)
    @test !r.in_support
end

end # module

TestSimulate.runtests()
