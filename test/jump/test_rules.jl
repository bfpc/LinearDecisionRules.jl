# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestRules

using JuMP
using Test

import Distributions
import Random
import Distributions: Uniform
import HiGHS
import LinearDecisionRules
import LinearDecisionRules:
    DecisionRule,
    DualLDR,
    FirstStage,
    Formulations,
    LDRModel,
    MvDiscreteNonParametric,
    PrimalLDR,
    SampledLDR,
    Uncertainty,
    decision_rule,
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

function newsvendor()
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @variable(model, d in Uncertainty(Uniform(80, 120)))
    @constraint(model, sell + ret <= buy)
    @constraint(model, sell <= d)
    @objective(model, Max, -10buy + 8ret + 15sell)
    return model
end

# `sell` depends on the price `p` through `sell <= d + 2p`.
function price_model()
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @variable(model, d in Uncertainty(Uniform(80, 120)))
    @variable(model, p in Uncertainty(Uniform(13, 17)))
    @constraint(model, sell + ret <= buy)
    @constraint(model, sell <= d + 2p)
    @objective(model, Max, -10buy + 8ret + 15sell)
    return model
end

function test_decision_rule()
    model = newsvendor()
    optimize!(model)
    d = model[:d]
    rule = decision_rule(model[:buy])
    @test rule isa DecisionRule
    @test rule.constant ≈ 120 && isempty(rule.coefficients)
    @test rule() ≈ 120
    rule = decision_rule(model[:sell])
    @test rule.constant ≈ 0 atol = 1e-8
    @test rule.coefficients == Dict(d => 1.0) || rule.coefficients[d] ≈ 1
    @test length(rule.coefficients) == 1
    @test rule(Dict(d => 90.0)) ≈ 90
    @test rule(d => 110.0) ≈ 110
    rule = decision_rule(model[:ret])
    @test rule.constant ≈ 120 && rule.coefficients[d] ≈ -1
    @test rule(d => 90.0) ≈ 30
    # The rule of an uncertainty is itself.
    rule = decision_rule(d)
    @test rule.constant == 0 && rule.coefficients == Dict(d => 1.0)
    return
end

function test_decision_rule_show()
    model = newsvendor()
    optimize!(model)
    @test sprint(show, decision_rule(model[:buy])) == "DecisionRule: 120.0"
    @test sprint(show, decision_rule(model[:ret])) == "DecisionRule: 120.0 - 1.0 d"
    return
end

function test_rules_of_each_formulation()
    model = newsvendor()
    set_attribute(model, Formulations(), [PrimalLDR(), DualLDR(), SampledLDR(100)])
    optimize!(model)
    # The dual rule is not a policy: it buys less.
    @test decision_rule(model[:buy]; formulation = DualLDR()).constant ≈ 320 / 3
    @test decision_rule(model[:buy]; formulation = PrimalLDR()).constant ≈ 120
    @test isfinite(decision_rule(model[:buy]; formulation = SampledLDR(100)).constant)
    set_attribute(model, Formulations(), [PrimalLDR()])
    optimize!(model)
    err = try
        decision_rule(model[:buy]; formulation = DualLDR())
    catch err
        err
    end
    @test err isa MOI.GetAttributeNotAllowed
    return
end

# A rule is plain data: it survives changes of the model, which discard the solution.
function test_decision_rule_survives_changes()
    model = newsvendor()
    optimize!(model)
    rule = decision_rule(model[:ret])
    d = model[:d]
    @variable(model, extra >= 0)
    delete(model, extra)
    @test rule(d => 90.0) ≈ 30
    @test_logs (:warn,) @test_throws OptimizeNotCalled decision_rule(model[:ret])
    # Even if the uncertainty is deleted.
    delete(model, d)
    @test rule(d => 90.0) ≈ 30
    # A new solve gives new rules.
    model = newsvendor()
    optimize!(model)
    set_upper_bound(model[:buy], 100)
    optimize!(model)
    @test decision_rule(model[:buy]).constant ≈ 100
    return
end

function test_expected_value_of_affine_expressions()
    model = newsvendor()
    optimize!(model)
    buy, sell, ret, d = model[:buy], model[:sell], model[:ret], model[:d]
    @test expected_value(buy) ≈ 120
    @test expected_value(sell) ≈ value(sell)
    @test expected_value(d) ≈ 100
    @test expected_value(-10buy + 8ret + 15sell) ≈ 460
    @test expected_value(2buy + 1) ≈ 241
    @test expected_value(model[:buy] + model[:sell]; formulation = DualLDR()) ≈
          320 / 3 +
          decision_rule(sell; formulation = DualLDR()).constant +
          100 * decision_rule(sell; formulation = DualLDR()).coefficients[d]
    @test expected_value(3.0) == 3.0
    @test expected_value(3) == 3.0
    @test expected_value(AffExpr(5.0)) == 5.0
    return
end

function test_expected_value_of_products()
    model = price_model()
    optimize!(model)
    sell, p, d = model[:sell], model[:p], model[:d]
    rule = decision_rule(sell)
    c, a, b = rule.constant, get(rule.coefficients, d, 0.0), get(rule.coefficients, p, 0.0)
    @test b != 0
    variance = 16 / 12
    # E[p * sell] with independent p and d, from the rule.
    exact = c * 15 + a * 100 * 15 + b * (variance + 15^2)
    @test expected_value(p * sell) ≈ exact
    # `value` plugs in the means, which is not the expectation of a product.
    @test value(p * sell) ≈ 15 * value(sell)
    @test abs(expected_value(p * sell) - value(p * sell)) ≈ abs(b * variance)
    # The second moment of the rule.
    @test expected_value(sell^2) ≈
          c^2 +
          a^2 * (Distributions.var(Uniform(80, 120)) + 100^2) +
          b^2 * (variance + 15^2) +
          2c * a * 100 +
          2c * b * 15 +
          2a * b * 100 * 15
    @test expected_value(p^2) ≈ variance + 15^2
    @test expected_value(2p * sell - 3sell + 4p + 5) ≈
          2expected_value(p * sell) - 3expected_value(sell) + 4 * 15 + 5
    return
end

# The expectation of a product by simulation, with the rule.
function test_expected_value_by_sampling()
    model = price_model()
    optimize!(model)
    sell, p, d = model[:sell], model[:p], model[:d]
    rule = decision_rule(sell)
    rng = Random.Xoshiro(1)
    draw() = Dict(d => rand(rng, Uniform(80, 120)), p => rand(rng, Uniform(13, 17)))
    scenarios = [draw() for _ in 1:200_000]
    estimate = sum(s[p] * rule(s) for s in scenarios) / length(scenarios)
    @test expected_value(p * sell) ≈ estimate rtol = 0.01
    return
end

function test_correlated_uncertainties()
    scenarios = [[80.0, 60.0], [100.0, 80.0], [120.0, 100.0], [80.0, 100.0], [120.0, 60.0]]
    law = MvDiscreteNonParametric(scenarios, [0.2, 0.3, 0.2, 0.15, 0.15])
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x >= 0, FirstStage)
    @variable(model, y >= 0)
    @variable(model, d[1:2] in Uncertainty(law))
    @constraint(model, y <= x)
    @constraint(model, y <= d[1] + d[2])
    @objective(model, Max, y - 0.1x)
    optimize!(model)
    mean = Distributions.mean(law)
    exact = Distributions.cov(law)[1, 2] + mean[1] * mean[2]
    @test expected_value(d[1] * d[2]) ≈ exact
    @test expected_value(d[1]^2) ≈ Distributions.var(law)[1] + mean[1]^2
    @test value(d[1]) ≈ mean[1] && value(d[2]) ≈ mean[2]
    return
end

end # module TestRules

TestRules.runtests()
