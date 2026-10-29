# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestRuleMatrix

using JuMP
using Test

import LinearDecisionRules
import LinearDecisionRules:
    DualLDR, PrimalLDR, RuleMatrix, decision_rule, lift, rule_coefficient, rule_matrix
import MathOptInterface as MOI

include(joinpath(@__DIR__, "..", "problems", "library.jl"))
include(joinpath(@__DIR__, "..", "problems", "legacy.jl"))

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
import .TaggedProblems: newsvendor, newsvendor_three_stages

function test_newsvendor()
    model = newsvendor()
    optimize!(model)
    buy, sell, d = model[:buy], model[:sell], model[:demand]
    @test rule_coefficient(buy) ≈ 120
    @test rule_coefficient(buy, nothing) ≈ 120
    @test rule_coefficient(buy, d) == 0.0
    @test rule_coefficient(sell, d) ≈ 1
    @test rule_coefficient(sell) ≈ 0 atol = 1e-9
    @test rule_coefficient(sell, d) == decision_rule(sell).coefficients[d]
    R = rule_matrix(model)
    @test R isa RuleMatrix
    @test R.columns == [:constant, d]
    @test R.decisions == [buy, sell, model[:ret]]
    @test size(R) == (3, 2)
    @test R.matrix[1, :] ≈ [120, 0] atol = 1e-9
    @test R[2, 2] ≈ 1
    text = sprint(show, R)
    @test occursin("constant", text) && occursin("sell", text)
    negative_zero = RuleMatrix([-0.0 1.0], [buy], [:constant, d], PrimalLDR(), R.lifting)
    @test !occursin("-0.0", sprint(show, negative_zero))
    return
end

# `x(ξ) = X lift(R.lifting, ξ)`: the lifting of the matrix gives the rules on the original
# uncertainties; without breakpoints, it is the identity.
function test_lifting()
    for points in (nothing, [90.0, 100.0, 110.0])
        model = newsvendor(; points)
        optimize!(model)
        d = model[:demand]
        R = rule_matrix(model)
        @test R.lifting.uncertainties == [index(d)]
        for value in (85.0, 100.0, 117.0)
            ξ = [1.0, value]
            points === nothing && @test lift(R.lifting, ξ) == ξ
            x = R.matrix * lift(R.lifting, ξ)
            for (i, v) in enumerate(R.decisions)
                @test x[i] ≈ decision_rule(v)(Dict(d => value)) atol = 1e-6
            end
        end
    end
    return
end

function test_three_stages()
    model = newsvendor_three_stages()
    optimize!(model)
    R = rule_matrix(model)
    names = [name(c) for c in R.columns[2:end]]
    @test R.columns[1] == :constant
    @test sort(names) == ["buy_cost2", "demand2", "demand3"]
    row(x) = R.matrix[findfirst(==(model[x]), R.decisions), :]
    column(η) = findfirst(==(model[η]), R.columns)
    # No coefficient on the columns of a later stage.
    @test row(:buy)[2:end] == zeros(3)
    for x in (:sell2, :ret2, :buy2), η in (:demand3,)
        @test row(x)[column(η)] == 0.0
        @test rule_coefficient(model[x], model[η]) == 0.0
    end
    @test row(:sell3)[column(:demand3)] ≈ 1
    @test rule_coefficient(model[:sell2], model[:demand2]) ≈ row(:sell2)[column(:demand2)]
    return
end

function test_breakpoints()
    for k in 1:4
        points, _ = Problems.NEWSVENDOR_BREAKPOINTS[k]
        legacy = Legacy.VALUES["newsvendor_breakpoints_$k"]["rules"]
        for (formulation, key) in ((PrimalLDR(), "primal"), (DualLDR(), "dual"))
            model = newsvendor(; points)
            optimize!(model)
            d = model[:demand]
            R = rule_matrix(model; formulation)
            @test R.columns == [:constant; [(d, i) for i in 1:length(points)+1]]
            for (x, rule) in legacy[key]
                v = model[Symbol(x)]
                expected = rule["coefficients"]["demand"]
                @test rule_coefficient(v, d; formulation) ≈ expected atol = 1e-6
                @test rule_coefficient(v; formulation) ≈ rule["constant"] atol = 1e-6
                i = findfirst(==(v), R.decisions)
                @test R.matrix[i, 2:end] ≈ expected atol = 1e-6
                @test R.matrix[i, 1] ≈ rule["constant"] atol = 1e-6
            end
        end
    end
    return
end

function test_dual_formulation()
    model = newsvendor()
    optimize!(model)
    R = rule_matrix(model; formulation = DualLDR())
    @test R.formulation == DualLDR()
    @test size(R) == (3, 2)
    @test rule_coefficient(model[:buy]; formulation = DualLDR()) ≈ R.matrix[1, 1]
    return
end

function test_errors()
    model = newsvendor()
    buy, d = model[:buy], model[:demand]
    @test_logs (:warn,) @test_throws OptimizeNotCalled rule_coefficient(buy)
    @test_logs (:warn,) @test_throws OptimizeNotCalled rule_matrix(model)
    optimize!(model)
    @test_throws ArgumentError rule_coefficient(d, d)
    @test_throws ArgumentError rule_coefficient(buy, buy)
    other = newsvendor()
    @test_throws ArgumentError rule_coefficient(buy, other[:demand])
    sampled = LinearDecisionRules.SampledLDR(5; seed = 1)
    @test_throws MOI.GetAttributeNotAllowed rule_coefficient(buy; formulation = sampled)
    @test_throws MOI.GetAttributeNotAllowed rule_matrix(model; formulation = sampled)
    return
end

end # module

TestRuleMatrix.runtests()
