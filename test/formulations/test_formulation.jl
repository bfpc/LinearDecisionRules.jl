# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestFormulation

using JuMP
using Test

import Distributions
import LinearAlgebra
import LinearDecisionRules
import LinearDecisionRules:
    DualLDR,
    FormulationCapabilities,
    InformationStructure,
    ModelInfo,
    PrimalLDR,
    SampledLDR,
    UncertaintyDeclaration,
    canonicalize,
    capabilities,
    moments,
    second_moment

const LDR = LinearDecisionRules

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

function test_sampled_ldr()
    @test SampledLDR(100) == SampledLDR(100, 1234)
    @test SampledLDR(10; seed = 5).seed == 5
    @test SampledLDR(10; seed = 5).num_scenarios == 10
    @test sprint(show, SampledLDR(10; seed = 5)) == "SampledLDR(10; seed = 5)"
    @test_throws ArgumentError SampledLDR(0)
    return
end

function test_capabilities()
    @test capabilities(PrimalLDR()) isa FormulationCapabilities
    @test capabilities(PrimalLDR()).rule_status == MOI.FEASIBLE_POINT
    @test !capabilities(PrimalLDR()).bound
    @test capabilities(DualLDR()).rule_status == MOI.NO_SOLUTION
    @test capabilities(DualLDR()).bound
    @test capabilities(SampledLDR(10)).rule_status == MOI.UNKNOWN_RESULT_STATUS
    @test !capabilities(SampledLDR(10)).bound
    return
end

# Three stages, one univariate and one bivariate law, and every kind of objective term.
function quadratic_model()
    model = Model()
    @variable(model, x1 >= 1, integer = true)
    @variable(model, x2 <= 4)
    @variable(model, x3)
    @variable(model, η1)
    @variable(model, η[1:2])
    @constraint(model, x2 + x3 == η1 + 2η[2])
    @constraint(model, x1 + x2 <= 10)
    @objective(
        model,
        Min,
        x1^2 + 2x1 * x2 + 3x3^2 - x2 * x3 + x2 * η1 + 4x3 * η[2] - x1 * η[1] +
        η1 * η[1] +
        2η[2]^2 +
        5η1 +
        7 +
        x1 - 3x3,
    )
    info = ModelInfo(;
        uncertainties = [
            UncertaintyDeclaration(Distributions.Uniform(0, 1), [index(η1)], [2]),
            UncertaintyDeclaration(
                Distributions.MvNormal([1.0, 2.0], [2.0 0.5; 0.5 1.0]),
                index.(η),
                [3, 3],
            ),
        ],
        decision_stages = Dict(index(x1) => 1, index(x2) => 2, index(x3) => 3),
    )
    return canonicalize(backend(model), info)
end

function test_rule_variables()
    form = quadratic_model()
    info = InformationStructure(form)
    model = Model()
    X = LDR._rule_variables!(model, form, info)
    @test length.(X) == [1, 2, 4]
    @test is_integer(X[1][1])
    # A decision that observes only the constant column keeps its bounds as variable bounds.
    @test lower_bound(X[1][1]) == 1
    @test !has_upper_bound(X[2][1])
    @test num_variables(model) == 7
    return
end

function test_equalities()
    form = quadratic_model()
    info = InformationStructure(form)
    model = Model()
    X = LDR._rule_variables!(model, form, info)
    LDR._add_equalities!(model, form, info, X)
    # x2 + x3 == η1 + 2η[2]: one row per column of ξ that x3 observes.
    @test num_constraints(model, AffExpr, MOI.EqualTo{Float64}) == 4
    return
end

# The structured objective equals `tr(X'PXM) + tr(C'XM) + E[η'Qη + d'η] + f` at random rules.
function test_objective()
    form = quadratic_model()
    mom = moments(form)
    info = InformationStructure(form)
    model = Model()
    X = LDR._rule_variables!(model, form, info)
    LDR._set_objective!(model, form, mom, X)
    @test objective_sense(model) == MOI.MIN_SENSE
    Xv = zeros(3, 4)
    point = Dict{VariableRef,Float64}()
    for i in 1:3, j in eachindex(X[i])
        Xv[i, j] = point[X[i][j]] = sin(7i + 3j)
    end
    M = second_moment(mom)
    P, C, Q = Matrix(form.P), Matrix(form.C), Matrix(form.Q)
    expected =
        LinearAlgebra.tr(Xv' * P * Xv * M) +
        LinearAlgebra.tr(C' * Xv * M) +
        LinearAlgebra.tr(Q * M[2:end, 2:end]) +
        form.d' * M[2:end, 1] +
        form.f
    # The means of the decisions are variables, each fixed by an equality row.
    for c in all_constraints(model, AffExpr, MOI.EqualTo{Float64})
        terms = constraint_object(c).func.terms
        (m,) = [v for v in keys(terms) if !haskey(point, v)]
        point[m] = -sum(a * point[v] for (v, a) in terms if v != m) / terms[m]
    end
    @test value(v -> point[v], objective_function(model)) ≈ expected
    return
end

function test_objective_feasibility()
    model = Model()
    @variable(model, x >= 0)
    form = canonicalize(backend(model), ModelInfo())
    inner = Model()
    X = LDR._rule_variables!(inner, form, InformationStructure(form))
    LDR._set_objective!(inner, form, moments(form), X)
    @test objective_sense(inner) == MOI.FEASIBILITY_SENSE
    return
end

end # module TestFormulation

TestFormulation.runtests()
