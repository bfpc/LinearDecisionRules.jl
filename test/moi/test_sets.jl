# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestSets

using JuMP
using Test

import Distributions
import Distributions: Uniform
import LinearDecisionRules
import LinearDecisionRules: Stage, Uncertainty

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

function test_uncertainty_constructor()
    s = Uncertainty(Distributions.Uniform(80, 120))
    @test s isa LinearDecisionRules.UnivariateUncertainty
    @test s isa MOI.AbstractScalarSet
    mv = Distributions.product_distribution(Uniform(0, 1), Uniform(0, 2))
    v = Uncertainty(mv)
    @test v isa LinearDecisionRules.MultivariateUncertainty
    @test v isa MOI.AbstractVectorSet
    @test MOI.dimension(v) == 2
    err = try
        Uncertainty(Distributions.LKJ(3, 1.0))
    catch err
        err
    end
    @test err isa ArgumentError
    @test occursin("must be univariate or multivariate", err.msg)
    return
end

function test_uncertainty_equality_and_copy()
    a = Uncertainty(Distributions.Uniform(80, 120))
    @test a == Uncertainty(Distributions.Uniform(80, 120))
    @test hash(a) == hash(Uncertainty(Distributions.Uniform(80, 120)))
    @test a != Uncertainty(Distributions.Uniform(80, 121))
    @test a != Uncertainty(Distributions.Normal(0, 1))
    @test copy(a) == a
    mv(x) = Uncertainty(Distributions.MvNormal([0.0, x], [1.0 0.0; 0.0 1.0]))
    @test mv(1.0) == mv(1.0)
    @test hash(mv(1.0)) == hash(mv(1.0))
    @test mv(1.0) != mv(2.0)
    @test copy(mv(1.0)) == mv(1.0)
    @test MOI.dimension(mv(1.0)) == 2
    @test Uncertainty(Distributions.Uniform(0, 1)) != Stage(1)
    return
end

function test_stage_set()
    @test Stage(2) == Stage(2)
    @test Stage(2) != Stage(3)
    @test copy(Stage(2)) == Stage(2)
    @test Stage(Int32(2)).t === 2
    @test sprint(show, Stage(2)) == "Stage(2)"
    @test LinearDecisionRules.FirstStage == Stage(1)
    for t in (0, -1)
        err = try
            Stage(t)
        catch err
            err
        end
        @test err isa ArgumentError
        @test occursin("stages start at 1", err.msg)
    end
    return
end

function test_metadata_constraint()
    f = LinearDecisionRules._is_metadata_constraint
    @test f(MOI.VariableIndex, typeof(Uncertainty(Distributions.Uniform(0, 1))))
    mv = Distributions.product_distribution(Uniform(0, 1), Uniform(0, 2))
    @test f(MOI.VectorOfVariables, typeof(Uncertainty(mv)))
    @test f(MOI.VariableIndex, Stage)
    @test !f(MOI.VariableIndex, MOI.GreaterThan{Float64})
    @test !f(MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64})
    return
end

function test_sets_in_a_jump_model()
    model = Model()
    @variable(model, d in Uncertainty(Distributions.Uniform(80, 120)))
    mv = Distributions.product_distribution(Uniform(0, 1), Uniform(0, 2))
    @variable(model, e[1:2] in Uncertainty(mv))
    @variable(model, x)
    @constraint(model, x in Stage(2))
    text = sprint(print, model)
    @test occursin(" d in Uncertainty(Distributions.Uniform{Float64}(a=80.0, b=120.0))
", text)
    @test occursin(" [e[1], e[2]] in Uncertainty(ProductDistribution of dimension 2)
", text)
    @test occursin(" x in Stage(2)
", text)
    src = backend(model)
    S = LinearDecisionRules.UnivariateUncertainty{Distributions.Uniform{Float64}}
    cis = MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex,S}())
    @test length(cis) == 1
    @test MOI.get(src, MOI.ConstraintFunction(), cis[1]) == index(d)
    @test MOI.get(src, MOI.ConstraintSet(), cis[1]) == Uncertainty(Distributions.Uniform(80, 120))
    @test num_constraints(model, VariableRef, Stage) == 1
    # A stage dies with its variable.
    delete(model, x)
    @test num_constraints(model, VariableRef, Stage) == 0
    return
end

function test_mof_write()
    model = Model()
    @variable(model, x)
    @constraint(model, x in Stage(2))
    @variable(model, d in Uncertainty(Distributions.Uniform(0, 1)))
    dest = MOI.FileFormats.Model(; format = MOI.FileFormats.FORMAT_MOF)
    MOI.copy_to(dest, backend(model))
    file = tempname() * ".mof.json"
    MOI.write_to_file(dest, file)
    text = read(file, String)
    rm(file)
    @test occursin("\"set\":{\"type\":\"Stage\",\"t\":2}", text)
    @test occursin("\"type\":\"UnivariateUncertainty\"", text)
    return
end

end # module TestSets

TestSets.runtests()
