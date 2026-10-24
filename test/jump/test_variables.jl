# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestVariables

using JuMP
using Test

import Distributions
import Distributions: Uniform
import LinearDecisionRules
import LinearDecisionRules:
    BreakPoints, FirstStage, Stage, Uncertainty, set_stage, stage, unset_stage

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

function test_stage_tag()
    model = Model()
    @variable(model, x, Stage(1))
    @variable(model, y >= 0, Stage(2))
    @variable(model, z)
    @test stage(x) == 1
    @test stage(y) == 2
    @test stage(z) === nothing
    @test lower_bound(y) == 0
    @test num_constraints(model, VariableRef, Stage) == 2
    text = sprint(print, model)
    @test occursin(" x in Stage(1)
", text) && occursin(" y in Stage(2)
", text)
    return
end

function test_first_stage_alias()
    model = Model()
    @variable(model, x >= 0, FirstStage)
    @test stage(x) == 1
    @variable(model, y[1:2], FirstStage)
    @test stage.(y) == [1, 1]
    return
end

function test_stage_composes_with_integrality_and_bounds()
    model = Model()
    @variable(model, a, Int, Stage(1))
    @variable(model, b, Stage(1), Int)
    @variable(model, c, Bin, Stage(1))
    @variable(model, 1 <= d <= 3, Stage(2))
    @variable(model, e, Stage(2), start = 2.0)
    @variable(model, f in MOI.Integer(), Stage(1))
    @variable(model, g[1:2] in MOI.ZeroOne(), Stage(1))
    @test is_integer(a) && is_integer(b) && is_binary(c) && is_integer(f)
    @test is_binary(g[1]) && is_binary(g[2])
    @test lower_bound(d) == 1 && upper_bound(d) == 3
    @test start_value(e) == 2.0
    @test stage.([a, b, c, d, e, f, g...]) == [1, 1, 1, 2, 2, 1, 1, 1]
    return
end

function test_stage_in_containers()
    model = Model()
    @variable(model, x[t = 1:3] >= 0, Stage(t))
    @test stage.(x) == [1, 2, 3]
    @variable(model, y[t = 1:2, s = 1:2], Stage(t + s))
    @test stage.(y) == [2 3; 3 4]
    @variable(model, z[i = 2:3], Stage(i), Int)
    @test stage(z[2]) == 2 && stage(z[3]) == 3
    @test all(is_integer, z)
    @variable(model, w[1:2, 1:2] >= 0, FirstStage)
    @test all(==(1), stage.(w))
    return
end

function test_uncertainty_with_stage()
    model = Model()
    d = Distributions.Uniform(80, 120)
    @variable(model, u in Uncertainty(d), Stage(3))
    @test stage(u) == 3
    @variable(model, v[t = 2:3] in Uncertainty(d), Stage(t))
    @test collect(stage.(v)) == [2, 3]
    @variable(model, w[1:2] in Uncertainty(d))
    @test stage.(w) == [nothing, nothing]
    mv = Distributions.product_distribution(Uniform(0, 1), Uniform(0, 2))
    @variable(model, p[1:2] in Uncertainty(mv), Stage(2))
    @test stage.(p) == [2, 2]
    @variable(model, q[t = 1:2] in Uncertainty(mv), Stage(t + 1))
    @test stage.(q) == [2, 3]
    @variable(model, r[1:2] in Uncertainty(mv))
    @test stage.(r) == [nothing, nothing]
    src = backend(model)
    S = LinearDecisionRules.UnivariateUncertainty{Distributions.Uniform{Float64}}
    @test length(MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex,S}())) == 1 + 2 + 2
    M = typeof(Uncertainty(mv))
    @test length(MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorOfVariables,M}())) == 3
    return
end

# A vector uncertainty over a named range is a `DenseAxisArray`, whose variables are those of the
# law, in column-major order.
function test_uncertainty_in_named_range()
    model = Model()
    law = Distributions.product_distribution(Uniform(0, 1), Uniform(0, 2))
    R = 1:2
    @variable(model, a[R] in Uncertainty(law))
    @test a isa JuMP.Containers.DenseAxisArray
    @test name.(a.data) == ["a[1]", "a[2]"]
    @test stage.(a) == [nothing, nothing]
    S = [:p, :q]
    @variable(model, b[S] in Uncertainty(law), Stage(3))
    @test b[:q] isa VariableRef && name(b[:q]) == "b[q]"
    @test [stage(b[s]) for s in S] == [3, 3]
    @variable(model, c[t = 2:3, 1:1] in Uncertainty(law), Stage(t))
    @test [stage(c[2, 1]), stage(c[3, 1])] == [2, 3]
    four = Distributions.product_distribution(fill(Uniform(0, 1), 4)...)
    @variable(model, e[1:2, 1:2] in Uncertainty(four))
    @test size(e) == (2, 2) && name(e[2, 1]) == "e[2,1]"
    @variable(model, f[S, 1:2] in Uncertainty(four))
    @test size(f) == (2, 2) && name(f[:q, 1]) == "f[q,1]"
    # One declaration per call: `a`'s two variables are the components of one law.
    src = backend(model)
    M = typeof(Uncertainty(law))
    @test length(MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorOfVariables,M}())) == 3
    return
end

function test_stage_helpers()
    model = Model()
    @variable(model, x)
    @test stage(x) === nothing
    set_stage(x, 2)
    @test stage(x) == 2
    set_stage(x, 3)
    @test stage(x) == 3
    @test num_constraints(model, VariableRef, Stage) == 1
    unset_stage(x)
    @test stage(x) === nothing
    @test num_constraints(model, VariableRef, Stage) == 0
    unset_stage(x)
    @test stage(x) === nothing
    @test_throws ArgumentError set_stage(x, 0)
    # A stage is deleted with its variable.
    @variable(model, y, Stage(2))
    delete(model, y)
    @test num_constraints(model, VariableRef, Stage) == 0
    return
end

function test_stage_survives_other_deletions()
    model = Model()
    @variable(model, x, Stage(1))
    @variable(model, y, Stage(2))
    @variable(model, z, Stage(3))
    delete(model, y)
    @test stage(x) == 1 && stage(z) == 3
    return
end

function test_break_points()
    model = Model()
    d = Distributions.Uniform(80, 120)
    @variable(model, u in Uncertainty(d))
    @variable(model, v in Uncertainty(d))
    @test get_attribute(u, BreakPoints()) === nothing
    set_attribute(u, BreakPoints(), [90.0, 100.0, 110.0])
    @test get_attribute(u, BreakPoints()) == [90.0, 100.0, 110.0]
    @test get_attribute(v, BreakPoints()) === nothing
    # They survive the deletion of another variable.
    delete(model, v)
    @test get_attribute(u, BreakPoints()) == [90.0, 100.0, 110.0]
    return
end

end # module TestVariables

TestVariables.runtests()
