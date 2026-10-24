# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestConstraints

using JuMP
using Test

import LinearDecisionRules
import LinearDecisionRules: ConstraintStage, Stage, Truncate, ValidInequality, XiRole
import LinearDecisionRules: set_stage, stage, unset_stage

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

function test_xi_roles()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    a = @constraint(model, x + y <= 3, ValidInequality())
    b = @constraint(model, x >= 1, Truncate())
    c = @constraint(model, y == 0)
    @test get_attribute(a, XiRole()) == ValidInequality()
    @test get_attribute(b, XiRole()) == Truncate()
    @test get_attribute(c, XiRole()) === nothing
    @test get_attribute(a, ConstraintStage()) === nothing
    # The tag does not change the row.
    @test sprint(print, constraint_object(a).func) == "x + y"
    @test constraint_object(a).set == MOI.LessThan(3.0)
    @test sprint(print, a) == "x + y <= 3"
    return
end

function test_xi_roles_in_containers()
    model = Model()
    @variable(model, x[1:3])
    a = @constraint(model, [i = 1:3], x[i] <= i, ValidInequality())
    @test all(get_attribute(c, XiRole()) == ValidInequality() for c in a)
    b = @constraint(model, x .>= 0, Truncate())
    @test all(get_attribute(c, XiRole()) == Truncate() for c in b)
    d = @constraint(model, sum(x) in MOI.Interval(0.0, 1.0), ValidInequality())
    @test get_attribute(d, XiRole()) == ValidInequality()
    e = @constraint(model, 2x[1] == 1, ValidInequality())
    @test get_attribute(e, XiRole()) == ValidInequality()
    return
end

# A vector row takes a role: a second-order cone, or the vector syntax of linear rows.
function test_xi_roles_on_vector_rows()
    model = Model()
    @variable(model, x[1:2])
    a = @constraint(model, [1; x .- 1] in SecondOrderCone(), Truncate())
    @test get_attribute(a, XiRole()) == Truncate()
    @test constraint_object(a).set == MOI.SecondOrderCone(3)
    b = @constraint(model, [x[1], x[2]] in SecondOrderCone(), ValidInequality())
    @test get_attribute(b, XiRole()) == ValidInequality()
    @test constraint_object(b).func == x
    c = @constraint(model, x .- 1 in MOI.Nonnegatives(2), ValidInequality())
    @test get_attribute(c, XiRole()) == ValidInequality()
    err = try
        @constraint(model, [1, x[1]^2] in SecondOrderCone(), ValidInequality())
    catch err
        err
    end
    @test occursin("must be affine", err.msg)
    return
end

function test_constraint_stage()
    model = Model()
    @variable(model, x)
    a = @constraint(model, x <= 1, Stage(2))
    b = @constraint(model, x >= 0)
    @test stage(a) == 2
    @test stage(b) === nothing
    @test get_attribute(a, XiRole()) === nothing
    set_stage(b, 3)
    @test stage(b) == 3
    set_stage(a, 1)
    @test stage(a) == 1
    unset_stage(a)
    @test stage(a) === nothing
    @test_throws ArgumentError set_stage(a, 0)
    c = @constraint(model, [t = 1:2], x <= t, Stage(t))
    @test stage.(c) == [1, 2]
    return
end

function test_tag_requires_affine_row()
    model = Model()
    @variable(model, x)
    err = try
        @constraint(model, x^2 <= 1, ValidInequality())
    catch err
        err
    end
    @test err isa ErrorException
    @test occursin("must be affine", err.msg)
    return
end

end # module TestConstraints

TestConstraints.runtests()
