# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestChecks

using JuMP
using Test

import Distributions
import Distributions: Uniform
import LinearDecisionRules
import LinearDecisionRules: Stage, Uncertainty

include(joinpath(@__DIR__, "..", "common.jl"))
import .TestCommon: error_of

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

# The message of the error thrown by `f`, with its type.
macro error_message(expr)
    return quote
        err = error_of(() -> $(esc(expr)))
        err === nothing ? "" : sprint(showerror, err)
    end
end

function test_unbounded_univariate()
    model = Model()
    msg = @error_message @variable(model, d in Uncertainty(Distributions.Normal(0, 1)))
    @test occursin("`@variable(model, d in Uncertainty(Distributions.Normal(0, 1)))`", msg)
    @test occursin("has an unbounded support", msg)
    @test occursin("must be bounded", msg)
    @test num_variables(model) == 0
    for dist in (Distributions.Exponential(1), Distributions.Uniform(0, Inf))
        msg = @error_message @variable(model, d in Uncertainty(dist), Stage(2))
        @test occursin("has an unbounded support", msg)
    end
    msg = @error_message @variable(model, d[1:3] in Uncertainty(Distributions.Normal(0, 1)))
    @test occursin("has an unbounded support", msg)
    normal = Uncertainty(Distributions.Normal(0, 1))
    msg = @error_message @variable(model, d[t = 1:3] in normal, Stage(t))
    @test occursin("has an unbounded support", msg)
    @test num_variables(model) == 0
    # Bounded laws, even a truncated unbounded one, are accepted.
    @variable(model, a in Uncertainty(Distributions.truncated(Distributions.Normal(0, 1), -2, 2)))
    @variable(model, b in Uncertainty(Distributions.Bernoulli(0.3)))
    @variable(model, c[1:2] in Uncertainty(Distributions.Beta(2, 2)))
    @test num_variables(model) == 4
    return
end

# The checks also apply to arrays that are not a `Vector`, such as a named range.
function test_named_range_checks()
    model = Model()
    normal = Distributions.MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0])
    R = 1:2
    msg = @error_message @variable(model, d[R] in Uncertainty(normal))
    @test occursin("has an unbounded support", msg)
    @test num_variables(model) == 0
    law = Distributions.product_distribution(Uniform(0, 1), Uniform(0, 2))
    msg = @error_message @variable(model, d[i in 1:3; i != 2] in Uncertainty(law))
    @test occursin("needs an array of variables", msg)
    @test occursin("cannot be declared over a condition", msg)
    @test num_variables(model) == 0
    return
end

function test_unbounded_multivariate()
    model = Model()
    normal = Distributions.MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0])
    msg = @error_message @variable(model, d[1:2] in Uncertainty(normal))
    @test occursin("The distribution of type MvNormal has an unbounded support", msg)
    msg = @error_message @variable(model, d[t = 1:2] in Uncertainty(normal), Stage(t + 1))
    @test occursin("has an unbounded support", msg)
    mixed = Distributions.product_distribution(Distributions.Uniform(0, 1), Distributions.Normal())
    msg = @error_message @variable(model, d[1:2] in Uncertainty(mixed))
    @test occursin("has an unbounded support", msg)
    @test num_variables(model) == 0
    ok = Distributions.product_distribution(Uniform(0, 1), Uniform(0, 2))
    @variable(model, e[1:2] in Uncertainty(ok), Stage(2))
    @variable(model, f[1:2] in Uncertainty(ok))
    discrete = LinearDecisionRules.MvDiscreteNonParametric([[0.0, 1.0], [1.0, 2.0]], [0.5, 0.5])
    @variable(model, g[1:2] in Uncertainty(discrete))
    @test num_variables(model) == 6
    return
end

# `Base.minimum(itr)` applies to every law, and throws a `MethodError` for these.
function test_multivariate_without_bounds()
    model = Model()
    for (law, name) in (
        (Distributions.Dirichlet([1.0, 1.0, 1.0]), "Dirichlet"),
        (Distributions.MvLogNormal([0.0, 0.0, 0.0], [1.0 0 0; 0 1 0; 0 0 1]), "MvLogNormal"),
    )
        msg = @error_message @variable(model, w[1:3] in Uncertainty(law))
        @test occursin("`@variable(model, w[1:3] in Uncertainty(law))`", msg)
        @test occursin("The distribution of type $name does not define `minimum`", msg)
        @test occursin("must be bounded", msg)
    end
    @test num_variables(model) == 0
    return
end

function test_bounds_on_an_uncertainty()
    model = Model()
    set = Uncertainty(Distributions.Uniform(80, 120))
    cases = (
        ("a lower bound", () -> @variable(model, d in set, lower_bound = 0)),
        ("a lower bound", () -> @variable(model, d in set, Stage(2), lower_bound = 0)),
        ("a lower bound", () -> @variable(model, d[1:2] in set, lower_bound = 0)),
        ("an upper bound", () -> @variable(model, d in set, upper_bound = 200)),
        ("an upper bound", () -> @variable(model, d[t = 1:2] in set, Stage(t), upper_bound = 200)),
        ("integrality", () -> @variable(model, d in set, integer = true)),
        ("integrality", () -> @variable(model, d[1:2] in set, Stage(2), integer = true)),
        ("integrality (binary)", () -> @variable(model, d in set, binary = true)),
        ("integrality (binary)", () -> @variable(model, d[1:2] in set, binary = true)),
    )
    for (what, f) in cases
        msg = sprint(showerror, error_of(f))
        @test occursin("An uncertainty cannot have $what", msg)
        @test occursin("ValidInequality()", msg)
    end
    msg = @error_message @variable(model, d in set, Int)
    @test occursin("An uncertainty cannot have integrality", msg)
    msg = @error_message @variable(model, d in set, Bin, Stage(2))
    @test occursin("An uncertainty cannot have integrality (binary)", msg)
    msg = @error_message @variable(model, d in set, lower_bound = 1, upper_bound = 1)
    @test occursin("An uncertainty cannot have a lower bound", msg)
    mv = Distributions.product_distribution(Uniform(0, 1), Uniform(0, 2))
    msg = @error_message @variable(model, d[1:2] in Uncertainty(mv), lower_bound = 0)
    @test occursin("An uncertainty cannot have a lower bound", msg)
    msg = @error_message @variable(model, d[t = 1:2] in Uncertainty(mv), Stage(t), upper_bound = 0)
    @test occursin("An uncertainty cannot have an upper bound", msg)
    msg = @error_message @variable(model, d in set, set_string_name = false, start = 1.0)
    @test msg == ""
    @test num_variables(model) == 1
    return
end

# JuMP gives no hook for these, short of type piracy: `canonicalize` reports them at `optimize!`.
function test_changes_after_declaration_are_not_intercepted()
    model = Model()
    @variable(model, d in Uncertainty(Distributions.Uniform(80, 120)))
    set_lower_bound(d, 0)
    set_upper_bound(d, 200)
    set_integer(d)
    @test has_lower_bound(d) && has_upper_bound(d) && is_integer(d)
    @variable(model, e in Uncertainty(Distributions.Uniform(80, 120)))
    fix(e, 100)
    @test is_fixed(e)
    @constraint(model, e >= 0)
    return
end

function test_two_stage_tags()
    model = Model()
    msg = @error_message @variable(model, x, Stage(1), Stage(2))
    @test occursin("A variable has one `Stage`", msg)
    msg = @error_message @variable(model, x, Stage(1), Stage(1), Int)
    @test occursin("A variable has one `Stage`", msg)
    msg = @error_message @variable(model, x in Stage(1), Stage(2))
    @test occursin("A variable has one `Stage`", msg)
    msg = @error_message @variable(model, x[1:2], Stage(1), Stage(2))
    @test occursin("A variable has one `Stage`", msg)
    msg = @error_message @variable(model, x in Uncertainty(Uniform(0, 1)), Stage(1), Stage(2))
    @test occursin("A variable has one `Stage`", msg)
    @test num_variables(model) == 0
    return
end

function test_invalid_stage()
    model = Model()
    for t in (0, -1)
        err = error_of(() -> @variable(model, x, Stage(t)))
        @test err isa ArgumentError && occursin("stages start at 1", err.msg)
        err = error_of(() -> @variable(model, x in Uncertainty(Uniform(0, 1)), Stage(t)))
        @test err isa ArgumentError
    end
    @test num_variables(model) == 0
    return
end

end # module TestChecks

TestChecks.runtests()
