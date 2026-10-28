# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The problems of `tagged.jl` and their twins of `library.jl` give the same canonical form, so that
# the tests of the core and those of the JuMP layer solve the same problems.

module TestTwins

using JuMP
using Test

import LinearDecisionRules
import LinearDecisionRules: CanonicalForm, canonicalize

include(joinpath(@__DIR__, "library.jl"))
include(joinpath(@__DIR__, "tagged.jl"))

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

const TWINS = [
    (TaggedProblems.newsvendor, Problems.newsvendor),
    (
        () -> TaggedProblems.newsvendor(; integer = true),
        () -> Problems.newsvendor(; integer = true),
    ),
    (TaggedProblems.newsvendor_random_price, Problems.newsvendor_random_price),
    (TaggedProblems.newsvendor_two_products, Problems.newsvendor_two_products),
    (TaggedProblems.newsvendor_three_stages, Problems.newsvendor_three_stages),
    (() -> TaggedProblems.inventory(4), () -> Problems.inventory(4)),
    (() -> TaggedProblems.inventory(8), () -> Problems.inventory(8)),
    (TaggedProblems.hydro_toy, Problems.hydro_toy),
    (TaggedProblems.newsvendor_mv_discrete, Problems.newsvendor_mv_discrete),
    (TaggedProblems.newsvendor_discrete_demand, Problems.newsvendor_discrete_demand),
]

# Equality of two canonical forms, field by field. The blocks of the laws are compared sorted by
# column, since their order is the order of declaration.
function same_form(a::CanonicalForm, b::CanonicalForm)
    for name in fieldnames(CanonicalForm)
        if name == :laws
            continue
        end
        if getfield(a, name) != getfield(b, name)
            return false
        end
    end
    key(law) = (law.columns, law.stages, law.distribution)
    return sort(key.(a.laws); by = first) == sort(key.(b.laws); by = first)
end

function test_twins_have_the_same_form()
    for (tagged, by_hand) in TWINS
        p = by_hand()
        @testset "$(p.name)" begin
            model = tagged()
            info = LinearDecisionRules.model_info(backend(model))
            @test same_form(canonicalize(backend(model), info), Problems.form(p))
        end
    end
    return
end

end # module TestTwins

TestTwins.runtests()
