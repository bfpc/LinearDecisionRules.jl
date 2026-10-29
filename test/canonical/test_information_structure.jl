# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestInformationStructure

using JuMP
using Test

import Distributions
import LinearDecisionRules:
    InformationStructure, ModelInfo, UncertaintyDeclaration, canonicalize, observes, visible_columns

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

function declare(dist, vars...; stages)
    return UncertaintyDeclaration(dist, collect(index.(vars)), stages)
end

function test_three_stages()
    model = Model()
    @variable(model, buy >= 0, Int)
    @variable(model, early >= 0)
    @variable(model, sell2 >= 0)
    @variable(model, sell3 >= 0)
    @variable(model, d1)
    @variable(model, d2)
    @variable(model, d3)
    @constraint(model, buy + early + sell2 + sell3 <= d1 + d2 + d3)
    # Declared out of stage order: columns are d1, d2, d3 after sorting.
    info = ModelInfo(;
        uncertainties = [
            declare(Distributions.Uniform(0, 1), d3; stages = [3]),
            declare(Distributions.Uniform(0, 1), d2; stages = [2]),
            declare(Distributions.Uniform(0, 1), d1; stages = [1]),
        ],
        decision_stages = Dict(
            index(buy) => 1,
            index(early) => 1,
            index(sell2) => 2,
            index(sell3) => 3,
        ),
    )
    form = canonicalize(backend(model), info)
    info = InformationStructure(form)
    @test info.column_stages == [1, 1, 2, 3]
    @test info.decision_stages == [1, 1, 2, 3]
    @test info.num_visible == [1, 2, 3, 4]
    @test [observes(info, i, j) for i in 1:4, j in 1:4] == Bool[
        1 0 0 0
        1 1 0 0
        1 1 1 0
        1 1 1 1
    ]
    @test visible_columns(info, 1) == 1:1
    @test visible_columns(info, 3) == 1:3
    return
end

function test_two_stages_without_uncertainty()
    info = InformationStructure([1, 2], [1], falses(2))
    @test info.num_visible == [1, 1]
    @test observes(info, 2, 1) && !observes(info, 2, 2)
    return
end

function test_unsorted_columns()
    @test_throws ArgumentError InformationStructure([1], [1, 3, 2], falses(1))
    return
end

end # module TestInformationStructure

TestInformationStructure.runtests()
