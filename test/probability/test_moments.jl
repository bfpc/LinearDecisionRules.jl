# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestMoments

using JuMP
using Test

import Distributions
import LinearDecisionRules:
    CovarianceBlock,
    ModelInfo,
    Moments,
    MvDiscreteNonParametric,
    UncertaintyDeclaration,
    canonicalize,
    moments,
    second_moment

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

# The moments of one or more declarations, each given as `(distribution, stages)`.
function moments_of(laws...)
    model = Model()
    @variable(model, x)
    decls = map(laws) do (dist, stages)
        vars = @variable(model, [1:length(stages)])
        return UncertaintyDeclaration(dist, index.(vars), stages)
    end
    info = ModelInfo(; uncertainties = collect(decls), decision_stages = Dict(index(x) => 1))
    form = canonicalize(backend(model), info)
    return moments(form)
end

function test_uniform()
    mom = moments_of((Distributions.Uniform(80, 120), [nothing]))
    @test mom isa Moments
    @test mom.mean == [1, 100]
    @test length(mom.blocks) == 1 && mom.blocks[1] isa CovarianceBlock
    @test mom.blocks[1].columns == [2]
    @test mom.blocks[1].covariance ≈ [40^2 / 12;;]
    @test second_moment(mom) ≈ [1 100; 100 100^2+40^2/12]
    return
end

function test_triangular()
    mom = moments_of((Distributions.TriangularDist(0, 3, 1), [nothing]))
    @test mom.mean ≈ [1, 4 / 3]
    @test mom.blocks[1].covariance ≈ [7 / 18;;]
    return
end

function test_product_distribution()
    dist = Distributions.product_distribution([
        Distributions.Uniform(0, 1),
        Distributions.Uniform(2, 4),
    ])
    mom = moments_of((dist, [nothing, nothing]))
    @test mom.mean ≈ [1, 0.5, 3]
    @test mom.blocks[1].covariance ≈ [1/12 0; 0 4/12]
    @test second_moment(mom) ≈ [1 0.5 3; 0.5 1/3 1.5; 3 1.5 9+1/3]
    return
end

function test_mv_discrete_non_parametric()
    dist = MvDiscreteNonParametric([[1.0, 2.0], [3.0, 6.0]], [0.25, 0.75])
    mom = moments_of((dist, [nothing, nothing]))
    @test mom.mean ≈ [1, 2.5, 5]
    @test mom.blocks[1].covariance ≈ [0.75 1.5; 1.5 3]
    # E[ξ ξᵀ] is the probability-weighted sum of the outer products of the scenarios.
    M = 0.25 * [1, 1, 2] * [1, 1, 2]' + 0.75 * [1, 3, 6] * [1, 3, 6]'
    @test second_moment(mom) ≈ M
    return
end

function test_independent_declarations_across_stages()
    # A multivariate law split over stages 3 and 2, and a stage-2 uniform: columns are sorted by
    # stage, so the multivariate law's columns are not contiguous.
    dist = MvDiscreteNonParametric([[1.0, 2.0], [3.0, 6.0]], [0.25, 0.75])
    mom = moments_of((dist, [3, 2]), (Distributions.Uniform(0, 6), [2]))
    # Columns: dist[2] (stage 2), uniform (stage 2), dist[1] (stage 3).
    @test mom.mean ≈ [1, 5, 3, 2.5]
    @test [block.columns for block in mom.blocks] == [[4, 2], [3]]
    @test mom.blocks[1].covariance ≈ [0.75 1.5; 1.5 3]
    @test mom.blocks[2].covariance ≈ [3;;]
    M = second_moment(mom)
    @test M ≈ M'
    @test M[2, 3] ≈ 5 * 3          # independent declarations
    @test M[2, 4] ≈ 1.5 + 5 * 2.5  # same declaration
    @test M[4, 4] ≈ 0.75 + 2.5^2
    @test M[3, 3] ≈ 3 + 9
    return
end

function test_no_finite_moments()
    model = Model()
    @variable(model, x)
    @variable(model, c)
    info = ModelInfo(; uncertainties = [UncertaintyDeclaration(Distributions.Cauchy(), [index(c)])])
    form = canonicalize(backend(model), info)
    @test_throws ErrorException(
        "The distribution of the uncertainty c has no finite mean or covariance.",
    ) moments(form)
    return
end

end # module TestMoments

TestMoments.runtests()
