# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestMvDiscreteNonParametric

using Test

import Distributions
import LinearAlgebra
import Random
import LinearDecisionRules: MvDiscreteNonParametric

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

function test_params()
    d = MvDiscreteNonParametric([[1.0, 2.0], [3.0, 4.0]], [0.4, 0.6])
    @test d isa Distributions.DiscreteMultivariateDistribution
    @test eltype(typeof(d)) == Float64
    xs, ps = Distributions.params(d)
    @test xs == [[1.0, 2.0], [3.0, 4.0]]
    @test ps == [0.4, 0.6]
    @test length(d) == 2
    return
end

function test_check_args()
    @test_throws DomainError MvDiscreteNonParametric([[1.0, 2.0]], [0.4, 0.6])
    @test_throws DomainError MvDiscreteNonParametric([[1.0, 2.0], [3.0, 4.0]], [0.5, 0.6])
    @test_throws DomainError MvDiscreteNonParametric([[1.0, 2.0], [3.0]], [0.5, 0.5])
    d = MvDiscreteNonParametric([[1.0, 2.0], [3.0, 4.0]], [0.5, 0.6]; check_args = false)
    @test d.p == [0.5, 0.6]
    return
end

function test_statistics()
    d = MvDiscreteNonParametric([[3.0, 1.0], [1.0, 3.0]], [0.5, 0.5])
    @test minimum(d) == [1.0, 1.0]
    @test maximum(d) == [3.0, 3.0]
    @test Distributions.mean(d) == [2.0, 2.0]
    @test Distributions.cov(d) == [1.0 -1.0; -1.0 1.0]
    @test Distributions.var(d) == LinearAlgebra.diag(Distributions.cov(d))
    @test Distributions.insupport(d, [3.0, 1.0])
    @test !Distributions.insupport(d, [0.0, 0.0])
    return
end

function test_rand()
    d = MvDiscreteNonParametric([[3.0, 1.0], [1.0, 3.0]], [0.5, 0.5])
    rng = Random.Xoshiro(42)
    draws = [rand(rng, d) for _ in 1:200]
    @test all(x -> x == [3.0, 1.0] || x == [1.0, 3.0], draws)
    @test any(==([3.0, 1.0]), draws) && any(==([1.0, 3.0]), draws)
    # A point with probability zero is never drawn.
    d = MvDiscreteNonParametric([[0.0], [1.0], [2.0]], [0.5, 0.0, 0.5])
    @test all(x -> x != [1.0], (rand(rng, d) for _ in 1:200))
    return
end

function test_integer_support()
    d = MvDiscreteNonParametric([[1, 2], [3, 4], [5, 6]], [0.2, 0.3, 0.5])
    @test eltype(typeof(d)) == Int
    @test length(d) == 2
    @test minimum(d) == [1, 2]
    @test maximum(d) == [5, 6]
    @test Distributions.mean(d) ≈ [3.6, 4.6]
    @test Distributions.cov(d) ≈ fill(0.2 * 2.6^2 + 0.3 * 0.6^2 + 0.5 * 1.4^2, 2, 2)
    @test Distributions.insupport(d, [3, 4])
    @test !Distributions.insupport(d, [2, 3])
    rng = Random.Xoshiro(123)
    @test all(x -> x in ([1, 2], [3, 4], [5, 6]), (rand(rng, d) for _ in 1:20))
    d = MvDiscreteNonParametric([Int32[10, 20], Int32[30, 40]], [0.5, 0.5])
    @test eltype(typeof(d)) == Int32
    @test minimum(d) == Int32[10, 20]
    @test maximum(d) == Int32[30, 40]
    @test all(x -> x == Int32[10, 20] || x == Int32[30, 40], (rand(rng, d) for _ in 1:20))
    return
end

end # module TestMvDiscreteNonParametric

TestMvDiscreteNonParametric.runtests()
