# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestLiftedDistribution

using Test

import Distributions
import LinearDecisionRules: AxisPiecewise, LiftedDistribution, lift, retraction
import Random

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

# Monte Carlo estimates of `E[ξ̃ ξ̃ᵀ]`, `ξ̃ = (1, η̃)`, and of their standard errors.
function monte_carlo(d::LiftedDistribution, N::Int; seed = 1)
    rng = Random.Xoshiro(seed)
    k = 1 + length(d)
    sum1, sum2 = zeros(k, k), zeros(k, k)
    ξ = ones(k)
    for _ in 1:N
        ξ[2:end] .= rand(rng, d)
        y = ξ * ξ'
        sum1 .+= y
        sum2 .+= y .^ 2
    end
    M = sum1 ./ N
    return M, sqrt.(max.(sum2 ./ N .- M .^ 2, 0.0) ./ N)
end

second_moment(d::LiftedDistribution) = [1 d.mean'; d.mean d.covariance+d.mean*d.mean']

# The quadrature and the exact formulas agree with Monte Carlo within 5 standard errors.
function check_monte_carlo(d::LiftedDistribution; N = 1_000_000)
    M, se = monte_carlo(d, N)
    @test all(abs.(M .- second_moment(d)) .<= 5 .* se .+ 1e-12)
    return
end

function test_uniform_exact()
    d = LiftedDistribution(Distributions.Uniform(80, 120), AxisPiecewise([100]))
    @test length(d) == 2
    @test Distributions.mean(d) ≈ [95, 5]
    # Var(min(η, 100)) = Var(max(η - 100, 0)) = 125/3 and their covariance is 25.
    @test Distributions.cov(d) ≈ [125/3 25; 25 125/3]
    @test Distributions.var(d) ≈ [125 / 3, 125 / 3]
    @test minimum(d) == [80, 0] && maximum(d) == [100, 20]
    @test Distributions.params(d) == (Distributions.Uniform(80, 120), AxisPiecewise([100]))
    @test sprint(show, d) ==
          "LiftedDistribution($(Distributions.Uniform(80, 120)), " * "AxisPiecewise([100.0]))"
    return
end

# The retraction of the lifted moments is the law of η: `R M̃ Rᵀ = M`.
function test_retraction_of_moments()
    laws = [
        Distributions.Uniform(80, 120),
        Distributions.truncated(Distributions.Normal(5, 0.5), 3, 8),
        Distributions.TriangularDist(0, 3, 1),
        Distributions.Beta(2, 5),
        Distributions.DiscreteNonParametric([80.0, 90, 100, 110, 120], [0.1, 0.2, 0.4, 0.2, 0.1]),
    ]
    for law in laws
        lo, hi = minimum(law), maximum(law)
        for points in ([lo + (hi - lo) / 3], lo .+ (hi - lo) .* [0.1, 0.25, 0.5, 0.9])
            l = AxisPiecewise(points)
            d = LiftedDistribution(law, l)
            R = Matrix(retraction(l))
            @test R * d.mean ≈ [Distributions.mean(law)]
            @test R * d.covariance * R' ≈ [Distributions.var(law);;] rtol = 1e-8
            @test d.covariance ≈ d.covariance'
        end
    end
    return
end

function test_uniform_monte_carlo()
    for points in ([100.0], [90.0, 100.0, 110.0], collect(85.0:5.0:115.0))
        check_monte_carlo(LiftedDistribution(Distributions.Uniform(80, 120), AxisPiecewise(points)))
    end
    return
end

# The case of `test/pwl_moments.jl` in the previous version: a truncated normal, with the last
# piece of small probability.
function test_truncated_normal_monte_carlo()
    law = Distributions.truncated(Distributions.Normal(5, 0.5), 3, 8)
    d = LiftedDistribution(law, AxisPiecewise([4, 6.2]))
    @test minimum(d) == [3, 0, 0] && maximum(d) ≈ [4, 2.2, 1.8]
    check_monte_carlo(d)
    return
end

function test_discrete_exact()
    law = Distributions.DiscreteNonParametric([80.0, 90, 100, 110, 120], [0.1, 0.2, 0.4, 0.2, 0.1])
    l = AxisPiecewise([95, 105])
    d = LiftedDistribution(law, l)
    lifted = [lift(l, x) for x in law.support]
    μ = sum(p * x for (p, x) in zip(law.p, lifted))
    @test d.mean ≈ μ
    @test d.covariance ≈ sum(p * (x - μ) * (x - μ)' for (p, x) in zip(law.p, lifted))
    # Lifted values are real even for an integer law with a fractional breakpoint.
    integer = LiftedDistribution(Distributions.DiscreteUniform(0, 3), AxisPiecewise([1.5]))
    @test eltype(integer) == Float64
    @test rand(Random.Xoshiro(1), integer) isa Vector{Float64}
    @test integer.mean ≈ [(0 + 1 + 1.5 + 1.5) / 4, (0.5 + 1.5) / 4]
    return
end

# A draw of the lifted law is the lift of one draw of the original law.
function test_rand()
    law = Distributions.Uniform(80, 120)
    l = AxisPiecewise([90, 100, 110])
    d = LiftedDistribution(law, l)
    a, b = Random.Xoshiro(3), Random.Xoshiro(3)
    for _ in 1:10
        @test rand(a, d) == lift(l, rand(b, law))
    end
    return
end

function test_errors()
    l = AxisPiecewise([100.0])
    @test_throws ArgumentError LiftedDistribution(Distributions.Normal(100, 1), l)
    @test_throws ArgumentError LiftedDistribution(Distributions.Uniform(100, 120), l)
    @test_throws ArgumentError LiftedDistribution(Distributions.Uniform(0, 1), l)
    return
end

end # module TestLiftedDistribution

TestLiftedDistribution.runtests()
