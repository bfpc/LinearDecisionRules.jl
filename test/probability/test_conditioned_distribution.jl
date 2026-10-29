# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestConditionedDistribution

using JuMP
using Test

import Distributions
import LinearDecisionRules as LDR
import LinearDecisionRules:
    ConditionedDistribution, ModelInfo, UncertaintyDeclaration, canonicalize, moments
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

# Two demands in [40, 80] with a budget `d[1] + d[2] ≤ 120`: the triangle with legs of length
# a = 40 at (40, 40), whose uniform law has mean 40 + a / 3, variance a² / 18 and covariance
# -a² / 36.
function budget()
    laws = [Distributions.Uniform(40, 80), Distributions.Uniform(40, 80)]
    return ConditionedDistribution(laws, [-1.0 -1.0], [-120.0]; names = ["budget"])
end

const TRIANGLE_MEAN = [40 + 40 / 3, 40 + 40 / 3]

const TRIANGLE_COVARIANCE = [40^2/18 -40^2/36; -40^2/36 40^2/18]

function test_construction()
    d = budget()
    @test length(d) == 2
    @test eltype(d) == Float64
    @test minimum(d) == [40, 40] && maximum(d) == [80, 80]
    @test sprint(show, d) ==
          "ConditionedDistribution(" *
          string(Distributions.Uniform(40, 80)) *
          ", " *
          string(Distributions.Uniform(40, 80)) *
          "; 1 rows)"
    # A multivariate law and a univariate one: three components.
    mv = Distributions.product_distribution([
        Distributions.Uniform(0, 1),
        Distributions.Uniform(0, 2),
    ])
    e = ConditionedDistribution([mv, Distributions.Uniform(0, 3)], [1.0 1.0 1.0], [1.0])
    @test length(e) == 3 && minimum(e) == [0, 0, 0] && maximum(e) == [1, 2, 3]
    @test_throws ArgumentError ConditionedDistribution(
        [Distributions.Uniform(0, 1)],
        [1.0 1.0],
        [0],
    )
    @test_throws ArgumentError ConditionedDistribution(
        [Distributions.Uniform(0, 1)],
        [1.0;;],
        [0.0];
        names = String[],
    )
    @test_throws ArgumentError ConditionedDistribution([d], [1.0 1.0], [0.0])
    return
end

# Equality is by value, without the names, since the moments are cached by the law.
function test_equality()
    d = budget()
    laws = [Distributions.Uniform(40, 80), Distributions.Uniform(40, 80)]
    same = ConditionedDistribution(laws, [-1.0 -1.0], [-120.0])
    @test d == same && hash(d) == hash(same)
    @test d != ConditionedDistribution(laws, [-1.0 -1.0], [-130.0])
    @test d != ConditionedDistribution(laws, [-1.0 -2.0], [-120.0])
    return
end

function test_rand_in_set()
    d = budget()
    rng = Random.Xoshiro(1)
    draws = rand(rng, d, 1000)
    @test size(draws) == (2, 1000)
    @test all(sum(draws; dims = 1) .<= 120)
    @test all(40 .<= draws .<= 80)
    x = rand(rng, d)
    @test x isa Vector{Float64} && sum(x) <= 120
    # The same seed gives the same draws.
    @test rand(Random.Xoshiro(5), d, 10) == rand(Random.Xoshiro(5), d, 10)
    return
end

# The disc ‖η - (60, 60)‖ ≤ 20 in a second-order cone: the uniform law on it has mean (60, 60)
# and covariance 20² / 4 I.
function test_conic_set()
    laws = [Distributions.Uniform(40, 80), Distributions.Uniform(40, 80)]
    W, h = [0.0 0.0; 1.0 0.0; 0.0 1.0], [-20.0, 60.0, 60.0]
    d = ConditionedDistribution(laws, W, h; cones = [MOI.SecondOrderCone(3)])
    @test d.cones == [MOI.SecondOrderCone(3)] && Distributions.params(d)[4] == d.cones
    draws = rand(Random.Xoshiro(1), d, 1000)
    @test all(sum((draws .- 60) .^ 2; dims = 1) .<= 400 * (1 + 1e-9))
    @test d != ConditionedDistribution(laws, W, h; cones = [MOI.Nonnegatives(3)])
    @test d == ConditionedDistribution(laws, W, h; cones = Any[MOI.SecondOrderCone(3)])
    mom = moments_of(d; max_samples = 100_000)
    @test isapprox(mom.mean[2:3], [60, 60]; atol = 0.2)
    @test isapprox(only(mom.blocks).covariance, [100 0; 0 100]; atol = 2)
    # A product of cones: the disc and `η₁ ≤ 60`, the left half of the disc.
    half = ConditionedDistribution(
        laws,
        [W; -1.0 0.0],
        [h; -60.0];
        cones = [MOI.SecondOrderCone(3), MOI.Nonnegatives(1)],
    )
    @test all(rand(Random.Xoshiro(2), half, 1000)[1, :] .<= 60)
    @test_throws ArgumentError ConditionedDistribution(laws, W, h; cones = [MOI.Nonnegatives(2)])
    @test_throws ArgumentError ConditionedDistribution(laws, W, h; cones = [MOI.Zeros(3)])
    return
end

# Atoms on the boundary of the set are kept despite rounding.
function test_discrete_boundary()
    law = Distributions.DiscreteNonParametric([0.1, 0.2, 0.3], [1 / 3, 1 / 3, 1 / 3])
    d = ConditionedDistribution([law, law], [1.0 1.0], [0.4])
    draws = rand(Random.Xoshiro(1), d, 2000)
    @test any(isapprox.(sum(draws; dims = 1), 0.4))
    @test all(sum(draws; dims = 1) .>= 0.4 - 1e-12)
    return
end

# The form of declarations of the laws `laws`, and their moments.
function form_of(laws...)
    model = Model()
    @variable(model, x)
    decls = map(enumerate(laws)) do (k, dist)
        vars = @variable(model, [1:length(dist)], base_name = "d$k")
        return UncertaintyDeclaration(dist, index.(vars))
    end
    info = ModelInfo(; uncertainties = collect(decls), decision_stages = Dict(index(x) => 1))
    return canonicalize(backend(model), info)
end

moments_of(laws...; kwargs...) = moments(form_of(laws...); kwargs...)

function test_moments_by_rejection()
    mom = moments_of(budget())
    @test only(mom.blocks).num_samples == 10_000
    @test LDR._is_estimated(mom)
    # Within 5 standard errors of the mean, about 0.47.
    @test isapprox(mom.mean[2:3], TRIANGLE_MEAN; atol = 0.5)
    @test isapprox(only(mom.blocks).covariance, TRIANGLE_COVARIANCE; rtol = 0.1)
    @test only(mom.blocks).covariance == only(mom.blocks).covariance'
    # More draws, closer.
    big = moments_of(budget(); max_samples = 400_000)
    @test only(big.blocks).num_samples == 400_000
    @test isapprox(big.mean[2:3], TRIANGLE_MEAN; atol = 0.08)
    @test isapprox(only(big.blocks).covariance, TRIANGLE_COVARIANCE; rtol = 0.02)
    # Exact laws are not estimated.
    @test !LDR._is_estimated(moments_of(Distributions.Uniform(0, 1)))
    return
end

# The sample is the same for the same seed, and each conditioned law has its own seed.
function test_seeds()
    a = moments_of(budget(), budget(); seed = 7)
    b = moments_of(budget(), budget(); seed = 7)
    @test a.mean == b.mean
    @test [block.covariance for block in a.blocks] == [block.covariance for block in b.blocks]
    @test a.mean[2:3] != a.mean[4:5]
    @test a.blocks[1].covariance != a.blocks[2].covariance
    c = moments_of(budget(), budget(); seed = 8)
    @test c.mean != a.mean
    # The seed of the k-th conditioned law, from the global one.
    seeds = rand(Random.Xoshiro(UInt64(7)), UInt64, 2)
    first = LDR._estimate(budget(), Random.Xoshiro(seeds[1]), 10_000, 10.0)
    @test a.mean[2:3] == first.mean
    # A negative seed is a seed.
    @test moments_of(budget(); seed = -1).blocks[1].num_samples == 10_000
    return
end

function test_impossible_conditioning()
    laws = [Distributions.Uniform(40, 80), Distributions.Uniform(40, 80)]
    d = ConditionedDistribution(laws, [-1.0 -1.0], [-50.0]; names = ["tiny"])
    start = time_ns()
    err = try
        moments_of(d)
    catch e
        e
    end
    # The upper bound of the acceptance rate stops it well before the time limit.
    @test (time_ns() - start) / 1e9 < 5
    msg = sprint(showerror, err)
    @test occursin("conditioned on the constraints tiny tagged `Truncate()`", msg)
    @test occursin("uncertainties d1[1], d1[2]", msg)
    @test occursin("kept 0 of", msg) && occursin("needs 100 within", msg)
    @test occursin("`RejectionSamplingTimeLimit()`", msg)
    # With a small target, the target is what it needs.
    msg = sprint(showerror, try
        moments_of(d; max_samples = 10, time_limit = 0.05)
    catch e
        e
    end)
    @test occursin("needs 10 within its time limit of 0.05 seconds", msg)
    # Drawing one outcome fails too, after many attempts: it is not tested here (ten seconds).
    return
end

# A time limit that stops the sampling before the target: one warning for every such law.
function test_small_sample_warning()
    pattern =
        r"Rejection sampling kept fewer than 100000000 draws .* for the conditioned laws of " *
        r"d1\[1\], d1\[2\] \(\d+ draws\); d2\[1\], d2\[2\] \(\d+ draws\): the estimates"
    mom = @test_logs (:warn, pattern) moments_of(
        budget(),
        budget();
        max_samples = 10^8,
        time_limit = 0.01,
    )
    @test all(100 <= block.num_samples < 10^8 for block in mom.blocks)
    return
end

function test_settings()
    @test_throws ArgumentError moments_of(budget(); time_limit = 0)
    @test_throws ArgumentError moments_of(budget(); max_samples = 0)
    return
end

# The time limit of each law: the explicit one, or an equal share of what remains of a budget
# (what `optimize!` gives the sampling out of `TimeLimitSec()`), or 10 seconds.
function test_sampling_time_limit()
    s = LDR._RejectionSampling(1234, nothing, 10^8)
    @test LDR._sampling_time_limit(s, 1.0, 0.25, 3) == 0.25
    @test LDR._sampling_time_limit(s, 1.0, 1.5, 1) == 0.0
    @test LDR._sampling_time_limit(s, nothing, 0.0, 2) == 10.0
    explicit = LDR._RejectionSampling(1234, 3.0, 10^8)
    @test LDR._sampling_time_limit(explicit, 1.0, 0.0, 2) == 3.0
    # Two laws that cannot reach their sample size share the budget, in about its time.
    form = form_of(budget(), budget())
    LDR._moments(form, LDR._RejectionSampling(1, nothing, 100), nothing; budget = 1.0)
    pattern = r"0.4 seconds shared by the laws, half of the time that remained of `TimeLimitSec"
    start = time_ns()
    mom = @test_logs (:warn, pattern) LDR._moments(form, s, nothing; budget = 0.4)
    # The budget is respected: 10^8 draws would take minutes. The slack is wide because other
    # processes may share the machine (0.2 s failed under the load of a parallel test suite).
    @test (time_ns() - start) / 1e9 < 0.4 + 2.0
    @test all(100 <= block.num_samples < 10^8 for block in mom.blocks)
    return
end

# A valid inequality on a conditioned law is not checked at the mean, which is not known yet.
function test_valid_row_on_conditioned_law()
    model = Model()
    @variable(model, x)
    @variable(model, d[1:2])
    row = @constraint(model, d[1] + d[2] <= 130)
    info = ModelInfo(;
        uncertainties = [UncertaintyDeclaration(budget(), index.(d))],
        decision_stages = Dict(index(x) => 1),
        valid_rows = Set{MOI.ConstraintIndex}([index(row)]),
    )
    form = canonicalize(backend(model), info)
    @test only(form.laws).distribution == budget()
    return
end

end # module TestConditionedDistribution

TestConditionedDistribution.runtests()
