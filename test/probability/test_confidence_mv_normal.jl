# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestConfidenceMvNormal

using JuMP
using Test

import Clarabel
import Distributions
import HiGHS
import LinearAlgebra
import LinearDecisionRules as LDR
import LinearDecisionRules:
    ConfidenceMvNormal,
    FirstStage,
    LDRModel,
    ModelInfo,
    Uncertainty,
    UncertaintyDeclaration,
    canonicalize,
    support_constraints
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

const μ = [1.0, 2.0]

const Σ = [1.0 0.5; 0.5 2.0]

function test_construction()
    d = ConfidenceMvNormal(μ, Σ, 0.9)
    @test length(d) == 2 && eltype(d) == Float64
    @test d.support == :auto
    @test Distributions.params(d) == (μ, Σ, 0.9)
    @test Distributions.mean(d) == μ
    ρ = sqrt(Distributions.quantile(Distributions.Chisq(2), 0.9))
    @test d.ρ ≈ ρ
    @test minimum(d) ≈ μ .- ρ .* sqrt.([1, 2]) && maximum(d) ≈ μ .+ ρ .* sqrt.([1, 2])
    @test d == ConfidenceMvNormal(μ, Σ, 0.9) && hash(d) == hash(ConfidenceMvNormal(μ, Σ, 0.9))
    @test d != ConfidenceMvNormal(μ, Σ, 0.9; support = :boxes)
    @test occursin("ConfidenceMvNormal(μ = [1.0, 2.0]", sprint(show, d))
    @test_throws ArgumentError ConfidenceMvNormal(μ, Σ, 1.0)
    @test_throws ArgumentError ConfidenceMvNormal(μ, Σ, 0.0)
    @test_throws ArgumentError ConfidenceMvNormal(μ, [1.0 0.5; 0.4 2.0], 0.9)
    @test_throws ArgumentError ConfidenceMvNormal(μ, [1.0 2.0; 2.0 1.0], 0.9)
    @test_throws ArgumentError ConfidenceMvNormal(μ, ones(3, 3), 0.9)
    @test_throws ArgumentError ConfidenceMvNormal(μ, Σ, 0.9; support = :disc)
    return
end

# The covariance is a multiple of Σ, below 1, as in the previous version; in dimension 1, it is
# the variance of the normal law truncated to [μ - ρσ, μ + ρσ].
function test_covariance()
    d = ConfidenceMvNormal(μ, Σ, 0.9)
    ratio = Distributions.cov(d) ./ Σ
    @test all(isapprox.(ratio, ratio[1, 1]; rtol = 1e-12))
    @test 0 < ratio[1, 1] < 1
    @test Distributions.var(d) ≈ LinearAlgebra.diag(Distributions.cov(d))
    σ = 2.0
    d1 = ConfidenceMvNormal([0.0], [σ^2;;], 0.95)
    truncated = Distributions.truncated(Distributions.Normal(0.0, σ), -d1.ρ * σ, d1.ρ * σ)
    @test only(Distributions.var(d1)) ≈ Distributions.var(truncated) rtol = 1e-10
    return
end

# The moments against Monte Carlo, in dimension 3: within 5 standard errors.
function test_moments_monte_carlo()
    m = [1.0, 2.0, 3.0]
    S = [2.0 0.5 0.1; 0.5 1.0 0.3; 0.1 0.3 1.5]
    d = ConfidenceMvNormal(m, S, 0.8)
    n = 200_000
    draws = rand(Random.Xoshiro(1), d, n)
    @test all(x -> Distributions.insupport(d, x), eachcol(draws))
    mean = vec(sum(draws; dims = 2)) / n
    C = Distributions.cov(d)
    @test all(abs.(mean .- m) .<= 5 .* sqrt.(LinearAlgebra.diag(C) ./ n))
    centered = draws .- m
    empirical = centered * centered' / n
    # The standard error of a product of two coordinates is below √(E[x²y²] / n) ≤ max(C) √(3 / n).
    @test all(abs.(empirical .- C) .<= 5 * maximum(C) * sqrt(3 / n))
    return
end

# The ellipsoid holds a fraction α of the normal law: the share of normal draws outside of it is
# about 1 - α.
function test_probability_outside()
    three = ([1.0, 2.0, 0.5], [1.0 0.5 -0.2; 0.5 2.0 0.3; -0.2 0.3 1.5], 0.9)
    for (m, S, α) in ((μ, Σ, 0.9), three)
        d = ConfidenceMvNormal(m, S, α)
        n = 1_000_000
        normal = Distributions.MvNormal(m, S)
        draws = rand(Random.Xoshiro(42), normal, n)
        outside = count(x -> !Distributions.insupport(d, x), eachcol(draws)) / n
        # Within 5 standard errors.
        @test abs(outside - (1 - α)) <= 5 * sqrt(α * (1 - α) / n)
    end
    return
end

# Points of the boundary of the ellipsoid, in random directions.
function boundary_points(d::ConfidenceMvNormal, rng; n = 1000)
    return map(1:n) do _
        z = randn(rng, length(d))
        return d.μ .+ d.L * (d.ρ .* z ./ LinearAlgebra.norm(z))
    end
end

# The boundary of the ellipsoid is in the support, and the bounding box touches it.
function test_insupport_and_box()
    d = ConfidenceMvNormal([100.0, 80.0], [100.0 40.0; 40.0 64.0], 0.95)
    boundary = boundary_points(d, Random.Xoshiro(3))
    @test all(x -> Distributions.insupport(d, x), boundary)
    @test all(x -> all(minimum(d) .- 1e-9 .<= x .<= maximum(d) .+ 1e-9), boundary)
    @test !Distributions.insupport(d, d.μ .+ [1000.0, 0.0])
    @test !Distributions.insupport(d, [1.0])
    for k in 1:2
        # The point of the boundary in the direction `L' e_k` attains the upper bound.
        e = [k == j ? 1.0 : 0.0 for j in 1:2]
        z = d.ρ * (d.L' * e) / LinearAlgebra.norm(d.L' * e)
        @test (d.μ.+d.L*z)[k] ≈ maximum(d)[k]
    end
    return
end

# The ellipsoid is one second-order cone row; the boxes are the principal axes, which contain it.
function test_support_constraints()
    d = ConfidenceMvNormal(μ, Σ, 0.9; support = :ellipsoid)
    rows = support_constraints(d)
    @test rows.cones == [MOI.SecondOrderCone(3)]
    @test support_constraints(ConfidenceMvNormal(μ, Σ, 0.9)) == rows
    boxes = support_constraints(ConfidenceMvNormal(μ, Σ, 0.9; support = :boxes))
    @test boxes.cones == [MOI.Nonnegatives(4)] && size(boxes.W) == (4, 2)
    boundary = boundary_points(d, Random.Xoshiro(4))
    on_cone(v) = isapprox(v[1], LinearAlgebra.norm(v[2:3]); atol = 1e-12)
    @test all(x -> on_cone(rows.W * x .- rows.h), boundary)
    @test all(x -> all(boxes.W * x .- boxes.h .>= -1e-9), boundary)
    # The boxes touch the ellipsoid on each principal axis.
    λ, V = LinearAlgebra.eigen(Σ)
    x = d.μ .+ d.ρ * sqrt(λ[1]) * V[:, 1]
    @test minimum(boxes.W * x .- boxes.h) ≈ 0 atol = 1e-12
    # A diagonal Σ: the principal box is the bounding box, so there are no rows.
    diagonal = ConfidenceMvNormal(μ, [1.0 0.0; 0.0 2.0], 0.9; support = :boxes)
    @test isempty(support_constraints(diagonal).h)
    # Other laws have none.
    @test isempty(support_constraints(Distributions.Uniform(0, 1)).h)
    @test size(support_constraints(Distributions.MvNormal(μ, Σ)).W) == (0, 2)
    return
end

function test_resolve_support()
    d = ConfidenceMvNormal(μ, Σ, 0.9)
    @test LDR._has_auto_support(d)
    @test LDR._resolve_support(d, true).support == :ellipsoid
    @test LDR._resolve_support(d, false).support == :boxes
    boxes = ConfidenceMvNormal(μ, Σ, 0.9; support = :boxes)
    @test !LDR._has_auto_support(boxes) && LDR._resolve_support(boxes, true) === boxes
    @test !LDR._has_auto_support(Distributions.Uniform(0, 1))
    return
end

# `canonicalize` chooses the ellipsoid with `conic_support` and without integer decisions.
function test_auto_support_in_canonicalize()
    for integer in (false, true), conic_support in (false, true)
        model = Model()
        @variable(model, x >= 0, integer = integer)
        @variable(model, d[1:2])
        @constraint(model, x >= sum(d))
        law = ConfidenceMvNormal(μ, Σ, 0.9)
        info = ModelInfo(;
            uncertainties = [UncertaintyDeclaration(law, index.(d))],
            decision_stages = Dict(index(x) => 1),
        )
        form = canonicalize(backend(model), info; conic_support)
        chosen = only(form.laws).distribution.support
        @test chosen == (conic_support && !integer ? :ellipsoid : :boxes)
    end
    return
end

# `optimize!` asks the inner optimizer: HiGHS has no second-order cones, Clarabel has them.
function test_auto_support_in_optimize()
    for (factory, chosen) in ((HiGHS.Optimizer, :boxes), (Clarabel.Optimizer, :ellipsoid))
        model = LDRModel(factory)
        set_silent(model)
        @variable(model, buy >= 0, FirstStage)
        @variable(model, sell[1:2] >= 0)
        @variable(model, d[1:2] in Uncertainty(ConfidenceMvNormal([100.0, 80.0], Σ, 0.9)))
        @constraint(model, sum(sell) <= buy)
        @constraint(model, sell .<= d)
        @objective(model, Max, -10buy + 15sum(sell))
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        @test only(unsafe_backend(model).form.laws).distribution.support == chosen
    end
    return
end

end # module TestConfidenceMvNormal

TestConfidenceMvNormal.runtests()
