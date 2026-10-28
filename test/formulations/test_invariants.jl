# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The invariants of the formulations on every problem of the library and on the edge models of the
# Phase 1 review: the primal value is on the attained side of the dual bound, the primal rule is
# feasible on samples of the support, and the sampled rule is feasible on its outcomes and at least
# as good there as the primal rule. The edge models are also checked against the dense oracles.

module TestInvariants

using JuMP
using Test

import HiGHS
import LinearAlgebra
import LinearDecisionRules
import LinearDecisionRules: DualLDR, PrimalLDR, SampledLDR, moments, second_moment
import LinearDecisionRules: solve_formulation
import Random

const LDR = LinearDecisionRules

include(joinpath(@__DIR__, "..", "oracles", "dense.jl"))
include(joinpath(@__DIR__, "problems.jl"))

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

# An exact MIP bound, so that the integer dual is a bound of the integer primal.
const SETTINGS = Dict("mip_rel_gap" => 0.0)

function solve(form, f)
    return solve_formulation(form, f, moments(form), HiGHS.Optimizer; raw_attributes = SETTINGS)
end

# The largest violation by the rule `X` of the rows and bounds of `form` at the outcome `ξ`,
# relative to the size of the terms.
function violation(form, X, ξ)
    x = X * ξ
    scale = 1 + maximum(abs, x; init = 0.0) + maximum(abs, ξ)
    v = maximum(abs, form.A_e * x - form.B_e * ξ; init = 0.0)
    v = max(v, maximum(form.A_u * x - form.B_u * ξ; init = 0.0))
    v = max(v, maximum(form.B_l * ξ - form.A_l * x; init = 0.0))
    v = max(v, maximum(x - form.upper; init = 0.0), maximum(form.lower - x; init = 0.0))
    return v / scale
end

# Outcomes drawn from the laws that satisfy the valid rows, so that they are in the support Ξ.
function support_samples(form, n::Int; seed::Int = 1)
    rng = Random.Xoshiro(seed)
    samples = Vector{Float64}[]
    for _ in 1:100n
        ξ = ones(1 + length(form.uncertainties))
        for law in form.laws
            ξ[law.columns] .= rand(rng, law.distribution)
        end
        if all(form.W_valid * ξ[2:end] .>= form.h_valid .- 1e-9)
            push!(samples, ξ)
        end
        length(samples) == n && break
    end
    return samples
end

# `E[x'Px + x'Cξ + η'Qη + d'η + f]` for the rule `X` and the moments `mom`.
function objective(form, X, mom)
    form.sense == MOI.FEASIBILITY_SENSE && return 0.0
    M = second_moment(mom)
    return LinearAlgebra.tr(X' * form.P * X * M) +
           LinearAlgebra.tr(form.C' * X * M) +
           LinearAlgebra.tr(form.Q * M[2:end, 2:end]) +
           form.d' * M[2:end, 1] +
           form.f
end

function check_invariants(form, primal, dual, sampled, f::SampledLDR)
    @test primal.primal_status == MOI.FEASIBLE_POINT
    @test sampled.primal_status == MOI.FEASIBLE_POINT
    # The primal value is attained and the dual bounds the optimum: primal ≤ dual for a Max.
    sign = form.sense == MOI.MAX_SENSE ? 1.0 : -1.0
    gap = sign * (dual.objective_bound - primal.objective_value)
    @test gap >= -1e-6 * (1 + abs(primal.objective_value))
    # The primal rule is feasible for every outcome of the support.
    samples = support_samples(form, 200)
    @test !isempty(samples)
    @test maximum(ξ -> violation(form, primal.X, ξ), samples) <= 1e-7
    # The sampled rule is feasible on its outcomes; where the primal rule is feasible too, the
    # sampled rule is at least as good as the primal rule on the empirical moments.
    outcomes = LDR._sample_outcomes(form, f)
    @test maximum(ξ -> violation(form, sampled.X, ξ), eachcol(outcomes)) <= 1e-7
    if maximum(ξ -> violation(form, primal.X, ξ), eachcol(outcomes)) <= 1e-9
        empirical = objective(form, primal.X, LDR._empirical_moments(outcomes))
        @test sign * (sampled.objective_value - empirical) >= -1e-6 * (1 + abs(empirical))
    end
    return
end

function test_library()
    f = SampledLDR(50)
    for p in Problems.library()
        @testset "$(p.name)" begin
            form = Problems.form(p)
            primal, dual, sampled = solve(form, PrimalLDR()), solve(form, DualLDR()), solve(form, f)
            check_invariants(form, primal, dual, sampled, f)
        end
    end
    return
end

# The edge models of the review agree with the dense oracles and satisfy the invariants.
function test_edge_cases()
    f = SampledLDR(50)
    for p in Problems.edge_cases()
        @testset "$(p.name)" begin
            form = Problems.form(p)
            mom = moments(form)
            primal = solve(form, PrimalLDR())
            # No warning: the correlated law of `law_spanning_stages` has linear conditional means.
            dual = @test_logs solve(form, DualLDR())
            sampled = solve(form, f)
            for (result, builder) in
                ((primal, DenseOracle.build_primal), (dual, DenseOracle.build_dual))
                oracle = DenseOracle.solve(builder, form, mom, HiGHS.Optimizer)
                @test result.termination_status == oracle.termination_status == MOI.OPTIMAL
                @test Problems.matches(result.objective_value, oracle.objective_value, p) ||
                      isapprox(result.objective_value, oracle.objective_value; rtol = 1e-7)
            end
            check_invariants(form, primal, dual, sampled, f)
        end
    end
    return
end

end # module TestInvariants

TestInvariants.runtests()
