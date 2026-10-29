# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The invariants of the formulations, and their agreement with the dense oracles, on every problem
# of the library, the edge models and the conic library: the primal value is on the attained side
# of the dual bound, and of the optimum when it is known; each rule is zero on the columns that its
# decision does not observe; the primal rule is feasible on samples of the support; the sampled
# rule is feasible on its outcomes and at least as good there as the primal rule.

module TestInvariants

using JuMP
using Test

import Clarabel
import HiGHS
import LinearAlgebra
import LinearDecisionRules
import LinearDecisionRules:
    DualLDR, InformationStructure, PrimalLDR, SampledLDR, moments, observes, second_moment
import LinearDecisionRules: solve_formulation
import Random

const LDR = LinearDecisionRules

include(joinpath(@__DIR__, "..", "oracles", "dense.jl"))
include(joinpath(@__DIR__, "..", "problems", "library.jl"))

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

# Each set of problems with its solver, and the tolerance of that solver: the relative agreement
# with the dense oracles and the violation of a row by a rule. Clarabel solves to about 1e-8.
const PROBLEM_SETS = [
    (Problems.library, HiGHS.Optimizer, 1e-7),
    (Problems.edge_cases, HiGHS.Optimizer, 1e-7),
    (Problems.conic_library, Clarabel.Optimizer, 1e-6),
]

# HiGHS with an exact MIP bound (see `Problems.solve`); Clarabel with its defaults.
function solve(form, f, optimizer)
    optimizer === HiGHS.Optimizer && return Problems.solve(form, f)
    return solve_formulation(form, f, moments(form), optimizer)
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

# Outcomes drawn from the laws that satisfy the valid rows and the conic rows, so that they are in
# the support Ξ.
function support_samples(form, n::Int; seed::Int = 1)
    rng = Random.Xoshiro(seed)
    samples = Vector{Float64}[]
    for _ in 1:100n
        ξ = ones(1 + length(form.uncertainties))
        for law in form.laws
            ξ[law.columns] .= rand(rng, law.distribution)
        end
        η = ξ[2:end]
        in_cones = all(form.conic_rows) do row
            return LDR._cone_violation(row.cone, row.W * η - row.h) <= 1e-9
        end
        if in_cones && all(form.W_valid * η .>= form.h_valid .- 1e-9)
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

function check_invariants(p, form, primal, dual, sampled, f::SampledLDR, tolerance)
    @test primal.primal_status == MOI.FEASIBLE_POINT
    @test sampled.primal_status == MOI.FEASIBLE_POINT
    # The primal value is attained and the dual bounds the optimum: primal ≤ optimum ≤ dual for a
    # Max.
    sign = form.sense == MOI.MAX_SENSE ? 1.0 : -1.0
    slack = 1e-6 * (1 + abs(primal.objective_value))
    @test sign * (dual.objective_bound - primal.objective_value) >= -slack
    if !isnan(p.optimum)
        @test sign * (p.optimum - primal.objective_value) >= -slack
        @test sign * (dual.objective_bound - p.optimum) >= -slack
    end
    # The stage mask: a rule is exactly zero on the columns that its decision does not observe.
    observed = InformationStructure(form)
    for result in (primal, dual, sampled), i in axes(result.X, 1), j in axes(result.X, 2)
        observes(observed, i, j) || @test result.X[i, j] == 0
    end
    # The primal rule is feasible for every outcome of the support.
    samples = support_samples(form, 200)
    @test !isempty(samples)
    @test maximum(ξ -> violation(form, primal.X, ξ), samples) <= tolerance
    # The sampled rule is feasible on its outcomes; where the primal rule is feasible too, the
    # sampled rule is at least as good as the primal rule on the empirical moments.
    outcomes = LDR._sample_outcomes(form, f)
    @test maximum(ξ -> violation(form, sampled.X, ξ), eachcol(outcomes)) <= tolerance
    if maximum(ξ -> violation(form, primal.X, ξ), eachcol(outcomes)) <= 1e-9
        empirical = objective(form, primal.X, LDR._empirical_moments(outcomes))
        @test sign * (sampled.objective_value - empirical) >= -1e-6 * (1 + abs(empirical))
    end
    return
end

# The primal and dual formulations agree with their dense oracles, and with the known values of
# the continuous problems (those of the integer ones, a MIP bound, are in `test_known_values.jl`).
function check_oracles(p, form, primal, dual, optimizer, tolerance)
    mom = moments(form)
    for (result, builder, known) in
        ((primal, DenseOracle.build_primal, p.primal), (dual, DenseOracle.build_dual, p.dual))
        oracle = DenseOracle.solve(builder, form, mom, optimizer)
        @test result.termination_status == oracle.termination_status == MOI.OPTIMAL
        @test isapprox(result.objective_value, oracle.objective_value; rtol = tolerance) ||
              Problems.matches(result.objective_value, oracle.objective_value, p)
        if !isnan(known) && !any(form.integer)
            @test isapprox(result.objective_value, known; rtol = tolerance) ||
                  Problems.matches(result.objective_value, known, p)
        end
    end
    return
end

function test_invariants_and_oracles()
    f = SampledLDR(50)
    for (problems, optimizer, tolerance) in PROBLEM_SETS, p in problems()
        @testset "$(p.name)" begin
            form = Problems.form(p)
            primal = solve(form, PrimalLDR(), optimizer)
            dual = solve(form, DualLDR(), optimizer)
            sampled = solve(form, f, optimizer)
            check_oracles(p, form, primal, dual, optimizer, tolerance)
            check_invariants(p, form, primal, dual, sampled, f, tolerance)
        end
    end
    return
end

# The correlated law of `law_spanning_stages` has linear conditional expectations across its
# stages, so the dual bound gives no warning.
function test_no_warning_for_linear_conditional_expectations()
    form = Problems.form(Problems.law_spanning_stages())
    @test_logs Problems.solve(form, DualLDR())
    return
end

# The primal values increase and the dual values decrease over nested breakpoints.
function test_nested_breakpoints()
    primal, dual = Float64[], Float64[]
    for p in [Problems.newsvendor(); Problems.newsvendor_breakpoints.(1:4)]
        form = Problems.form(p)
        push!(primal, Problems.solve(form, PrimalLDR()).objective_value)
        push!(dual, Problems.solve(form, DualLDR()).objective_value)
    end
    @test all(diff(primal) .>= -1e-7)
    @test all(diff(dual) .<= 1e-7)
    @test all(primal .<= dual .+ 1e-7)
    return
end

end # module TestInvariants

TestInvariants.runtests()
