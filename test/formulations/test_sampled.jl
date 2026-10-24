# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestSampled

using JuMP
using Test

import HiGHS
import LinearDecisionRules
import LinearDecisionRules:
    DualLDR, PrimalLDR, SampledLDR, moments, second_moment, solve_formulation
import TOML

const LDR = LinearDecisionRules

include(joinpath(@__DIR__, "problems.jl"))

const LEGACY = TOML.parsefile(joinpath(@__DIR__, "..", "reference", "legacy_values.toml"))

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

function solve(p::Problems.Problem, f)
    return (form = Problems.form(p); solve_formulation(form, f, moments(form), HiGHS.Optimizer))
end

# The legacy sampler drew `rand(Xoshiro(1234), Uniform(80, 120))` scenario by scenario, as here.
function test_legacy_value()
    result = solve(Problems.newsvendor(), SampledLDR(1000))
    @test result.termination_status == MOI.OPTIMAL
    @test result.objective_value ≈ LEGACY["sampled_newsvendor_N1000_seed1234"]["sampled"]
    return
end

# As in the tests of the previous version: the sampled value is near [primal, dual].
function test_sandwich()
    p = Problems.newsvendor()
    primal = solve(p, PrimalLDR()).objective_value
    dual = solve(p, DualLDR()).objective_value
    sampled = solve(p, SampledLDR(50))
    @test primal - 5 <= sampled.objective_value <= dual + 1
    @test sampled.X[1, 2] == 0  # buy is in stage 1
    p = Problems.inventory(4; dual = 340)
    sampled = solve(p, SampledLDR(200)).objective_value
    @test p.dual - 1 <= sampled <= p.primal + 5
    return
end

function test_reproducible()
    p = Problems.newsvendor_two_products()
    a, b = solve(p, SampledLDR(30)), solve(p, SampledLDR(30))
    @test a.objective_value == b.objective_value
    @test a.X == b.X
    c = solve(p, SampledLDR(30; seed = 1))
    @test c.objective_value != a.objective_value
    return
end

function test_empirical_moments()
    form = Problems.form(Problems.newsvendor_three_stages())
    Ξ = LDR._sample_outcomes(form, SampledLDR(20; seed = 7))
    @test size(Ξ) == (4, 20)
    @test all(==(1), Ξ[1, :])
    @test all(80 .<= Ξ[2, :] .<= 120) && all(8 .<= Ξ[3, :] .<= 14) && all(80 .<= Ξ[4, :] .<= 120)
    @test Ξ == LDR._sample_outcomes(form, SampledLDR(20; seed = 7))
    @test second_moment(LDR._empirical_moments(Ξ)) ≈ Ξ * Ξ' ./ 20
    return
end

# Equality rows are matched once, not per scenario; with N equalities per row, interior-point
# solvers such as Ipopt failed with "too few degrees of freedom" (previous version).
function test_equalities_once()
    p = Problems.hydro_toy()
    for N in (4, 5, 50)
        result = solve(p, SampledLDR(N))
        @test result.termination_status == MOI.OPTIMAL
        @test num_constraints(result.model, AffExpr, MOI.EqualTo{Float64}) == 8
        # gt + gh == 0.3 for every outcome.
        @test result.X[3, 1] + result.X[4, 1] ≈ 0.3
        @test result.X[3, 2:end] ≈ -result.X[4, 2:end] atol = 1e-9
    end
    return
end

function test_deterministic_rows_once()
    result = solve(Problems.deterministic(), SampledLDR(10))
    @test num_constraints(result.model, AffExpr, MOI.LessThan{Float64}) +
          num_constraints(result.model, AffExpr, MOI.GreaterThan{Float64}) == 1
    @test result.objective_value ≈ 6
    return
end

function test_integer()
    result = solve(Problems.newsvendor(; integer = true), SampledLDR(100))
    @test result.termination_status == MOI.OPTIMAL
    @test result.X[1, 1] ≈ round(result.X[1, 1])
    @test isfinite(result.objective_bound)
    return
end

end # module TestSampled

TestSampled.runtests()
