# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestSampled

using JuMP
using Test

import HiGHS
import LinearAlgebra
import LinearDecisionRules
import LinearDecisionRules:
    DualLDR, PrimalLDR, SampledLDR, moments, second_moment, solve_formulation
import TOML

const LDR = LinearDecisionRules

include(joinpath(@__DIR__, "..", "problems", "library.jl"))

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

# The value of the rule `X` on the empirical moments of `outcomes`, for a linear objective.
function empirical_value(form, X, outcomes)
    return LinearAlgebra.dot(form.C, X * second_moment(LDR._empirical_moments(outcomes))) + form.f
end

# On its outcomes, the sampled rule is at least as good as the primal rule (which is feasible
# there), and at most as good as the decisions that know the outcome (wait and see).
function test_sandwich()
    p = Problems.newsvendor()
    form = Problems.form(p)
    f = SampledLDR(50)
    outcomes = LDR._sample_outcomes(form, f)
    primal = solve(p, PrimalLDR())
    sampled = solve(p, f)
    # Knowing the demand d, the newsvendor buys and sells d: 5d.
    wait_and_see = 5 * sum(outcomes[2, :]) / 50
    @test empirical_value(form, primal.X, outcomes) <= sampled.objective_value + 1e-6
    @test sampled.objective_value <= wait_and_see + 1e-6
    @test sampled.X[1, 2] == 0  # buy is in stage 1
    # The inventory (Min) orders each demand when it knows it: Σ_t d_t.
    p = Problems.inventory(4; dual = 340)
    form = Problems.form(p)
    f = SampledLDR(200)
    outcomes = LDR._sample_outcomes(form, f)
    primal = solve(p, PrimalLDR())
    sampled = solve(p, f)
    wait_and_see = sum(outcomes[2:end, :]) / 200
    @test wait_and_see - 1e-6 <= sampled.objective_value
    @test sampled.objective_value <= empirical_value(form, primal.X, outcomes) + 1e-6
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
    ξs = LDR._sample_outcomes(form, SampledLDR(20; seed = 7))
    @test size(ξs) == (4, 20)
    @test all(==(1), ξs[1, :])
    @test all(80 .<= ξs[2, :] .<= 120) && all(8 .<= ξs[3, :] .<= 14) && all(80 .<= ξs[4, :] .<= 120)
    @test ξs == LDR._sample_outcomes(form, SampledLDR(20; seed = 7))
    @test second_moment(LDR._empirical_moments(ξs)) ≈ ξs * ξs' ./ 20
    return
end

# Equality rows are matched once, not per scenario; with N equalities per row, interior-point
# solvers such as Ipopt failed with "too few degrees of freedom" (previous version).
function test_equalities_once()
    p = Problems.hydro_toy()
    for N in (4, 5, 50)
        result = solve(p, SampledLDR(N))
        @test result.termination_status == MOI.OPTIMAL
        # 8 rows of coefficient matching, and the means of `gt` and `vf` in the quadratic objective.
        @test num_constraints(result.model, AffExpr, MOI.EqualTo{Float64}) == 8 + 2
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
