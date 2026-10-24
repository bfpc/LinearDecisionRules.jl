# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestProblems

using JuMP
using Test

import HiGHS
import LinearDecisionRules: DualLDR, PrimalLDR, SampledLDR, moments, solve_formulation
import TOML

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

# An exact MIP bound, so that the integer dual is the known value.
const SETTINGS = Dict("mip_rel_gap" => 0.0)

function solve(p::Problems.Problem, f)
    form = Problems.form(p)
    result = solve_formulation(form, f, moments(form), HiGHS.Optimizer; raw_attributes = SETTINGS)
    return form, result
end

# `x1` in the legacy file is `x[1]` here.
function _index(names, legacy::String)
    i = findfirst(==(legacy), names)
    if i === nothing
        m = match(r"^([a-z_]+)(\d+)$", legacy)
        i = findfirst(==("$(m[1])[$(m[2])]"), names)
    end
    return i
end

# The legacy rules of the variables listed in `legacy_values.toml`.
function check_rules(form, result, rules)
    for (name, rule) in rules
        i = _index(form.decision_names, name)
        @test result.X[i, 1] ≈ rule["constant"] atol = 1e-6
        for (u, coefficient) in rule["coefficients"]
            j = 1 + _index(form.uncertainty_names, u)
            @test result.X[i, j] ≈ coefficient atol = 1e-6
        end
    end
    return
end

function test_known_values()
    for p in Problems.library()
        @testset "$(p.name)" begin
            form, primal = solve(p, PrimalLDR())
            _, dual = solve(p, DualLDR())
            @test primal.termination_status == MOI.OPTIMAL
            @test primal.primal_status == MOI.FEASIBLE_POINT
            @test dual.termination_status == MOI.OPTIMAL
            # With integer decisions, the dual's bound is the inner MIP's bound.
            isnan(p.primal) || @test Problems.matches(primal.objective_value, p.primal, p)
            isnan(p.dual) || @test Problems.matches(dual.objective_bound, p.dual, p)
            if !isempty(p.legacy)
                legacy = LEGACY[p.legacy]
                @test Problems.matches(primal.objective_value, legacy["primal"], p)
                @test Problems.matches(dual.objective_bound, legacy["dual"], p)
            end
            if p.legacy in ("newsvendor", "newsvendor_integer", "inventory_T4", "inventory_T8")
                check_rules(form, primal, LEGACY[p.legacy]["rules"]["primal"])
            end
            _, sampled = solve(p, SampledLDR(100))
            @test sampled.termination_status == MOI.OPTIMAL
            @test isfinite(sampled.objective_value)
        end
    end
    return
end

function test_integer_bound()
    p = Problems.newsvendor(; integer = true)
    form = Problems.form(p)
    for f in (PrimalLDR(), DualLDR())
        result = solve_formulation(form, f, moments(form), HiGHS.Optimizer)
        @test result.X[1, 1] ≈ round(result.X[1, 1])
        # With the default MIP gap, the bound is still on the right side of the incumbent.
        @test result.objective_bound >= result.objective_value - 1e-6
    end
    _, dual = solve(p, DualLDR())
    @test dual.objective_bound ≈ 486
    @test dual.X[1, :] ≈ [107, 0]
    return
end

# A decision has no coefficient on an uncertainty of a later stage.
function test_three_stages_mask()
    p = Problems.newsvendor_three_stages()
    for f in (PrimalLDR(), DualLDR(), SampledLDR(50))
        form, result = solve(p, f)
        stage = form.decision_stages
        later = [1 + j for j in eachindex(form.uncertainties)]
        for i in eachindex(form.decisions), j in later
            if form.uncertainty_stages[j-1] > stage[i]
                @test result.X[i, j] == 0
            end
        end
        sell2 = findfirst(==("sell2"), form.decision_names)
        demand3 = 1 + findfirst(==("demand3"), form.uncertainty_names)
        @test result.X[sell2, demand3] == 0
    end
    return
end

# A valid inequality shrinks Ξ: the primal can only improve, the dual can only tighten.
function test_valid_inequality()
    with, without = Problems.shared_stock(), Problems.shared_stock(; valid = false)
    p_with = solve(with, PrimalLDR())[2].objective_value
    p_without = solve(without, PrimalLDR())[2].objective_value
    d_with = solve(with, DualLDR())[2].objective_value
    d_without = solve(without, DualLDR())[2].objective_value
    @test p_without <= p_with + 1e-6
    @test p_with <= d_with + 1e-6
    @test d_with < d_without
    return
end

function test_infeasible()
    p = Problems.infeasible()
    for f in (PrimalLDR(), DualLDR(), SampledLDR(10))
        _, result = solve(p, f)
        @test result.termination_status == MOI.INFEASIBLE
        @test result.primal_status != MOI.FEASIBLE_POINT
        @test isnan(result.objective_value)
        @test all(isnan, result.X)
    end
    return
end

# Spike audit, bug 2: a ray of an unbounded problem is not reported as a solution.
function test_unbounded()
    problems = (Problems.unbounded(), Problems.unbounded_without_rows())
    for p in problems, f in (PrimalLDR(), DualLDR(), SampledLDR(10))
        _, result = solve(p, f)
        @test result.termination_status in (MOI.DUAL_INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
        @test result.primal_status != MOI.FEASIBLE_POINT
        @test isnan(result.objective_value)
        @test all(isnan, result.X)
        # No dual bound exists: the bound of a maximization is `Inf`.
        if result.termination_status == MOI.DUAL_INFEASIBLE
            @test result.objective_bound == Inf
        end
    end
    return
end

function test_feasibility_sense()
    model = Model()
    @variable(model, x <= 3)
    form = canonicalize_model(model)
    for f in (PrimalLDR(), DualLDR(), SampledLDR(10))
        result = solve_formulation(form, f, moments(form), HiGHS.Optimizer)
        @test result.termination_status == MOI.OPTIMAL
        @test result.objective_value == 0
        @test result.X[1, 1] <= 3
    end
    return
end

function canonicalize_model(model)
    return Problems.form(Problems.Problem("", "", model, Problems.ModelInfo(), NaN, NaN))
end

end # module TestProblems

TestProblems.runtests()
