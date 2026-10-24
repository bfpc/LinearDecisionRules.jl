# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestDual

using JuMP
using Test

import Distributions
import HiGHS
import LinearDecisionRules: DualLDR, PrimalLDR, moments, solve_formulation

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

function test_dense_oracle()
    for p in Problems.library()
        form = Problems.form(p)
        mom = moments(form)
        @testset "$(p.name)" begin
            result = solve_formulation(form, DualLDR(), mom, HiGHS.Optimizer)
            oracle = DenseOracle.solve(DenseOracle.build_dual, form, mom, HiGHS.Optimizer)
            @test result.termination_status == MOI.OPTIMAL
            @test isapprox(result.objective_value, oracle.objective_value; rtol = 1e-7)
        end
    end
    return
end

# Spike audit, bug 1: without bounded uncertainty, the inequalities stay (Max 2x, x ≤ 3 is 6).
function test_deterministic()
    for p in (Problems.deterministic(), Problems.deterministic(; bound = true))
        form = Problems.form(p)
        result = solve_formulation(form, DualLDR(), moments(form), HiGHS.Optimizer)
        @test result.termination_status == MOI.OPTIMAL
        @test result.objective_value ≈ 6
        @test result.objective_bound ≈ 6
        @test result.X ≈ [3;;]
    end
    return
end

# A row on an unbounded block keeps `E[s(ξ)] ≥ 0`, so the dual is bounded.
function test_unbounded_support()
    p = Problems.unbounded_support()
    form = Problems.form(p)
    result = solve_formulation(form, DualLDR(), moments(form), HiGHS.Optimizer)
    @test result.objective_value ≈ 6
    return
end

# S is masked by history, and the cone rows of a block appear only in the rows that see it.
function test_size()
    p = Problems.newsvendor_three_stages()
    form = Problems.form(p)
    model = solve_formulation(form, DualLDR(), moments(form), HiGHS.Optimizer).model
    # Rows (structural columns of S): sell2 + ret2 <= buy + buy2 (3), sell2 <= demand2 (3),
    # sell2 + ret2 + sell3 + ret3 <= buy + buy2 (4), sell3 <= demand3 (4), and the lower bounds of
    # sell2, ret2, buy2 (3 each) and sell3, ret3 (4 each).
    num_S = 3 + 3 + 4 + 4 + 3 * 3 + 2 * 4
    num_X = 1 + 3 * 3 + 2 * 4
    num_rows = 4 + 5
    @test num_variables(model) == num_X + num_S + num_rows
    # Each column of each block has 2 box rows; a stage-2 row sees 2 blocks, a stage-3 row 3.
    @test num_constraints(model, AffExpr, MOI.GreaterThan{Float64}) == 2 * (2 * 5 + 3 * 4)
    @test num_constraints(model, AffExpr, MOI.EqualTo{Float64}) == num_S + num_rows
    return
end

# The bound needs stagewise independence: a law that correlates two stages gets a warning.
function test_warns_when_a_law_correlates_stages()
    p = Problems.law_spanning_stages()
    form = Problems.form(p)
    mom = moments(form)
    warning = (:warn, r"in the law of d\[1\], d\[2\] \(stages 2, 3\)\. Stagewise independence")
    result = @test_logs warning solve_formulation(form, DualLDR(), mom, HiGHS.Optimizer)
    @test result.termination_status == MOI.OPTIMAL
    @test_logs solve_formulation(form, PrimalLDR(), mom, HiGHS.Optimizer)
    # Independent components in different stages, or correlated ones in one stage, are fine.
    for (law, t) in (
        (Distributions.product_distribution([Distributions.Uniform(80, 120) for _ in 1:2]), [2, 3]),
        (Problems.CORRELATED, [2, 2]),
    )
        info = Problems.ModelInfo(;
            uncertainties = [Problems.declare(law, p.model[:d]...; stages = t)],
            decision_stages = p.info.decision_stages,
        )
        form = Problems.canonicalize(backend(p.model), info)
        @test_logs solve_formulation(form, DualLDR(), moments(form), HiGHS.Optimizer)
    end
    return
end

end # module TestDual

TestDual.runtests()
