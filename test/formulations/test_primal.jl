# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestPrimal

using JuMP
using Test

import Clarabel
import HiGHS
import LinearDecisionRules
import LinearDecisionRules: DualLDR, FormulationResult, PrimalLDR, moments, solve_formulation

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
            result = solve_formulation(form, PrimalLDR(), mom, HiGHS.Optimizer)
            oracle = DenseOracle.solve(DenseOracle.build_primal, form, mom, HiGHS.Optimizer)
            @test result.termination_status == MOI.OPTIMAL
            @test isapprox(result.objective_value, oracle.objective_value; rtol = 1e-7)
        end
    end
    return
end

# Second-order cone rows of Ξ: multipliers in the second-order cone. Clarabel solves to about 1e-8.
function test_conic_dense_oracle()
    for p in Problems.conic_library()
        form = Problems.form(p)
        mom = moments(form)
        @testset "$(p.name)" begin
            result = solve_formulation(form, PrimalLDR(), mom, Clarabel.Optimizer)
            oracle = DenseOracle.solve(DenseOracle.build_primal, form, mom, Clarabel.Optimizer)
            @test result.termination_status == MOI.OPTIMAL
            @test isapprox(result.objective_value, oracle.objective_value; rtol = 1e-6)
            isnan(p.primal) || @test isapprox(result.objective_value, p.primal; rtol = 1e-6)
        end
    end
    return
end

# A disc that contains the box leaves Ξ, and the value, unchanged; a smaller one tightens Ξ, so
# the primal improves. The multipliers of the cone are named `λ[r,b,2]`.
function test_conic_support()
    box = Problems.form(Problems.shared_stock(; valid = false))
    lp = solve_formulation(box, PrimalLDR(), moments(box), HiGHS.Optimizer)
    form = Problems.form(Problems.shared_stock_disc(30))
    result = solve_formulation(form, PrimalLDR(), moments(form), Clarabel.Optimizer; names = true)
    @test isapprox(result.objective_value, lp.objective_value; rtol = 1e-6)
    @test variable_by_name(result.model, "λ[1,1,2][3]") !== nothing
    # One per inequality that touches the block: the stock, sell .<= d and sell .>= 0.
    @test num_constraints(result.model, Vector{VariableRef}, MOI.SecondOrderCone) == 5
    tight = Problems.form(Problems.shared_stock_disc(20; truncate = true, cut = true))
    mom = moments(tight)
    disc = solve_formulation(tight, PrimalLDR(), mom, Clarabel.Optimizer)
    # The same moments with Ξ the box: a worse primal.
    loose = LinearDecisionRules._replace(tight; conic_rows = LinearDecisionRules.ConicRow[])
    box_value = solve_formulation(loose, PrimalLDR(), mom, HiGHS.Optimizer).objective_value
    @test disc.objective_value > box_value + 1
    return
end

function test_result()
    p = Problems.newsvendor()
    form = Problems.form(p)
    result = solve_formulation(form, PrimalLDR(), moments(form), HiGHS.Optimizer)
    @test result isa FormulationResult
    @test result.formulation == PrimalLDR()
    @test result.primal_status == MOI.FEASIBLE_POINT
    @test result.raw_status isa String
    @test result.objective_value ≈ 460
    @test result.objective_bound ≈ 460
    @test result.solve_time >= 0
    # buy = 120, sell = demand, ret = 120 - demand.
    @test result.X ≈ [120 0; 0 1; 120 -1]
    @test result.model isa Model
    @test objective_sense(result.model) == MOI.MAX_SENSE
    @test occursin("PrimalLDR(): OPTIMAL, FEASIBLE_POINT", sprint(show, result))
    return
end

function test_solver_settings()
    p = Problems.newsvendor()
    form = Problems.form(p)
    result = solve_formulation(
        form,
        PrimalLDR(),
        moments(form),
        HiGHS.Optimizer;
        time_limit = 10.0,
        raw_attributes = Dict("presolve" => "off"),
    )
    @test get_attribute(result.model, MOI.Silent())
    # The build takes some of the limit.
    @test 0 < time_limit_sec(result.model) < 10
    @test get_attribute(result.model, "presolve") == "off"
    @test result.objective_value ≈ 460
    return
end

# Slacks are expressions: no slack variables, and multipliers only for the touched blocks.
function test_size()
    p = Problems.newsvendor()
    form = Problems.form(p)
    model = solve_formulation(form, PrimalLDR(), moments(form), HiGHS.Optimizer).model
    # X: buy (1), sell (2), ret (2); λ: 2 for each of the 2 rows and the 2 lower bounds.
    @test num_variables(model) == 5 + 8
    # Per row: one equality for the column of demand, and the coupling inequality.
    @test num_constraints(model, AffExpr, MOI.EqualTo{Float64}) == 4
    @test num_constraints(model, AffExpr, MOI.GreaterThan{Float64}) == 4
    return
end

# Names slow the build down: the inner problem has them only on request.
function test_names()
    p = Problems.newsvendor()
    form = Problems.form(p)
    for f in (PrimalLDR(), DualLDR())
        model = solve_formulation(form, f, moments(form), HiGHS.Optimizer).model
        @test all(isempty ∘ name, all_variables(model))
        model = solve_formulation(form, f, moments(form), HiGHS.Optimizer; names = true).model
        @test all(!isempty ∘ name, all_variables(model))
        # sell is decision 2; its rule on the demand is X[2,2].
        @test value(variable_by_name(model, "X[2,2]")) ≈ 1
        @test variable_by_name(model, f == PrimalLDR() ? "λ[1,1][1]" : "S[1,2]") !== nothing
    end
    return
end

end # module TestPrimal

TestPrimal.runtests()
