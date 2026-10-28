# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestPrimal

using JuMP
using Test

import Clarabel
import Distributions
import HiGHS
import LinearDecisionRules
import LinearDecisionRules: DualLDR, FormulationResult, PrimalLDR, moments, solve_formulation

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
    f = PrimalLDR(; box_closed_form = false)
    model = solve_formulation(form, f, moments(form), HiGHS.Optimizer).model
    # X: buy (1), sell (2), ret (2); λ: 2 for each of the 2 rows and the 2 lower bounds.
    @test num_variables(model) == 5 + 8
    # Per row: one equality for the column of demand, and the coupling inequality.
    @test num_constraints(model, AffExpr, MOI.EqualTo{Float64}) == 4
    @test num_constraints(model, AffExpr, MOI.GreaterThan{Float64}) == 4
    # The demand is a box: one auxiliary `w ≥ ±s` per row instead of two multipliers and an
    # equality.
    model = solve_formulation(form, PrimalLDR(), moments(form), HiGHS.Optimizer).model
    @test num_variables(model) == 5 + 4
    @test num_constraints(model, AffExpr, MOI.EqualTo{Float64}) == 0
    @test num_constraints(model, AffExpr, MOI.GreaterThan{Float64}) == 4 + 4
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
        @test variable_by_name(model, f == PrimalLDR() ? "w[1,2]" : "S[1,2]") !== nothing
    end
    f = PrimalLDR(; box_closed_form = false)
    model = solve_formulation(form, f, moments(form), HiGHS.Optimizer; names = true).model
    @test variable_by_name(model, "λ[1,1][1]") !== nothing
    return
end

function test_box_closed_form_switch()
    @test PrimalLDR().box_closed_form
    f = PrimalLDR(; box_closed_form = false)
    @test !f.box_closed_form
    @test f == PrimalLDR()
    @test sprint(show, f) == "PrimalLDR(box_closed_form = false)"
    f = PrimalLDR(; attributes = ["presolve" => "off"], box_closed_form = false)
    @test sprint(show, f) ==
          "PrimalLDR(attributes = [\"presolve\" => \"off\"], box_closed_form = false)"
    return
end

# Whether the rule `X` is optimal for `f`: fixed in the inner problem of `f`, it keeps the value.
function attains(form, mom, f, X, value, optimizer; rtol)
    model = Model(optimizer)
    set_silent(model)
    rules = LinearDecisionRules._build!(model, f, form, mom)
    for i in eachindex(rules), j in eachindex(rules[i])
        fix(rules[i][j], X[i, j]; force = true)
    end
    optimize!(model)
    is_solved_and_feasible(model) || return false
    fixed = form.sense == MOI.FEASIBILITY_SENSE ? 0.0 : objective_value(model)
    return isapprox(fixed, value; rtol = rtol, atol = rtol)
end

# The closed form on box blocks, the general form and the dense oracle have the same value. Their
# rules are the same, or, when the inner problem is degenerate (or a QP solved to a tolerance), the
# rule of each form is optimal for the other.
function compare_box_closed_form(problems, optimizer; rtol)
    general = PrimalLDR(; box_closed_form = false)
    for p in problems
        form = Problems.form(p)
        mom = moments(form)
        @testset "$(p.name)" begin
            closed = solve_formulation(form, PrimalLDR(), mom, optimizer)
            result = solve_formulation(form, general, mom, optimizer)
            oracle = DenseOracle.solve(DenseOracle.build_primal, form, mom, optimizer)
            @test closed.termination_status == MOI.OPTIMAL
            @test isapprox(closed.objective_value, result.objective_value; rtol = rtol)
            @test isapprox(closed.objective_value, oracle.objective_value; rtol = rtol)
            if !isapprox(closed.X, result.X; atol = 1e-6)
                value = closed.objective_value
                @test attains(form, mom, general, closed.X, value, optimizer; rtol = rtol)
                @test attains(form, mom, PrimalLDR(), result.X, value, optimizer; rtol = rtol)
                @test attains(form, mom, PrimalLDR(), oracle.X, value, optimizer; rtol = rtol)
            end
        end
    end
    return
end

function test_box_closed_form()
    problems = [Problems.library(); Problems.edge_cases(); box_edge_cases()]
    compare_box_closed_form(problems, HiGHS.Optimizer; rtol = 1e-7)
    return
end

function test_conic_box_closed_form()
    compare_box_closed_form(Problems.conic_library(), Clarabel.Optimizer; rtol = 1e-6)
    return
end

# The cases of the closed form: a slack constant on a box (`x` does not observe `e`), a point
# (`p`), a law bounded on one side (`g`, the general form) and a zero endpoint (`e`). The rule
# `x = 10`, `y = 13 + g` is optimal: the value is 10 + 13 + E[g] = 24.
function box_edge_cases()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @variable(model, e)
    @variable(model, p)
    @variable(model, g)
    @constraint(model, x <= e + 10)
    @constraint(model, y <= 2p + g + 3)
    @constraint(model, y >= g)
    @objective(model, Max, x + y)
    info = Problems.ModelInfo(;
        uncertainties = [
            Problems.declare(Distributions.Uniform(0, 4), e; stages = [2]),
            Problems.declare(Distributions.Dirac(5.0), p; stages = [2]),
            Problems.declare(Distributions.Exponential(1.0), g; stages = [2]),
        ],
        decision_stages = Problems.stages(x => 1, y => 2),
    )
    return [Problems.Problem("box_edge_cases", "", model, info, 24, NaN)]
end

function test_box_edge_cases()
    p = only(box_edge_cases())
    form = Problems.form(p)
    mom = moments(form)
    blocks = LinearDecisionRules.support_blocks(form)
    @test [LinearDecisionRules._box_bounds(b) for b in blocks] == [([0.0], [4.0]), ([5.0], [5.0]), nothing]
    result = solve_formulation(form, PrimalLDR(), mom, HiGHS.Optimizer)
    @test result.objective_value ≈ p.primal
    # X: x (1), y (4). The two rows of `y` have a `w` on `e` and a `λ` for the lower bound of `g`;
    # the constant slack of `x ≤ e + 10` on `e` and the point `p` need none.
    @test num_variables(result.model) == 1 + 4 + 2 * 2
    @test num_constraints(result.model, AffExpr, MOI.EqualTo{Float64}) == 2
    return
end

end # module TestPrimal

TestPrimal.runtests()
