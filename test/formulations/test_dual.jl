# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestDual

using JuMP
using Test

import Distributions
import Clarabel
import HiGHS
import LinearDecisionRules
import LinearDecisionRules: DualLDR, PrimalLDR, moments, solve_formulation

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

# Second-order cone rows of Ξ: cone rows of the dual in the second-order cone.
function test_conic_dense_oracle()
    for p in Problems.conic_library()
        form = Problems.form(p)
        mom = moments(form)
        @testset "$(p.name)" begin
            result = solve_formulation(form, DualLDR(), mom, Clarabel.Optimizer)
            oracle = DenseOracle.solve(DenseOracle.build_dual, form, mom, Clarabel.Optimizer)
            @test result.termination_status == MOI.OPTIMAL
            @test isapprox(result.objective_value, oracle.objective_value; rtol = 1e-6)
            isnan(p.dual) || @test isapprox(result.objective_value, p.dual; rtol = 1e-6)
            primal = solve_formulation(form, PrimalLDR(), mom, Clarabel.Optimizer)
            tol = 1e-6 * abs(result.objective_value)
            @test primal.objective_value <= result.objective_value + tol
        end
    end
    return
end

# A disc that contains the box leaves the bound unchanged; a smaller one tightens it.
function test_conic_support()
    box = Problems.form(Problems.shared_stock(; valid = false))
    lp = solve_formulation(box, DualLDR(), moments(box), HiGHS.Optimizer)
    form = Problems.form(Problems.shared_stock_disc(30))
    result = solve_formulation(form, DualLDR(), moments(form), Clarabel.Optimizer)
    @test isapprox(result.objective_value, lp.objective_value; rtol = 1e-6)
    @test num_constraints(result.model, Vector{AffExpr}, MOI.SecondOrderCone) > 0
    tight = Problems.form(Problems.shared_stock_disc(20; truncate = true, cut = true))
    mom = moments(tight)
    disc = solve_formulation(tight, DualLDR(), mom, Clarabel.Optimizer)
    # The same moments with Ξ the box: a weaker bound.
    loose = LinearDecisionRules._replace(tight; conic_rows = LinearDecisionRules.ConicRow[])
    box_bound = solve_formulation(loose, DualLDR(), mom, HiGHS.Optimizer).objective_value
    @test disc.objective_value < box_bound - 1
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

# The bound of an inner LP solved to optimality is its dual objective value, which is valid for an
# interior point without crossover too: it matches the simplex bound.
function test_bound_is_the_dual_objective_of_an_lp()
    for p in (Problems.inventory(4), Problems.inventory(8), Problems.newsvendor_three_stages())
        form = Problems.form(p)
        mom = moments(form)
        result = solve_formulation(form, DualLDR(), mom, HiGHS.Optimizer)
        @test result.termination_status == MOI.OPTIMAL
        @test result.objective_bound == dual_objective_value(result.model)
        @test isapprox(result.objective_bound, result.objective_value; rtol = 1e-9)
    end
    return
end

# After a solver error the dual point is feasible up to a tolerance only, so it gives no bound;
# after a limit it does. Found on the hydro benchmark (T = 12), where HiGHS's interior point
# without crossover ended in OTHER_ERROR with a dual objective above the optimum of a minimization.
function test_no_bound_after_a_solver_error()
    form = Problems.form(Problems.inventory(4))
    for status in (MOI.OTHER_ERROR, MOI.TIME_LIMIT)
        mock = () -> MOI.Utilities.MockOptimizer(MOI.Utilities.Model{Float64}())
        model = Model(mock)
        @variable(model, x >= 0)
        @objective(model, Min, x)
        optimize!(model)
        inner = unsafe_backend(model)
        MOI.set(inner, MOI.TerminationStatus(), status)
        MOI.set(inner, MOI.PrimalStatus(), MOI.NO_SOLUTION)
        MOI.set(inner, MOI.DualStatus(), MOI.FEASIBLE_POINT)
        bound = LinearDecisionRules._objective_bound(form, model, NaN)
        if status == MOI.OTHER_ERROR
            @test isnan(bound)
        else
            # The mock computes the dual objective from its (zero) duals.
            @test bound == dual_objective_value(model)
        end
    end
    return
end

function test_interior_point_without_crossover()
    attributes = ["solver" => "ipm", "run_crossover" => "off"]
    for p in (Problems.inventory(4), Problems.inventory(8))
        form = Problems.form(p)
        mom = moments(form)
        simplex = solve_formulation(form, DualLDR(), mom, HiGHS.Optimizer)
        ipm = solve_formulation(form, DualLDR(; attributes), mom, HiGHS.Optimizer)
        @test ipm.termination_status == MOI.OPTIMAL
        @test get_attribute(ipm.model, "solver") == "ipm"
        @test get_attribute(ipm.model, "run_crossover") == "off"
        @test ipm.objective_bound == dual_objective_value(ipm.model)
        @test isapprox(ipm.objective_bound, simplex.objective_bound; rtol = 1e-6)
        # The inner primal point is feasible up to a tolerance only, so its value is not the bound.
        @test isapprox(ipm.objective_value, simplex.objective_bound; rtol = 1e-6)
    end
    return
end

# A quadratic inner problem keeps its primal objective: HiGHS's dual objective of a QP leaves out
# the quadratic term.
function test_quadratic_bound_is_the_primal_objective()
    form = Problems.form(Problems.hydro_toy())
    result = solve_formulation(form, DualLDR(), moments(form), HiGHS.Optimizer)
    @test result.objective_bound == result.objective_value
    @test isapprox(result.objective_bound, -0.412778; atol = 1e-6)
    return
end

# `attributes` of a formulation are set on its inner problem after the `raw_attributes`, and a
# formulation is the same whatever its attributes.
function test_formulation_attributes()
    f = DualLDR(; attributes = ["solver" => "ipm", "run_crossover" => "off"])
    @test f.attributes == ["solver" => "ipm", "run_crossover" => "off"]
    @test DualLDR(; attributes = Dict(:solver => "ipm")).attributes == ["solver" => "ipm"]
    @test isempty(DualLDR().attributes) && isempty(PrimalLDR().attributes)
    @test f == DualLDR() && hash(f) == hash(DualLDR())
    @test PrimalLDR(; attributes = ["solver" => "ipx"]) == PrimalLDR()
    @test f != PrimalLDR()
    @test sprint(show, DualLDR()) == "DualLDR()"
    @test sprint(show, f) ==
          "DualLDR(attributes = [\"solver\" => \"ipm\", \"run_crossover\" => \"off\"])"
    form = Problems.form(Problems.newsvendor())
    mom = moments(form)
    raw = ["solver" => "simplex", "presolve" => "off"]
    result = solve_formulation(form, f, mom, HiGHS.Optimizer; raw_attributes = raw)
    @test get_attribute(result.model, "solver") == "ipm"
    @test get_attribute(result.model, "presolve") == "off"
    primal = PrimalLDR(; attributes = ["presolve" => "off"])
    result = solve_formulation(form, primal, mom, HiGHS.Optimizer)
    @test get_attribute(result.model, "presolve") == "off"
    @test result.termination_status == MOI.OPTIMAL
    return
end

# The formulations of the optimizer, and the lookups of their results, ignore the attributes.
function test_formulation_attributes_through_the_optimizer()
    model = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, buy >= 0, LinearDecisionRules.FirstStage)
    @variable(model, sell >= 0)
    @variable(model, d in LinearDecisionRules.Uncertainty(Distributions.Uniform(80, 120)))
    @constraint(model, sell <= buy)
    @constraint(model, sell <= d)
    @objective(model, Max, 2sell - buy)
    dual = DualLDR(; attributes = ["presolve" => "off"])
    set_attribute(model, LinearDecisionRules.Formulations(), [PrimalLDR(), dual])
    optimize!(model)
    @test termination_status(model, DualLDR()) == MOI.OPTIMAL
    @test get_attribute(LinearDecisionRules.inner_model(model, dual), "presolve") == "off"
    @test objective_value(model, dual) == objective_value(model, DualLDR())
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

# Uncorrelated, but the second component is a function of the first, `60 + (d[1] - 80)²`: its
# conditional expectation is not affine.
const SQUARE = LinearDecisionRules.MvDiscreteNonParametric(
    [[79.0, 61.0], [80.0, 60.0], [81.0, 61.0]],
    [1 / 3, 1 / 3, 1 / 3],
)

const ELLIPSOID = LinearDecisionRules.ConfidenceMvNormal(
    [100.0, 80.0],
    [100.0 20.0; 20.0 64.0],
    0.9;
    support = :boxes,
)

# A law whose second component is in a stage after the first one, and its outcomes in
# `law_spanning_stages`.
function form_spanning_stages(law; stages = [2, 3])
    p = Problems.law_spanning_stages()
    info = Problems.ModelInfo(;
        uncertainties = [Problems.declare(law, p.model[:d]...; stages)],
        decision_stages = p.info.decision_stages,
    )
    return Problems.canonicalize(backend(p.model), info)
end

# The trait of each law: known for univariate, product, normal and confidence-ellipsoid normal
# laws, checked exactly for discrete laws, unknown for other multivariate laws across stages.
function test_linear_conditional_expectations()
    lce = LinearDecisionRules._linear_conditional_expectations
    uniform = Distributions.Uniform(0, 1)
    @test lce(uniform, [2]) === true
    @test lce(Distributions.product_distribution([uniform, uniform]), [2, 3]) === true
    @test lce(Distributions.ProductDistribution((uniform, uniform)), [2, 3]) === true
    @test lce(Distributions.MvNormal([0.0, 0.0], [1.0 0.5; 0.5 1.0]), [2, 3]) === true
    @test lce(ELLIPSOID, [2, 3]) === true
    # Correlated, but the conditional mean of `d[2]` is affine in `d[1]`: (460 + d[1]) / 7.
    @test lce(Problems.CORRELATED, [2, 3]) === true
    @test lce(SQUARE, [2, 3]) === false
    @test lce(SQUARE, [2, 2]) === true
    # Three stages: affine given the first stage, but not given the first two.
    atoms = [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    three = LinearDecisionRules.MvDiscreteNonParametric(atoms, fill(0.25, 4))
    @test lce(three, [1, 2, 2]) === true
    @test lce(three, [1, 2, 3]) === false
    # The triangle `d[1] + d[2] ≤ 1`.
    W, h = [-1.0 -1.0], [-1.0]
    conditioned = LinearDecisionRules.ConditionedDistribution([uniform, uniform], W, h)
    @test lce(conditioned, [2, 3]) === nothing
    @test lce(conditioned, [2, 2]) === true
    return
end

# The bound needs linear conditional expectations of later stages given earlier ones: a law
# spanning several stages without them, or not known to have them, gets a warning, and the bound is
# not certified.
function test_warns_when_a_law_spanning_stages_may_not_have_linear_conditional_expectations()
    form = form_spanning_stages(SQUARE)
    mom = moments(form)
    warning = (:warn, r"linear \(as for independent stages\), which fails for the law of d\[1\], d")
    result = @test_logs warning solve_formulation(form, DualLDR(), mom, HiGHS.Optimizer)
    @test result.termination_status == MOI.OPTIMAL
    @test !result.certified
    reason = "[not certified: conditional expectations across stages are not linear]"
    @test endswith(result.raw_status, reason)
    # The value of the primal rule is exact: only bounds need linear conditional expectations.
    primal = @test_logs solve_formulation(form, PrimalLDR(), mom, HiGHS.Optimizer)
    @test primal.certified && !occursin("not certified", primal.raw_status)
    # A conditioned law across stages is not known to have them.
    uniforms = [Distributions.Uniform(80, 120), Distributions.Uniform(60, 100)]
    conditioned = LinearDecisionRules.ConditionedDistribution(uniforms, [-1.0 1.0], [-30.0])
    form = form_spanning_stages(conditioned)
    warning = (:warn, r"which is not known for the law of d\[1\], d\[2\] \(stages 2, 3\)\.$")
    result = @test_logs warning solve_formulation(form, DualLDR(), moments(form), HiGHS.Optimizer)
    reason = "conditional expectations across stages are not known to be linear]"
    @test !result.certified && endswith(result.raw_status, reason)
    # Correlated laws with linear conditional expectations (the confidence-ellipsoid normal law, and
    # a discrete law whose conditional mean is affine), independent components in different
    # stages, or the uncorrelated but dependent law in one stage, are fine.
    for (law, stages) in (
        (ELLIPSOID, [2, 3]),
        (Problems.CORRELATED, [2, 3]),
        (Distributions.product_distribution([Distributions.Uniform(80, 120) for _ in 1:2]), [2, 3]),
        (SQUARE, [2, 2]),
    )
        form = form_spanning_stages(law; stages)
        result = @test_logs solve_formulation(form, DualLDR(), moments(form), HiGHS.Optimizer)
        @test result.termination_status == MOI.OPTIMAL
        @test result.certified
    end
    return
end

end # module TestDual

TestDual.runtests()
