# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestRescaling

using JuMP
using Test

import Clarabel
import Distributions
import Distributions: Uniform
import HiGHS
import LinearDecisionRules as LDR
import Random

include(joinpath(@__DIR__, "..", "problems", "tagged.jl"))
include(joinpath(@__DIR__, "..", "problems", "production.jl"))

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

# The value of result `result` and the bound (if `bound`) of `build()`, solved with and without
# `RescaleUncertainties()`, agree.
function _same_results(build; rtol = 1e-7, result = 1, bound = true)
    results = map((true, false)) do rescale
        model = build()
        set_attribute(model, LDR.RescaleUncertainties(), rescale)
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        return model, objective_value(model; result), bound ? objective_bound(model) : NaN
    end
    (scaled, value, b), (plain, value0, b0) = results
    @test value ≈ value0 rtol = rtol
    bound && @test b ≈ b0 rtol = rtol
    return scaled, plain
end

function test_default_and_setting()
    model = TaggedProblems.ldr_model()
    @test get_attribute(model, LDR.RescaleUncertainties()) === true
    set_attribute(model, LDR.RescaleUncertainties(), false)
    @test get_attribute(model, LDR.RescaleUncertainties()) === false
    return
end

# The rules are reported in the uncertainties of the model: those of the newsvendor are unique.
function test_newsvendor_rules()
    scaled, plain = _same_results(TaggedProblems.newsvendor)
    @test LDR.rule_matrix(scaled).matrix ≈ LDR.rule_matrix(plain).matrix atol = 1e-9
    @test LDR.decision_rule(scaled[:sell])(scaled[:demand] => 95.0) ≈ 95
    return
end

# `SampledLDR` draws the same outcomes, moved to the rescaled uncertainties.
function test_sampled()
    function build()
        return TaggedProblems.newsvendor(
            TaggedProblems.ldr_model(; formulations = [LDR.SampledLDR(20), LDR.DualLDR()]),
        )
    end
    scaled, plain = _same_results(build; result = 2)
    @test LDR.rule_matrix(scaled; formulation = LDR.SampledLDR(20)).matrix ≈
          LDR.rule_matrix(plain; formulation = LDR.SampledLDR(20)).matrix atol = 1e-7
    return
end

# Breakpoints: the lifted coordinates are rescaled, one by one. The combinations span stages, which
# `DualLDR` refuses.
function test_breakpoints()
    _same_results(() -> TaggedProblems.inventory(4; points = 2))
    function combinations()
        model = TaggedProblems.ldr_model(; formulations = [LDR.PrimalLDR()])
        return TaggedProblems.inventory(4, model; combinations = (:pair, 1))
    end
    _same_results(combinations; bound = false)
    return
end

# The internal choices of the rescaling of the pieces (the study of `benchmark/scaling/`) give the
# same values, and their centers and scales on the pieces of one demand.
function test_lifted_rescaling()
    build() = TaggedProblems.inventory(3; points = 2)
    model = build()
    form = LDR.canonicalize(backend(model), LDR.model_info(backend(model)))
    mom = LDR.moments(form)
    law = first(l for l in form.laws if l.distribution isa LDR.LiftedDistribution)
    η = law.columns .- 1
    lo, hi = LDR._support_box(law.distribution)
    s = (hi[1] - lo[1] + sum(hi[2:end])) / 2
    values = map((:per_piece, :before_lifting, :hybrid)) do mode
        LDR._LIFTED_RESCALING[] = mode
        try
            r = LDR._rescaling(form, mom)
            if mode == :per_piece
                @test r.center[η] ≈ (lo + hi) / 2
                @test r.scale[η] ≈ (hi - lo) / 2
            elseif mode == :before_lifting
                @test r.center[η] ≈ [lo[1] + s; zeros(length(η) - 1)]
                @test r.scale[η] ≈ fill(s, length(η))
            else
                @test r.center[η] ≈ [(lo[1] + hi[1]) / 2; zeros(length(η) - 1)]
                @test r.scale[η] ≈ [(hi[1] - lo[1]) / 2; hi[2:end]]
            end
            m = build()
            optimize!(m)
            @test termination_status(m) == MOI.OPTIMAL
            return objective_value(m), objective_bound(m)
        finally
            LDR._LIFTED_RESCALING[] = :per_piece
        end
    end
    @test all(v -> all(v .≈ values[1]), values)
    @test_throws ArgumentError LDR._rescale_pieces!(zeros(2), ones(2), [1, 2], lo, hi, :other)
    return
end

# A random price multiplies a decision (the matrix `C`), and the objective has the terms `Q` and
# `d` in the uncertainties; a `ValidInequality` row couples the demands (`W_valid`).
function test_objective_and_valid_rows()
    function build()
        model = TaggedProblems.ldr_model()
        @variable(model, buy[1:2] >= 0, LDR.FirstStage)
        @variable(model, sell[1:2] >= 0)
        @variable(model, demand[1:2] in LDR.Uncertainty(Uniform(80, 120)))
        @variable(model, price in LDR.Uncertainty(Uniform(13, 17)))
        @constraint(model, demand[1] + demand[2] <= 220, LDR.ValidInequality())
        @constraint(model, [i = 1:2], sell[i] <= buy[i])
        @constraint(model, [i = 1:2], sell[i] <= demand[i])
        @objective(
            model,
            Max,
            price * sell[1] + 15 * sell[2] - 10 * sum(buy) + 0.01 * demand[1]^2 + 2 * demand[2]
        )
        return model
    end
    _same_results(build)
    return
end

# Discrete laws: the box of the atoms.
function test_discrete()
    _same_results(TaggedProblems.newsvendor_mv_discrete)
    return
end

# Second-order cones: the support rows of a law (`support_constraints`) and a valid row.
function test_cones()
    function build()
        model = LDR.LDRModel(Clarabel.Optimizer)
        set_silent(model)
        law = LDR.ConfidenceMvNormal([100.0, 80.0], [100.0 20.0; 20.0 64.0], 0.95)
        @variable(model, buy[1:2] >= 0, LDR.FirstStage)
        @variable(model, sell[1:2] >= 0)
        @variable(model, demand[1:2] in LDR.Uncertainty(law))
        @variable(model, price in LDR.Uncertainty(Uniform(13, 17)))
        @constraint(
            model,
            [3.0; price - 15; demand[1] - 100] in SecondOrderCone(),
            LDR.ValidInequality()
        )
        @constraint(model, [i = 1:2], sell[i] <= buy[i])
        @constraint(model, [i = 1:2], sell[i] <= demand[i])
        @objective(model, Max, price * sell[1] + 15 * sell[2] - 10 * sum(buy))
        return model
    end
    _same_results(build; rtol = 1e-6)
    return
end

# A parameter alone in a row is a fixed variable of the inner problems, which a new value of the
# parameter only fixes again; the rescaled constants keep that.
function test_parameters_resolve()
    model = TaggedProblems.inventory(3; parameter = true, points = 1)
    optimize!(model)
    builds = unsafe_backend(model).inner_cache.num_builds
    set_parameter_value(model[:stock], 25.0)
    optimize!(model)
    @test unsafe_backend(model).inner_cache.num_builds == builds
    fresh = TaggedProblems.inventory(3; x0 = 25.0, points = 1)
    set_attribute(fresh, LDR.RescaleUncertainties(), false)
    optimize!(fresh)
    @test objective_value(model) ≈ objective_value(fresh) rtol = 1e-7
    @test objective_bound(model) ≈ objective_bound(fresh) rtol = 1e-7
    return
end

# Demands of order 10⁴: with a breakpoint, HiGHS's dual simplex may fail on the problem in the
# demands (in some runs, depending on the order of the rows), and solves the rescaled one.
function test_production()
    model = production_model(3; formulations = [LDR.PrimalLDR(), LDR.DualLDR()], breakpoints = 1)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ 154603.707679 rtol = 1e-7
    @test objective_bound(model) ≈ 157113.201756 rtol = 1e-7
    return
end

# The rows of Ξ are divided by their largest coefficient, and a right-hand side that is zero but for
# rounding is zero: the hull of an `AxisPiecewise` lifting of a demand of order 10⁴, whose rows have
# coefficients of order 10⁶ in ζ before. The vertices of the lifted support stay in it.
function test_rescaled_rows()
    lo, hi = 0.8 * 27_345.678, 1.2 * 27_345.678
    points = lo .+ (hi - lo) * [0.25, 0.5, 0.75]
    l = LDR.AxisPiecewise(points)
    W, h = LDR._hull_rows(l, [lo], [hi])
    box_lo, box_hi = LDR._lifted_bounds(l, [lo], [hi])
    center, scale = (box_lo + box_hi) / 2, (box_hi - box_lo) / 2
    Ws, hs = LDR._rescale_rows(W, h, center, scale)
    @test vec(maximum(abs, Ws; dims = 2)) ≈ ones(3)
    @test hs == zeros(3)
    for v in [lo; points; hi]
        ζ = (LDR.lift(l, [v]) - center) ./ scale
        @test minimum(Ws * ζ - hs) >= -1e-12
    end
    return
end

# The centers and scales; `T` and `T⁻¹` give the same decisions.
function test_rescaling_map()
    model = TaggedProblems.newsvendor_random_price()
    form = LDR.canonicalize(backend(model), LDR.model_info(backend(model)))
    mom = LDR.moments(form)
    r = LDR._rescaling(form, mom)
    @test sort(r.center) == [15.0, 100.0]
    @test sort(r.scale) == [2.0, 20.0]
    scaled = LDR._rescale(mom, r)
    @test scaled.mean[2:end] ≈ zeros(2) atol = 1e-12
    @test all(b -> b.covariance ≈ fill(1 / 3, 1, 1), scaled.blocks)
    T = LDR._rescaling_matrix(r)
    X̃ = randn(Random.Xoshiro(1), 3, 3)
    result = LDR.FormulationResult(
        LDR.PrimalLDR(),
        MOI.OPTIMAL,
        MOI.FEASIBLE_POINT,
        "",
        0.0,
        0.0,
        true,
        0.0,
        0.0,
        0.0,
        X̃,
        Model(),
    )
    X = LDR._unscale(result, r).X
    @test X * T ≈ X̃
    return
end

end # module TestRescaling

TestRescaling.runtests()
