# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Models with a `ConfidenceMvNormal` demand: the ellipsoid as a second-order cone (Clarabel) or
# its boxes as a polyhedron (HiGHS), against the previous version and the dense oracles.

module TestConicSupport

using JuMP
using Test

import Clarabel
import Distributions
import HiGHS
import LinearDecisionRules:
    ConfidenceMvNormal, FirstStage, LDRModel, Uncertainty, evaluate, simulate
import Random

include(joinpath(@__DIR__, "..", "oracles", "dense.jl"))

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

const μ = [100.0, 80.0]

# The optimizer for a representation of the support.
optimizer(support::Symbol) = support == :ellipsoid ? Clarabel.Optimizer : HiGHS.Optimizer

# Two products with their own stock (`shared = false`, the newsvendor of the previous version's
# tests) or a shared one, and a `ConfidenceMvNormal` demand at the 95% level.
function newsvendor(Σ; support = :auto, shared = false, solver = optimizer(support))
    model = LDRModel(solver)
    set_silent(model)
    law = ConfidenceMvNormal(μ, Σ, 0.95; support)
    @variable(model, d[1:2] in Uncertainty(law))
    @variable(model, sell[1:2] >= 0)
    if shared
        @variable(model, buy >= 0, FirstStage)
        @constraint(model, sum(sell) <= buy)
        @constraint(model, sell .<= d)
        @objective(model, Max, -10buy + 15sell[1] + 12sell[2])
    else
        @variable(model, buy[1:2] >= 0, FirstStage)
        @constraint(model, sell .<= buy)
        @constraint(model, sell .<= d)
        @objective(model, Max, sum(-10buy[i] + 15sell[i] for i in 1:2))
    end
    optimize!(model)
    return model
end

# The values of the previous version, whose support was the boxes, with HiGHS (`:auto` chooses the
# boxes, since HiGHS has no second-order cones).
function test_previous_version()
    for Σ in ([100.0 20.0; 20.0 64.0], [100.0 0.0; 0.0 64.0])
        model = newsvendor(Σ)
        @test termination_status(model) == MOI.OPTIMAL
        @test objective_value(model) ≈ 679.7027852387265 rtol = 1e-7
        @test objective_bound(model) ≈ 869.0287866864696 rtol = 1e-7
    end
    for (Σ, primal, dual) in (
        ([100.0 20.0; 20.0 64.0], 582.3518736706494, 650.5167012183156),
        ([100.0 -40.0; -40.0 64.0], 606.6974500277304, 654.123055605186),
    )
        model = newsvendor(Σ; shared = true)
        @test objective_value(model) ≈ primal rtol = 1e-7
        @test objective_bound(model) ≈ dual rtol = 1e-7
    end
    return
end

# The ellipsoid is inside the boxes: a better primal and a tighter dual, both between the bounds
# of the boxes, and each equal to its dense oracle.
function test_ellipsoid_and_boxes()
    for Σ in ([100.0 20.0; 20.0 64.0], [100.0 -40.0; -40.0 64.0]), shared in (false, true)
        values = Dict{Symbol,Tuple{Float64,Float64}}()
        for support in (:ellipsoid, :boxes)
            model = newsvendor(Σ; support, shared)
            @test termination_status(model) == MOI.OPTIMAL
            primal, dual = objective_value(model), objective_bound(model)
            values[support] = (primal, dual)
            @test primal <= dual + 1e-6 * abs(dual)
            ldr = unsafe_backend(model)
            form, mom = ldr.form, ldr.moments
            solver = optimizer(support)
            oracle = DenseOracle.solve(DenseOracle.build_primal, form, mom, solver)
            @test oracle.objective_value ≈ primal rtol = 1e-6
            oracle = DenseOracle.solve(DenseOracle.build_dual, form, mom, solver)
            @test oracle.objective_value ≈ dual rtol = 1e-6
        end
        (primal_e, dual_e), (primal_b, dual_b) = values[:ellipsoid], values[:boxes]
        @test primal_b <= primal_e + 1e-6 * abs(primal_e)
        @test dual_e <= dual_b + 1e-6 * abs(dual_b)
        if shared
            # The stock covers `d[1] + d[2]`, whose maximum is smaller on the ellipsoid.
            @test primal_e > primal_b + 1
        end
    end
    return
end

# The draws of `simulate` are in the ellipsoid: `y = d` records them. A scenario at a corner of
# the bounding box is outside the support of the law.
function test_simulate_in_ellipsoid()
    Σ = [100.0 40.0; 40.0 64.0]
    for support in (:ellipsoid, :boxes)
        model = LDRModel(optimizer(support))
        set_silent(model)
        law = ConfidenceMvNormal(μ, Σ, 0.95; support)
        @variable(model, d[1:2] in Uncertainty(law))
        @variable(model, buy >= 0, FirstStage)
        @variable(model, sell[1:2] >= 0)
        @variable(model, y[1:2])
        @constraint(model, y .== d)
        @constraint(model, sum(sell) <= buy)
        @constraint(model, sell .<= d)
        @objective(model, Max, -10buy + 15sell[1] + 12sell[2])
        optimize!(model)
        s = simulate(model, 2000; keep_decisions = true, rng = Random.Xoshiro(1))
        draws = [[s.decisions[y[1]][k], s.decisions[y[2]][k]] for k in 1:2000]
        @test all(x -> Distributions.insupport(law, x), draws)
        @test s.violation_fraction == 0
        corner = maximum(law)
        msg = try
            evaluate(model, d => corner)
            ""
        catch err
            sprint(showerror, err)
        end
        @test occursin("is outside the support of its law", msg)
        @test !evaluate(model, d => corner; outside_support = :ignore).in_support
    end
    return
end

end # module TestConicSupport

TestConicSupport.runtests()
