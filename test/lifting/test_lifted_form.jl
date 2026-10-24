# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestLiftedForm

using JuMP
using Test

import Distributions
import HiGHS
import LinearDecisionRules
import LinearDecisionRules:
    AxisPiecewise,
    DualLDR,
    InformationStructure,
    LiftedDistribution,
    LiftingMap,
    ModelInfo,
    PrimalLDR,
    canonicalize,
    lift,
    moments,
    retraction,
    second_moment,
    solve_formulation,
    support_blocks

include(joinpath(@__DIR__, "..", "formulations", "problems.jl"))

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

function error_message(f)
    try
        f()
    catch err
        return sprint(showerror, err)
    end
    return ""
end

# The newsvendor with breakpoints [90, 100, 110] on the demand.
function test_newsvendor_structure()
    p = Problems.newsvendor()
    form = Problems.form(p)
    lifted = Problems.form(Problems.newsvendor_breakpoints(2))
    pieces = ["demand[piece $k]" for k in 1:4]
    @test lifted.uncertainty_names == pieces
    @test lifted.uncertainty_stages == fill(2, 4)
    @test all(==(form.uncertainties[1]), lifted.uncertainties)
    @test lifted.decisions == form.decisions
    law = only(lifted.laws)
    @test law.distribution isa LiftedDistribution
    @test law.columns == 2:5 && law.stages == fill(2, 4)
    # η = Σ pieces, so `sell <= demand` has the same coefficient on each piece.
    @test lifted.B_u == form.B_u * [1 0 0 0 0; 0 1 1 1 1]
    @test lifted.C[:, 1] == form.C[:, 1] && iszero(lifted.C[:, 2:end])
    @test lifted.f == form.f && lifted.d == zeros(4) && iszero(lifted.Q)
    # The hull rows, with the support box of the lifted law.
    @test lifted.W_valid == [10 -10 0 0; 0 10 -10 0; 0 0 10 -10]
    @test lifted.h_valid == [800, 0, 0]
    @test lifted.valid_constraints == [nothing, nothing, nothing]
    block = only(support_blocks(lifted))
    @test block.columns == 2:5
    @test block.roles == [fill(:box, 8); fill(:hull, 3)]
    # The map back to the original uncertainty.
    map = lifted.lifting
    @test map isa LiftingMap
    @test form.lifting === nothing
    @test map.uncertainties == form.uncertainties
    @test map.uncertainty_names == ["demand"] && map.uncertainty_stages == [2]
    @test map.kept == [1, 0]
    @test only(map.blocks).inputs == [2] && only(map.blocks).columns == 2:5
    @test retraction(map) == [1 0 0 0 0; 0 1 1 1 1]
    @test lift(map, [1.0, 104.0]) == [1, 90, 10, 4, 0]
    @test lift(map, [1.0 1.0; 85.0 120.0]) == [1 1; 85 90; 0 10; 0 10; 0 10]
    @test_throws DimensionMismatch lift(map, [104.0])
    return
end

# `R L(ξ) = ξ`, and the lifted moments retract to the original ones: `R M̃ Rᵀ = M`.
function test_retraction()
    for p in Problems.library()
        lifted = Problems.form(p)
        lifted.lifting === nothing && continue
        @testset "$(p.name)" begin
            form = canonicalize(backend(p.model), ModelInfo(; p.info.uncertainties,
                p.info.decision_stages, p.info.valid_rows))
            R = retraction(lifted.lifting)
            @test size(R) == (1 + length(form.uncertainties), 1 + length(lifted.uncertainties))
            M, M̃ = second_moment(moments(form)), second_moment(moments(lifted))
            @test R * M̃ * R' ≈ M
            ξ = M[:, 1]
            @test R * lift(lifted.lifting, ξ) ≈ ξ
            # The objective constant `E[η'Qη + d'η + f]` does not change.
            @test LinearDecisionRules._objective_constant(lifted, moments(lifted)) ≈
                  LinearDecisionRules._objective_constant(form, moments(form))
        end
    end
    return
end

# Breakpoints on the stage-2 demand of three stages: the pieces stay in stage 2, the columns are
# sorted by stage, and a stage-2 decision does not observe the stage-3 demand.
function test_multistage_columns()
    lifted = Problems.form(Problems.newsvendor_three_stages_breakpoints())
    @test lifted.uncertainty_names == [
        "demand2[piece 1]",
        "demand2[piece 2]",
        "demand2[piece 3]",
        "demand2[piece 4]",
        "buy_cost2",
        "demand3",
    ]
    @test lifted.uncertainty_stages == [2, 2, 2, 2, 2, 3]
    @test [law.columns for law in lifted.laws] == [2:5, [6], [7]]
    @test lifted.lifting.kept == [1, 0, 6, 7]
    info = InformationStructure(lifted)
    stage(name) = lifted.decision_stages[findfirst(==(name), lifted.decision_names)]
    visible(name) = info.num_visible[findfirst(==(name), lifted.decision_names)]
    @test visible("buy") == 1
    @test visible("sell2") == 6 && stage("sell2") == 2
    @test visible("sell3") == 7
    # The bound of the dual is certified: no law spans two stages.
    @test_logs solve_formulation(lifted, DualLDR(), moments(lifted), HiGHS.Optimizer)
    return
end

# A lifting with several pieces in a valid row: `W ← WR` on the lifted columns.
function test_valid_row()
    p = only(filter(p -> p.name == "valid_row_breakpoints", Problems.edge_cases()))
    form = Problems.form(p)
    @test form.uncertainty_names[3:end] == ["d[3][piece 1]", "d[3][piece 2]", "d[3][piece 3]"]
    # `d[1] + d[3] <= 210` on the pieces, then the two hull rows of `d[3]`.
    @test form.W_valid[1, :] == [-1, 0, -1, -1, -1]
    @test form.valid_constraints[1] !== nothing
    @test form.valid_constraints[2:3] == [nothing, nothing]
    blocks = support_blocks(form)
    # `d[2]` is coupled to no other coordinate by a row.
    @test [block.columns for block in blocks] == [[2, 4, 5, 6], [3]]
    @test blocks[1].roles == [fill(:box, 8); :valid; :hull; :hull]
    return
end

# The number of breakpoints gives equally spaced breakpoints inside the support; no breakpoint
# means no lifting.
function test_breakpoint_count()
    p = Problems.newsvendor()
    demand = index(variable_by_name(p.model, "demand"))
    function lifted(points)
        info = ModelInfo(; p.info.uncertainties, p.info.decision_stages,
            breakpoints = Dict(demand => points))
        return canonicalize(backend(p.model), info)
    end
    @test only(lifted(3).lifting.blocks).lifting == AxisPiecewise([90, 100, 110])
    @test only(lifted([110.0, 90, 100]).lifting.blocks).lifting == AxisPiecewise([90, 100, 110])
    @test lifted(0).lifting === nothing
    @test lifted(Float64[]).lifting === nothing
    @test occursin("must be nonnegative", error_message(() -> lifted(-1)))
    return
end

function test_errors()
    p = Problems.newsvendor()
    demand = index(variable_by_name(p.model, "demand"))
    form = Problems.form(p)
    l = AxisPiecewise([100])
    lifted = lift(form, Dict(demand => l))
    @test occursin("already lifted", error_message(() -> lift(lifted, Dict(demand => l))))
    @test occursin(
        "index 1 has a lifting, but it is not an uncertainty",
        error_message(() -> lift(form, Dict(form.decisions[1] => l))),
    )
    for points in ([80.0], [100.0, 130.0])
        msg = error_message(() -> lift(form, Dict(demand => AxisPiecewise(points))))
        @test occursin("The lifting of the uncertainty demand is invalid", msg)
        @test occursin("strictly inside the support [80.0, 120.0]", msg)
    end
    info(points) = ModelInfo(; p.info.uncertainties, p.info.decision_stages,
        breakpoints = Dict(demand => points))
    msg = error_message(() -> canonicalize(backend(p.model), info([100.0, 100.0])))
    @test occursin("The lifting of the uncertainty demand is invalid", msg)
    @test occursin("must be distinct", msg)
    # A component of a multivariate law.
    two = Problems.newsvendor_two_products()
    d1 = index(variable_by_name(two.model, "demand[1]"))
    msg = error_message(() -> lift(Problems.form(two), Dict(d1 => l)))
    @test occursin("demand[1] has a lifting, but it is a component of the law of", msg)
    # An unbounded law.
    unbounded = Problems.unbounded_support()
    η = index(variable_by_name(unbounded.model, "η"))
    msg = error_message(() -> lift(Problems.form(unbounded), Dict(η => AxisPiecewise([0]))))
    @test occursin("bounded support", msg)
    unbounded_info = ModelInfo(; unbounded.info.uncertainties, breakpoints = Dict(η => 2))
    msg = error_message(() -> canonicalize(backend(unbounded.model), unbounded_info))
    @test occursin("η has 2 breakpoints, but its support is unbounded", msg)
    return
end

# The primal values increase and the dual values decrease over nested breakpoints.
function test_nested_breakpoints_monotone()
    primal, dual = Float64[], Float64[]
    for p in [Problems.newsvendor(); Problems.newsvendor_breakpoints.(1:4)]
        form = Problems.form(p)
        mom = moments(form)
        push!(primal, solve_formulation(form, PrimalLDR(), mom, HiGHS.Optimizer).objective_value)
        push!(dual, solve_formulation(form, DualLDR(), mom, HiGHS.Optimizer).objective_value)
    end
    @test all(diff(primal) .>= -1e-7)
    @test all(diff(dual) .<= 1e-7)
    @test all(primal .<= dual .+ 1e-7)
    return
end

end # module TestLiftedForm

TestLiftedForm.runtests()
