# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The columns of ξ are sorted by stage, before and after lifting, when the uncertainties are
# declared out of stage order: a law over stages 2 to 5, then scalars at stages 4, 3 and 2, then an
# unmarked one. What is expected is computed from the declarations (see `stage_order_models.jl`).

module TestStageOrder

using JuMP
using Test

import HiGHS
import LinearDecisionRules
import LinearDecisionRules:
    AxisPiecewise,
    BreakPoints,
    CanonicalForm,
    ConditionedDistribution,
    LiftedDistribution,
    MvDiscreteNonParametric,
    canonicalize,
    lift,
    moments,
    second_moment,
    support_blocks
import MathOptInterface as MOI
import Random

include(joinpath(@__DIR__, "stage_order_models.jl"))

import .StageOrderModels:
    Staged,
    change_stage!,
    check_columns,
    joint_law_message,
    check_lifting,
    check_moments,
    check_support,
    delete_uncertainty!,
    draw_scenario,
    inventory,
    randomize!,
    refs,
    set_breakpoints!

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

# The canonical form of the model of `s`, as `optimize!` reads it. `z` and `spill` are unmarked in
# a model with five stages, hence the warning.
function form_of(s::Staged)
    src = backend(s.model)
    info = LinearDecisionRules.model_info(src)
    return @test_logs (:warn, r"without a stage") match_mode = :any canonicalize(src, info)
end

# The form of `s` without its breakpoints, and with them.
function base_and_lifted(s::Staged)
    for x in keys(s.breakpoints)
        set_attribute(x, BreakPoints(), nothing)
    end
    base = form_of(s)
    for (x, points) in s.breakpoints
        set_attribute(x, BreakPoints(), points)
    end
    return base, form_of(s)
end

function check_all(s::Staged, form::CanonicalForm; base = nothing)
    rng = Random.Xoshiro(1)
    scenarios = [draw_scenario(rng, s) for _ in 1:20]
    check_columns(s, form)
    check_moments(s, form, moments(form))
    check_support(s, form, scenarios)
    base === nothing || check_lifting(s, base, form, scenarios)
    return
end

function test_declared_out_of_stage_order()
    for law in (:walk, :product, :confidence)
        @testset "$law" begin
            s = inventory(; law, optimizer = HiGHS.Optimizer)
            form = form_of(s)
            w, u2, z = s.model[:w], s.model[:u2], s.model[:z]
            # Stable by stage: the order of the declarations within a stage is that of MOI's types.
            @test form.uncertainty_stages == [2, 2, 3, 3, 4, 4, 5, 5]
            @test Set(refs(s, form.uncertainties[1:2])) == Set([w[2], u2])
            @test Set(refs(s, form.uncertainties[7:8])) == Set([w[5], z])
            check_all(s, form)
        end
    end
    return
end

# Lifting the scalar in Stage(4) declared after the law over stages 2 to 5, then one in Stage(3),
# then both, through `BreakPoints` (as `optimize!` does) and through `lift`.
function test_lifting_one_then_the_other()
    s = inventory(; valid = true, optimizer = HiGHS.Optimizer)
    base = form_of(s)
    check_all(s, base)
    v, u3 = s.model[:v], s.model[:u3]
    lift_v = Dict(index(v) => AxisPiecewise([2.0, 3.5]))
    lift_u3 = Dict(index(u3) => AxisPiecewise([4 / 3, 8 / 3]))
    set_breakpoints!(s, v, [3.5, 2.0])
    form = form_of(s)
    @test form.uncertainty_stages == [2, 2, 3, 3, 4, 4, 4, 4, 5, 5]
    @test count(==(index(v)), form.uncertainties) == 3
    check_all(s, form; base)
    @test lift(base, lift_v).uncertainties == form.uncertainties
    set_breakpoints!(s, v, nothing)
    # Two equally spaced breakpoints in the support [0, 4] of `u3`.
    set_attribute(u3, BreakPoints(), 2)
    s.breakpoints[u3] = [4 / 3, 8 / 3]
    form = form_of(s)
    @test form.uncertainty_stages == [2, 2, 3, 3, 3, 3, 4, 4, 5, 5]
    check_all(s, form; base)
    @test lift(base, lift_u3).uncertainties == form.uncertainties
    set_breakpoints!(s, v, [2.0, 3.5])
    form = form_of(s)
    @test form.uncertainty_stages == [2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5]
    check_all(s, form; base)
    both = lift(base, merge(lift_v, lift_u3))
    @test both.uncertainties == form.uncertainties
    @test both.lifting.R == form.lifting.R
    @test second_moment(moments(both)) ≈ second_moment(moments(form))
    @test_throws ErrorException("The canonical form is already lifted.") lift(form, lift_v)
    return
end

# A component of a law over several stages cannot be lifted, by design.
function test_breakpoints_on_a_component()
    s = inventory(; optimizer = HiGHS.Optimizer)
    w = s.model[:w]
    base = form_of(s)
    msg = joint_law_message("w[3]", "w[2], w[3], w[4], w[5]")
    @test_throws ErrorException(msg) lift(base, Dict(index(w[3]) => AxisPiecewise([12.0])))
    set_attribute(w[3], BreakPoints(), [12.0])
    src = backend(s.model)
    info = LinearDecisionRules.model_info(src)
    @test_logs (:warn, r"without a stage") match_mode = :any begin
        @test_throws ErrorException(msg) canonicalize(src, info)
    end
    return
end

# Rows of Ξ that couple uncertainties of different stages: a valid row and a disc (stages 2 and 4),
# a truncation of the two scalars of stages 3 and 5, and one of the law over stages 2 to 5 with the
# scalar of stage 5. Each case also lifts `v` (stage 4), which is in the valid row and the disc.
function test_rows_across_stages()
    cases = [
        "valid" => (; valid = true),
        "conic" => (; conic = true),
        "truncate scalars, discrete" => (; truncate = :scalars, discrete = true),
        "truncate scalars, continuous" => (; truncate = :scalars),
        "truncate across, discrete" => (; truncate = :across, discrete = true),
        "truncate across, continuous" => (; truncate = :across),
    ]
    for (name, kwargs) in cases
        @testset "$name" begin
            s = inventory(; kwargs..., optimizer = HiGHS.Optimizer)
            base = form_of(s)
            check_all(s, base)
            u2, v, u3, z, w = (s.model[k] for k in (:u2, :v, :u3, :z, :w))
            column(x) = 1 + findfirst(==(index(x)), base.uncertainties)
            blocks = support_blocks(base)
            block_of(x) = only(b for (b, block) in enumerate(blocks) if column(x) in block.columns)
            if haskey(kwargs, :valid) || haskey(kwargs, :conic)
                @test block_of(u2) == block_of(v)
            end
            discrete = get(kwargs, :discrete, false)
            truncate = get(kwargs, :truncate, :none)
            if truncate == :scalars
                law = only(l for l in base.laws if column(u3) in l.columns)
                @test sort(law.columns) == sort([column(u3), column(z)])
                @test law.stages == base.uncertainty_stages[law.columns.-1]
                @test block_of(u3) == block_of(z)
                T = discrete ? MvDiscreteNonParametric : ConditionedDistribution
                @test law.distribution isa T
            elseif truncate == :across
                law = only(l for l in base.laws if column(z) in l.columns)
                @test sort(law.columns) == sort([column(z); [column(w[t]) for t in 2:5]])
                @test law.stages == base.uncertainty_stages[law.columns.-1]
                @test block_of(z) == block_of(w[2])
            end
            set_breakpoints!(s, v, [2.0, 3.5])
            form = form_of(s)
            check_all(s, form; base)
            lifted = only(l for l in form.laws if l.distribution isa LiftedDistribution)
            @test lifted.stages == [4, 4, 4]
        end
    end
    return
end

# `set_stage` moves columns: an uncertainty declared alone, a component of the law over several
# stages (whose columns are then not in the order of its components) and a decision.
function test_set_stage()
    s = inventory(; valid = true, optimizer = HiGHS.Optimizer)
    v, w, u3, buy = s.model[:v], s.model[:w], s.model[:u3], s.model[:buy]
    set_breakpoints!(s, u3, [1.0])
    base, form = base_and_lifted(s)
    check_all(s, form; base)
    change_stage!(s, v, 2)
    change_stage!(s, w[5], 3)
    change_stage!(s, buy[3], 4)
    base, form = base_and_lifted(s)
    @test form.uncertainty_stages == [2, 2, 2, 3, 3, 3, 3, 4, 5]
    law = only(l for l in base.laws if l.distribution isa MvDiscreteNonParametric)
    @test law.stages == [2, 3, 4, 3]
    @test !issorted(law.columns)
    check_all(s, form; base)
    return
end

# Deleting the scalar of Stage(3), declared between others; a component of a multivariate law
# cannot be deleted, and the model stays as it was.
function test_delete_an_uncertainty()
    s = inventory(; optimizer = HiGHS.Optimizer)
    u3, v, w = s.model[:u3], s.model[:v], s.model[:w]
    set_breakpoints!(s, v, [2.0, 3.5])
    delete_uncertainty!(s, u3)
    base, form = base_and_lifted(s)
    @test form.uncertainty_stages == [2, 2, 3, 4, 4, 4, 4, 5, 5]
    check_all(s, form; base)
    @test_throws MOI.DeleteNotAllowed delete(s.model, w[4])
    @test is_valid(s.model, w[4])
    base, form = base_and_lifted(s)
    check_all(s, form; base)
    return
end

# Random stages for every variable (the components of the law of `w` in any order) and random
# breakpoints, with rows of Ξ across stages.
function test_random_stages()
    rng = Random.Xoshiro(4)
    for k in 1:8
        @testset "$k" begin
            truncate = k % 4 == 0 ? :across : :none
            discrete = isodd(k ÷ 2)
            s = inventory(;
                valid = true,
                conic = isodd(k),
                truncate,
                discrete,
                optimizer = HiGHS.Optimizer,
            )
            randomize!(s, rng)
            base, form = base_and_lifted(s)
            check_all(s, form; base = form.lifting === nothing ? nothing : base)
        end
    end
    return
end

end # module TestStageOrder

TestStageOrder.runtests()
