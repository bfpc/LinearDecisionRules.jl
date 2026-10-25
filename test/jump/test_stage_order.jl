# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# End to end, with uncertainties declared out of stage order (see
# `test/canonical/stage_order_models.jl`): the rules of every formulation have exact zeros on the
# columns of later stages, `decision_rule`, `evaluate` and the rule matrix agree on scenarios, and
# the values match the dense oracles on the (lifted) forms, through breakpoints, rows of Ξ across
# stages, deletions and changes of stage between solves.

module TestStageOrderEndToEnd

using JuMP
using Test

import Clarabel
import HiGHS
import LinearDecisionRules
import LinearDecisionRules:
    BreakPoints,
    DualLDR,
    Formulations,
    InnerOptimizer,
    PrimalLDR,
    SampledLDR,
    decision_rule,
    evaluate,
    expected_value,
    rule_coefficient,
    rule_matrix,
    simulate
import MathOptInterface as MOI
import Random

include(joinpath(@__DIR__, "..", "oracles", "dense.jl"))
include(joinpath(@__DIR__, "..", "canonical", "stage_order_models.jl"))

import .StageOrderModels:
    Staged,
    by_hand,
    change_stage!,
    check_columns,
    check_moments,
    delete_uncertainty!,
    draw_scenario,
    inventory,
    randomize!,
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

const FORMULATIONS = [PrimalLDR(), DualLDR(), SampledLDR(30)]

function staged(; optimizer = HiGHS.Optimizer, kwargs...)
    s = inventory(; optimizer, kwargs...)
    set_attribute(s.model, Formulations(), FORMULATIONS)
    return s
end

# `z` and `spill` are unmarked in a model with five stages, hence the warning; laws over several
# stages may also warn that the bound is not certified.
function solve!(s::Staged)
    @test_logs (:warn, r"without a stage") match_mode = :any optimize!(s.model)
    for f in FORMULATIONS
        @test termination_status(s.model, f) == MOI.OPTIMAL
    end
    return
end

# The rule matrix of every formulation: its columns are those of the canonical form, and it is
# exactly zero on the columns of stages later than the decision of the row; `rule_coefficient`
# agrees.
function check_rule_zeros(s::Staged; nontrivial::Bool = true)
    model = s.model
    form = unsafe_backend(model).form
    variable(label) = label isa Tuple ? label[1] : label
    uncertainties = [x for law in s.laws for x in law.variables]
    for f in FORMULATIONS
        R = rule_matrix(model; formulation = f)
        @test index.(R.decisions) == form.decisions
        @test R.columns[1] == :constant
        @test [index(variable(c)) for c in R.columns[2:end]] == form.uncertainties
        later = [
            c > 1 && s.stages[variable(R.columns[c])] > s.stages[x] for x in R.decisions,
            c in eachindex(R.columns)
        ]
        @test all(iszero, R.matrix[later])
        # The rules are not trivial: some decision uses an uncertainty that it observes.
        nontrivial && @test any(!iszero, R.matrix[:, 2:end])
        pairs = [(x, η) for x in R.decisions for η in uncertainties if s.stages[η] > s.stages[x]]
        @test all(all(iszero, rule_coefficient(x, η; formulation = f)) for (x, η) in pairs)
    end
    return
end

# On scenarios of the support: `decision_rule(x)(scenario)`, `evaluate(model, scenario)` and the
# rule matrix times the lifted outcome computed by hand agree, and the primal rule is feasible.
function check_rules_on_scenarios(s::Staged; num_scenarios::Int = 5)
    model = s.model
    rng = Random.Xoshiro(2)
    for _ in 1:num_scenarios
        scenario = draw_scenario(rng, s)
        for f in FORMULATIONS
            R = rule_matrix(model; formulation = f)
            x = R.matrix * by_hand(s, R.columns, scenario)
            rules = [decision_rule(d; formulation = f)(scenario) for d in R.decisions]
            @test rules ≈ x
            f isa DualLDR && continue
            e = evaluate(model, scenario; formulation = f)
            @test [e[d] for d in R.decisions] ≈ x
            f isa PrimalLDR && @test isempty(e.violations)
        end
    end
    return
end

# The values of the primal and dual rules are those of the dense oracles on the canonical form of
# the solve, which is lifted when there are breakpoints. The objective value of the primal rule,
# computed with the moments, is its expected objective: `expected_value` of the objective, and the
# mean of a simulation, drawn from the original laws.
function check_values(s::Staged, optimizer)
    model = s.model
    o = unsafe_backend(model)
    primal = DenseOracle.solve(DenseOracle.build_primal, o.form, o.moments, optimizer)
    dual = DenseOracle.solve(DenseOracle.build_dual, o.form, o.moments, optimizer)
    value = objective_value(model, PrimalLDR())
    @test isapprox(value, primal.objective_value; rtol = 1e-6)
    @test isapprox(objective_bound(model), dual.objective_value; rtol = 1e-6)
    @test value <= objective_bound(model) + 1e-6
    @test isapprox(expected_value(objective_function(model)), value; rtol = 1e-6)
    simulation = simulate(model, 20_000; rng = Random.Xoshiro(3))
    @test simulation.violation_fraction == 0
    @test abs(simulation.objective_mean - value) <= 5 * simulation.objective_stderr
    return
end

function check_solve(s::Staged; nontrivial::Bool = true)
    solve!(s)
    o = unsafe_backend(s.model)
    check_columns(s, o.form)
    check_moments(s, o.form, o.moments)
    check_rule_zeros(s; nontrivial)
    check_rules_on_scenarios(s)
    check_values(s, get_attribute(s.model, InnerOptimizer()))
    return
end

function test_declared_out_of_stage_order()
    for law in (:walk, :product, :confidence)
        @testset "$law" begin
            check_solve(staged(; law, valid = true))
        end
    end
    return
end

# Breakpoints on the scalar of Stage(4), declared after the law over stages 2 to 5, then on the one
# of Stage(3) instead, then on both, solving each time.
function test_breakpoints_one_then_the_other()
    s = staged(; valid = true)
    v, u3, w = s.model[:v], s.model[:u3], s.model[:w]
    check_solve(s)
    set_breakpoints!(s, v, [3.5, 2.0])
    check_solve(s)
    set_breakpoints!(s, v, nothing)
    set_breakpoints!(s, u3, [1.0, 2.5])
    check_solve(s)
    set_breakpoints!(s, v, [2.0, 3.5])
    check_solve(s)
    pieces = filter(c -> c isa Tuple, rule_matrix(s.model).columns)
    @test pieces == [(u3, 1), (u3, 2), (u3, 3), (v, 1), (v, 2), (v, 3)]
    # A component of a law over several stages cannot be lifted, by design.
    set_attribute(w[4], BreakPoints(), [12.0])
    msg =
        "The uncertainty w[4] has a lifting, but it is a component of the law of w[2], w[3], " *
        "w[4], w[5]. Liftings apply to uncertainties declared alone, with a univariate law."
    @test_logs (:warn, r"without a stage") match_mode = :any begin
        @test_throws ErrorException(msg) optimize!(s.model)
    end
    set_attribute(w[4], BreakPoints(), nothing)
    check_solve(s)
    return
end

# Rows of Ξ across stages, with `v` (Stage(4)) lifted: a disc with `u2` (Stage(2)), and the
# truncation of the law over stages 2 to 5 by `z` (Stage(5)), exact for discrete laws.
function test_rows_across_stages()
    @testset "conic" begin
        s = staged(; conic = true, valid = true, optimizer = Clarabel.Optimizer)
        set_breakpoints!(s, s.model[:v], [2.0, 3.5])
        check_solve(s)
    end
    @testset "truncate across" begin
        s = staged(;
            truncate = :across,
            discrete = true,
            quadratic = true,
            optimizer = Clarabel.Optimizer,
        )
        set_breakpoints!(s, s.model[:v], [2.0, 3.5])
        check_solve(s)
    end
    # A law conditioned by a row across stages cannot be lifted: it is no longer declared alone.
    @testset "truncate scalars" begin
        s = staged(; truncate = :scalars, discrete = true)
        set_breakpoints!(s, s.model[:v], [2.0, 3.5])
        check_solve(s)
        set_attribute(s.model[:u3], BreakPoints(), [2.0])
        msg =
            "The uncertainty u3 has a lifting, but it is a component of the law of u3, z. " *
            "Liftings apply to uncertainties declared alone, with a univariate law."
        @test_logs (:warn, r"without a stage") match_mode = :any begin
            @test_throws ErrorException(msg) optimize!(s.model)
        end
    end
    return
end

# Deleting the uncertainty of Stage(3), declared between others, after a solve.
function test_delete_and_solve_again()
    s = staged(; quadratic = true, optimizer = Clarabel.Optimizer)
    set_breakpoints!(s, s.model[:v], [2.0, 3.5])
    check_solve(s)
    delete_uncertainty!(s, s.model[:u3])
    check_solve(s)
    @test unsafe_backend(s.model).form.uncertainty_stages == [2, 2, 3, 4, 4, 4, 4, 5, 5]
    return
end

# Changing stages after a solve: the scalar of Stage(4) to Stage(2), a component of the law over
# stages 2 to 5 to Stage(3) (so that its columns are not in the order of its components), and a
# decision; then every variable of Stage(5) to Stage(4), so that the last stage, and with it the
# stage of the unmarked variables, is 4.
function test_set_stage_and_solve_again()
    s = staged(; valid = true, quadratic = true, optimizer = Clarabel.Optimizer)
    v, w, buy, stock, sell = (s.model[k] for k in (:v, :w, :buy, :stock, :sell))
    set_breakpoints!(s, v, [2.0, 3.5])
    check_solve(s)
    change_stage!(s, v, 2)
    change_stage!(s, w[5], 3)
    change_stage!(s, buy[3], 4)
    check_solve(s)
    @test unsafe_backend(s.model).form.uncertainty_stages == [2, 2, 2, 2, 2, 3, 3, 3, 4, 5]
    for x in (buy[5], stock[5], sell[5])
        change_stage!(s, x, 4)
    end
    s.stages[s.model[:z]] = s.stages[s.model[:spill]] = 4
    check_solve(s)
    @test unsafe_backend(s.model).form.uncertainty_stages == [2, 2, 2, 2, 2, 3, 3, 3, 4, 4]
    return
end

# Random stages for every variable (the components of the law of `w` in any order) and random
# breakpoints, end to end.
function test_random_stages()
    rng = Random.Xoshiro(5)
    for k in 1:4
        @testset "$k" begin
            optimizer = k == 3 ? Clarabel.Optimizer : HiGHS.Optimizer
            truncate = k == 4 ? :across : :none
            s = staged(;
                valid = true,
                discrete = iseven(k),
                truncate,
                quadratic = k == 3,
                optimizer,
            )
            randomize!(s, rng)
            check_solve(s; nontrivial = false)
        end
    end
    return
end

end # module TestStageOrderEndToEnd

TestStageOrderEndToEnd.runtests()
