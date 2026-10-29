# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Re-solves with JuMP parameters: the optimizer keeps its inner problems and updates their
# parameters when nothing else changed. The cache must never change a result: every re-solve is
# compared with a fresh model, and every other change must build the inner problems again.

module TestParameters

using JuMP
using Test

import Distributions: Uniform
import HiGHS
import LinearDecisionRules
import LinearDecisionRules:
    DualLDR, FirstStage, Formulations, LDRModel, PrimalLDR, SampledLDR, Uncertainty, ValidInequality

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

const FORMULATIONS = [PrimalLDR(), DualLDR(), SampledLDR(20; seed = 3)]

include(joinpath(@__DIR__, "..", "problems", "tagged.jl"))

ldr_model() = TaggedProblems.ldr_model(; formulations = FORMULATIONS)

num_builds(model) = unsafe_backend(model).inner_cache.num_builds

results(model) = unsafe_backend(model).results

# The newsvendor with a capacity, a parameter alone in a row, and a fixed income `2 capacity` in
# the objective; with `parameter = false`, the capacity is a number.
function newsvendor(capacity; parameter = true)
    model = ldr_model()
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @variable(model, demand in Uncertainty(Uniform(80, 120)))
    c = parameter ? @variable(model, capacity in Parameter(capacity)) : capacity
    @constraint(model, limit, buy <= c)
    @constraint(model, sell + ret <= buy)
    @constraint(model, sell <= demand)
    @objective(model, Max, -10buy + 8ret + 15sell + 2c)
    return model
end

# The inventory with backlog of `benchmark/timing.jl`, with an initial stock.
function inventory(T, x0; parameter = true)
    return TaggedProblems.inventory(T, ldr_model(); x0, parameter)
end

# The same results, formulation by formulation: status, objective, bound and rules.
function check_same_results(model, fresh)
    @test length(results(model)) == length(results(fresh))
    for (a, b) in zip(results(model), results(fresh))
        @test a.formulation == b.formulation
        @test a.termination_status == b.termination_status
        @test a.primal_status == b.primal_status
        scale = max(1.0, abs(b.objective_value))
        @test isapprox(a.objective_value, b.objective_value; atol = 1e-6 * scale)
        @test isapprox(a.objective_bound, b.objective_bound; atol = 1e-6 * scale)
        @test isapprox(a.X, b.X; atol = 1e-6 * max(1.0, maximum(abs, b.X)))
        @test a.certified == b.certified
    end
    @test isapprox(objective_value(model), objective_value(fresh); atol = 1e-6)
    @test isapprox(objective_bound(model), objective_bound(fresh); atol = 1e-6)
    return
end

# After each change of the parameter, the cached re-solve equals a fresh model with the same
# value, and a model where the value is a number.
function check_resolves(build, parameter, values)
    model = build(first(values))
    optimize!(model)
    @test num_builds(model) == length(FORMULATIONS)
    for value in values
        set_parameter_value(model[parameter], value)
        optimize!(model)
        @test num_builds(model) == length(FORMULATIONS)
        fresh = build(value)
        optimize!(fresh)
        check_same_results(model, fresh)
        constant = build(value; parameter = false)
        optimize!(constant)
        check_same_results(model, constant)
    end
    return model
end

function test_newsvendor_capacity()
    model = check_resolves(newsvendor, :capacity, [150.0, 90.0, 105.0, 0.0, 200.0, 90.0])
    # The capacity binds at 90: 460 - 45 for the lost sales, plus the income 2 * 90.
    @test objective_value(model) ≈ 595
    return
end

function test_inventory_initial_stock()
    model = check_resolves((x0; kw...) -> inventory(4, x0; kw...), :stock, [0.0, 50.0, -20.0, 0.0])
    set_parameter_value(model[:stock], 0.0)
    optimize!(model)
    @test objective_value(model) ≈ 90 * 4 + 20
    @test objective_bound(model) ≈ 340
    return
end

# A new value is set in place: HiGHS keeps its copy of the inner problem and starts from its basis,
# so it needs fewer iterations than a fresh solve.
function test_warm_start()
    model = inventory(8, 0.0)
    set_attribute(model, Formulations(), [PrimalLDR()])
    optimize!(model)
    inner = only(results(model)).model
    set_parameter_value(model[:stock], 30.0)
    optimize!(model)
    @test only(results(model)).model === inner
    @test MOI.Utilities.state(backend(inner)) == MOI.Utilities.ATTACHED_OPTIMIZER
    set_parameter_value(model[:stock], -10.0)
    optimize!(model)
    @test MOI.Utilities.state(backend(inner)) == MOI.Utilities.ATTACHED_OPTIMIZER
    fresh = inventory(8, -10.0)
    set_attribute(fresh, Formulations(), [PrimalLDR()])
    optimize!(fresh)
    cold = simplex_iterations(only(results(fresh)).model)
    @test simplex_iterations(inner) < cold
    @test objective_value(model) ≈ objective_value(fresh)
    return
end

# A model where nothing changes is solved again from the same inner problems.
function test_resolve_without_change()
    model = newsvendor(90.0)
    optimize!(model)
    inner = [r.model for r in results(model)]
    optimize!(model)
    @test num_builds(model) == 3
    @test all(r.model === m for (r, m) in zip(results(model), inner))
    @test objective_value(model) ≈ 595
    return
end

# Every change that is not the value of a parameter alone builds the inner problems again, and the
# results are those of a fresh model.
function test_other_changes_rebuild()
    changes = [
        "coefficient" => m -> set_normalized_coefficient(m[:limit], m[:buy], 2.0),
        "right-hand side" => m -> set_normalized_rhs(m[:limit], 95.0),
        "bound" => m -> set_upper_bound(m[:sell], 70.0),
        "deleted constraint" => m -> delete(m, m[:limit]),
        "objective" => m -> set_objective_coefficient(m, m[:ret], 7.0),
        "law" => function (m)
            S = LinearDecisionRules.UnivariateUncertainty{Uniform{Float64}}
            ci = only(all_constraints(m, VariableRef, S))
            MOI.set(m, MOI.ConstraintSet(), ci, Uncertainty(Uniform(70, 130)))
            return
        end,
        "new constraint" => m -> @constraint(m, m[:sell] <= 95),
        "valid inequality" => m -> @constraint(m, m[:demand] <= 115, ValidInequality()),
        "inner attribute" => m -> set_attribute(m, "presolve", "off"),
        "inner optimizer" =>
            m ->
                set_attribute(m, LinearDecisionRules.InnerOptimizer(), () -> HiGHS.Optimizer()),
    ]
    for (name, change!) in changes
        @testset "$name" begin
            model = newsvendor(90.0)
            optimize!(model)
            set_parameter_value(model[:capacity], 105.0)
            change!(model)
            optimize!(model)
            @test num_builds(model) == 6
            fresh = newsvendor(105.0)
            change!(fresh)
            optimize!(fresh)
            check_same_results(model, fresh)
        end
    end
    return
end

# A formulation whose attributes change is built again, the others are reused.
function test_formulation_changes_rebuild()
    model = newsvendor(90.0)
    optimize!(model)
    primal = PrimalLDR(; attributes = ["presolve" => "off"])
    set_attribute(model, Formulations(), [primal, DualLDR(), SampledLDR(20; seed = 3)])
    optimize!(model)
    @test num_builds(model) == 4
    @test objective_value(model) ≈ 595
    set_attribute(model, Formulations(), [primal, DualLDR(), SampledLDR(20; seed = 4)])
    optimize!(model)
    @test num_builds(model) == 5
    set_attribute(model, Formulations(), [primal, DualLDR(), SampledLDR(21; seed = 4)])
    optimize!(model)
    @test num_builds(model) == 6
    fresh = newsvendor(90.0)
    set_attribute(fresh, Formulations(), [primal, DualLDR(), SampledLDR(21; seed = 4)])
    optimize!(fresh)
    check_same_results(model, fresh)
    # Dropped formulations are not kept.
    set_attribute(model, Formulations(), [DualLDR()])
    optimize!(model)
    @test length(unsafe_backend(model).inner_cache.problems) == 1
    @test num_builds(model) == 6
    return
end

# A parameter multiplied by a decision changes a coefficient of the form: the inner problems are
# built again, with the new value.
function test_parameter_in_a_product_rebuilds()
    function build(value)
        m = newsvendor(150.0)
        @variable(m, fraction in Parameter(value))
        @constraint(m, m[:sell] <= fraction * m[:buy])
        return m
    end
    model = build(0.8)
    optimize!(model)
    for (k, fraction) in enumerate([0.9, 2.0, 0.5])
        set_parameter_value(model[:fraction], fraction)
        optimize!(model)
        @test num_builds(model) == 3 * (k + 1)
        fresh = build(fraction)
        optimize!(fresh)
        check_same_results(model, fresh)
    end
    # The capacity alone is still updated in place.
    set_parameter_value(model[:capacity], 95.0)
    optimize!(model)
    @test num_builds(model) == 12
    fresh = build(0.5)
    set_parameter_value(fresh[:capacity], 95.0)
    optimize!(fresh)
    check_same_results(model, fresh)
    return
end

# A time limit is set on the inner problems for one solve and removed for the next.
function test_time_limit_on_resolve()
    model = newsvendor(90.0)
    set_time_limit_sec(model, 100.0)
    optimize!(model)
    inner = unsafe_backend(model).inner_cache.problems
    @test all(p -> p.second.time_limited, inner)
    set_time_limit_sec(model, nothing)
    set_parameter_value(model[:capacity], 105.0)
    optimize!(model)
    @test num_builds(model) == 3
    @test !any(p -> p.second.time_limited, inner)
    @test objective_value(model) ≈ 647.5
    return
end

# The structural comparison of the fingerprint.
function test_same_value()
    same = LinearDecisionRules._same_value
    @test same(Uniform(0, 1), Uniform(0, 1))
    @test !same(Uniform(0, 1), Uniform(0, 2))
    @test same([Uniform(0, 1)], [Uniform(0, 1)])
    @test same(Dict("a" => [1.0]), Dict("a" => [1.0]))
    @test !same(Dict("a" => [1.0]), Dict("a" => [2.0]))
    @test same(PrimalLDR(; attributes = ["a" => 1]), PrimalLDR(; attributes = ["a" => 1]))
    @test !same(PrimalLDR(; attributes = ["a" => 1]), PrimalLDR())
    @test !same(1, 1.0)
    @test same(HiGHS.Optimizer, HiGHS.Optimizer)
    @test !same(Main, Base)
    return
end

# A parameter multiplied by a decision in the objective, the cost of `buy`: the re-solve gives the
# results of a fresh model with the new cost.
function test_parameter_times_decision_in_the_objective()
    function build(cost)
        model = ldr_model()
        @variable(model, c in Parameter(cost))
        @variable(model, buy >= 0, FirstStage)
        @variable(model, sell >= 0)
        @variable(model, ret >= 0)
        @variable(model, demand in Uncertainty(Uniform(80, 120)))
        @constraint(model, sell + ret <= buy)
        @constraint(model, sell <= demand)
        @objective(model, Max, -c * buy + 8ret + 15sell)
        return model, c
    end
    model, c = build(10.0)
    optimize!(model)
    @test objective_value(model) ≈ 460
    @test value(c) == 10
    set_parameter_value(c, 12.0)
    optimize!(model)
    fresh, _ = build(12.0)
    optimize!(fresh)
    @test objective_value(model) < 460 - 1
    @test objective_value(model) ≈ objective_value(fresh)
    @test objective_bound(model) ≈ objective_bound(fresh)
    @test value(c) == 12
    return
end

# A product of a parameter and a decision in a row is a quadratic row that `canonicalize` reads
# as affine.
function test_parameter_times_decision_in_a_row()
    function build(capacity)
        model = TaggedProblems.newsvendor(ldr_model())
        @variable(model, c in Parameter(capacity))
        @constraint(model, model[:sell] <= c * model[:buy])
        return model, c
    end
    model, c = build(0.8)
    @test num_constraints(model, QuadExpr, MOI.LessThan{Float64}) == 1
    optimize!(model)
    fresh = TaggedProblems.newsvendor(ldr_model())
    @constraint(fresh, fresh[:sell] <= 0.8 * fresh[:buy])
    optimize!(fresh)
    @test objective_value(model) < 460 - 1
    @test objective_value(model) ≈ objective_value(fresh)
    @test objective_bound(model) ≈ objective_bound(fresh)
    set_parameter_value(c, 2.0)
    optimize!(model)
    @test objective_value(model) ≈ 460
    return
end

end # module TestParameters

TestParameters.runtests()
