# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Laws conditioned by constraints tagged `Truncate()`, end to end: the examples of
# `test_newsvendor_with_rejection_sampling` (test_core.jl) and of the tutorial
# `advanced_distributions.jl` of the previous version.

module TestTruncate

using JuMP
using Test

import Distributions
import Distributions: Uniform
import HiGHS
import LinearDecisionRules as LDR
import LinearDecisionRules:
    ConditionedDistribution,
    DualLDR,
    FirstStage,
    Formulations,
    LDRModel,
    MvDiscreteNonParametric,
    PrimalLDR,
    RejectionSamplingMaxSamples,
    RejectionSamplingSeed,
    RejectionSamplingTimeLimit,
    SampledLDR,
    Truncate,
    Uncertainty,
    is_certified,
    simulate
import Random

include(joinpath(@__DIR__, "..", "common.jl"))
import .TestCommon: error_message

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

include(joinpath(@__DIR__, "..", "problems", "tagged.jl"))

ldr_model() = TaggedProblems.ldr_model(; time_limit = 60.0)

# The newsvendor of the previous version; `demand` returns the demand from the model.
function newsvendor(demand::Function)
    model = ldr_model()
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    d = demand(model)
    @constraint(model, sell + ret <= buy)
    @constraint(model, sell <= d)
    @objective(model, Max, -10buy + 8ret + 15sell)
    return model
end

moments_of(model) = MOI.get(model, LDR.UncertaintyMoments())

form_of(model) = JuMP.unsafe_backend(model).form

# Uniform(80, 120) given 100 ≤ d ≤ 110: exactly Uniform(100, 110), whose mean is 105.
function exact_newsvendor()
    return newsvendor() do model
        @variable(model, demand in Uncertainty(Uniform(80, 120)))
        @constraint(model, demand >= 100, Truncate())
        @constraint(model, demand <= 110, Truncate())
        return demand
    end
end

# The same law as a one-component product law, which is truncated by rejection sampling.
function rejection_newsvendor()
    return newsvendor() do model
        law = Distributions.product_distribution([Uniform(80, 120)])
        @variable(model, demand[1:1] in Uncertainty(law))
        @constraint(model, demand[1] >= 100, Truncate())
        @constraint(model, demand[1] <= 110, Truncate())
        return demand[1]
    end
end

function plain_newsvendor(a, b)
    return newsvendor() do model
        @variable(model, demand in Uncertainty(Uniform(a, b)))
        return demand
    end
end

# The old moment check (mean 100, then about 105), by both paths.
function test_newsvendor_mean()
    untruncated = plain_newsvendor(80, 120)
    optimize!(untruncated)
    @test moments_of(untruncated).mean == [1, 100]
    exact = exact_newsvendor()
    optimize!(exact)
    @test moments_of(exact).mean == [1, 105]
    @test only(form_of(exact).laws).distribution == Uniform(100, 110)
    reference = plain_newsvendor(100, 110)
    optimize!(reference)
    @test objective_value(exact) ≈ objective_value(reference)
    @test objective_bound(exact) ≈ objective_bound(reference)
    # Truncating the demand to a narrower support improves both values.
    @test objective_value(untruncated) < objective_value(exact)
    @test objective_bound(untruncated) < objective_bound(exact)
    rejection = rejection_newsvendor()
    optimize!(rejection)
    @test only(form_of(rejection).laws).distribution isa ConditionedDistribution
    # The standard error of the mean is 10 / √12 / √10⁴ ≈ 0.029.
    @test isapprox(moments_of(rejection).mean[2], 105; atol = 0.1)
    @test isapprox(only(moments_of(rejection).blocks).covariance[1], 100 / 12; rtol = 0.05)
    # Both paths agree within the Monte Carlo error.
    @test objective_value(rejection) ≈ objective_value(exact) rtol = 1e-2
    @test objective_bound(rejection) ≈ objective_bound(exact) rtol = 1e-2
    @test value(rejection[:demand][1]) ≈ moments_of(rejection).mean[2]
    return
end

# Whether the values of each formulation of `model` are certified, in the order of `Formulations`.
certified(model) = [is_certified(model, f) for f in get_attribute(model, Formulations())]

# Values computed with moments estimated by Monte Carlo are not certified; exact ones are.
function test_certified()
    exact = exact_newsvendor()
    set_attribute(exact, Formulations(), [PrimalLDR(), DualLDR(), SampledLDR(100)])
    optimize!(exact)
    @test certified(exact) == [true, true, false]
    @test !occursin("not certified", raw_status(exact))
    rejection = rejection_newsvendor()
    set_attribute(rejection, Formulations(), [PrimalLDR(), DualLDR(), SampledLDR(100)])
    optimize!(rejection)
    @test !any(certified(rejection))
    note = "[not certified: moments of conditioned laws estimated by rejection sampling]"
    @test endswith(raw_status(rejection, DualLDR()), note)
    @test endswith(raw_status(rejection, PrimalLDR()), note)
    @test !occursin("not certified", raw_status(rejection, SampledLDR(100)))
    @test termination_status(rejection, DualLDR()) == MOI.OPTIMAL
    return
end

# Two independent demands, each truncated alone: two independent laws.
function test_separate_truncations_are_independent()
    model = rejection_newsvendor()
    law = Distributions.product_distribution([Uniform(80, 120)])
    @variable(model, other[1:1] in Uncertainty(law))
    @constraint(model, 100 <= other[1] <= 110, Truncate())
    @constraint(model, model[:sell] <= other[1])
    optimize!(model)
    mom = moments_of(model)
    @test [block.columns for block in mom.blocks] == [[2], [3]]
    @test all(block.num_samples == 10_000 for block in mom.blocks)
    # Each law has its own draws.
    @test mom.mean[2] != mom.mean[3]
    @test isapprox(mom.mean[3], 105; atol = 0.1)
    M = LDR.second_moment(mom)
    @test M[2, 3] == mom.mean[2] * mom.mean[3]
    return
end

# Part 2 of the tutorial: two demands in [40, 80] with a total of at most 120.
function budget(; total = 120.0)
    model = ldr_model()
    @variable(model, buy[1:2] >= 0, FirstStage)
    @variable(model, sell[1:2] >= 0)
    @variable(model, leftover[1:2] >= 0)
    law = Distributions.product_distribution([Uniform(40, 80), Uniform(40, 80)])
    @variable(model, demand[1:2] in Uncertainty(law))
    @constraint(model, budget, demand[1] + demand[2] <= total, Truncate())
    @constraint(model, stock[i in 1:2], sell[i] + leftover[i] <= buy[i])
    @constraint(model, [i in 1:2], sell[i] <= demand[i])
    @objective(model, Max, sum(-10buy[i] + 15sell[i] + 5leftover[i] for i in 1:2))
    return model
end

function test_budget()
    model = budget()
    set_attribute(model, Formulations(), [PrimalLDR(), DualLDR(), SampledLDR(500)])
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test termination_status(model, SampledLDR(500)) == MOI.OPTIMAL
    mom = moments_of(model)
    # The uniform law on the triangle: mean 40 + 40 / 3, covariance -40² / 36 < 0.
    @test isapprox(mom.mean[2:3], fill(40 + 40 / 3, 2); atol = 0.5)
    Σ = only(mom.blocks).covariance
    @test isapprox(Σ, [40^2/18 -40^2/36; -40^2/36 40^2/18]; rtol = 0.1)
    @test objective_value(model) <= objective_bound(model) + 1e-6
    # The independent demands have a larger mean, so more to sell.
    free = budget(; total = 160.0)
    optimize!(free)
    @test objective_value(model) < objective_value(free)
    @test value.(model[:demand]) ≈ mom.mean[2:3]
    return
end

# The scenarios of `SampledLDR` and of `simulate` are drawn from the conditioned law.
function test_scenarios_in_conditioning_set()
    model = budget()
    set_attribute(model, Formulations(), [PrimalLDR(), SampledLDR(500; seed = 3)])
    optimize!(model)
    outcomes = LDR._sample_outcomes(form_of(model), SampledLDR(500; seed = 3))
    @test all(outcomes[2, :] + outcomes[3, :] .<= 120)
    @test outcomes == LDR._sample_outcomes(form_of(model), SampledLDR(500; seed = 3))
    @test outcomes != LDR._sample_outcomes(form_of(model), SampledLDR(500; seed = 4))
    sim = MOI.get(model, LDR.SimulationData(PrimalLDR()))
    ξ = LDR._draw_chunk(Random.Xoshiro(1), sim, 2000)
    @test all(ξ[2, :] + ξ[3, :] .<= 120)
    @test all(40 .<= ξ[2:3, :] .<= 80)
    # The primal rule is feasible on Ξ, which contains the conditioning set.
    s = simulate(model, 2000; rng = Random.Xoshiro(1))
    @test s.violation_fraction == 0
    @test isapprox(s.objective_mean, objective_value(model); atol = 4 * s.objective_stderr + 1)
    # A scenario outside the conditioning set is outside Ξ.
    d = model[:demand]
    msg = error_message(() -> simulate(model, [Dict(d[1] => 70.0, d[2] => 70.0)]))
    @test occursin("the constraint tagged `Truncate()` budget is violated by 20", msg)
    return
end

# Part 3 of the tutorial: three reservoirs, with five constraints on their inflows.
function test_reservoir()
    model = ldr_model()
    capacity = [100.0, 80.0, 120.0]
    benefit = [5.0, 4.0, 6.0]
    @variable(model, 0 <= storage[i in 1:3] <= capacity[i], FirstStage)
    @variable(model, release[1:3] >= 0)
    @variable(model, spill[1:3] >= 0)
    law = Distributions.product_distribution([Uniform(10, 50), Uniform(10, 40), Uniform(10, 60)])
    @variable(model, inflow[1:3] in Uncertainty(law))
    rows = [
        @constraint(model, inflow[1] + inflow[2] + inflow[3] <= 120, Truncate()),
        @constraint(model, inflow[1] - inflow[2] <= 20, Truncate()),
        @constraint(model, inflow[2] - inflow[1] <= 20, Truncate()),
        @constraint(model, inflow[2] - inflow[3] <= 25, Truncate()),
        @constraint(model, inflow[3] - inflow[2] <= 25, Truncate()),
    ]
    @constraint(model, [i in 1:3], release[i] + spill[i] <= storage[i] + inflow[i])
    @objective(model, Max, sum(benefit[i] * release[i] - spill[i] for i in 1:3))
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) <= objective_bound(model) + 1e-6
    d = only(form_of(model).laws).distribution
    @test d isa ConditionedDistribution && length(d.h) == 5
    # The estimated mean is in the conditioning set.
    μ = moments_of(model).mean[2:4]
    @test all(d.W * μ .>= d.h)
    @test value.(inflow) ≈ μ
    return
end

# Part 4 of the tutorial: seven scenarios, the one with a total above 160 excluded.
function test_discrete_scenarios()
    scenarios = [[60.0, 60], [60, 80], [80, 60], [80, 80], [100, 60], [60, 100], [100, 100]]
    model = ldr_model()
    @variable(model, buy[1:2] >= 0, FirstStage)
    @variable(model, sell[1:2] >= 0)
    law = MvDiscreteNonParametric(scenarios, fill(1 / 7, 7))
    @variable(model, demand[1:2] in Uncertainty(law))
    @constraint(model, demand[1] + demand[2] <= 160, Truncate())
    @constraint(model, [i in 1:2], sell[i] <= buy[i])
    @constraint(model, [i in 1:2], sell[i] <= demand[i])
    @objective(model, Max, sum(-10buy[i] + 15sell[i] for i in 1:2))
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    # The six remaining scenarios are equally likely, exactly: no sampling, and certified values.
    mom = moments_of(model)
    @test mom.mean[2:3] ≈ sum(scenarios[1:6]) / 6
    @test only(mom.blocks).num_samples == 0
    @test all(certified(model))
    @test !occursin("not certified", raw_status(model))
    # The law of the rejection sampling of the previous versions: the same within its error.
    rejection = ConditionedDistribution([law], [-1.0 -1.0], [-160.0])
    draws = rand(Random.Xoshiro(1), rejection, 10_000)
    # The standard deviation of each demand is 14.9, so the standard error is 0.15.
    @test isapprox(vec(sum(draws; dims = 2)) ./ 10_000, mom.mean[2:3]; atol = 0.6)
    return
end

function test_rejection_sampling_attributes()
    model = budget()
    @test get_attribute(model, RejectionSamplingSeed()) == 1234
    # No time limit by default: it depends on `TimeLimitSec()`.
    @test get_attribute(model, RejectionSamplingTimeLimit()) === nothing
    @test get_attribute(model, RejectionSamplingMaxSamples()) == 10_000
    set_attribute(model, RejectionSamplingSeed(), -5)
    set_attribute(model, RejectionSamplingTimeLimit(), 20)
    set_attribute(model, RejectionSamplingMaxSamples(), 2_000)
    @test get_attribute(model, RejectionSamplingSeed()) == -5
    @test get_attribute(model, RejectionSamplingTimeLimit()) === 20.0
    @test get_attribute(model, RejectionSamplingMaxSamples()) == 2_000
    @test_throws ArgumentError set_attribute(model, RejectionSamplingTimeLimit(), 0)
    @test_throws ArgumentError set_attribute(model, RejectionSamplingMaxSamples(), 0)
    optimize!(model)
    @test only(moments_of(model).blocks).num_samples == 2_000
    # The same settings give the same estimates as `moments`.
    expected = LDR.moments(form_of(model); seed = -5, max_samples = 2_000)
    @test moments_of(model).mean == expected.mean
    set_attribute(model, RejectionSamplingTimeLimit(), nothing)
    @test get_attribute(model, RejectionSamplingTimeLimit()) === nothing
    return
end

# Rejection sampling counts against `TimeLimitSec()`: without `RejectionSamplingTimeLimit()`, it
# gets half of the time that remains, shared by the conditioned laws, and the formulations the
# rest; an explicit limit wins.
function test_rejection_sampling_shares_the_time_limit()
    model = budget()
    set_attribute(model, RejectionSamplingMaxSamples(), 10^8)
    set_time_limit_sec(model, 2.0)
    shared = r"seconds shared by the laws, half of the time that remained of `TimeLimitSec\(\)`"
    start = time_ns()
    @test_logs (:warn, shared) optimize!(model)
    # 10^8 draws would take minutes; the slack is for a loaded machine.
    @test (time_ns() - start) / 1e9 < 2.0 + 2.0
    @test termination_status(model, DualLDR()) == MOI.OPTIMAL
    @test 100 <= only(moments_of(model).blocks).num_samples < 10^8
    # Without time left, the laws are estimated from one batch of draws, if it has enough.
    set_time_limit_sec(model, 1e-6)
    @test_logs (:warn, shared) optimize!(model)
    @test termination_status(model, DualLDR()) == MOI.TIME_LIMIT
    # A law with an acceptance rate of 3% has too few in one batch: an error that says why.
    model = budget(; total = 90.0)
    set_time_limit_sec(model, 1e-6)
    message = error_message(() -> optimize!(model))
    @test occursin("needs 100 within its time limit of 0.0 seconds", message)
    @test occursin("half of the time that remained of `TimeLimitSec()`", message)
    # An explicit limit for each law wins, and its time counts against `TimeLimitSec()`.
    model = budget()
    set_attribute(model, RejectionSamplingMaxSamples(), 10^8)
    set_attribute(model, RejectionSamplingTimeLimit(), 0.5)
    set_time_limit_sec(model, 0.2)
    explicit = r"within its time limit \(0.5 seconds per law, `RejectionSamplingTimeLimit\(\)`\)"
    start = time_ns()
    @test_logs (:warn, explicit) optimize!(model)
    @test (time_ns() - start) / 1e9 >= 0.5
    @test termination_status(model, PrimalLDR()) == MOI.TIME_LIMIT
    @test termination_status(model, DualLDR()) == MOI.TIME_LIMIT
    return
end

num_estimates(model) = JuMP.unsafe_backend(model).moments_cache.num_estimates

# A re-solve samples again only when the inputs of the estimates change.
function test_cache()
    model = budget()
    optimize!(model)
    first = moments_of(model)
    @test num_estimates(model) == 1
    # Data that do not change the conditioned law: the right-hand side of a decision row, the
    # objective, a new decision.
    set_normalized_rhs(model[:stock][1], 5.0)
    set_objective_coefficient(model, model[:buy][1], -11.0)
    @variable(model, extra >= 0)
    @constraint(model, extra <= model[:buy][1])
    optimize!(model)
    @test num_estimates(model) == 1
    @test moments_of(model).mean == first.mean
    @test only(moments_of(model).blocks).covariance == only(first.blocks).covariance
    # A new seed samples again, and so does the previous seed then: the cache keeps the estimates
    # of the last solve only.
    set_attribute(model, RejectionSamplingSeed(), 99)
    optimize!(model)
    @test num_estimates(model) == 2
    @test moments_of(model).mean != first.mean
    set_attribute(model, RejectionSamplingSeed(), 1234)
    optimize!(model)
    @test num_estimates(model) == 3
    @test moments_of(model).mean == first.mean
    # So do a new sample size and a new conditioning row.
    set_attribute(model, RejectionSamplingMaxSamples(), 5_000)
    optimize!(model)
    @test num_estimates(model) == 4
    set_normalized_rhs(model[:budget], 130.0)
    optimize!(model)
    @test num_estimates(model) == 5
    @test moments_of(model).mean[2] > first.mean[2]
    # A sample cut by the time limit is not kept: it depends on the machine.
    set_attribute(model, RejectionSamplingMaxSamples(), 10^8)
    set_attribute(model, RejectionSamplingTimeLimit(), 0.01)
    @test_logs (:warn, r"Rejection sampling kept fewer than") optimize!(model)
    @test_logs (:warn, r"Rejection sampling kept fewer than") optimize!(model)
    @test num_estimates(model) == 7
    return
end

# Laws coupled by a row tagged `Truncate()` merge; the dual warns that the conditioned law, which
# spans two stages, is not known to have linear conditional expectations.
function test_coupled_declarations()
    model = ldr_model()
    @variable(model, 0 <= buy <= 1, FirstStage)
    @variable(model, early, LDR.Stage(2))
    @variable(model, late, LDR.Stage(3))
    @variable(model, a in Uncertainty(Uniform(0, 1)), LDR.Stage(2))
    @variable(model, b in Uncertainty(Uniform(0, 1)), LDR.Stage(3))
    @constraint(model, a + b <= 1, Truncate())
    @constraint(model, early <= buy + a)
    @constraint(model, late <= early + b)
    @objective(model, Max, late - 0.5buy)
    @test_logs (:warn, r"is not known for the law of a, b \(stages 2, 3\)") optimize!(model)
    mom = moments_of(model)
    @test [block.columns for block in mom.blocks] == [[2, 3]]
    # The triangle: mean 1 / 3, covariance -1 / 36.
    @test isapprox(mom.mean[2:3], [1 / 3, 1 / 3]; atol = 0.02)
    @test isapprox(only(mom.blocks).covariance[1, 2], -1 / 36; atol = 0.003)
    return
end

end # module TestTruncate

TestTruncate.runtests()
