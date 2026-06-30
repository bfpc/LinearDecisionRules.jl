module SampledTests

include("common.jl")

function test_solve_sampled_newsvendor()
    buy_cost = 10
    return_value = 8
    sell_value = 15
    demand_max = 120
    demand_min = 80
    demand_distr = Uniform(demand_min, demand_max)

    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell >= 0)
    @variable(ldr, ret >= 0)
    @variable(
        ldr,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = demand_distr)
    )

    @constraint(ldr, sell + ret <= buy)
    @constraint(ldr, sell <= demand)
    @objective(
        ldr,
        Max,
        -buy_cost * buy + return_value * ret + sell_value * sell
    )

    set_attribute(ldr, LinearDecisionRules.SolveSampled(), true)
    set_attribute(ldr, LinearDecisionRules.NumScenarios(), 50)
    optimize!(ldr)

    # Sampled objective should be close to primal (both are inner approximations)
    ldr_p_obj = objective_value(ldr)
    ldr_s_obj = objective_value(ldr; sampled = true)
    ldr_d_obj = objective_value(ldr; dual = true)

    # Sampled should be between primal and dual (approximately)
    @test ldr_s_obj <= ldr_d_obj + 1.0
    @test ldr_s_obj >= ldr_p_obj - 5.0  # allow tolerance for sampling noise

    # First-stage decision should not depend on uncertainty
    @test LinearDecisionRules.get_decision(ldr, buy, demand; sampled = true) ≈ 0 atol =
        1e-6

    # get_decision for constant term should work
    buy_val = LinearDecisionRules.get_decision(ldr, buy; sampled = true)
    @test buy_val > 0

    return nothing
end

function test_solve_sampled_attributes()
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)

    # Defaults
    @test get_attribute(ldr, LinearDecisionRules.SolveSampled()) == false
    @test get_attribute(ldr, LinearDecisionRules.NumScenarios()) === nothing

    # Set/get
    set_attribute(ldr, LinearDecisionRules.SolveSampled(), true)
    @test get_attribute(ldr, LinearDecisionRules.SolveSampled()) == true
    set_attribute(ldr, LinearDecisionRules.NumScenarios(), 200)
    @test get_attribute(ldr, LinearDecisionRules.NumScenarios()) == 200

    return nothing
end

function test_solve_sampled_num_scenarios_required()
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, x >= 0)
    @variable(
        ldr,
        η in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0.0, 1.0))
    )
    @constraint(ldr, x >= η)
    @objective(ldr, Min, x)

    set_attribute(ldr, LinearDecisionRules.SolvePrimal(), false)
    set_attribute(ldr, LinearDecisionRules.SolveDual(), false)
    set_attribute(ldr, LinearDecisionRules.SolveSampled(), true)
    # NumScenarios not set -> should error
    @test_throws ErrorException optimize!(ldr)

    return nothing
end

function test_solve_sampled_with_rejection()
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, buy[1:2] >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell[1:2] >= 0)
    @variable(ldr, ret[1:2] >= 0)
    @variable(
        ldr,
        demand[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = product_distribution([
                Uniform(80.0, 120.0),
                Uniform(80.0, 120.0),
            ]),
        )
    )

    @constraint(ldr, [i = 1:2], sell[i] + ret[i] <= buy[i])
    @constraint(ldr, [i = 1:2], sell[i] <= demand[i])
    @objective(
        ldr,
        Max,
        sum(-10 * buy[i] + 8 * ret[i] + 15 * sell[i] for i in 1:2)
    )

    # Add polytope constraints that require rejection sampling
    @constraint(ldr, demand[1] <= 110)
    @constraint(ldr, demand[1] >= 100)
    @constraint(ldr, demand[2] >= 100)
    @constraint(ldr, demand[2] <= 110)

    set_attribute(ldr, LinearDecisionRules.SolvePrimal(), false)
    set_attribute(ldr, LinearDecisionRules.SolveDual(), false)
    set_attribute(ldr, LinearDecisionRules.SolveSampled(), true)
    set_attribute(ldr, LinearDecisionRules.NumScenarios(), 20)

    optimize!(ldr)

    @test termination_status(ldr; sampled = true) == MOI.OPTIMAL
    obj = objective_value(ldr; sampled = true)
    @test obj > 0

    # First-stage should not depend on uncertainty
    for i in 1:2, j in 1:2
        @test LinearDecisionRules.get_decision(
            ldr,
            buy[i],
            demand[j];
            sampled = true,
        ) ≈ 0 atol = 1e-6
    end

    return nothing
end

function test_solve_sampled_only()
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell >= 0)
    @variable(ldr, ret >= 0)
    @variable(
        ldr,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(80.0, 120.0))
    )

    @constraint(ldr, sell + ret <= buy)
    @constraint(ldr, sell <= demand)
    @objective(ldr, Max, -10 * buy + 8 * ret + 15 * sell)

    set_attribute(ldr, LinearDecisionRules.SolvePrimal(), false)
    set_attribute(ldr, LinearDecisionRules.SolveDual(), false)
    set_attribute(ldr, LinearDecisionRules.SolveSampled(), true)
    set_attribute(ldr, LinearDecisionRules.NumScenarios(), 30)

    optimize!(ldr)

    @test termination_status(ldr; sampled = true) == MOI.OPTIMAL
    obj = objective_value(ldr; sampled = true)
    @test obj > 0

    # Primal/dual should error
    @test_throws ErrorException objective_value(ldr)
    @test_throws ErrorException objective_value(ldr; dual = true)
    @test_throws ErrorException LinearDecisionRules.get_decision(ldr, buy)
    @test_throws ErrorException LinearDecisionRules.get_decision(
        ldr,
        buy;
        dual = true,
    )

    # Sampled get_decision should work
    buy_val = LinearDecisionRules.get_decision(ldr, buy; sampled = true)
    @test buy_val > 0

    return nothing
end

function test_solve_sampled_piecewise()
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell >= 0)
    @variable(ldr, ret >= 0)
    @variable(
        ldr,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(80.0, 120.0))
    )

    @constraint(ldr, sell + ret <= buy)
    @constraint(ldr, sell <= demand)
    @objective(ldr, Max, -10 * buy + 8 * ret + 15 * sell)

    set_attribute(demand, LinearDecisionRules.BreakPoints(), 2)

    set_attribute(ldr, LinearDecisionRules.SolvePrimal(), false)
    set_attribute(ldr, LinearDecisionRules.SolveDual(), false)
    set_attribute(ldr, LinearDecisionRules.SolveSampled(), true)
    set_attribute(ldr, LinearDecisionRules.NumScenarios(), 30)

    optimize!(ldr)

    @test termination_status(ldr; sampled = true) == MOI.OPTIMAL

    # With 2 breakpoints we have 3 pieces
    for piece in 1:3
        coef = LinearDecisionRules.get_decision(
            ldr,
            sell,
            demand;
            sampled = true,
            piece = piece,
        )
        @test isfinite(coef)
    end

    # Constant term should work
    buy_val = LinearDecisionRules.get_decision(ldr, buy; sampled = true)
    @test buy_val > 0

    return nothing
end

function test_get_decision_dual_and_sampled_error()
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell >= 0)
    @variable(
        ldr,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(80.0, 120.0))
    )

    @constraint(ldr, sell <= buy)
    @constraint(ldr, sell <= demand)
    @objective(ldr, Max, -10 * buy + 15 * sell)

    set_attribute(ldr, LinearDecisionRules.SolveSampled(), true)
    set_attribute(ldr, LinearDecisionRules.NumScenarios(), 100)
    optimize!(ldr)

    # dual=true and sampled=true should error for all query functions
    @test_throws ErrorException LinearDecisionRules.get_decision(
        ldr,
        buy;
        dual = true,
        sampled = true,
    )
    @test_throws ErrorException LinearDecisionRules.get_decision(
        ldr,
        sell,
        demand;
        dual = true,
        sampled = true,
    )
    @test_throws ErrorException objective_value(
        ldr;
        dual = true,
        sampled = true,
    )
    @test_throws ErrorException termination_status(
        ldr;
        dual = true,
        sampled = true,
    )

    return nothing
end

function test_solve_sampled_quadratic_objective()
    # Tests the M̂ path in _prepare_data (needs_M̂ = true)
    #
    # HiGHS crashes with std::bad_alloc on 32-bit QP (upstream bug)
    # On the other hand, Ipopt complains (on any platform) that this problem
    # has too few degrees of freedom, so this test is skipped on 32-bit
    if Sys.WORD_SIZE == 32
        return
    end
    initial_volume = 0.5
    demand = 0.3

    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, vi == initial_volume)
    @variable(m, 0 <= vf <= 1)
    @variable(m, gh >= 0.0)
    @variable(m, gt >= 0.0)
    @variable(
        m,
        inflow in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 0.2))
    )

    @constraint(m, balance, vf == vi - gh + inflow)
    @constraint(m, gt + gh == demand)
    @objective(m, Min, gt^2 + vf^2 / 2 - vf)

    set_attribute(m, LinearDecisionRules.SolvePrimal(), false)
    set_attribute(m, LinearDecisionRules.SolveDual(), false)
    set_attribute(m, LinearDecisionRules.SolveSampled(), true)
    set_attribute(m, LinearDecisionRules.NumScenarios(), 50)
    optimize!(m)

    # Verify the M̂ path was taken (not the μ̂ path)
    @test haskey(m.ext, :_LDR_M_empirical)
    @test !haskey(m.ext, :_LDR_μ_empirical)
    @test termination_status(m; sampled = true) in
          (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)

    # Sampled decision rules should satisfy balance constraint
    gh_const = LinearDecisionRules.get_decision(m, gh; sampled = true)
    gt_const = LinearDecisionRules.get_decision(m, gt; sampled = true)
    @test gh_const + gt_const ≈ demand atol = 1e-4

    return nothing
end

function test_solve_sampled_reoptimize()
    # Tests the rebuild path in _solve_sampled_ldr
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell >= 0)
    @variable(ldr, ret >= 0)
    @variable(
        ldr,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(80.0, 120.0))
    )

    @constraint(ldr, sell + ret <= buy)
    @constraint(ldr, sell <= demand)
    @objective(ldr, Max, -10 * buy + 8 * ret + 15 * sell)

    set_attribute(ldr, LinearDecisionRules.SolvePrimal(), false)
    set_attribute(ldr, LinearDecisionRules.SolveDual(), false)
    set_attribute(ldr, LinearDecisionRules.SolveSampled(), true)
    set_attribute(ldr, LinearDecisionRules.NumScenarios(), 30)

    optimize!(ldr)
    obj1 = objective_value(ldr; sampled = true)
    @test obj1 > 0

    # Re-optimize should rebuild the sampled model
    optimize!(ldr)
    obj2 = objective_value(ldr; sampled = true)
    @test obj2 > 0

    return nothing
end

function test_solve_sampled_minimization()
    # All other sampled tests use Max; this tests Min
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, x >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, y >= 0)
    @variable(
        ldr,
        η in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0.0, 1.0))
    )

    @constraint(ldr, y >= η)
    @constraint(ldr, x + y >= 1)
    @objective(ldr, Min, x + y)

    set_attribute(ldr, LinearDecisionRules.SolvePrimal(), false)
    set_attribute(ldr, LinearDecisionRules.SolveDual(), false)
    set_attribute(ldr, LinearDecisionRules.SolveSampled(), true)
    set_attribute(ldr, LinearDecisionRules.NumScenarios(), 50)

    optimize!(ldr)

    @test termination_status(ldr; sampled = true) == MOI.OPTIMAL
    @test objective_value(ldr; sampled = true) >= 1.0 - 1e-6

    return nothing
end

function test_solve_sampled_discrete_nonparametric()
    # Tests free sampling path (no rejection) with DiscreteNonParametric
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell >= 0)
    @variable(
        ldr,
        demand in LinearDecisionRules.Uncertainty(;
            distribution = DiscreteNonParametric(
                [80.0, 100.0, 120.0],
                [0.25, 0.5, 0.25],
            ),
        )
    )

    @constraint(ldr, sell <= buy)
    @constraint(ldr, sell <= demand)
    @objective(ldr, Max, -10 * buy + 15 * sell)

    set_attribute(ldr, LinearDecisionRules.SolvePrimal(), false)
    set_attribute(ldr, LinearDecisionRules.SolveDual(), false)
    set_attribute(ldr, LinearDecisionRules.SolveSampled(), true)
    set_attribute(ldr, LinearDecisionRules.NumScenarios(), 50)

    optimize!(ldr)
    @test termination_status(ldr; sampled = true) == MOI.OPTIMAL
    @test objective_value(ldr; sampled = true) > 0

    return nothing
end

function test_sampled_status_functions()
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        ldr,
        η in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0.0, 1.0))
    )
    @constraint(ldr, x >= η)
    @objective(ldr, Min, x)

    # Sampled disabled: status should reflect not-called/no-solution
    @test termination_status(ldr; sampled = true) == MOI.OPTIMIZE_NOT_CALLED
    @test primal_status(ldr; sampled = true) == MOI.NO_SOLUTION
    @test raw_status(ldr; sampled = true) == MOI.NO_SOLUTION
    @test dual_status(ldr; sampled = true) == MOI.NO_SOLUTION

    # Enable sampled and solve
    set_attribute(ldr, LinearDecisionRules.SolveSampled(), true)
    set_attribute(ldr, LinearDecisionRules.NumScenarios(), 20)
    optimize!(ldr)

    @test termination_status(ldr; sampled = true) == MOI.OPTIMAL
    @test primal_status(ldr; sampled = true) == MOI.FEASIBLE_POINT
    @test dual_status(ldr; sampled = true) == MOI.NO_SOLUTION

    return nothing
end

function test_sampled_solution_summary()
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        ldr,
        η in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0.0, 1.0))
    )
    @constraint(ldr, x >= η)
    @objective(ldr, Min, x)

    # Without sampled: summary.sampled should be nothing
    optimize!(ldr)
    s = solution_summary(ldr)
    @test s.sampled === nothing
    io = IOBuffer()
    show(io, s)
    @test occursin("Sampled", String(take!(io)))

    # With sampled enabled
    set_attribute(ldr, LinearDecisionRules.SolveSampled(), true)
    set_attribute(ldr, LinearDecisionRules.NumScenarios(), 20)
    optimize!(ldr)
    s2 = solution_summary(ldr)
    @test s2.sampled !== nothing
    io2 = IOBuffer()
    show(io2, s2)
    @test occursin("Sampled", String(take!(io2)))

    return nothing
end

function test_sampled_disabled_errors()
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell >= 0)
    @variable(
        ldr,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(80.0, 120.0))
    )

    @constraint(ldr, sell <= buy)
    @constraint(ldr, sell <= demand)
    @objective(ldr, Max, -10 * buy + 15 * sell)

    # Sampled not enabled -> errors on sampled queries
    optimize!(ldr)
    @test_throws ErrorException objective_value(ldr; sampled = true)
    @test_throws ErrorException LinearDecisionRules.get_decision(
        ldr,
        buy;
        sampled = true,
    )
    @test_throws ErrorException LinearDecisionRules.get_decision(
        ldr,
        sell,
        demand;
        sampled = true,
    )

    return nothing
end

function test_solve_sampled_vector_distribution()
    # Tests _sample_scenarios with MvDiscreteNonParametric (no rejection)
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, buy[1:2] >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell[1:2] >= 0)
    @variable(
        ldr,
        demand[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = LinearDecisionRules.MvDiscreteNonParametric(
                [[80.0, 80.0], [120.0, 120.0]],
                [0.5, 0.5],
            ),
        )
    )

    @constraint(ldr, [i = 1:2], sell[i] <= buy[i])
    @constraint(ldr, [i = 1:2], sell[i] <= demand[i])
    @objective(ldr, Max, sum(-10 * buy[i] + 15 * sell[i] for i in 1:2))

    set_attribute(ldr, LinearDecisionRules.SolvePrimal(), false)
    set_attribute(ldr, LinearDecisionRules.SolveDual(), false)
    set_attribute(ldr, LinearDecisionRules.SolveSampled(), true)
    set_attribute(ldr, LinearDecisionRules.NumScenarios(), 50)

    optimize!(ldr)
    @test termination_status(ldr; sampled = true) == MOI.OPTIMAL
    @test objective_value(ldr; sampled = true) > 0

    return nothing
end

end # module

SampledTests.runtests()
