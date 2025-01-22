module TestMain

using Test
using Random

using LinearDecisionRules

using JuMP
using Ipopt
using HiGHS
using Distributions

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_no_random()
    m = LinearDecisionRules.LDRModel(Ipopt.Optimizer)
    set_silent(m)
    @variable(m, x)
    @constraint(m, x == 1)
    @objective(m, Min, 0)
    set_attribute(m, LinearDecisionRules.SolveDual(), false)
    @test get_attribute(m, LinearDecisionRules.SolveDual()) == false
    optimize!(m)
    @test_throws OptimizeNotCalled() value(x) # Also prints a warning
    @test primal_status(m) == MOI.FEASIBLE_POINT
    @test termination_status(m) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test LinearDecisionRules.get_decision(m, x) == 1
    @test_throws ErrorException LinearDecisionRules.get_decision(m, x, dual = true)
    # set new config
    set_attribute(m, LinearDecisionRules.SolveDual(), true)
    @test get_attribute(m, LinearDecisionRules.SolveDual()) == true
    set_attribute(m, LinearDecisionRules.SolvePrimal(), false)
    @test get_attribute(m, LinearDecisionRules.SolvePrimal()) == false
    optimize!(m)
    @test_throws ErrorException LinearDecisionRules.get_decision(m, x)
    @test LinearDecisionRules.get_decision(m, x, dual = true) == 1
    # set new config
    set_attribute(m, LinearDecisionRules.SolveDual(), false)
    @test get_attribute(m, LinearDecisionRules.SolveDual()) == false
    set_attribute(m, LinearDecisionRules.SolvePrimal(), false)
    @test get_attribute(m, LinearDecisionRules.SolvePrimal()) == false
    @test_throws ErrorException optimize!(m)
    # set new config
    set_attribute(m, LinearDecisionRules.SolveDual(), true)
    @test get_attribute(m, LinearDecisionRules.SolveDual()) == true
    set_attribute(m, LinearDecisionRules.SolvePrimal(), true)
    @test get_attribute(m, LinearDecisionRules.SolvePrimal()) == true
    optimize!(m)
    @test primal_status(m) == MOI.FEASIBLE_POINT
    @test primal_status(m, dual = true) == MOI.FEASIBLE_POINT
    @test termination_status(m) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test termination_status(m, dual = true) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test LinearDecisionRules.get_decision(m, x) == 1
    @test LinearDecisionRules.get_decision(m, x, dual = true) == 1
    return nothing
end

function test_newsvendor()
    buy_cost = 10
    return_value = 8
    sell_value = 15

    demand_max = 120
    demand_min = 80
    demand_distr = Uniform(demand_min, demand_max)

    # SAA

    scenarios = 1000

    rng = Random.MersenneTwister(123)
    demand = rand(rng, demand_distr, scenarios)

    saa = Model(HiGHS.Optimizer)
    set_silent(saa)

    @variable(saa, buy >= 0)
    @variable(saa, sell[i in 1:scenarios] >= 0)
    @variable(saa, ret[i in 1:scenarios] >= 0)

    @constraint(saa, [i in 1:scenarios], sell[i] + ret[i] <= buy)

    @constraint(saa, [i in 1:scenarios], sell[i] <= demand[i])

    @objective(saa, Max,
        - buy_cost * buy
        + (1/scenarios) * sum(
            return_value * ret[i]
            + sell_value * sell[i]
            for i in 1:scenarios
        )
    )

    optimize!(saa)

    saa_obj = objective_value(saa)

    # LDR

    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell >= 0)
    @variable(ldr, ret >= 0)
    @variable(ldr, demand in
        LinearDecisionRules.Uncertainty(
            distribution = demand_distr
        )
    )

    @constraint(ldr, sell + ret <= buy)

    @constraint(ldr, sell <= demand)

    @objective(ldr, Max,
        - buy_cost * buy
        + return_value * ret
        + sell_value * sell
    )

    optimize!(ldr)

    # Primal LDR is feasible:
    # underperforms adaptive solution (SAA) up to estimation errors
    ldr_p_obj = objective_value(ldr)
    @test saa_obj >= ldr_p_obj - 1e-6

    # First-stage decision do not depend on the uncertainty
    @test LinearDecisionRules.get_decision(ldr, buy, demand) == 0

    # Dual LDR is a performance bound:
    # SAA cannot yield better objective, up to estimation errors
    ldr_d_obj = objective_value(ldr, dual = true)
    @test saa_obj <= ldr_d_obj + 1e-6

    # First-stage decision do not depend on the uncertainty
    @test LinearDecisionRules.get_decision(ldr, buy, demand, dual = true) == 0

    return
end

function test_double_newsvendor()

    buy_cost = 10
    return_value = 8
    sell_value = 15

    demand_max = 120
    demand_min = 80

    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, buy[1:2] >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell[1:2] >= 0)
    @variable(ldr, ret[1:2] >= 0)
    @variable(ldr, demand[1:2] in
        LinearDecisionRules.Uncertainty(
            distribution = product_distribution(
                [
                    Uniform(demand_min, demand_max),
                    Uniform(demand_min, demand_max),
                ]
            )
        )
    )

    @constraint(ldr, [i=1:2], sell[i] + ret[i] <= buy[i])

    @constraint(ldr, [i=1:2], sell[i] <= demand[i])

    @objective(ldr, Max,
        sum(
            - buy_cost * buy[i]
            + return_value * ret[i]
            + sell_value * sell[i]
            for i in 1:2
        )
    )

    optimize!(ldr)

    ldr_p_obj = objective_value(ldr)

    # First-stage decisions do not depend on uncertainties
    for i in 1:2, j in 1:2
        @test LinearDecisionRules.get_decision(ldr, buy[i], demand[j]) == 0
        @test LinearDecisionRules.get_decision(ldr, buy[i], demand[j], dual = true) == 0
    end

    ldr_d_obj = objective_value(ldr, dual = true)

    return
end

function test_double_newsvendor_nonparametric()

    buy_cost = 10
    return_value = 8
    sell_value = 15

    demand_max = 120
    demand_min = 80

    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, buy[1:2] >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell[1:2] >= 0)
    @variable(ldr, ret[1:2] >= 0)
    @variable(ldr, demand[1:2] in
        LinearDecisionRules.Uncertainty(
            distribution = LinearDecisionRules.MvDiscreteNonParametric(
                [[demand_min, demand_min], [demand_max, demand_max]],
                [0.5, 0.5],
            )
        )
    )

    @constraint(ldr, [i=1:2], sell[i] + ret[i] <= buy[i])

    @constraint(ldr, [i=1:2], sell[i] <= demand[i])

    @objective(ldr, Max,
        sum(
            - buy_cost * buy[i]
            + return_value * ret[i]
            + sell_value * sell[i]
            for i in 1:2
        )
    )

    optimize!(ldr)

    ldr_p_obj = objective_value(ldr)

    # First-stage decisions do not depend on uncertainties
    for i in 1:2, j in 1:2
        @test LinearDecisionRules.get_decision(ldr, buy[i], demand[j]) == 0
        @test LinearDecisionRules.get_decision(ldr, buy[i], demand[j], dual = true) == 0
    end

    ldr_d_obj = objective_value(ldr, dual = true)

    return
end

function test_0_uniform()

    initial_volume = 0.5
    demand = 0.3

    m = LinearDecisionRules.LDRModel()
    set_silent(m)
    @variable(m, vi == initial_volume)
    @variable(m, 0 <= vf <= 1)
    @variable(m, gh >= 0.0)
    @variable(m, gt >= 0.0)
    @variable(m, inflow in LinearDecisionRules.Uncertainty(distribution=Uniform(0, 0.2)))

    @constraint(m, balance, vf == vi - gh + inflow)
    @constraint(m, gt + gh == demand)

    @objective(m, Min, gt^2 + vf^2/2 - vf)

    @test m[:vi] == vi

    data = LinearDecisionRules.matrix_data(m.cache_model)
    @test data.variables == [vi; vf; gh; gt; inflow]
    @test data.Q == LinearDecisionRules.SparseArrays.sparse([2, 4], [2, 4], [1/2, 1], 5, 5)
    @test data.sense == MOI.MIN_SENSE

    set_optimizer(m, Ipopt.Optimizer)
    LinearDecisionRules._prepare_data(m)
    LinearDecisionRules._solve_primal_ldr(m)

    LinearDecisionRules.get_decision(m, vf, inflow)
    LinearDecisionRules.get_decision(m, vf)

    @test LinearDecisionRules.get_decision(m, gh) + LinearDecisionRules.get_decision(m, gt) ≈ demand atol=1e-6
    @test LinearDecisionRules.get_decision(m, gh, inflow) + LinearDecisionRules.get_decision(m, gt, inflow) ≈ 0 atol=1e-6

    @test LinearDecisionRules.get_decision(m, vi) ≈ initial_volume atol=1e-6
    @test LinearDecisionRules.get_decision(m, vi, inflow) ≈ 0 atol=1e-6

end

function test_0_non_parametric()

    initial_volume = 0.5
    demand = 0.3

    m = LinearDecisionRules.LDRModel()
    set_silent(m)
    @variable(m, vi == initial_volume)
    @variable(m, 0 <= vf <= 1)
    @variable(m, gh >= 0.0)
    @variable(m, gt >= 0.0)
    @variable(m, inflow in LinearDecisionRules.Uncertainty(distribution=DiscreteNonParametric([0.0 , 0.2], [0.5, 0.5])))

    @constraint(m, balance, vf == vi - gh + inflow)
    @constraint(m, gt + gh == demand)

    @objective(m, Min, gt^2 + vf^2/2 - vf)

    set_optimizer(m, Ipopt.Optimizer)
    optimize!(m)

    LinearDecisionRules.get_decision(m, vf, inflow)
    LinearDecisionRules.get_decision(m, vf)

    @test LinearDecisionRules.get_decision(m, gh) + LinearDecisionRules.get_decision(m, gt) ≈ demand atol=1e-6
    @test LinearDecisionRules.get_decision(m, gh, inflow) + LinearDecisionRules.get_decision(m, gt, inflow) ≈ 0 atol=1e-6

    @test LinearDecisionRules.get_decision(m, vi) ≈ initial_volume atol=1e-6
    @test LinearDecisionRules.get_decision(m, vi, inflow) ≈ 0 atol=1e-6

end


function test_0_truncated_normal()

    initial_volume = 0.5
    demand = 0.3

    m = LinearDecisionRules.LDRModel()
    set_silent(m)
    @variable(m, vi == initial_volume)
    @variable(m, 0 <= vf <= 1)
    @variable(m, gh >= 0.0)
    @variable(m, gt >= 0.0)
    @variable(m, inflow in LinearDecisionRules.Uncertainty(distribution=truncated(Normal(0.1, 0.01), 0.0, 0.2)))

    @constraint(m, balance, vf == vi - gh + inflow)
    @constraint(m, gt + gh == demand)

    @objective(m, Min, gt^2 + vf^2/2 - vf)

    set_optimizer(m, Ipopt.Optimizer)
    optimize!(m)

    LinearDecisionRules.get_decision(m, vf, inflow)
    LinearDecisionRules.get_decision(m, vf)

    @test LinearDecisionRules.get_decision(m, gh) + LinearDecisionRules.get_decision(m, gt) ≈ demand atol=1e-6
    @test LinearDecisionRules.get_decision(m, gh, inflow) + LinearDecisionRules.get_decision(m, gt, inflow) ≈ 0 atol=1e-6

    @test LinearDecisionRules.get_decision(m, vi) ≈ initial_volume atol=1e-6
    @test LinearDecisionRules.get_decision(m, vi, inflow) ≈ 0 atol=1e-6

end

function test_1()

    initial_volume = 0.5
    demand = 0.3
    
    m = LinearDecisionRules.LDRModel()
    set_silent(m)
    @variable(m, vi in LinearDecisionRules.Uncertainty(distribution=Uniform(0, initial_volume)))
    @variable(m, 0 <= vf <= 1)
    @variable(m, gh >= 0.0)
    @variable(m, gt >= 0.0)
    @variable(m, inflow in LinearDecisionRules.Uncertainty(distribution=Uniform(0, 0.2)))
    
    @constraint(m, balance, vf == vi - gh + inflow)
    @constraint(m, gt + gh == demand)
    
    @objective(m, Min, gt^2 + vf^2/2 - vf)
    
    @test m[:vi] == vi
    
    data = LinearDecisionRules.matrix_data(m.cache_model)
    @test data.variables == [vi; vf; gh; gt; inflow]
    @test data.Q == LinearDecisionRules.SparseArrays.sparse([2, 4], [2, 4], [1/2, 1], 5, 5)
    @test data.sense == MOI.MIN_SENSE
    
    set_optimizer(m, Ipopt.Optimizer)
    LinearDecisionRules._prepare_data(m)
    LinearDecisionRules._solve_primal_ldr(m)
    
    LinearDecisionRules.get_decision(m, vf, inflow)
    LinearDecisionRules.get_decision(m, vf)
    
    @test LinearDecisionRules.get_decision(m, gh) + LinearDecisionRules.get_decision(m, gt) ≈ demand atol=1e-6
    @test LinearDecisionRules.get_decision(m, gh, inflow) + LinearDecisionRules.get_decision(m, gt, inflow) ≈ 0 atol=1e-6
    
end

# testing array based uncertainty output
function test_2()

    initial_volume = 0.5
    demand = 0.3

    m = LinearDecisionRules.LDRModel()
    set_silent(m)
    @variable(m, vi == initial_volume)
    @variable(m, 0 <= vf <= 1)
    @variable(m, gh >= 0.0)
    @variable(m, gt >= 0.0)
    @variable(m, inflow[i = 1:3] in LinearDecisionRules.Uncertainty(distribution=Uniform(0, 0.1 * i)))

    @constraint(m, balance, vf == vi - gh + sum(inflow[i] for i in 1:3))
    @constraint(m, gt + gh == demand)

    @objective(m, Min, gt^2 + vf^2/2 - vf)

    @test m[:vi] == vi

    set_optimizer(m, Ipopt.Optimizer)
    optimize!(m)
end

end # TestMain module

TestMain.runtests()