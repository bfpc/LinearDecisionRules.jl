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

    # This problem is separable, so decision rules are independent by "product"
    for (i,j) in [(1,2), (2,1)]
        @test LinearDecisionRules.get_decision(ldr, sell[i], demand[j]) == 0
        @test LinearDecisionRules.get_decision(ldr, ret[i], demand[j]) == 0
        @test LinearDecisionRules.get_decision(ldr, sell[i], demand[j], dual = true) == 0
        @test LinearDecisionRules.get_decision(ldr, ret[i], demand[j], dual = true) == 0
    end

    ldr_d_obj = objective_value(ldr, dual = true)

    return
end

function test_double_newsvendor_with_rejection()

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

    @constraint(ldr, demand[1] <= 110)
    @constraint(ldr, demand[1] >= 100)
    @constraint(ldr, demand[2] >= 100)
    @constraint(ldr, demand[2] <= 110)

    optimize!(ldr)

    ldr_p_obj = objective_value(ldr)

    # First-stage decisions do not depend on uncertainties
    for i in 1:2, j in 1:2
        @test LinearDecisionRules.get_decision(ldr, buy[i], demand[j]) == 0
        @test LinearDecisionRules.get_decision(ldr, buy[i], demand[j], dual = true) == 0
    end

    # This problem is separable, so decision rules are independent by "product"
    for (i,j) in [(1,2), (2,1)]
        @test LinearDecisionRules.get_decision(ldr, sell[i], demand[j]) == 0
        @test LinearDecisionRules.get_decision(ldr, ret[i], demand[j]) == 0
        @test LinearDecisionRules.get_decision(ldr, sell[i], demand[j], dual = true) == 0
        @test LinearDecisionRules.get_decision(ldr, ret[i], demand[j], dual = true) == 0
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

function test_newsvendor_with_rejection_sampling()

    buy_cost = 10
    return_value = 8
    sell_value = 15

    demand_max = 120
    demand_min = 80

    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell >= 0)
    @variable(ldr, ret >= 0)
    @variable(ldr, demand in
        LinearDecisionRules.Uncertainty(
            distribution = Uniform(demand_min, demand_max),
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

    ldr_p_obj = objective_value(ldr)
    ldr_d_obj = objective_value(ldr, dual = true)

    M1 = ldr.ext[:_LDR_M] # TODO add function to query this ??? (will need a map)

    @constraint(ldr, demand <= 110)
    @constraint(ldr, demand >= 100)

    optimize!(ldr)

    ldr_p_obj2 = objective_value(ldr)
    @test ldr_p_obj < ldr_p_obj2
    ldr_d_obj2 = objective_value(ldr, dual = true)
    @test ldr_d_obj < ldr_d_obj2

    M2 = ldr.ext[:_LDR_M] 

    @test M1[1,1] == M2[1,1] == 1
    @test M1[1,2] == M1[2,1] == 100
    @test M2[1,2] == M2[2,1]
    @test M2[1,2] ≈ 105.0 atol = 1e-1

    @variable(ldr, demand2 in
        LinearDecisionRules.Uncertainty(
            distribution = Uniform(demand_min, demand_max),
        )
    )

    @constraint(ldr, demand2 <= 110)
    @constraint(ldr, demand2 >= 100)

    optimize!(ldr)

    ldr_p_obj3 = objective_value(ldr)
    @test ldr_p_obj3 == ldr_p_obj2

    ldr_d_obj3 = objective_value(ldr, dual = true) 
    @test ldr_d_obj3 == ldr_d_obj2

    @show M3 = ldr.ext[:_LDR_M] 
    @test M3[1,1] == 1
    @test M3[3,2] == M3[2,3] == M3[3,1] * M3[1,2]
    @test M3[3,3] == M3[2,2]

    @constraint(ldr, sell <= demand2)
    optimize!(ldr)
    @show objective_value(ldr, dual = true) 

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

    data = LinearDecisionRules.matrix_data(m.cache_model.model)
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
    
    data = LinearDecisionRules.matrix_data(m.cache_model.model)
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

function test_piecewise_distribution()

    o_dist = Distributions.Uniform(1.0, 4.0)

    p_dist = LinearDecisionRules.UnivariatePieceWise(o_dist, [2.0, 3.0])

    len = length(p_dist)

    nu_mean = Distributions.mean(p_dist)
    nu_cov = Distributions.cov(p_dist)

    rng = Random.MersenneTwister(123)

    N = 1_000_000

    mc_mean = zeros(len)
    mc_cov = zeros(len, len)
    x = zeros(len)
    for i in 1:N
        fill!(x, 0.0)
        Random.rand!(rng, p_dist, x)
        mc_mean .+= x
        mc_cov .+= x * x'
    end
    mc_mean ./= N
    mc_cov ./= N
    mc_cov .-= mc_mean * mc_mean'

    # @show nu_mean
    # @show mc_mean
    # @show nu_cov
    # @show mc_cov

    @test nu_mean ≈ mc_mean atol = 1e-2
    @test nu_cov ≈ mc_cov atol = 1e-2

    return
end

function test_newsvendor_piecewise()
    buy_cost = 10
    return_value = 8
    sell_value = 15

    demand_max = 120
    demand_min = 80
    demand_distr = Distributions.Uniform(demand_min, demand_max)

    # LDR

    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)
    # unset_silent(ldr)

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

    ldr_p_obj = Float64[]
    push!(ldr_p_obj, objective_value(ldr))
    ldr_d_obj = Float64[]
    push!(ldr_d_obj, objective_value(ldr, dual = true))

    @test ldr_p_obj[] <= ldr_d_obj[] + 1e-6

    for n_intervals in 2:12
        set_attribute(demand, LinearDecisionRules.BreakPoints(), n_intervals-1)
        optimize!(ldr)
        push!(ldr_p_obj, objective_value(ldr))
        push!(ldr_d_obj, objective_value(ldr, dual = true))
        @test ldr_p_obj[end] <= ldr_d_obj[end] + 1e-6
        for j in 1:n_intervals÷2
            # Divisibility for LDR improvement
            if n_intervals % j != 0
                continue
            end
            @test ldr_p_obj[end] >= ldr_p_obj[j] - 1e-6
            @test ldr_d_obj[end] <= ldr_d_obj[j] + 1e-6
        end
    end

    return
end

function test_newsvendor_integer()
    buy_cost = 10
    return_value = 8
    sell_value = 15

    demand_max = 120
    demand_min = 80
    demand_distr = Distributions.Uniform(demand_min, demand_max)

    # LDR

    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)
    # unset_silent(ldr)

    @variable(ldr, buy >= 0, LinearDecisionRules.FirstStage, integer = true)
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

    ldr_p_obj = Float64[]
    push!(ldr_p_obj, objective_value(ldr))
    ldr_d_obj = Float64[]
    push!(ldr_d_obj, objective_value(ldr, dual = true))

    @test ldr_p_obj[] <= ldr_d_obj[] + 1e-6

    return
end

end # TestMain module

TestMain.runtests()
