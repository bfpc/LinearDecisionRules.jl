module TestCore

include("common.jl")

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
    @test_throws ErrorException LinearDecisionRules.get_decision(
        m,
        x,
        dual = true,
    )
    # set new config
    set_attribute(m, LinearDecisionRules.SolveDual(), true)
    @test get_attribute(m, LinearDecisionRules.SolveDual()) == true
    set_attribute(m, LinearDecisionRules.SolvePrimal(), false)
    @test get_attribute(m, LinearDecisionRules.SolvePrimal()) == false
    optimize!(m)
    @test_throws ErrorException LinearDecisionRules.get_decision(m, x)
    @test LinearDecisionRules.get_decision(m, x; dual = true) == 1
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
    @test primal_status(m; dual = true) == MOI.FEASIBLE_POINT
    @test termination_status(m) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test termination_status(m; dual = true) in
          (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test LinearDecisionRules.get_decision(m, x) == 1
    @test LinearDecisionRules.get_decision(m, x; dual = true) == 1
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

    @objective(
        saa,
        Max,
        -buy_cost * buy +
        (1 / scenarios) *
        sum(return_value * ret[i] + sell_value * sell[i] for i in 1:scenarios)
    )

    optimize!(saa)

    saa_obj = objective_value(saa)

    # LDR

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

    optimize!(ldr)

    # Primal LDR is feasible:
    # underperforms adaptive solution (SAA) up to estimation errors
    ldr_p_obj = objective_value(ldr)
    @test saa_obj >= ldr_p_obj - 1e-6

    # First-stage decision do not depend on the uncertainty
    @test LinearDecisionRules.get_decision(ldr, buy, demand) == 0

    # Dual LDR is a performance bound:
    # SAA cannot yield better objective, up to estimation errors
    ldr_d_obj = objective_value(ldr; dual = true)
    @test saa_obj <= ldr_d_obj + 1e-6

    # First-stage decision do not depend on the uncertainty
    @test LinearDecisionRules.get_decision(ldr, buy, demand; dual = true) == 0

    return
end

function test_newsvendor_random_price()
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
            distribution = Uniform(demand_min, demand_max)
        )
    )
    @variable(ldr, sell_value2 in
            LinearDecisionRules.Uncertainty(
            distribution = Uniform(sell_value-2, sell_value+2)
        )
    )

    @constraint(ldr, sell + ret <= buy)
    @constraint(ldr, sell <= demand)

    @objective(ldr, Max,
        - buy_cost * buy
        + return_value * ret
        + sell_value2 * sell
    )

    optimize!(ldr)


    ldr_p_obj = objective_value(ldr)
    @test ldr_p_obj ≈ 460.0
    @test LinearDecisionRules.get_decision(ldr, buy, demand) == 0

    ldr_d_obj = objective_value(ldr, dual = true)
    @test ldr_d_obj ≈ 486.6666666666665
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
    @variable(
        ldr,
        demand[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = product_distribution([
                Uniform(demand_min, demand_max),
                Uniform(demand_min, demand_max),
            ]),
        )
    )

    @constraint(ldr, [i = 1:2], sell[i] + ret[i] <= buy[i])

    @constraint(ldr, [i = 1:2], sell[i] <= demand[i])

    @objective(
        ldr,
        Max,
        sum(
            -buy_cost * buy[i] + return_value * ret[i] + sell_value * sell[i]
            for i in 1:2
        )
    )

    optimize!(ldr)

    ldr_p_obj = objective_value(ldr)

    # First-stage decisions do not depend on uncertainties
    for i in 1:2, j in 1:2
        @test LinearDecisionRules.get_decision(ldr, buy[i], demand[j]) == 0
        @test LinearDecisionRules.get_decision(
            ldr,
            buy[i],
            demand[j];
            dual = true,
        ) == 0
    end

    # This problem is separable, so decision rules are independent by "product"
    for (i, j) in [(1, 2), (2, 1)]
        @test LinearDecisionRules.get_decision(ldr, sell[i], demand[j]) == 0
        @test LinearDecisionRules.get_decision(ldr, ret[i], demand[j]) == 0
        @test LinearDecisionRules.get_decision(
            ldr,
            sell[i],
            demand[j];
            dual = true,
        ) == 0
        @test LinearDecisionRules.get_decision(
            ldr,
            ret[i],
            demand[j];
            dual = true,
        ) == 0
    end

    ldr_d_obj = objective_value(ldr; dual = true)

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
    @variable(
        ldr,
        demand[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = product_distribution([
                Uniform(demand_min, demand_max),
                Uniform(demand_min, demand_max),
            ]),
        )
    )

    @constraint(ldr, [i = 1:2], sell[i] + ret[i] <= buy[i])

    @constraint(ldr, [i = 1:2], sell[i] <= demand[i])

    @objective(
        ldr,
        Max,
        sum(
            -buy_cost * buy[i] + return_value * ret[i] + sell_value * sell[i]
            for i in 1:2
        )
    )

    @constraint(ldr, demand[1] <= 110)
    @constraint(ldr, demand[1] >= 100)
    @constraint(ldr, demand[2] >= 100)
    @constraint(ldr, demand[2] <= 110)

    @test_logs (:warn, "Rejection sampling required") begin
      optimize!(ldr)
    end

    ldr_p_obj = objective_value(ldr)

    # First-stage decisions do not depend on uncertainties
    for i in 1:2, j in 1:2
        @test LinearDecisionRules.get_decision(ldr, buy[i], demand[j]) == 0
        @test LinearDecisionRules.get_decision(
            ldr,
            buy[i],
            demand[j];
            dual = true,
        ) == 0
    end

    # This problem is separable, so decision rules are independent by "product"
    for (i, j) in [(1, 2), (2, 1)]
        @test LinearDecisionRules.get_decision(ldr, sell[i], demand[j]) == 0
        @test LinearDecisionRules.get_decision(ldr, ret[i], demand[j]) == 0
        @test LinearDecisionRules.get_decision(
            ldr,
            sell[i],
            demand[j];
            dual = true,
        ) == 0
        @test LinearDecisionRules.get_decision(
            ldr,
            ret[i],
            demand[j];
            dual = true,
        ) == 0
    end

    ldr_d_obj = objective_value(ldr; dual = true)

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
    @variable(
        ldr,
        demand[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = LinearDecisionRules.MvDiscreteNonParametric(
                [[demand_min, demand_min], [demand_max, demand_max]],
                [0.5, 0.5],
            ),
        )
    )

    @constraint(ldr, [i = 1:2], sell[i] + ret[i] <= buy[i])

    @constraint(ldr, [i = 1:2], sell[i] <= demand[i])

    @objective(
        ldr,
        Max,
        sum(
            -buy_cost * buy[i] + return_value * ret[i] + sell_value * sell[i]
            for i in 1:2
        )
    )

    optimize!(ldr)

    ldr_p_obj = objective_value(ldr)

    # First-stage decisions do not depend on uncertainties
    for i in 1:2, j in 1:2
        @test LinearDecisionRules.get_decision(ldr, buy[i], demand[j]) == 0
        @test LinearDecisionRules.get_decision(
            ldr,
            buy[i],
            demand[j];
            dual = true,
        ) == 0
    end

    ldr_d_obj = objective_value(ldr; dual = true)

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
    @variable(
        ldr,
        demand in LinearDecisionRules.Uncertainty(;
            distribution = Uniform(demand_min, demand_max),
        )
    )

    @constraint(ldr, sell + ret <= buy)
    @constraint(ldr, sell <= demand)
    @objective(
        ldr,
        Max,
        -buy_cost * buy + return_value * ret + sell_value * sell
    )
    optimize!(ldr)

    ldr_p_obj = objective_value(ldr)
    ldr_d_obj = objective_value(ldr; dual = true)

    M1 = ldr.ext[:_LDR_M] # TODO add function to query this ??? (will need a map)

    @constraint(ldr, demand <= 110)
    @constraint(ldr, demand >= 100)

    @test_logs (:warn, "Rejection sampling required") begin
      optimize!(ldr)
    end

    ldr_p_obj2 = objective_value(ldr)
    @test ldr_p_obj < ldr_p_obj2
    ldr_d_obj2 = objective_value(ldr; dual = true)
    @test ldr_d_obj < ldr_d_obj2

    M2 = ldr.ext[:_LDR_M]

    @test M1[1, 1] == M2[1, 1] == 1
    @test M1[1, 2] == M1[2, 1] == 100
    @test M2[1, 2] == M2[2, 1]
    @test M2[1, 2] ≈ 105.0 atol = 1e-1

    @variable(
        ldr,
        demand2 in LinearDecisionRules.Uncertainty(;
            distribution = Uniform(demand_min, demand_max),
        )
    )

    @constraint(ldr, demand2 <= 110)
    @constraint(ldr, demand2 >= 100)

    @test_logs (:warn, "Rejection sampling required") begin
      optimize!(ldr)
    end

    ldr_p_obj3 = objective_value(ldr)
    @test ldr_p_obj3 ≈ ldr_p_obj2 rtol = 1e-2

    ldr_d_obj3 = objective_value(ldr; dual = true)
    @test ldr_d_obj3 ≈ ldr_d_obj2 rtol = 1e-2

    M3 = ldr.ext[:_LDR_M]
    @test M3[1, 1] == 1
    @test M3[3, 2] ≈ M3[2, 3] rtol = 1e-2
    @test M3[3, 2] ≈ M3[3, 1] * M3[1, 2] rtol = 1e-2
    @test M3[3, 3] ≈ M3[2, 2] rtol = 1e-2

    @constraint(ldr, sell <= demand2)

    @test_logs (:warn, "Rejection sampling required") begin
      optimize!(ldr)
    end

    objective_value(ldr; dual = true)

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
    @variable(
        m,
        inflow in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 0.2))
    )

    @constraint(m, balance, vf == vi - gh + inflow)
    @constraint(m, gt + gh == demand)

    @objective(m, Min, gt^2 + vf^2 / 2 - vf)

    @test m[:vi] == vi

    data = LinearDecisionRules.matrix_data(m.cache_model.model)
    @test data.variables == [vi; vf; gh; gt; inflow]
    @test data.Q == LinearDecisionRules.SparseArrays.sparse(
        [2, 4],
        [2, 4],
        [1 / 2, 1],
        5,
        5,
    )
    @test data.sense == MOI.MIN_SENSE

    set_optimizer(m, Ipopt.Optimizer)
    LinearDecisionRules._prepare_data(m)
    LinearDecisionRules._solve_primal_ldr(m)

    LinearDecisionRules.get_decision(m, vf, inflow)
    LinearDecisionRules.get_decision(m, vf)

    @test LinearDecisionRules.get_decision(m, gh) +
          LinearDecisionRules.get_decision(m, gt) ≈ demand atol = 1e-6
    @test LinearDecisionRules.get_decision(m, gh, inflow) +
          LinearDecisionRules.get_decision(m, gt, inflow) ≈ 0 atol = 1e-6

    @test LinearDecisionRules.get_decision(m, vi) ≈ initial_volume atol = 1e-6
    @test LinearDecisionRules.get_decision(m, vi, inflow) ≈ 0 atol = 1e-6
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
    @variable(
        m,
        inflow in LinearDecisionRules.Uncertainty(;
            distribution = DiscreteNonParametric([0.0, 0.2], [0.5, 0.5]),
        )
    )

    @constraint(m, balance, vf == vi - gh + inflow)
    @constraint(m, gt + gh == demand)

    @objective(m, Min, gt^2 + vf^2 / 2 - vf)

    set_optimizer(m, Ipopt.Optimizer)
    optimize!(m)

    LinearDecisionRules.get_decision(m, vf, inflow)
    LinearDecisionRules.get_decision(m, vf)

    @test LinearDecisionRules.get_decision(m, gh) +
          LinearDecisionRules.get_decision(m, gt) ≈ demand atol = 1e-6
    @test LinearDecisionRules.get_decision(m, gh, inflow) +
          LinearDecisionRules.get_decision(m, gt, inflow) ≈ 0 atol = 1e-6

    @test LinearDecisionRules.get_decision(m, vi) ≈ initial_volume atol = 1e-6
    @test LinearDecisionRules.get_decision(m, vi, inflow) ≈ 0 atol = 1e-6
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
    @variable(
        m,
        inflow in LinearDecisionRules.Uncertainty(;
            distribution = truncated(Normal(0.1, 0.01), 0.0, 0.2),
        )
    )

    @constraint(m, balance, vf == vi - gh + inflow)
    @constraint(m, gt + gh == demand)

    @objective(m, Min, gt^2 + vf^2 / 2 - vf)

    set_optimizer(m, Ipopt.Optimizer)
    optimize!(m)

    LinearDecisionRules.get_decision(m, vf, inflow)
    LinearDecisionRules.get_decision(m, vf)

    @test LinearDecisionRules.get_decision(m, gh) +
          LinearDecisionRules.get_decision(m, gt) ≈ demand atol = 1e-6
    @test LinearDecisionRules.get_decision(m, gh, inflow) +
          LinearDecisionRules.get_decision(m, gt, inflow) ≈ 0 atol = 1e-6

    @test LinearDecisionRules.get_decision(m, vi) ≈ initial_volume atol = 1e-6
    @test LinearDecisionRules.get_decision(m, vi, inflow) ≈ 0 atol = 1e-6
end

function test_1()
    initial_volume = 0.5
    demand = 0.3

    m = LinearDecisionRules.LDRModel()
    set_silent(m)
    @variable(
        m,
        vi in LinearDecisionRules.Uncertainty(;
            distribution = Uniform(0, initial_volume),
        )
    )
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

    @test m[:vi] == vi

    data = LinearDecisionRules.matrix_data(m.cache_model.model)
    @test data.variables == [vi; vf; gh; gt; inflow]
    @test data.Q == LinearDecisionRules.SparseArrays.sparse(
        [2, 4],
        [2, 4],
        [1 / 2, 1],
        5,
        5,
    )
    @test data.sense == MOI.MIN_SENSE

    set_optimizer(m, Ipopt.Optimizer)
    LinearDecisionRules._prepare_data(m)
    LinearDecisionRules._solve_primal_ldr(m)

    LinearDecisionRules.get_decision(m, vf, inflow)
    LinearDecisionRules.get_decision(m, vf)

    @test LinearDecisionRules.get_decision(m, gh) +
          LinearDecisionRules.get_decision(m, gt) ≈ demand atol = 1e-6
    @test LinearDecisionRules.get_decision(m, gh, inflow) +
          LinearDecisionRules.get_decision(m, gt, inflow) ≈ 0 atol = 1e-6
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
    @variable(
        m,
        inflow[i = 1:3] in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 0.1 * i))
    )

    @constraint(m, balance, vf == vi - gh + sum(inflow[i] for i in 1:3))
    @constraint(m, gt + gh == demand)

    @objective(m, Min, gt^2 + vf^2 / 2 - vf)

    @test m[:vi] == vi

    set_optimizer(m, Ipopt.Optimizer)
    return optimize!(m)
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

    optimize!(ldr)

    ldr_p_obj = Float64[]
    push!(ldr_p_obj, objective_value(ldr))
    ldr_d_obj = Float64[]
    push!(ldr_d_obj, objective_value(ldr; dual = true))

    @test ldr_p_obj[] <= ldr_d_obj[] + 1e-6

    return
end

function test_recursion()
    initial_volume = 0.5
    demand = 0.3

    m = LinearDecisionRules.LDRModel()

    set_silent(m)

    @variable(
        m,
        vi in LinearDecisionRules.Uncertainty(;
            distribution = Distributions.Uniform(
                initial_volume * 0.9,
                initial_volume * 1.1,
            ),
        )
    )
    @variable(m, 0 <= vf <= 1)
    @variable(m, gh >= 0.0)
    @variable(m, gt >= 0.0)
    @variable(
        m,
        inflow[i = 1:3] in LinearDecisionRules.Uncertainty(;
            distribution = Distributions.Uniform(0, 0.1 * i),
        )
    )

    @constraint(m, balance, vf == vi - gh + sum(inflow[i] for i in 1:3))
    @constraint(m, gt + gh == demand)

    @objective(m, Min, gt^2 + vf^2 / 2 - vf)

    set_optimizer(m, HiGHS.Optimizer)
    optimize!(m)

    m_0 = LinearDecisionRules.LDRModel()
    set_silent(m_0)
    LinearDecisionRules.set_parametric_objective!(m_0, m, Dict())

    m_2 = LinearDecisionRules.LDRModel()
    set_silent(m_2)

    @variable(m_2, vi_2 == initial_volume)
    @variable(m_2, 0 <= vf_2 <= 1)
    @variable(m_2, gh_2 >= 0.0)
    @variable(m_2, gt_2 >= 0.0)
    @variable(
        m_2,
        inflow_2[i = 1:3] in LinearDecisionRules.Uncertainty(;
            distribution = Distributions.Uniform(0, 0.1 * i),
        )
    )

    @constraint(
        m_2,
        balance_2,
        vf_2 == vi_2 - gh_2 + sum(inflow_2[i] for i in 1:3)
    )
    @constraint(m_2, gt_2 + gh_2 == demand)

    @objective(m_2, Min, gt_2^2)

    LinearDecisionRules.set_parametric_objective!(m_2, m, Dict(vi => vf_2))

    set_optimizer(m_2, Ipopt.Optimizer)
    optimize!(m_2)

    return
end

function test_recursion_pwl()
    initial_volume = 0.5
    demand = 0.3

    m = LinearDecisionRules.LDRModel()

    set_silent(m)

    @variable(
        m,
        vi in LinearDecisionRules.Uncertainty(;
            distribution = Distributions.Uniform(
                initial_volume * 0.9,
                initial_volume * 1.1,
            ),
        )
    )
    @variable(m, 0 <= vf <= 1)
    @variable(m, gh >= 0.0)
    @variable(m, gt >= 0.0)
    @variable(
        m,
        inflow[i = 1:3] in LinearDecisionRules.Uncertainty(;
            distribution = Distributions.Uniform(0, 0.1 * i),
        )
    )

    @constraint(m, balance, vf == vi - gh + sum(inflow[i] for i in 1:3))
    @constraint(m, gt + gh == demand)

    @objective(m, Min, gt^2 + vf^2 / 2 - vf)

    set_attribute(vi, LinearDecisionRules.BreakPoints(), 3)

    set_optimizer(m, HiGHS.Optimizer)
    optimize!(m)

    m_0 = LinearDecisionRules.LDRModel()
    set_silent(m_0)
    LinearDecisionRules.set_parametric_objective!(m_0, m, Dict())

    m_2 = LinearDecisionRules.LDRModel()
    set_silent(m_2)

    @variable(m_2, vi_2 == initial_volume)
    @variable(m_2, 0 <= vf_2 <= 1)
    @variable(m_2, gh_2 >= 0.0)
    @variable(m_2, gt_2 >= 0.0)
    @variable(
        m_2,
        inflow_2[i = 1:3] in LinearDecisionRules.Uncertainty(;
            distribution = Distributions.Uniform(0, 0.1 * i),
        )
    )

    @constraint(
        m_2,
        balance_2,
        vf_2 == vi_2 - gh_2 + sum(inflow_2[i] for i in 1:3)
    )
    @constraint(m_2, gt_2 + gh_2 == demand)

    @objective(m_2, Min, gt_2^2)

    LinearDecisionRules.set_parametric_objective!(m_2, m, Dict(vi => vf_2))

    set_optimizer(m_2, HiGHS.Optimizer)
    optimize!(m_2)

    return
end

function test_get_decision_invalid_inputs()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(m, y >= 0)
    @variable(
        m,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1)),
    )
    @objective(m, Min, y)
    optimize!(m)

    # η is not an uncertainty variable
    @test_throws ArgumentError LinearDecisionRules.get_decision(m, y, x)
    # x is an uncertainty variable (passed as the decision)
    @test_throws ArgumentError LinearDecisionRules.get_decision(
        m,
        demand,
        demand,
    )
    # single-arg form: x is an uncertainty variable
    @test_throws ArgumentError LinearDecisionRules.get_decision(m, demand)

    return nothing
end

function test_delete_not_allowed()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @constraint(m, con, x >= 0)

    @test_throws MOI.DeleteNotAllowed JuMP.delete(m, x)
    @test_throws MOI.DeleteNotAllowed JuMP.delete(m, con)

    return nothing
end

function test_raw_status()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @objective(m, Min, x)
    optimize!(m)

    @test raw_status(m) == raw_status(m.primal_model)
    @test raw_status(m; dual = true) == raw_status(m.dual_model)
    set_attribute(m, LinearDecisionRules.SolvePrimal(), false)
    @test raw_status(m) == MOI.NO_SOLUTION
    return nothing
end

function test_unset_silent()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @objective(m, Min, x)
    optimize!(m)
    unset_silent(m)
    @test m.silent == false
    return nothing
end

function test_solution_summary()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @objective(m, Min, x)
    optimize!(m)

    s = solution_summary(m)
    @test s.primal == solution_summary(m.primal_model)
    @test s.dual == solution_summary(m.dual_model)

    set_attribute(m, LinearDecisionRules.SolvePrimal(), false)
    optimize!(m)
    s2 = solution_summary(m)
    @test s2.primal === nothing
    io = IOBuffer()
    show(io, s2)
    @test occursin("disabled", String(take!(io)))
    return nothing
end

function test_objective_value_disabled()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @objective(m, Min, x)

    set_attribute(m, LinearDecisionRules.SolvePrimal(), false)
    optimize!(m)
    @test_throws ErrorException objective_value(m)

    set_attribute(m, LinearDecisionRules.SolvePrimal(), true)
    set_attribute(m, LinearDecisionRules.SolveDual(), false)
    optimize!(m)
    @test_throws ErrorException objective_value(m; dual = true)
    return nothing
end

function test_optimize_no_solver()
    m = LinearDecisionRules.LDRModel()
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @objective(m, Min, x)
    @test_throws ErrorException optimize!(m)
    return nothing
end

function test_uncertainty_invalid()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    @test_throws MethodError LinearDecisionRules.Uncertainty()
    @test_throws ErrorException @variable(
        m,
        xinv in LinearDecisionRules.Uncertainty(; distribution = Normal(0, 1)),
    )
    m2 = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    @test_throws ErrorException @variable(
        m2,
        y >= 0,
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1)),
    )
    return nothing
end

function test_vector_uncertainty_bounds_errors()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    d = product_distribution([Uniform(0, 1), Uniform(0, 1)])
    @test_throws ErrorException @variable(
        m,
        xvec[1:2] >= 0,
        LinearDecisionRules.Uncertainty(; distribution = d),
    )
    return nothing
end

function test_get_decision_disabled_modes()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @objective(m, Min, x)

    set_attribute(m, LinearDecisionRules.SolveDual(), false)
    optimize!(m)
    @test_throws ErrorException LinearDecisionRules.get_decision(
        m,
        x,
        demand;
        dual = true,
    )

    set_attribute(m, LinearDecisionRules.SolveDual(), true)
    set_attribute(m, LinearDecisionRules.SolvePrimal(), false)
    optimize!(m)
    @test_throws ErrorException LinearDecisionRules.get_decision(m, x, demand)
    return nothing
end

function test_cross_model_get_decision()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @objective(m, Min, x)
    optimize!(m)

    m2 = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    @variable(m2, y >= 0)
    @test_throws ArgumentError LinearDecisionRules.get_decision(m, y, demand)
    @test_throws ArgumentError LinearDecisionRules.get_decision(m, y)
    return nothing
end

function test_integer_not_first_stage()
    m_bin = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m_bin)
    @variable(m_bin, x, Bin)
    @variable(
        m_bin,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1)),
    )
    @objective(m_bin, Min, x)
    @test_throws ErrorException optimize!(m_bin)

    m_int = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m_int)
    @variable(m_int, y, Int)
    @variable(
        m_int,
        demand2 in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1)),
    )
    @objective(m_int, Min, y)
    @test_throws ErrorException optimize!(m_int)
    return nothing
end

function test_recursion_value_function_twice()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(
        m,
        vi in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0.4, 0.6)),
    )
    @variable(m, 0 <= vf <= 1)
    @objective(m, Min, vf^2)
    optimize!(m)

    m_new = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m_new)
    @variable(m_new, vi_new == 0.5)
    @variable(m_new, 0 <= vf_new <= 1)
    @objective(m_new, Min, 0)
    LinearDecisionRules.set_parametric_objective!(m_new, m, Dict(vi => vf_new))
    @test_throws ErrorException LinearDecisionRules.set_parametric_objective!(
        m_new,
        m,
        Dict(vi => vf_new),
    )
    return nothing
end

function test_model_api_queries()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @constraint(m, con, x >= 0.5)
    @objective(m, Min, x)

    @test JuMP.num_variables(m) == 2
    @test length(JuMP.all_variables(m)) == 2
    @test JuMP.variable_by_name(m, "x") == x
    @test JuMP.constraint_by_name(m, "con") !== nothing
    @test JuMP.objective_sense(m) == MOI.MIN_SENSE
    @test JuMP.objective_function_type(m) == JuMP.VariableRef
    @test JuMP.objective_function(m) == x
    @test JuMP.objective_function(m, JuMP.VariableRef) == x

    @test Base.broadcastable(m) isa Ref
    @test JuMP.is_valid(m, con)
    @test JuMP.dual_status(m) == MOI.NO_SOLUTION
    @test JuMP.variable_ref_type(m) == JuMP.VariableRef

    cs = JuMP.all_constraints(m, JuMP.VariableRef, MOI.GreaterThan{Float64})
    @test length(cs) >= 1
    JuMP.delete(m, JuMP.VariableRef[])
    JuMP.delete(m, typeof(con)[])
    @test JuMP.num_constraints(m, JuMP.VariableRef, MOI.GreaterThan{Float64}) >=
          1
    @test JuMP.num_constraints(m; count_variable_in_set_constraints = true) >= 1

    @variable(m, z in MOI.ZeroOne())  # VariableConstrainedOnCreation path
    @test JuMP.is_valid(m, z)

    io = IOBuffer()
    print(io, m)
    @test !isempty(String(take!(io)))
    return nothing
end

function test_status_functions_disabled()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @objective(m, Min, x)

    set_attribute(m, LinearDecisionRules.SolveDual(), false)
    optimize!(m)
    @test termination_status(m; dual = true) == MOI.OPTIMIZE_NOT_CALLED
    @test primal_status(m; dual = true) == MOI.NO_SOLUTION
    @test raw_status(m; dual = true) == MOI.NO_SOLUTION

    set_attribute(m, LinearDecisionRules.SolveDual(), true)
    set_attribute(m, LinearDecisionRules.SolvePrimal(), false)
    optimize!(m)
    @test termination_status(m) == MOI.OPTIMIZE_NOT_CALLED
    @test primal_status(m) == MOI.NO_SOLUTION
    return nothing
end

function test_solution_summary_dual_disabled_show()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @objective(m, Min, x)
    set_attribute(m, LinearDecisionRules.SolveDual(), false)
    optimize!(m)

    s = solution_summary(m)
    @test s.dual === nothing
    io = IOBuffer()
    show(io, s)
    @test occursin("disabled", String(take!(io)))
    return nothing
end

function test_first_stage_binary()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x, Bin, LinearDecisionRules.FirstStage)
    @variable(
        m,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @objective(m, Min, x)
    optimize!(m)
    @test termination_status(m) == MOI.OPTIMAL
    @test LinearDecisionRules.get_decision(m, x) in [0.0, 1.0]
    return nothing
end

function test_lower_bound_constraint()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, 0.0 <= x <= 10.0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @constraint(m, x >= 0.5 * demand)
    @objective(m, Min, x)
    optimize!(m)
    @test termination_status(m) == MOI.OPTIMAL
    return nothing
end

function test_get_decision_cross_model_eta()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @objective(m, Min, x)
    optimize!(m)

    m2 = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    @variable(
        m2,
        demand2 in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1)),
    )
    @test_throws ArgumentError LinearDecisionRules.get_decision(m, x, demand2)
    return nothing
end

function test_vector_distribution_infinite_bounds()
    # MvNormal has -Inf lower bounds, which are not supported
    m1 = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    @test_throws ErrorException @variable(
        m1,
        mv[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]),
        )
    )

    # Exponential has +Inf upper bound, which is not supported
    m2 = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    @test_throws ErrorException @variable(
        m2,
        pe[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = product_distribution([
                Uniform(0, 1),
                Exponential(1),
            ]),
        )
    )
    return nothing
end

function test_set_objective_after_parametric()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(
        m,
        vi in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0.4, 0.6))
    )
    @variable(m, 0 <= vf <= 1)
    @objective(m, Min, vf^2)
    optimize!(m)

    m_new = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m_new)
    @variable(m_new, vi_new == 0.5)
    @variable(m_new, 0 <= vf_new <= 1)
    @objective(m_new, Min, 0)
    LinearDecisionRules.set_parametric_objective!(m_new, m, Dict(vi => vf_new))
    # setting objective again after parametric objective has been set should error
    @test_throws ErrorException JuMP.set_objective_function(m_new, vf_new + 1.0)
    return nothing
end

function test_jump_show_wrappers()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @constraint(m, x >= 0.5)
    @objective(m, Min, x)

    io = IOBuffer()
    JuMP.show_backend_summary(io, m)
    JuMP.show_objective_function_summary(io, m)
    JuMP.show_constraints_summary(io, m)
    @test !isempty(String(take!(io)))
    r1 = JuMP.objective_function_string(MIME("text/plain"), m)
    @test !isempty(r1)
    r2 = JuMP.constraints_string(MIME("text/plain"), m)
    @test length(r2) >= 1
    return nothing
end

function test_exponential_uncertainty_error()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    @test_throws ErrorException @variable(
        m,
        d in LinearDecisionRules.Uncertainty(; distribution = Exponential(1))
    )
    return nothing
end

function test_interval_constraint()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x, LinearDecisionRules.FirstStage)  # no bounds so MOI.Interval works
    @variable(m, y >= 0)
    @variable(
        m,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @constraint(m, x in MOI.Interval(0.0, 2.0))   # variable interval constraint
    @constraint(m, 0.0 <= 1.0 * x <= 2.0)          # affine interval constraint
    @constraint(m, y <= demand)
    @objective(m, Max, y - x)
    optimize!(m)
    @test termination_status(m) == MOI.OPTIMAL
    return nothing
end

function test_feasibility_sense_error()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        d in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    # no @objective → FEASIBILITY_SENSE, which is not supported
    @test_throws ErrorException optimize!(m)
    return nothing
end

function test_uncertainty_constraint_warnings()
    # equality constraint on uncertainty is not valid and should warn
    m1 = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m1)
    @variable(m1, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m1,
        d in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @constraint(m1, d == 0.5)
    @objective(m1, Min, x)
    @test_logs (:warn, "pure equality constraint on uncertainty variables is not valid") begin
      optimize!(m1)
    end
    @test termination_status(m1) == MOI.OPTIMAL

    # interval constraint on uncertainty is not valid and should warn
    m2 = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m2)
    @variable(m2, x2 >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m2,
        d2 in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @constraint(m2, 0.2 <= d2 <= 0.8)
    @objective(m2, Min, x2)
    @test_logs (:warn, "pure interval constraint on uncertainty variables is not valid") begin
      optimize!(m2)
    end
    @test termination_status(m2) == MOI.OPTIMAL
    return nothing
end

function test_confidence_mv_normal_with_pwl()
    # ConfidenceMvNormal as vector uncertainty + scalar PWL breakpoints in same model.
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, buy >= 0, LinearDecisionRules.FirstStage)
    @variable(m, sell >= 0)
    μ = [80.0, 60.0]
    Σ = [100.0 0.0; 0.0 100.0]
    @variable(
        m,
        mv[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = LinearDecisionRules.ConfidenceMvNormal(μ, Σ, 0.90),
        )
    )
    @variable(
        m,
        scalar_d in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(70.0, 100.0)),
    )
    @constraint(m, sell <= buy)
    @constraint(m, sell <= mv[1] + scalar_d)
    @objective(m, Max, -buy + sell)
    set_attribute(scalar_d, LinearDecisionRules.BreakPoints(), 2)
    optimize!(m)
    @test termination_status(m) == MOI.OPTIMAL
    return nothing
end

function test_get_decision_invalid_eta()
    # η exists in the uncertainty dict but has been deleted from the inner model
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1)),
    )
    @objective(m, Min, x)
    # Manually destroy: delete demand from the inner model while keeping it in the dict
    JuMP.delete(m.cache_model.model, demand)
    # demand IS in uncertainty_to_distribution but is_valid(m, demand) = false
    @test_throws ArgumentError LinearDecisionRules.get_decision(m, x, demand)
    return nothing
end

function test_vector_uncertainty_non_multivariate()
    # VectorUncertainty requires a MultivariateDistribution
    @test_throws ErrorException LinearDecisionRules.VectorUncertainty(
        Uniform(0, 1),
    )
    return nothing
end

function test_variables_constrained_on_creation()
    # variables constrained on creation (e.g. MOI.Nonnegatives) are forwarded to inner model
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x[1:2] in MOI.Nonnegatives(2))
    @test length(x) == 2
    return nothing
end

function test_matrix_data_unsupported_constraint()
    # nonlinear constraints are not supported by matrix_data
    inner = JuMP.Model()
    @variable(inner, x)
    @constraint(inner, sin(x) <= 1)
    @objective(inner, Min, x)
    @test_throws ErrorException LinearDecisionRules.matrix_data(inner)
    return nothing
end

function test_matrix_data_unsupported_objective()
    # nonlinear objectives are not supported by matrix_data
    inner = JuMP.Model()
    @variable(inner, x)
    @objective(inner, Min, sin(x))
    @test_throws ErrorException LinearDecisionRules.matrix_data(inner)
    return nothing
end

end # TestCore module

TestCore.runtests()
