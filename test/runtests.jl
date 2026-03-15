module TestMain

using Test
using Random
using Logging

using LinearDecisionRules

using JuMP
using Ipopt
using HiGHS
using Distributions
using LinearAlgebra

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

    optimize!(ldr)

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

    optimize!(ldr)

    ldr_p_obj3 = objective_value(ldr)
    @test ldr_p_obj3 == ldr_p_obj2

    ldr_d_obj3 = objective_value(ldr; dual = true)
    @test ldr_d_obj3 == ldr_d_obj2

    @show M3 = ldr.ext[:_LDR_M]
    @test M3[1, 1] == 1
    @test M3[3, 2] == M3[2, 3] == M3[3, 1] * M3[1, 2]
    @test M3[3, 3] == M3[2, 2]

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

    for n_intervals in 2:12
        set_attribute(
            demand,
            LinearDecisionRules.BreakPoints(),
            n_intervals - 1,
        )
        optimize!(ldr)
        push!(ldr_p_obj, objective_value(ldr))
        push!(ldr_d_obj, objective_value(ldr; dual = true))
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
    @show LinearDecisionRules.set_parametric_objective!(m_0, m, Dict())

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

    @show LinearDecisionRules.set_parametric_objective!(
        m_2,
        m,
        Dict(vi => vf_2),
    )

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
    @show LinearDecisionRules.set_parametric_objective!(m_0, m, Dict())

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

    @show LinearDecisionRules.set_parametric_objective!(
        m_2,
        m,
        Dict(vi => vf_2),
    )

    set_optimizer(m_2, HiGHS.Optimizer)
    optimize!(m_2)

    return
end

function test_confidence_mv_normal()
    # --- Unit tests on the distribution object ---
    μ = [1.0, 2.0]
    Σ = [1.0 0.5; 0.5 2.0]
    α = 0.90
    d = LinearDecisionRules.ConfidenceMvNormal(μ, Σ, α)

    @test Distributions.length(d) == 2
    @test Distributions.mean(d) ≈ μ

    # Covariance should be a positive scalar multiple of Σ
    cov_d = Distributions.cov(d)
    ratio = cov_d ./ Σ
    @test all(isapprox.(ratio, ratio[1, 1]; rtol = 1e-10))
    # Scaling must be < 1 (truncation reduces variance)
    @test ratio[1, 1] < 1.0
    @test ratio[1, 1] > 0.0

    # Bounds: μ_k ± ρ·√Σ_kk
    ρ = sqrt(Distributions.quantile(Distributions.Chisq(2), α))
    @test Distributions.minimum(d) ≈ μ - ρ .* sqrt.(LinearAlgebra.diag(Σ))
    @test Distributions.maximum(d) ≈ μ + ρ .* sqrt.(LinearAlgebra.diag(Σ))

    # insupport: centre is always in the ellipsoid
    @test Distributions.insupport(d, μ)
    # A point far outside should not be in support
    @test !Distributions.insupport(d, μ + [1000.0, 1000.0])

    # Sampling: empirical mean should be close to μ
    rng = Random.MersenneTwister(42)
    N = 50_000
    samples = [rand(rng, d) for _ in 1:N]
    emp_mean = sum(samples) / N
    @test norm(emp_mean - μ) < 0.05

    # --- 1-D case: compare with analytic truncated normal ---
    # For d=1, Σ=[σ²], α gives ρ = σ·z_{(1+α)/2}
    σ = 2.0
    d1 = LinearDecisionRules.ConfidenceMvNormal([0.0], [σ^2;;], 0.95)
    trunc_dist = truncated(Normal(0.0, σ), -d1.ρ * σ, d1.ρ * σ)
    @test Distributions.var(d1)[1] ≈ Distributions.var(trunc_dist) rtol = 1e-4
    @test Distributions.params(d)[1] == μ

    # --- Integration with LDR: 2-D newsvendor ---
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)
    set_attribute(ldr, LinearDecisionRules.SolveDual(), false)

    μ_d = [100.0, 80.0]
    Σ_d = [100.0 20.0; 20.0 64.0]
    dist_ldr = LinearDecisionRules.ConfidenceMvNormal(μ_d, Σ_d, 0.95)

    @variable(ldr, buy[1:2] >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell[1:2] >= 0)
    @variable(
        ldr,
        demand[1:2] in
        LinearDecisionRules.Uncertainty(; distribution = dist_ldr)
    )

    @constraint(ldr, [i = 1:2], sell[i] <= buy[i])
    @constraint(ldr, [i = 1:2], sell[i] <= demand[i])
    @objective(ldr, Max, sum(-10 * buy[i] + 15 * sell[i] for i in 1:2))

    optimize!(ldr)
    @test termination_status(ldr) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    return nothing
end

function test_confidence_mv_normal_bounds()
    # The ellipsoid in original space is {x : (x-μ)'Σ⁻¹(x-μ) ≤ ρ²}.
    # Via Σ = LL', this is the image of the ball B(0,ρ) under x = μ + Lz.
    # The bounding box is lb[k] = μ[k] - ρ√Σ_kk, ub[k] = μ[k] + ρ√Σ_kk.
    # We test two properties:
    #   1. Containment: every point in the ellipsoid lies within [lb, ub].
    #   2. Tightness:   each face of the box is touched by the ellipsoid.
    test_cases = [
        ([0.0, 0.0], [1.0 0.0; 0.0 1.0], 0.95),           # identity covariance
        ([100.0, 80.0], [100.0 40.0; 40.0 64.0], 0.95),    # correlated 2-D
        ([1.0, 2.0, 3.0], [2.0 0.5 0.1; 0.5 1.0 0.3; 0.1 0.3 1.5], 0.80),  # 3-D
    ]
    for (μ_tc, Σ_tc, α) in test_cases
        dist = LinearDecisionRules.ConfidenceMvNormal(μ_tc, Σ_tc, α)
        lb = Distributions.minimum(dist)
        ub = Distributions.maximum(dist)
        n = length(μ_tc)
        ρ = dist.ρ
        L = dist.L

        # --- Containment check 1: deterministic grid ---
        # Sweep a uniform grid over the unit ball in z-space (‖z‖ ≤ 1),
        # scale by ρ, then map to x = μ + L(ρz).  Every resulting x must
        # lie inside [lb, ub].
        grid = range(-1, 1; length = 30)
        violations = 0
        for z_unit in Iterators.product(fill(grid, n)...)
            z = collect(z_unit)
            norm(z) > 1 && continue   # keep only points inside the unit ball
            x = μ_tc + L * (ρ .* z)
            if any(x .< lb .- 1e-10) || any(x .> ub .+ 1e-10)
                violations += 1
            end
        end
        @test violations == 0

        # --- Containment check 2: random ellipsoid boundary ---
        # Draw a random unit vector, scale to ‖z‖ = ρ (boundary of the ball),
        # then map to x = μ + Lz.  Boundary points are the hardest cases for
        # the box constraint, so this is a strong probabilistic check.
        rng = Random.MersenneTwister(42)
        violations_sampled = 0
        for _ in 1:1000
            z = randn(rng, n)
            z .*= ρ / norm(z)   # scale to ellipsoid boundary
            x = μ_tc + L * z
            if any(x .< lb .- 1e-10) || any(x .> ub .+ 1e-10)
                violations_sampled += 1
            end
        end
        @test violations_sampled == 0

        # --- Containment check 3: samples from the distribution ---
        # rand(dist) uses rejection sampling internally (draw z ~ N(0,I),
        # accept if ‖z‖² ≤ ρ², return μ + Lz).  Samples are strictly inside
        # the ellipsoid, so they must also be inside [lb, ub].
        violations_dist = 0
        for _ in 1:1000
            x = rand(rng, dist)
            if any(x .< lb .- 1e-10) || any(x .> ub .+ 1e-10)
                violations_dist += 1
            end
        end
        @test violations_dist == 0

        # --- Tightness: each bound is achieved by an analytic tangent point ---
        # To show ub[k] = μ[k] + ρ√Σ_kk is tight we need a point on the
        # ellipsoid boundary whose k-th coordinate equals ub[k].
        #
        # Maximise eₖ'x = eₖ'(μ + Lz) subject to ‖z‖ = ρ.
        # The maximum of a linear function on a sphere is achieved in the
        # direction of the gradient, so z* = ρ · L'eₖ / ‖L'eₖ‖.
        #
        # The achieved value is:
        #   x*[k] = μ[k] + eₖ'L z* = μ[k] + ρ · eₖ'LL'eₖ / ‖L'eₖ‖
        #         = μ[k] + ρ · ‖L'eₖ‖ = μ[k] + ρ√Σ_kk = ub[k].
        for k in 1:n
            ek = zeros(n)
            ek[k] = 1.0
            Lt_ek = L' * ek                        # gradient direction in z-space
            z_star = ρ * Lt_ek / norm(Lt_ek)       # tangent point on ball boundary
            x_star = μ_tc + L * z_star             # map back to original space
            @test x_star[k] ≈ ub[k] rtol = 1e-10
        end
    end
    return nothing
end

function test_confidence_mv_normal_rotated_box()
    # The rotated (principal-axis) box gives a tighter outer approximation of the
    # ellipsoidal support than the axis-aligned box.  This test verifies that:
    #   1. No rejection-sampling warning is emitted (implied constraints bypass
    #      _compute_groups entirely).
    #   2. The dual is solved successfully with a finite bound.
    #   3. Wu_implied has the correct shape: 2d rows (d upper + d lower halfspaces).
    #   4. For a correlated distribution the dual bound is at least as tight as
    #      that obtained with an uncorrelated distribution of the same marginals.
    μ = [100.0, 80.0]
    Σ_corr = [100.0 40.0; 40.0 64.0]   # off-diagonal → rotated box strictly tighter
    α = 0.95

    # --- (a) correlated demands ---
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)
    dist = LinearDecisionRules.ConfidenceMvNormal(μ, Σ_corr, α)
    @variable(ldr, buy[1:2] >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell[1:2] >= 0)
    @variable(ldr, demand[1:2] in LinearDecisionRules.Uncertainty(distribution = dist))
    @constraint(ldr, [i = 1:2], sell[i] <= buy[i])
    @constraint(ldr, [i = 1:2], sell[i] <= demand[i])
    @objective(ldr, Max, sum(-10 * buy[i] + 15 * sell[i] for i in 1:2))

    # Verify no rejection-sampling warning is emitted.
    @test_logs min_level = Logging.Warn optimize!(ldr)

    @test termination_status(ldr; dual = true) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    obj_corr = objective_value(ldr; dual = true)
    @test isfinite(obj_corr)

    # Wu_implied should have 2d = 4 rows and d = 2 uncertainty columns.
    ABC = ldr.ext[:_LDR_ABC]
    @test size(ABC.Wu_implied, 1) == 4
    @test size(ABC.Wu_implied, 2) == 2

    # --- (b) uncorrelated demands (Σ diagonal) ---
    # With diagonal Σ the rotated box coincides with the axis-aligned box, so
    # Wu_implied adds no new information beyond what lb/ub already encode.
    Σ_diag = [100.0 0.0; 0.0 64.0]
    ldr2 = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr2)
    dist2 = LinearDecisionRules.ConfidenceMvNormal(μ, Σ_diag, α)
    @variable(ldr2, buy2[1:2] >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr2, sell2[1:2] >= 0)
    @variable(ldr2, demand2[1:2] in LinearDecisionRules.Uncertainty(distribution = dist2))
    @constraint(ldr2, [i = 1:2], sell2[i] <= buy2[i])
    @constraint(ldr2, [i = 1:2], sell2[i] <= demand2[i])
    @objective(ldr2, Max, sum(-10 * buy2[i] + 15 * sell2[i] for i in 1:2))
    @test_logs min_level = Logging.Warn optimize!(ldr2)
    @test termination_status(ldr2; dual = true) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    obj_diag = objective_value(ldr2; dual = true)

    # The correlated problem has strictly smaller feasible set (tighter polytope),
    # so its dual bound should be ≤ the uncorrelated one.
    @test obj_corr <= obj_diag + 1e-6

    return nothing
end

function test_tutorials()
    # Automatically find and run all tutorial .jl files
    tutorials_dir = joinpath(dirname(@__DIR__), "docs", "src", "tutorials")
    for (root, dirs, files) in walkdir(tutorials_dir)
        for file in files
            if endswith(file, ".jl")
                tutorial_path = joinpath(root, file)
                @testset "$file" begin
                    # Include in a sandbox module to avoid polluting the namespace
                    mod = @eval module $(gensym("TutorialTest")) end
                    Base.include(mod, tutorial_path)
                end
            end
        end
    end
    return nothing
end

end # TestMain module

TestMain.runtests()
