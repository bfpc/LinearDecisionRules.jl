module TestMain

using Test
using Random

using LinearDecisionRules

using JuMP
using Ipopt
using HiGHS
using Distributions
using LinearAlgebra

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            if Sys.WORD_SIZE == 32
                # On 32-bit, only run the quadratic test to debug OOM
                if name != :test_solve_sampled_quadratic_objective
                    continue
                end
                GC.gc()
                println("DEBUG [runtests] before $(name): free_memory=$(Sys.free_memory() / 1024^2) MB")
            end
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
    @test ldr_p_obj3 ≈ ldr_p_obj2 rtol = 1e-2

    ldr_d_obj3 = objective_value(ldr; dual = true)
    @test ldr_d_obj3 ≈ ldr_d_obj2 rtol = 1e-2

    M3 = ldr.ext[:_LDR_M]
    @test M3[1, 1] == 1
    @test M3[3, 2] ≈ M3[2, 3] rtol = 1e-2
    @test M3[3, 2] ≈ M3[3, 1] * M3[1, 2] rtol = 1e-2
    @test M3[3, 3] ≈ M3[2, 2] rtol = 1e-2

    @constraint(ldr, sell <= demand2)
    optimize!(ldr)
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

function test_rejection_sampling_attributes()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)

    # defaults
    @test get_attribute(m, LinearDecisionRules.RejectionSamplingTimeLimit()) ==
          10.0
    @test get_attribute(m, LinearDecisionRules.RejectionSamplingSeed()) == 1234
    @test get_attribute(
        m,
        LinearDecisionRules.RejectionSamplingMaxIterations(),
    ) == 1000
    @test get_attribute(
        m,
        LinearDecisionRules.RejectionSamplingWarnAttempts(),
    ) == 1000

    # round-trip set/get
    set_attribute(m, LinearDecisionRules.RejectionSamplingTimeLimit(), 5.0)
    set_attribute(m, LinearDecisionRules.RejectionSamplingSeed(), 42)
    set_attribute(m, LinearDecisionRules.RejectionSamplingMaxIterations(), 500)
    set_attribute(m, LinearDecisionRules.RejectionSamplingWarnAttempts(), 200)

    @test get_attribute(m, LinearDecisionRules.RejectionSamplingTimeLimit()) ==
          5.0
    @test get_attribute(m, LinearDecisionRules.RejectionSamplingSeed()) == 42
    @test get_attribute(
        m,
        LinearDecisionRules.RejectionSamplingMaxIterations(),
    ) == 500
    @test get_attribute(
        m,
        LinearDecisionRules.RejectionSamplingWarnAttempts(),
    ) == 200

    return nothing
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

function test_univariate_piecewise_constructor_errors()
    o_dist = Distributions.Uniform(1.0, 4.0)

    # break point at or above upper bound
    @test_throws ArgumentError LinearDecisionRules.UnivariatePieceWise(
        o_dist,
        [3.5, 4.5],
    )
    # break point at or below lower bound
    @test_throws ArgumentError LinearDecisionRules.UnivariatePieceWise(
        o_dist,
        [0.5, 2.0],
    )
    # empty break_points
    @test_throws ArgumentError LinearDecisionRules.UnivariatePieceWise(
        o_dist,
        Float64[],
    )
    # NaN break point
    @test_throws ArgumentError LinearDecisionRules.UnivariatePieceWise(
        o_dist,
        [NaN],
    )

    return nothing
end

function test_break_points_getter()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(
        m,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1)),
    )

    # before setting: returns nothing
    @test get_attribute(demand, LinearDecisionRules.BreakPoints()) === nothing

    # after setting: returns the vector
    set_attribute(demand, LinearDecisionRules.BreakPoints(), [0.5])
    @test get_attribute(demand, LinearDecisionRules.BreakPoints()) == [0.5]

    # after clearing: returns nothing again
    set_attribute(demand, LinearDecisionRules.BreakPoints(), nothing)
    @test get_attribute(demand, LinearDecisionRules.BreakPoints()) === nothing

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

function test_break_points_errors_extended()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        s in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @variable(
        m,
        v[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = product_distribution([Uniform(0, 1), Uniform(0, 1)]),
        )
    )

    @test_throws ErrorException set_attribute(
        x,
        LinearDecisionRules.BreakPoints(),
        [0.5],
    )
    @test_throws ErrorException set_attribute(
        x,
        LinearDecisionRules.BreakPoints(),
        nothing,
    )
    @test_throws ErrorException set_attribute(
        x,
        LinearDecisionRules.BreakPoints(),
        2,
    )

    @test_throws ErrorException set_attribute(
        v[1],
        LinearDecisionRules.BreakPoints(),
        [0.5],
    )

    @test_throws ErrorException set_attribute(
        s,
        LinearDecisionRules.BreakPoints(),
        Float64[],
    )
    @test_throws ErrorException set_attribute(
        s,
        LinearDecisionRules.BreakPoints(),
        0,
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

function test_get_decision_pwl_piece_errors()
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)
    @variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell >= 0)
    @variable(
        ldr,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(80.0, 120.0)),
    )
    @constraint(ldr, sell <= buy)
    @constraint(ldr, sell <= demand)
    @objective(ldr, Max, -10buy + 15sell)
    set_attribute(demand, LinearDecisionRules.BreakPoints(), 2)
    optimize!(ldr)

    @test_throws ErrorException LinearDecisionRules.get_decision(
        ldr,
        sell,
        demand,
    )
    @test_throws ErrorException LinearDecisionRules.get_decision(
        ldr,
        sell,
        demand;
        piece = 10,
    )
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

function test_mv_discrete_stats()
    d = LinearDecisionRules.MvDiscreteNonParametric(
        [[3.0, 1.0], [1.0, 3.0]],
        [0.5, 0.5],
    )
    @test Distributions.minimum(d) == [1.0, 1.0]
    @test Distributions.var(d) == LinearAlgebra.diag(Distributions.cov(d))
    @test Distributions.insupport(d, [3.0, 1.0])
    @test !Distributions.insupport(d, [0.0, 0.0])

    # rand must return one of the two support points
    rng = Random.Xoshiro(42)
    for _ in 1:20
        x = rand(rng, d)
        @test x == [3.0, 1.0] || x == [1.0, 3.0]
    end

    # Integer eltype
    d_int = LinearDecisionRules.MvDiscreteNonParametric(
        [[1, 2], [3, 4], [5, 6]],
        [0.2, 0.3, 0.5],
    )
    @test Base.eltype(typeof(d_int)) == Int
    @test Distributions.length(d_int) == 2
    @test Distributions.minimum(d_int) == [1, 2]
    @test Distributions.maximum(d_int) == [5, 6]
    @test Distributions.insupport(d_int, [3, 4])
    @test !Distributions.insupport(d_int, [2, 3])

    rng2 = Random.Xoshiro(123)
    for _ in 1:20
        x = rand(rng2, d_int)
        @test x == [1, 2] || x == [3, 4] || x == [5, 6]
    end

    # Int32 eltype
    d_i32 = LinearDecisionRules.MvDiscreteNonParametric(
        [Int32[10, 20], Int32[30, 40]],
        [0.5, 0.5],
    )
    @test Base.eltype(typeof(d_i32)) == Int32
    @test Distributions.minimum(d_i32) == Int32[10, 20]
    @test Distributions.maximum(d_i32) == Int32[30, 40]

    rng3 = Random.Xoshiro(7)
    for _ in 1:20
        x = rand(rng3, d_i32)
        @test x == Int32[10, 20] || x == Int32[30, 40]
    end

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

function test_breakpoints_getter_and_vector_nothing()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        s in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @variable(
        m,
        v[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = product_distribution([Uniform(0, 1), Uniform(0, 1)]),
        )
    )
    @test_throws ErrorException get_attribute(
        x,
        LinearDecisionRules.BreakPoints(),
    )
    @test_throws ErrorException get_attribute(
        v[1],
        LinearDecisionRules.BreakPoints(),
    )
    @test_throws ErrorException set_attribute(
        v[1],
        LinearDecisionRules.BreakPoints(),
        nothing,
    )
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

function test_mv_discrete_params()
    d = LinearDecisionRules.MvDiscreteNonParametric(
        [[1.0, 2.0], [3.0, 4.0]],
        [0.4, 0.6],
    )
    @test Base.eltype(typeof(d)) == Float64
    xs, ps = Distributions.params(d)
    @test length(xs) == 2
    @test sum(ps) ≈ 1.0
    return nothing
end

function test_univariate_piecewise_errors()
    @test_throws ArgumentError LinearDecisionRules.UnivariatePieceWise(
        Normal(0, 1),
        [0.0],
    )
    @test_throws ArgumentError LinearDecisionRules.UnivariatePieceWise(
        Uniform(0, 1),
        [-Inf],
    )
    d = LinearDecisionRules.UnivariatePieceWise(Uniform(0, 1), [0.3, 0.7])
    @test LinearDecisionRules._original(d) isa Uniform
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

function test_vector_uncertainty_with_pwl()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, buy >= 0, LinearDecisionRules.FirstStage)
    @variable(m, sell >= 0)
    @variable(
        m,
        scalar_d in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1)),
    )
    @variable(
        m,
        vec_d[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = product_distribution([Uniform(0, 1), Uniform(0, 1)]),
        )
    )
    @constraint(m, sell <= buy)
    @constraint(m, sell <= scalar_d + vec_d[1])
    @objective(m, Min, buy - sell)
    set_attribute(scalar_d, LinearDecisionRules.BreakPoints(), 2)
    optimize!(m)
    @test termination_status(m) == MOI.OPTIMAL
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

function test_breakpoint_integer_on_vector_uncert()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    @variable(
        m,
        v[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = product_distribution([Uniform(0, 1), Uniform(0, 1)]),
        )
    )
    @test_throws ErrorException set_attribute(
        v[1],
        LinearDecisionRules.BreakPoints(),
        2,
    )
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

function test_univariate_piecewise_more_errors()
    # eltype mismatch: Float64 distribution, Float32 breakpoints
    @test_throws ArgumentError LinearDecisionRules.UnivariatePieceWise(
        Uniform(0.0, 1.0),
        Float32[0.3f0],
    )
    # max is +Inf (Exponential: min=0, max=Inf)
    @test_throws ArgumentError LinearDecisionRules.UnivariatePieceWise(
        Exponential(1),
        [0.5],
    )
    # breakpoint at +Inf
    @test_throws ArgumentError LinearDecisionRules.UnivariatePieceWise(
        Uniform(0, 1),
        [Inf],
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

function test_mixed_rejection_sampling()
    # d1 constrained (rejection sampling) + d2 free: mixed grouped/ungrouped uncertainties
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        d1 in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @variable(
        m,
        d2 in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @constraint(m, d1 <= 0.8)
    @objective(m, Min, x)
    optimize!(m)
    @test termination_status(m) == MOI.OPTIMAL

    # free vector distribution alongside constrained scalar
    m2 = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m2)
    @variable(m2, x2 >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m2,
        d3 in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @variable(
        m2,
        vec_d[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = product_distribution([Uniform(0, 1), Uniform(0, 1)]),
        )
    )
    @constraint(m2, d3 <= 0.8)
    @objective(m2, Min, x2)
    optimize!(m2)
    @test termination_status(m2) == MOI.OPTIMAL
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
    optimize!(m1)
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
    optimize!(m2)
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

function test_rejection_sampling_warnings()
    # warn fires when rejection sampling exceeds max attempts
    m1 = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m1)
    @variable(m1, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m1,
        d in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @constraint(m1, d <= 0.5)
    @objective(m1, Min, x)
    set_attribute(m1, LinearDecisionRules.RejectionSamplingWarnAttempts(), 0)
    optimize!(m1)
    @test termination_status(m1) == MOI.OPTIMAL

    # warn fires when _attempts reaches warn_attempts threshold
    # d1+d2 >= 0.01 is always satisfied by Uniform(0,1), so first attempt triggers the warn
    m2 = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m2)
    @variable(m2, x2 >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m2,
        d1 in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @variable(
        m2,
        d2 in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @constraint(m2, d1 + d2 >= 0.01)
    @objective(m2, Min, x2)
    set_attribute(m2, LinearDecisionRules.RejectionSamplingWarnAttempts(), 1)
    optimize!(m2)
    @test termination_status(m2) == MOI.OPTIMAL

    # time limit warn fires when RejectionSamplingTimeLimit is 0.0
    m3 = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m3)
    @variable(m3, x3 >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m3,
        d3 in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @constraint(m3, d3 <= 0.5)
    @objective(m3, Min, x3)
    set_attribute(m3, LinearDecisionRules.RejectionSamplingTimeLimit(), 0.0)
    optimize!(m3)
    @test termination_status(m3) == MOI.OPTIMAL
    return nothing
end

function test_univariate_piecewise_params()
    # Distributions.params should return original params plus break_points
    d = LinearDecisionRules.UnivariatePieceWise(Uniform(0.0, 1.0), [0.3, 0.7])
    ps = Distributions.params(d)
    @test length(ps) == 3  # (a, b) from Uniform + break_points
    @test ps[end] == [0.3, 0.7]
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
    # HiGHS supports QP objectives needed for the sampled sub-model
    println("DEBUG [quadratic] WORD_SIZE=$(Sys.WORD_SIZE)")
    println("DEBUG [quadratic] total_memory=$(Sys.total_memory() / 1024^2) MB")
    println("DEBUG [quadratic] free_memory=$(Sys.free_memory() / 1024^2) MB")
    GC.gc()
    println("DEBUG [quadratic] free_memory after GC=$(Sys.free_memory() / 1024^2) MB")

    initial_volume = 0.5
    demand = 0.3

    println("DEBUG [quadratic] creating model...")
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
    println("DEBUG [quadratic] free_memory before optimize=$(Sys.free_memory() / 1024^2) MB")
    println("DEBUG [quadratic] JuMP model before optimize!:")
    println(m)
    flush(stdout)
    optimize!(m)
    println("DEBUG [quadratic] optimize! done")
    sm = m.sampled_model
    println("DEBUG [quadratic] num_variables=$(num_variables(sm))")
    println("DEBUG [quadratic] num_constraints=$(num_constraints(sm; count_variable_in_set_constraints=true))")
    flush(stdout)
    println("DEBUG [quadratic] optimize! done")
    println("DEBUG [quadratic] free_memory after optimize=$(Sys.free_memory() / 1024^2) MB")

    # Verify the M̂ path was taken (not the μ̂ path)
    @test haskey(m.ext, :_LDR_M_empirical)
    @test !haskey(m.ext, :_LDR_μ_empirical)
    @test termination_status(m; sampled = true) == MOI.OPTIMAL

    # Sampled decision rules should satisfy balance constraint
    gh_const = LinearDecisionRules.get_decision(m, gh; sampled = true)
    gt_const = LinearDecisionRules.get_decision(m, gt; sampled = true)
    @test gh_const + gt_const ≈ demand atol = 1e-4

    println("DEBUG [quadratic] test passed")
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

end # TestMain module

TestMain.runtests()
