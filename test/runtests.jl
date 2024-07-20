using Test
using LinearDecisionRules
using JuMP
using Ipopt
using Distributions
using SparseArrays

function test_no_random()
    m = LinearDecisionRules.LDRModel(Ipopt.Optimizer)
    set_silent(m)
    @variable(m, x)
    @constraint(m, x == 1)
    @objective(m, Min, 0)
    set_attribute(m, LinearDecisionRules.SolveDual(), false)
    @test get_attribute(m, LinearDecisionRules.SolveDual()) == false
    optimize!(m)
    @test_throws OptimizeNotCalled() value(x)
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

test_no_random()

function test_0()

    initial_volume = 0.5
    demand = 0.3

    m = LinearDecisionRules.LDRModel()
    set_silent(m)
    @variable(m, vi == initial_volume)
    @variable(m, 0 <= vf <= 1)
    @variable(m, gh >= 0.0)
    @variable(m, gt >= 0.0)
    @variable(m, 0 <= inflow <= 0.2, LinearDecisionRules.Uncertainty, distribution=Uniform(0, 0.2))
    # @variable(m, 0 <= inflow[i=1:3] <= ub[i], LinearDecisionRules.Uncertainty, Uniform)

    @constraint(m, balance, vf == vi - gh + inflow)
    @constraint(m, gt + gh == demand)

    @objective(m, Min, gt^2 + vf^2/2 - vf)

    @test m.cache_uncertainty == Dict(inflow => Uniform(0, 0.2))
    @test m[:vi] == vi

    data = LinearDecisionRules.matrix_data(m.cache_model)
    @test data.variables == [vi; vf; gh; gt; inflow]
    @test data.Q == SparseArrays.sparse([2, 4], [2, 4], [1/2, 1], 5, 5)
    @test data.sense == MOI.MIN_SENSE

    # can = LinearDecisionRules._canonical(data, m.cache_uncertainty)

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

test_0()

function test_1()

    initial_volume = 0.5
    demand = 0.3
    
    m = LinearDecisionRules.LDRModel()
    set_silent(m)
    @variable(m, 0 <= vi <= initial_volume, LinearDecisionRules.Uncertainty, distribution=Uniform(0, 0.2))
    @variable(m, 0 <= vf <= 1)
    @variable(m, gh >= 0.0)
    @variable(m, gt >= 0.0)
    @variable(m, 0 <= inflow <= 0.2, LinearDecisionRules.Uncertainty, distribution=Uniform(0, 0.2))
    
    @constraint(m, balance, vf == vi - gh + inflow)
    @constraint(m, gt + gh == demand)
    
    @objective(m, Min, gt^2 + vf^2/2 - vf)
    
    @test m[:vi] == vi
    
    data = LinearDecisionRules.matrix_data(m.cache_model)
    @test data.variables == [vi; vf; gh; gt; inflow]
    @test data.Q == SparseArrays.sparse([2, 4], [2, 4], [1/2, 1], 5, 5)
    @test data.sense == MOI.MIN_SENSE
    
    set_optimizer(m, Ipopt.Optimizer)
    LinearDecisionRules._prepare_data(m)
    LinearDecisionRules._solve_primal_ldr(m)
    
    LinearDecisionRules.get_decision(m, vf, inflow)
    LinearDecisionRules.get_decision(m, vf)
    
    @test LinearDecisionRules.get_decision(m, gh) + LinearDecisionRules.get_decision(m, gt) ≈ demand atol=1e-6
    @test LinearDecisionRules.get_decision(m, gh, inflow) + LinearDecisionRules.get_decision(m, gt, inflow) ≈ 0 atol=1e-6
    
end

test_1()

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
    @variable(m, 0 <= inflow[i = 1:3] <= 0.1 * i, LinearDecisionRules.Uncertainty, distribution=Uniform(0, 0.1 * i))
    # @variable(m, 0 <= inflow[i=1:3] <= ub[i], LinearDecisionRules.Uncertainty, Uniform)

    @constraint(m, balance, vf == vi - gh + sum(inflow[i] for i in 1:3))
    @constraint(m, gt + gh == demand)

    @objective(m, Min, gt^2 + vf^2/2 - vf)

    # @test m.cache_uncertainty == Dict(inflow => Uniform(0, 0.2))
    @test m[:vi] == vi

    # data = LinearDecisionRules.matrix_data(m.cache_model)
    # @test data.variables == [vi; vf; gh; gt; inflow]
    # @test data.Q == SparseArrays.sparse([2, 4], [2, 4], [1/2, 1], 5, 5)
    # @test data.sense == MOI.MIN_SENSE

    # can = LinearDecisionRules._canonical(data, m.cache_uncertainty)

    set_optimizer(m, Ipopt.Optimizer)
    optimize!(m)

    # LinearDecisionRules.get_decision(m, vf, inflow)
    # LinearDecisionRules.get_decision(m, vf)

    # @test LinearDecisionRules.get_decision(m, gh) + LinearDecisionRules.get_decision(m, gt) ≈ demand atol=1e-6
    # @test LinearDecisionRules.get_decision(m, gh, inflow) + LinearDecisionRules.get_decision(m, gt, inflow) ≈ 0 atol=1e-6

    # @test LinearDecisionRules.get_decision(m, vi) ≈ initial_volume atol=1e-6
    # @test LinearDecisionRules.get_decision(m, vi, inflow) ≈ 0 atol=1e-6

end

test_2()

