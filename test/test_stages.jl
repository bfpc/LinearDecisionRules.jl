module TestStages

include("common.jl")

Stage = LinearDecisionRules.Stage

function test_stage_uncertainty_forms()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)

    # keyword stage form
    @variable(
        m,
        d_kw in LinearDecisionRules.Uncertainty(;
            distribution = Uniform(0.0, 1.0),
            stage = 1,
        )
    )
    # attribute stage form
    @variable(
        m,
        d_attr in LinearDecisionRules.Uncertainty(; distribution = Uniform(0.0, 1.0))
    )
    set_attribute(d_attr, Stage(1))

    @test get_attribute(d_kw, Stage()) == 1
    @test get_attribute(d_attr, Stage()) == 1

    # @variable(m, x >= 0, Stage(1)) does not work
    @variable(m, x >= 0)
    set_attribute(x, Stage(1))
    @constraint(m, x >= d_kw + d_attr)
    @objective(m, Min, x)
    optimize!(m)
    @test termination_status(m) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    return nothing
end

function test_stage_constraint_validation()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(
        m,
        d2 in LinearDecisionRules.Uncertainty(;
            distribution = Uniform(0.0, 1.0),
            stage = 2,
        )
    )
    @variable(m, x >= 0)
    # Maybe also add stage for x?
    @constraint(m, con_bad, x >= d2)
    set_attribute(con_bad, Stage(1))
    @objective(m, Min, x)
    @test_throws ErrorException optimize!(m)
    return nothing
end

function test_first_stage_no_warning_legacy_mode()
    @test_logs min_level = Logging.Warn begin
        m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
        set_silent(m)
        @variable(m, x >= 0, LinearDecisionRules.FirstStage)
        @variable(m, y >= 0)
        @variable(
            m,
            d in LinearDecisionRules.Uncertainty(; distribution = Uniform(0.0, 1.0))
        )
        @constraint(m, 2x + y >= 0.5 * d)
        @objective(m, Min, x + y)
        optimize!(m)
    end
    return nothing
end

function test_first_stage_mixed_with_stage_warns()
    @test_logs (:warn, r"Mixing FirstStage with Stage\(n\) is deprecated") begin
        m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
        set_silent(m)
        @variable(m, x >= 0, LinearDecisionRules.FirstStage)
        @variable(m, y >= 0)
        set_attribute(y, Stage(2))
    end
    return nothing
end

function test_3_stage_newsvendor()
    # Parameters
    buy_cost = 10
    return_value = 8
    sell_value = 15

    demand_max = 120
    demand_min = 80
    demand_distr = Distributions.Uniform(demand_min, demand_max)
    cost_stage2_distr = Distributions.Uniform(8.0, 14.0)

    # Model
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, buy >= 0); set_attribute(buy, Stage(1))

    @variable(ldr, sell2 >= 0); set_attribute(sell2, Stage(2))
    @variable(ldr, ret2 >= 0); set_attribute(ret2, Stage(2))
    @variable(ldr, demand2 in
        LinearDecisionRules.Uncertainty(
            distribution = demand_distr,
            stage = 2
            )
    )
    @variable(ldr, buy2 >= 0); set_attribute(buy2, Stage(2))
    @variable(ldr, buy_cost2 in
        LinearDecisionRules.Uncertainty(
            distribution = cost_stage2_distr,
            stage = 2
            )
    )

    @variable(ldr, sell3 >= 0); set_attribute(sell3, Stage(3))
    @variable(ldr, ret3 >= 0); set_attribute(ret3, Stage(3))
    @variable(ldr, demand3 in
        LinearDecisionRules.Uncertainty(
            distribution = demand_distr,
            stage = 3
            )
    )

    @constraint(ldr, sell2 + ret2 <= buy + buy2)
    @constraint(ldr, sell2 <= demand2)

    @constraint(ldr, sell2 + ret2 + sell3 + ret3 <= buy + buy2)
    @constraint(ldr, sell3 <= demand3)


    @objective(ldr, Max,
        - buy_cost * buy
        - buy_cost2 * buy2
        + return_value * ret2
        + sell_value * sell2
        + return_value * ret3
        + sell_value * sell3
    )

    optimize!(ldr)

    @test termination_status(ldr) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)

    @test objective_value(ldr) ≈ 940.0
    @test objective_value(ldr; dual=true) ≈ 1000.0

    # Simple no-dependence check on the decision rules
    @test LinearDecisionRules.get_decision(ldr, ldr[:buy], ldr[:demand2]) == 0.0
    @test LinearDecisionRules.get_decision(ldr, ldr[:buy], ldr[:demand3]) == 0.0
    for var in (:sell2, :ret2, :buy2)
        @test LinearDecisionRules.get_decision(ldr, ldr[var], ldr[:demand3]) == 0.0
    end

    return nothing
end

end # TestStages module

TestStages.runtests()
