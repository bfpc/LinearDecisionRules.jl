module SizeTests

include("common.jl")

function test_size_newsvendor_piecewise(; slack_as_expr = true)
    buy_cost = 10
    return_value = 8
    sell_value = 15

    demand_max = 120
    demand_min = 80
    demand_distr = Distributions.Uniform(demand_min, demand_max)

    # LDR
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)

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

    set_attribute(ldr, LinearDecisionRules.SlackAsExpr(), slack_as_expr)
    optimize!(ldr)

    ldr_p_obj = Float64[]
    push!(ldr_p_obj, objective_value(ldr))
    ldr_d_obj = Float64[]
    push!(ldr_d_obj, objective_value(ldr; dual = true))

    for n_intervals in 2:12
        set_attribute(
            demand,
            LinearDecisionRules.BreakPoints(),
            n_intervals - 1,
        )
        optimize!(ldr)
        push!(ldr_p_obj, objective_value(ldr))
        push!(ldr_d_obj, objective_value(ldr; dual = true))
    end

    return ldr_p_obj, ldr_d_obj
end

function test_size_confidence_mv_normal()
    # --- 2-D newsvendor ---
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
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

    set_attribute(ldr, LinearDecisionRules.SlackAsExpr(), false)
    optimize!(ldr)
    set_attribute(ldr, LinearDecisionRules.SlackAsExpr(), true)
    optimize!(ldr)

    return
end

end # module

DistributionsTests.runtests()
