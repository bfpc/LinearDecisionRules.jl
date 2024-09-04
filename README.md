# LinearDecisionRules.jl

## Example

```julia

buy_cost = 10
return_value = 8
sell_value = 12

demand_max = 120
demand_min = 80

ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
set_silent(ldr)

@variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
@variable(ldr, sell >= 0)
@variable(ldr, ret >= 0)
@variable(ldr, demand_min <= demand <= demand_max,
    LinearDecisionRules.Uncertainty,
    distribution = Uniform(demand_min, demand_max)
)

@constraint(ldr, sell + ret <= buy)

@constraint(ldr, sell <= demand)

@objective(ldr, Max,
    - buy_cost * buy
    + return_value * ret
    + sell_value * sell
)

optimize!(ldr)

@show objective_value(ldr)

@show LinearDecisionRules.get_decision(ldr, buy)

@show objective_value(ldr, dual = true)

@show LinearDecisionRules.get_decision(ldr, buy, dual = true)
```