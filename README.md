# LinearDecisionRules.jl

See presentation [video](https://youtu.be/ERO6vyTOOoI) and [slides](https://jump.dev/assets/jump-dev-workshops/2024/bfpc_ldr.pdf) at JuMP-dev 2024.

[![Build Status](https://github.com/bfpc/LinearDecisionRules.jl/workflows/CI/badge.svg?branch=main)](https://github.com/bfpc/LinearDecisionRules.jl/actions?query=workflow%3ACI)

[![codecov](https://codecov.io/gh/bfpc/LinearDecisionRules.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/bfpc/LinearDecisionRules.jl)

## Install

Install | `import Pkg; Pkg.add("LinearDecisionRules")`

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
