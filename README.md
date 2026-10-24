# LinearDecisionRules.jl

Linear decision rules for multistage stochastic programs written in [JuMP](https://jump.dev).

> This is the development branch of a rewrite (version 0.4). The design is in [DESIGN.md](DESIGN.md).
> Released versions 0.1 and 0.2 live in the history of the `legacy` branch.

```julia
using JuMP, HiGHS, Distributions, LinearDecisionRules
model = LDRModel(HiGHS.Optimizer)
@variable(model, buy >= 0, Stage(1))
@variable(model, d in Uncertainty(Uniform(80, 120)))
@variable(model, sell >= 0)
@variable(model, ret >= 0)
@constraint(model, sell + ret <= buy)
@constraint(model, sell <= d)
@objective(model, Max, -10buy + 8ret + 15sell)
optimize!(model)
objective_value(model)   # 460: value of the primal LDR policy
objective_bound(model)   # 1460/3: dual LDR bound on the optimal value
decision_rule(sell)      # sell = d
```
