# LinearDecisionRules.jl

Linear decision rules for multistage stochastic programs written in [JuMP](https://jump.dev).

> This is the development branch of a rewrite (version 0.4). How the package works inside is
> explained in the [developer documentation](docs/src/developer/index.md).
> Released versions 0.1 and 0.2, and the unregistered 0.3.1 (tag `v0.3.1`), live in the history
> of the `legacy` branch.

```julia
using JuMP, HiGHS, Distributions, LinearDecisionRules
model = LDRModel(HiGHS.Optimizer)
@variable(model, buy >= 0, FirstStage)   # the same as `Stage(1)`
@variable(model, d in Uncertainty(Uniform(80, 120)))
@variable(model, sell >= 0)
@variable(model, ret >= 0)
@constraint(model, sell + ret <= buy)
@constraint(model, sell <= d)
@objective(model, Max, -10buy + 8ret + 15sell)
optimize!(model)
objective_value(model)   # 460: value of the primal LDR policy
objective_bound(model)   # 1460/3: dual LDR bound on the optimal value
decision_rule(buy)       # buy = 120
decision_rule(sell)      # sell = d
value(sell)              # 100: E[sell(d)]
expected_value(d * sell) # exact for affine and quadratic expressions
```
