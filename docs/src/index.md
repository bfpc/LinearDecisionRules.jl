# LinearDecisionRules.jl Documentation

```@meta
CurrentModule = LinearDecisionRules
```

The `LinearDecisionRules.jl` package provides a simple `JuMP` abstraction to represent decision rules in (stochastic) optimization problems as linear functions of random variables.

The problems the package deals with are of the form
```math
\begin{array}{rl}
\min \ & E[ c(ξ)^\top x(ξ) + x(ξ)^\top Q x(ξ) + r ] \\[0.5ex]
\text{s.t.} & A_e x(ξ) = b_e(ξ) \\
& A_u x(ξ) ≤ b_u(ξ) \\
& A_l x(ξ) ≥ b_l(ξ) \\
& x(ξ) ≤ x_u \\
& x(ξ) ≥ x_l \\
& x_i(ξ) \text{ is non-anticipative, for } i ∈ I \\
& ∀ ξ ∈ Ξ
\end{array}
```
where $Ξ ⊂ ℝ^m$ is a polytope described by
```math
\begin{align*}
Ξ & = \{\, ξ = (1, η) ∈ ℝ^m \mid W_u η ≤ h_u, W_l η ≥ h_l, lb ≤ η ≤ ub \,\} \\
& = \{\, ξ ∈ ℝ^m \mid W ξ ≥ h \,\}.
\end{align*}
```
Variable $η$ cannot appear in equality constraints, since the linear span of $Ξ$ must be all of $ℝ^m$.

Non-anticipative variables are not allowed to depend on the random variable $ξ$.
This is enforced by fixing their decision rules to have coefficient equal to zero, except for the constant term.

## Example

Consider the following classical "Newsvendor" problem:
- A retailer must decide how many units of a product to buy (at a cost of \$10).
- The demand is uniformly distributed between 80 and 120 units, and unavailable at buying time; units are sold for \$12.
- Leftover units can be returned (for \$8).

This leads to the following optimization problem:
```math
\begin{array}{rl}
\max \ & - 10 \cdot \text{buy} + E [ 8 \cdot \text{return} + 12 \cdot \text{sell}] \\[0.5ex]
\text{s.t.} & \text{sell}(ξ) + \text{return}(ξ) ≤ \text{buy} \\
& \text{sell}(ξ) ≤ \text{demand}(ξ) \\
& 0 ≤ \text{sell}(ξ), \text{return}(ξ), \text{buy}
\end{array}
```
where we indicate that `buy` is a first-stage decision, and `sell` and `return` are second-stage decisions, depending on the scenario $\xi$ that fixes value of the random variable `demand`.

```julia
using JuMP
using LinearDecisionRules
using HiGHS
using Distributions

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
@variable(ldr, demand, LinearDecisionRules.Uncertainty,
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

```@docs
get_decision
```
