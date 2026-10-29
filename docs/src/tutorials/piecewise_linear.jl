# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# # [Piecewise linear decision rules](@id piecewise_linear_tutorial)

# ```@meta
# CurrentModule = LinearDecisionRules
# ```

# This tutorial introduces **piecewise linear decision rules**. If we generalize the linear
# decision rule to a piecewise linear one, we get a better approximation of the optimal decision
# rule, and a smaller gap between the primal value and the dual bound.

# We build on the newsvendor of [Getting started](@ref): she buys `buy` newspapers at 10 before
# the demand `d`, uniform on `[80, 120]`, is known; afterwards she sells `sell ≤ d` at 15 and
# returns the rest at 8.

# ## Why piecewise linear?

# A linear decision rule makes every decision an affine function of the uncertainty ``\eta``:
# ```math
# x(\eta) = x_0 + x_1 \eta
# ```
# However, many optimal policies are **piecewise linear**. In the newsvendor, once `buy` is
# chosen, the optimal sale is
# ```math
# \text{sell}(d) = \min(\text{buy}, d),
# ```
# which has a kink at `d = buy`: no affine function of `d` can follow it.

# ## Breakpoints and lifted variables

# To capture piecewise linear policies, we choose **breakpoints**
# ``\eta_0 = \eta_{\min} < \eta_1 < \dots < \eta_k = \eta_{\max}`` that partition the support of
# the uncertainty into segments. The uncertainty is represented as the sum of **lifted
# variables**,
# ```math
# \eta = \tilde{\eta}_1 + \tilde{\eta}_2 + \dots + \tilde{\eta}_k,
# ```
# where each ``\tilde{\eta}_i`` is the contribution of ``\eta`` from the ``i``-th segment:
# ```math
# \tilde{\eta}_i = \begin{cases}
#   \min(\eta, \eta_1) & i = 1 \\
#   \max\!\left(0,\, \min\!\left(\eta - \eta_{i-1},\, \eta_i - \eta_{i-1}\right)\right) & i \geq 2
# \end{cases}
# ```
# The first lifted variable satisfies ``\eta_{\min} \leq \tilde{\eta}_1 \leq \eta_1``, while
# ``0 \leq \tilde{\eta}_i \leq \eta_i - \eta_{i-1}`` for ``i \geq 2``.

# The decision rule is then a linear function of these lifted variables,
# ```math
# x(\eta) = \tilde{X} \tilde{\xi}, \qquad \tilde{\xi} = (1, \tilde{\eta}_1, \dots, \tilde{\eta}_k),
# ```
# so that it has a different slope in each segment. In the model, the breakpoints are an attribute
# of the uncertainty, [`BreakPoints`](@ref): everything else is the same model. See
# [`AxisPiecewise`](@ref) and [`lift`](@ref) for the lifting itself.

# ## Setup

using JuMP
using HiGHS
using Distributions
using LinearDecisionRules
using Random

# ## Part 1: a single breakpoint

# The newsvendor, as a function of the breakpoints of the demand. `optimize!` solves the primal
# rule and the dual bound, the default of the [`Formulations`](@ref) attribute.

function newsvendor(breakpoints = nothing)
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, buy >= 0, FirstStage)
    @variable(model, d in Uncertainty(Uniform(80, 120)))
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @constraint(model, sell + ret <= buy)
    @constraint(model, sell <= d)
    @objective(model, Max, -10buy + 8ret + 15sell)
    set_attribute(d, BreakPoints(), breakpoints)
    optimize!(model)
    return model
end

# Without breakpoints, this is the linear decision rule. The primal value is the value of a policy,
# and the dual value is an upper bound on the optimal value of the (maximization) problem:

model = newsvendor()
objective_value(model), objective_bound(model)

# Now with one breakpoint, which gives two linear pieces:

model = newsvendor(1)
objective_value(model), objective_bound(model)

# !!! note
#     `set_attribute(d, BreakPoints(), n)` places `n` breakpoints **equally spaced** inside the
#     support of the uncertainty. With one breakpoint, it is the midpoint, 100. A vector sets the
#     breakpoints at the given positions, strictly inside the support; `nothing`, `0` or an empty
#     vector mean no breakpoints. The uncertainty must be declared alone, with a univariate law.

# ### The piecewise rule

# The rule of a decision is a [`DecisionRule`](@ref): a constant, and for the lifted uncertainty,
# the slope on each of its pieces.

sell, ret, buy = model[:sell], model[:ret], model[:buy]
decision_rule(sell)

# [`rule_coefficient`](@ref) returns these slopes as a vector, one per piece (there is no `piece`
# keyword anymore); the constant is `rule_coefficient(x)`. The slope of `sell` is 1 below 100, as
# in `min(buy, d)` for `buy = 100`, and 0 above it. The rule of `ret` is the mirror image:

rule_coefficient(sell, model[:d]), rule_coefficient(ret, model[:d]), rule_coefficient(ret)

# A rule is callable on a scenario, in the original coordinates, so that the lifting is done for
# us:

decision_rule(sell)(model[:d] => 90.0), decision_rule(sell)(model[:d] => 110.0)

# ## Part 2: more breakpoints

# Let us see how the bounds improve as we add breakpoints. We print a table of the primal value,
# the dual bound and the gap:

function print_table(numbers)
    println("breakpoints   primal     dual       gap")
    for n in numbers
        model = newsvendor(n)
        primal, dual = objective_value(model), objective_bound(model)
        row = [round(v; digits = 3) for v in (primal, dual, dual - primal)]
        println(rpad(n, 14), rpad(row[1], 11), rpad(row[2], 11), row[3])
    end
    return
end

print_table(0:10)

# !!! tip
#     As the breakpoints refine the support, the **primal value** (an inner approximation)
#     increases and the **dual bound** (an outer approximation) decreases, so that the gap
#     shrinks. This is guaranteed when the new set of breakpoints is a **superset** of the
#     previous one: adding a point to an existing set never worsens the bounds. It is not
#     guaranteed for equally spaced points that are not nested. The sets for 0, 1, 3 and 7
#     breakpoints are nested (each one adds the midpoints of the previous pieces), and the
#     primal value never decreases along them; but 4 equally spaced breakpoints give a smaller
#     primal value than 3.

# ## Part 3: choosing the breakpoints

# Instead of equally spaced breakpoints, we can give their positions, which is useful when we know
# where the policy has kinks. In the newsvendor, the kink of `sell = min(buy, d)` is at `d = buy`,
# and the best `buy` is easy to compute: it is the quantile of the demand at the ratio
# ``(15 - 10) / (15 - 8) = 5/7``,
# ```math
# \text{buy}^\star = 80 + 40 \cdot \tfrac{5}{7} = \tfrac{760}{7} \approx 108.57.
# ```
# The linear rule cannot do this: it sells exactly `d`, which is affine, and for that it has to
# buy 120, the largest demand. So we put one breakpoint at ``760/7``.

model = newsvendor([760 / 7])
objective_value(model), 3300 / 7

# The primal value is now the optimal value of the problem, ``3300/7``, since the policy that
# we know to be optimal is piecewise linear with this kink, and the rule finds it:

decision_rule(model[:buy]), decision_rule(model[:sell])

# !!! note
#     The dual bound is not tight, even if the primal is optimal:

objective_bound(model)

# The dual LDR is a different approximation of the problem, so its bound does not close on the
# optimum with the primal. More breakpoints, in a superset of `[760 / 7]`, bring it down:

objective_bound(newsvendor([100.0, 760 / 7])),
objective_bound(newsvendor([90.0, 100.0, 760 / 7, 110.0]))

# Another choice, with three breakpoints, at 90, 100 and 110:

three = newsvendor([90.0, 100.0, 110.0])
objective_value(three), objective_bound(three)

# ### Evaluating the rule

# To see how well the rules follow `min(buy, d)`, we apply them to a grid of demands. The rule
# with the breakpoint at ``760/7`` is exactly `min(760/7, d)`. The three breakpoints cannot put a
# kink at 108.57, so the best rule that they allow has its kink at the next one, 110, and buys
# 110 instead:

rule_coefficient(three[:buy]), rule_coefficient(model[:buy])

#-

println("demand   min(760/7, d)   one breakpoint   three breakpoints")
for d in 80:5:120
    exact = min(760 / 7, d)
    one = decision_rule(model[:sell])(model[:d] => d)
    many = decision_rule(three[:sell])(three[:d] => d)
    println(
        rpad(d, 9),
        rpad(round(exact; digits = 3), 16),
        rpad(round(one; digits = 3), 17),
        round(many; digits = 3),
    )
end

# [`simulate`](@ref) applies the rule to scenarios drawn from the law of the demand. The mean of the
# objective of the primal rule estimates `objective_value`, and the rule is feasible for all of
# them.

simulate(model, 10_000; rng = Xoshiro(1))

# ## Part 4: comparing the formulations

# The [`SampledLDR`](@ref) formulation (see [Sampled decision rules](@ref sampled_tutorial)) also
# works with breakpoints. Let us compare the three formulations for nested breakpoints. The value
# of the sampled rule, solved on 500 scenarios, is an estimate and not a bound; we also evaluate
# its rule on 10 000 new scenarios with [`simulate`](@ref), which is the honest value of that rule.

function compare(breakpoints; scenarios = 500)
    model = newsvendor(breakpoints)
    sampled = SampledLDR(scenarios; seed = 1)
    set_attribute(model, Formulations(), [PrimalLDR(), DualLDR(), sampled])
    optimize!(model)
    out_of_sample = simulate(model, 10_000; formulation = sampled, rng = Xoshiro(1))
    return (
        objective_value(model),
        objective_bound(model),
        objective_value(model; result = 2),
        out_of_sample.objective_mean,
        out_of_sample.violation_fraction,
    )
end

println("breakpoints   primal     dual       sampled    out of sample   violations")
for n in (0, 1, 3, 7)
    r = compare(n)
    v = round.(r[1:4]; digits = 3)
    println(rpad(n, 14), rpad(v[1], 11), rpad(v[2], 11), rpad(v[3], 11), rpad(v[4], 16), r[5])
end

# The primal value and the dual bound enclose the optimal value, 3300/7, and both improve with the
# breakpoints. The sampled value is an in-sample estimate: it is optimistic, and it can even be
# above the dual bound, as with 7 breakpoints. The rule itself may be infeasible for some new
# scenarios (the fraction in the last column), so that its value out of sample is not comparable
# with the primal value, which has no violations on the support.

# ## Key API summary

# | Function | Description |
# |----------|-------------|
# | `set_attribute(d, BreakPoints(), n::Int)` | `n` equally spaced breakpoints for the uncertainty `d` |
# | `set_attribute(d, BreakPoints(), v::Vector)` | breakpoints at the positions `v` |
# | `decision_rule(x)` | the rule of `x`, callable on a scenario |
# | `rule_coefficient(x)` | the constant of the rule |
# | `rule_coefficient(x, d)` | the slopes of the rule of `x` on the pieces of `d` |
# | `rule_matrix(model)` | the whole matrix, in the lifted coordinates |

# ## What's next?

# * Learn about the other laws and supports of the uncertainties in [Distributions and liftings](@ref).
# * The [multistage](@ref multistage_tutorial) tutorial puts breakpoints on an uncertainty of a later
#   stage.
