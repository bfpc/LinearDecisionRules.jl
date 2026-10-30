# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# # [Sampled (SAA) decision rules](@id sampled_tutorial)

# ```@meta
# CurrentModule = LinearDecisionRules
# ```

# This tutorial introduces the **sampled** (Sample Average Approximation) formulation. Instead of
# solving the robust primal and dual formulations, [`SampledLDR`](@ref) draws scenarios from the
# laws of the uncertainties and optimizes the coefficients of the rule over them.

# We build on the newsvendor of [Getting started](@ref).

# ## Why sampled?

# The primal and dual LDR formulations provide **bounds** on the optimal objective. The primal is
# an inner approximation (a lower bound for a maximization), and the dual is an outer
# approximation (an upper bound). Both rely on the second moment matrix
# ``M = E[\xi \xi^\top]`` of the uncertainties, and enforce the constraints robustly over the
# whole support.

# The **sampled** formulation instead:
# 1. draws ``N`` scenarios ``\xi_1, \dots, \xi_N`` from the laws;
# 2. enforces the constraints **per scenario** (no dual multiplier matrices);
# 3. optimizes the **average objective** over these scenarios.

# This trades the conservatism of the robust formulation for a data-driven approach that handles
# complex laws naturally, at the expense of a possibly larger problem when many scenarios are
# needed for high-dimensional uncertainty, and of the sensitivity of SAA to the exact scenarios
# that were drawn.

# Two things are different from the primal and the dual, and the model says so:
# * the sampled rule is feasible on its ``N`` scenarios **only**, so its primal status is
#   `UNKNOWN_RESULT_STATUS`, and not `FEASIBLE_POINT`;
# * its objective value is an estimate, and not a bound, since it is the value *in the sample*,
#   which is optimistic: the rule is chosen to be good on these very scenarios.
# The [`simulate`](@ref) function evaluates the rule on new scenarios, which is the way to measure
# it.

# ## Setup

using JuMP
using HiGHS
using Distributions
using LinearDecisionRules
using Random
using Test #src

# ## The newsvendor

# The model is the one of the previous tutorials, as a function of the formulations that we ask
# for.

function newsvendor(formulations)
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, buy >= 0, FirstStage)
    @variable(model, d in Uncertainty(Uniform(80, 120)))
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @constraint(model, sell + ret <= buy)
    @constraint(model, sell <= d)
    @objective(model, Max, -10buy + 8ret + 15sell)
    set_attribute(model, Formulations(), formulations)
    optimize!(model)
    return model
end

# ## Part 1: a sampled solution

# `SampledLDR(N; seed)` takes the number of scenarios and the seed of their draws, so that a
# solution can be reproduced. It is a formulation like the others: we ask for
# it in the [`Formulations`](@ref) attribute, beside the primal and the dual.

sampled = SampledLDR(500; seed = 1)
model = newsvendor([PrimalLDR(), DualLDR(), sampled])
result_count(model)

# The primal rule is the result 1 and the sampled rule is the result 2, with their own status
# and objective value. The dual, which is a bound, is `objective_bound`:

primal_status(model; result = 1), primal_status(model; result = 2)
@test primal_status(model; result = 2) == UNKNOWN_RESULT_STATUS #src

#-

objective_value(model; result = 1), objective_value(model; result = 2), objective_bound(model)

# ### Extracting the decision rule

# The rules of the sampled formulation are read with the same functions as the primal rule, by
# giving the formulation. It must be the one of the model, with the same number of scenarios and
# seed.

buy, sell, d = model[:buy], model[:sell], model[:d]
decision_rule(buy; formulation = sampled), decision_rule(sell; formulation = sampled)

#-

rule_coefficient(buy; formulation = sampled), rule_coefficient(sell, d; formulation = sampled)
@test rule_coefficient(sell, d; formulation = sampled) ≈ 1 rtol = 1e-6 #src
@test 119 < rule_coefficient(buy; formulation = sampled) < 120 #src

# The sampled rule is linear too: it sells `d`, and it buys the largest demand of the sample, which
# is a little below 120, the largest demand of the law and the `buy` of the primal rule. (The
# optimal policy buys 760/7 and sells `min(buy, d)`: it is piecewise linear, see the
# [piecewise linear](@ref piecewise_linear_tutorial) tutorial.)

# ## Part 2: comparing the formulations

# The primal value and the dual bound enclose the optimal value. The sampled value is not a bound,
# and as it is the best value on the sample, it is above what the rule gets on new scenarios.
# [`simulate`](@ref) draws new scenarios from the laws and applies the rule:

simulate(model, 10_000; formulation = sampled, rng = Xoshiro(1))
out = simulate(model, 10_000; formulation = sampled, rng = Xoshiro(1)) #src
@test out.objective_mean < objective_value(model; result = 2) #src
@test out.violation_fraction > 0 #src

# The sample mean of the objective, the **out-of-sample** value of the rule, is below the sampled
# objective. The rule also has violations out of sample: since `buy` is the largest sampled demand,
# the constraint `sell <= d` fails, by a little, for the few new demands above it. The rule of the
# primal formulation has no violation, by construction:

simulate(model, 10_000; rng = Xoshiro(1))
@test simulate(model, 10_000; rng = Xoshiro(1)).violation_fraction == 0 #src

# ## Part 3: the effect of the number of scenarios

# The quality of the sampled rule depends on the number of scenarios. The primal and dual values
# do not depend on ``N``, since they use the moments of the law and not draws. We print the
# sampled value in the sample, its value out of sample, and the fraction of new scenarios in
# which the rule violates a constraint:

reference = newsvendor([PrimalLDR(), DualLDR()])
objective_value(reference), objective_bound(reference)

#-

function sampled_summary(N; seed = 1, evaluation = 10_000)
    formulation = SampledLDR(N; seed)
    model = newsvendor([formulation])
    s = simulate(model, evaluation; formulation, rng = Xoshiro(1))
    return objective_value(model; result = 2), s.objective_mean, s.violation_fraction
end

fractions = Float64[] #src
println("N      in sample   out of sample   violations")
for N in [10, 25, 50, 100, 200, 500, 1000]
    in_sample, out_of_sample, violations = sampled_summary(N)
    push!(fractions, violations) #src
    println(
        rpad(N, 7),
        rpad(round(in_sample; digits = 2), 12),
        rpad(round(out_of_sample; digits = 2), 16),
        round(violations; digits = 4),
    )
end
@test fractions[end] < fractions[1] #src

# !!! tip
#     With more scenarios the sampled value in the sample and out of the sample get closer, and
#     the violations are rarer. The primal and dual values are independent of ``N``. The seed
#     fixes the stream of draws, so that a larger sample contains the smaller ones; as the rule
#     buys the largest demand drawn, some values of ``N`` give the same rule.

# ## Part 4: SAA variance

# For a small number of scenarios, the sampled objective (and the decisions) have a significant
# variance. We solve the problem for 10 seeds for each ``N``, and print the extreme and mean
# values of the sampled objective, to compare with the primal value and the dual bound:

println("N     smallest   mean       largest")
for N in [2, 5, 10, 25]
    values = [sampled_summary(N; seed = 100N + k, evaluation = 1)[1] for k in 1:10]
    N <= 10 && @test maximum(values) > objective_bound(reference) #src
    println(
        rpad(N, 6),
        rpad(round(minimum(values); digits = 1), 11),
        rpad(round(sum(values) / length(values); digits = 1), 11),
        round(maximum(values); digits = 1),
    )
end
@test objective_bound(reference) ≈ 1460 / 3 rtol = 1e-6 #src

# The sampled values scatter, and for up to 10 scenarios some of them are above the dual bound
# 486.67, which is a bound for the true optimal value: for few scenarios, the in-sample value is
# not reliable. The rule, in the end, has to be measured out of sample.

# ## API summary

# | Function | Description |
# |----------|-------------|
# | `SampledLDR(N; seed)` | the formulation with `N` scenarios and the given seed |
# | `set_attribute(model, Formulations(), [PrimalLDR(), SampledLDR(N)])` | solve both |
# | `objective_value(model; result = 2)` | the sampled objective (in sample) |
# | `primal_status(model; result = 2)` | `UNKNOWN_RESULT_STATUS` |
# | `decision_rule(x; formulation = f)` | the sampled rule of `x` |
# | `rule_coefficient(x, d; formulation = f)` | the coefficient on `d` of the sampled rule |
# | `simulate(model, n; formulation = f)` | the sampled rule on new scenarios |

# ## What's next?

# The [multistage](@ref multistage_tutorial) tutorial writes a problem over several periods, in
# which each decision sees only the uncertainties revealed before it.
