# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# # [Multistage problems: an inventory over time](@id multistage_tutorial)

# ```@meta
# CurrentModule = LinearDecisionRules
# ```

# This tutorial shows how to write a **multistage** problem, in which the information arrives
# over time and each decision can use only what was revealed before it. We write a ``T``-period
# inventory problem with [`Stage`](@ref) containers, we look at the **nonanticipativity** of the
# decision rules in the matrix of the rules, we put breakpoints on an uncertainty of a later stage,
# and we simulate the policy.

# ## Setup

using JuMP
using HiGHS
using Distributions
using LinearDecisionRules
using Random

# ## Stages and information

# In a multistage problem, the decisions and the uncertainties have a **stage**, tagged with
# `Stage(t)`: the uncertainties of stage ``t`` are revealed before the decisions of stage ``t``,
# which then see all the uncertainties of the stages up to ``t``, and none of the later ones.
# The decisions of stage 1 (also written [`FirstStage`](@ref)) are made here and now, with no
# information. A decision rule has coefficients only on the uncertainties that its decision
# observes: this is what is called nonanticipativity, and the package enforces it.
#
# An untagged variable is in the last stage ``T``, the largest declared stage (and at least 2), with
# a warning when ``T > 2``: declare the stages of all the variables of a multistage model.

# ## The inventory problem

# A company orders ``x_t`` units at the beginning of each period ``t = 1, \dots, T``, at a unit cost
# of 1, up to a capacity of 150. Then the demand ``d_t``, uniform on `[60, 100]`, is revealed.
# The inventory at the end of the period is ``I_t = I_{t-1} + x_t - d_t`` (with ``I_0 = 0``),
# which is negative when there is a backlog. The cost of the period is ``y_t = \max(0.5 I_t, -3 I_t)``:
# 0.5 per unit of inventory, and 3 per unit of backlog. We minimize the expected total cost.

# The order ``x_t`` is made before ``d_t`` is known, so it is of stage ``t``, while ``d_t`` is
# revealed at stage ``t + 1``, together with the inventory and the cost of the period that depend
# on it. The stages are written with the index of the container:

function inventory(T; breakpoints = nothing)
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, d[t = 1:T] in Uncertainty(Uniform(60, 100)), Stage(t + 1))
    @variable(model, 0 <= x[t = 1:T] <= 150, Stage(t))
    @variable(model, I[t = 1:T], Stage(t + 1))
    @variable(model, y[t = 1:T] >= 0, Stage(t + 1))
    @constraint(model, I[1] == x[1] - d[1])
    @constraint(model, [t in 2:T], I[t] == I[t-1] + x[t] - d[t])
    @constraint(model, [t in 1:T], y[t] >= 0.5 * I[t])
    @constraint(model, [t in 1:T], y[t] >= -3 * I[t])
    @objective(model, Min, sum(x[t] + y[t] for t in 1:T))
    if breakpoints !== nothing
        set_attribute(d[2], BreakPoints(), breakpoints)
    end
    set_attribute(model, Formulations(), [PrimalLDR(), DualLDR()])
    optimize!(model)
    return model
end

# The stages are part of the model: they are `VariableIndex`-in-`Stage` constraints, which are
# printed with the model (here, for ``T = 2``) and read with [`stage`](@ref):

print(inventory(2))

# ## Solving

# For ``T = 4``, the primal value is the expected cost of the linear policy, and the dual value is
# a lower bound on the optimal cost (the problem is a minimization):

model = inventory(4)
x, I, y, d = model[:x], model[:I], model[:y], model[:d]
objective_value(model), objective_bound(model), relative_gap(model)

# The primal value is `90T + 20` for any ``T``: 100 units in the first period, then in each period
# the quantity of the last demand (80 on average), and a cost of 10 per period for the inventory.
# The bound is smaller:

println("T   primal   dual")
for T in 2:2:8
    m = inventory(T)
    println(
        rpad(T, 4),
        rpad(round(objective_value(m); digits = 2), 9),
        round(objective_bound(m); digits = 2),
    )
end

# ## Nonanticipativity

# [`rule_matrix`](@ref) is the matrix of the rules: one row per decision (all the variables
# that are not uncertainties), and one column for the constant and for each uncertainty. The
# entries are zero where a decision does not observe the uncertainty. We see it for the orders:

rule_matrix(model)

# The orders have a staircase pattern. `x[1]` can only be a constant, `x[2]` may depend on `d[1]`,
# `x[3]` on `d[1]` and `d[2]`, and so on: the zeros above the staircase are forced, and the
# entries below it are free. The optimal rule uses the last demand only. The rule of each decision
# is also available alone:

decision_rule(x[1]), decision_rule(x[3])

# Here, the order of a period is the demand of the previous period. We can check that no rule uses
# an uncertainty that its decision has not seen, with [`rule_coefficient`](@ref):

all(rule_coefficient(x[t], d[s]) == 0 for t in 1:4 for s in t:4)

# ## Breakpoints on a later stage

# The breakpoints of a later uncertainty are an attribute of the variable, as in the
# [piecewise linear](@ref piecewise_linear_tutorial) tutorial. With breakpoints on `d[2]`, which is
# of stage 3, the rules of stage 3 and later are piecewise linear on it, and the primal value
# improves, as the dual bound does:

println("breakpoints on d[2]   primal       dual")
for n in [0, 1, 3, 7]
    m = inventory(4; breakpoints = n)
    println(
        rpad(n, 22),
        rpad(round(objective_value(m); digits = 3), 13),
        round(objective_bound(m); digits = 3),
    )
end

# The pieces of `d[2]` are in the stage of `d[2]`: they are lifted coordinates, whose stage is the
# stage of the uncertainty that they come from. Therefore the rules of stages 1 and 2 do not see
# them, which we read in the matrix of the rules, in the columns `d[2][k]` of the `k`-th piece:

model3 = inventory(4; breakpoints = 3)
rule_matrix(model3)

# `x[2]` has zeros on the pieces of `d[2]`, and `x[3]`, which is of stage 3, has coefficients on
# them: it is piecewise linear in `d[2]`, with the breakpoints 70, 80 and 90. [`rule_coefficient`](@ref)
# gives the vector of the slopes on the pieces:

rule_coefficient(model3[:x][2], model3[:d][2]), rule_coefficient(model3[:x][3], model3[:d][2])

# The new rule explains the improvement. The cost `y[2]` is the maximum of two lines in the
# inventory `I[2]`, with a kink at `I[2] = 0`, and the rule makes `I[2] = 90 - d[2]`: its kink is at
# `d[2] = 90`, which is a breakpoint. The slopes of `y[2]` on the four pieces are those of the
# cost, -0.5 below 90 and 3 above (the backlog), where a linear rule could only have one slope:

rule_coefficient(model3[:y][2], model3[:d][2])

# ## Simulating the policy

# [`simulate`](@ref) applies the primal rule to scenarios of all the uncertainties, drawn from their
# laws (in the original coordinates, not the pieces). The mean of the objective estimates
# `objective_value`, and the rule has no violation on the support:

sim = simulate(model3, 10_000; keep_decisions = true, rng = Xoshiro(1))

#-

objective_value(model3), sim.objective_mean, sim.objective_stderr

# With `keep_decisions = true`, the value of every decision at every scenario is kept. The mean
# order of each period:

[sum(sim.decisions[model3[:x][t]]) / 10_000 for t in 1:4]

# [`evaluate`](@ref) applies the rule to one scenario, and gives the whole trajectory of the
# decisions. Let us follow the orders and the inventory with the demands 70, 90, 70 and 80:

scenario = model3[:d] .=> [70.0, 90.0, 70.0, 80.0]
trajectory = evaluate(model3, scenario...)

#-

println("t   demand   order    inventory   cost")
for t in 1:4
    demand = trajectory[model3[:d][t]]
    row = [trajectory[model3[v][t]] for v in (:x, :I, :y)]
    println(
        rpad(t, 4),
        rpad(demand, 9),
        rpad(round(row[1]; digits = 2), 9),
        rpad(round(row[2]; digits = 2), 12),
        round(row[3]; digits = 2),
    )
end
objective_value(trajectory)

# The scenarios can also come from elsewhere, for example the simulations of another method such as
# SDDP: a scenario maps each container of uncertainties to an array of values of the same size.
# We evaluate two extreme scenarios, with low and high demands:

low, high = fill(60.0, 4), fill(100.0, 4)
simulate(model3, [Dict(model3[:d] => low), Dict(model3[:d] => high)])

# ## What's next?

# * Learn about the other laws in the [distributions](@ref distributions_tutorial) tutorial.
# * See [`rule_matrix`](@ref), [`decision_rule`](@ref), [`evaluate`](@ref) and [`simulate`](@ref) in
#   the API.
