# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Hydrothermal benchmark: the 4-reservoir, 5-subsystem problem of Shapiro et al. (2013), written
# three ways so that the syntax can be compared side by side.
#
#   ldr_model              one monolithic model with `Uncertainty` and `Stage` (linear decision rules)
#   average_scenario_model plain JuMP, the inflows at their means
#   scenario_tree_model    plain JuMP, a tree with `branches` joint inflow vectors per stage
#
# Stage convention (the one of the previous repository's `monolithic_vs_sddp.jl`): stage t is month
# t. Its decisions (generation, exchange, deficit, spill, and the stored energy at the end of the
# month) are taken after the inflows of month t are known; stage 1 has the known inflows
# `data.inflow_initial`, stages 2:T have random inflows. So there are T - 1 uncertain vectors of
# dimension 4, independent across stages, and stage t's inflow vector is uncertainty of stage t.
# Stage t's inflows follow the 82 equiprobable joint scenarios of month `inflow_month(t)`.
#
# This file defines functions only; include it after `data.jl`.

using JuMP
import Distributions
import HiGHS
import LinearDecisionRules as LDR
import Random

const REES = 1:4
const SUBSYS = 1:5

month_of(t) = t % 12 == 0 ? 12 : t % 12

# Month whose scenarios are the inflows of stage t >= 2.
inflow_month(t) = (t - 1) % 12 == 0 ? 12 : (t - 1) % 12

# The joint law of the four inflows of stage t: 82 equiprobable scenarios.
function inflow_law(data, t)
    r = inflow_month(t)
    n = length(data.inflow_scenarios[1][r])
    atoms = [[data.inflow_scenarios[i][r][ω] for i in REES] for ω in 1:n]
    return LDR.MvDiscreteNonParametric(atoms, fill(1 / n, n))
end

# The decisions of one stage. They are anonymous containers, so that a model can hold many
# stages (one per node in a tree).
function stage_variables(m, data, t; name = "")
    month = month_of(t)
    n_thermal = length.(data.thermal_ub)
    spill = @variable(m, [REES], lower_bound = 0, base_name = "spill$name")
    hydro = @variable(m, [i in REES], lower_bound = 0, upper_bound = data.hydro_ub[i], base_name = "hydro$name")
    thermal = @variable(
        m,
        [i in REES, j in 1:n_thermal[i]],
        lower_bound = data.thermal_lb[i][j],
        upper_bound = data.thermal_ub[i][j],
        base_name = "thermal$name",
    )
    exchange = @variable(
        m,
        [i in SUBSYS, j in SUBSYS],
        lower_bound = 0,
        upper_bound = data.exchange_ub[i][j],
        base_name = "exchange$name",
    )
    deficit = @variable(
        m,
        [i in REES, j in 1:4],
        lower_bound = 0,
        upper_bound = data.demand[month][i] * data.deficit_ub[j],
        base_name = "deficit$name",
    )
    stored = @variable(
        m,
        [i in REES],
        lower_bound = 0,
        upper_bound = data.stored_energy_ub[i],
        base_name = "stored$name",
    )
    return (; spill, hydro, thermal, exchange, deficit, stored)
end

# Energy balance of each subsystem, water balance of each reservoir (`inflow[i]` may be an
# expression in uncertainties), and the balance of the transshipment node.
function stage_constraints(m, data, t, v, stored_in, inflow)
    month = month_of(t)
    scale = data.water_scale / data.energy_scale
    @constraint(m, [i in REES], sum(v.deficit[i, :]) + scale * v.hydro[i] + sum(v.thermal[i, :]) + sum(v.exchange[:, i]) - sum(v.exchange[i, :]) == data.demand[month][i])
    @constraint(m, [i in REES], v.stored[i] + v.spill[i] + v.hydro[i] - stored_in[i] == inflow[i])
    @constraint(m, sum(v.exchange[:, 5]) == sum(v.exchange[5, :]))
    return
end

function stage_cost(data, v)
    deficit = sum(data.deficit_obj[j] * sum(v.deficit[:, j]) for j in 1:4)
    thermal = sum(data.thermal_obj[i][j] * v.thermal[i, j] for i in REES for j in 1:length(data.thermal_obj[i]))
    return deficit + thermal
end

# The monolithic multistage model. `formulations` is the list given to `LDR.Formulations`.
function ldr_model(data, T; formulations = [LDR.PrimalLDR()])
    m = LDR.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    set_attribute(m, LDR.Formulations(), formulations)
    cost = AffExpr(0.0)
    stored_in = data.stored_energy_initial
    for t in 1:T
        v = stage_variables(m, data, t; name = "$t")
        for x in (v.spill, v.hydro, v.thermal, v.exchange, v.deficit, v.stored)
            LDR.set_stage.(x, t)
        end
        if t == 1
            inflow = data.inflow_initial
        else
            inflow = @variable(m, [1:4] in LDR.Uncertainty(inflow_law(data, t)), LDR.Stage(t), base_name = "inflow$t")
        end
        stage_constraints(m, data, t, v, stored_in, inflow)
        add_to_expression!(cost, stage_cost(data, v))
        stored_in = v.stored
    end
    @objective(m, Min, cost)
    return m
end

# The deterministic model with every inflow at its mean.
function average_scenario_model(data, T)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    cost = AffExpr(0.0)
    stored_in = data.stored_energy_initial
    for t in 1:T
        v = stage_variables(m, data, t; name = "$t")
        inflow = t == 1 ? data.inflow_initial : Distributions.mean(inflow_law(data, t))
        stage_constraints(m, data, t, v, stored_in, inflow)
        add_to_expression!(cost, stage_cost(data, v))
        stored_in = v.stored
    end
    @objective(m, Min, cost)
    return m
end

# Indices (into the 82 scenarios) of the `branches` joint inflow vectors of each stage 2:T: drawn
# without replacement, from a fixed seed, so that trees with the same `branches` agree up to T.
function tree_scenarios(data, T, branches; seed = 1234)
    rng = Random.Xoshiro(seed)
    return [Random.randperm(rng, length(data.inflow_scenarios[1][inflow_month(t)]))[1:branches] for t in 2:T]
end

# A scenario tree: every node of stage t has `branches` children, each with one joint inflow
# vector (all four reservoirs branch together) and probability 1 / branches. A node has its own
# copy of the stage variables, so the policy is nonanticipative by construction. The tree has
# branches^(T-1) leaves.
function scenario_tree_model(data, T, branches; seed = 1234)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    scenarios = tree_scenarios(data, T, branches; seed)
    cost = AffExpr(0.0)
    # Adds the node of stage t with the given inflow, and its subtree.
    function add_node!(t, stored_in, inflow, probability, name)
        v = stage_variables(m, data, t; name)
        stage_constraints(m, data, t, v, stored_in, inflow)
        add_to_expression!(cost, probability, stage_cost(data, v))
        t == T && return
        r = inflow_month(t + 1)
        for b in 1:branches
            ω = scenarios[t][b]
            next_inflow = [data.inflow_scenarios[i][r][ω] for i in REES]
            add_node!(t + 1, v.stored, next_inflow, probability / branches, "$(name)_$b")
        end
        return
    end
    add_node!(1, data.stored_energy_initial, data.inflow_initial, 1.0, "")
    @objective(m, Min, cost)
    return m
end
