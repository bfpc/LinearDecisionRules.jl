# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The problems written as a user would, with `LDRModel` and the tags. Each one is the twin of a
# problem of `library.jl`, which the core tests build with a `ModelInfo` by hand, so that they do
# not depend on the JuMP layer; `test_twins.jl` checks that both give the same canonical form.
#
# Every builder adds the problem to `model` (by default `ldr_model()`), so that a test can choose
# the optimizer and its settings.

module TaggedProblems

using JuMP

import Distributions
import Distributions: Uniform
import HiGHS
import LinearDecisionRules:
    BreakPoints,
    Experimental,
    FirstStage,
    Formulations,
    LDRModel,
    MvDiscreteNonParametric,
    Stage,
    Uncertainty

# A silent `LDRModel` with HiGHS; the keywords set the formulations, a time limit and attributes
# of the inner optimizer, such as `"mip_rel_gap" => 0.0`.
function ldr_model(; formulations = nothing, time_limit = nothing, attributes = ())
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    if formulations !== nothing
        set_attribute(model, Formulations(), formulations)
    end
    if time_limit !== nothing
        set_time_limit_sec(model, time_limit)
    end
    for (name, value) in attributes
        set_attribute(model, name, value)
    end
    return model
end

# Buy before the demand is known, then sell or return. The law and the breakpoints of the demand
# can change; the known values of `library.jl` are those of the default law.
function newsvendor(model = ldr_model(); integer = false, law = Uniform(80, 120), points = nothing)
    @variable(model, buy >= 0, FirstStage, integer = integer)
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @variable(model, demand in Uncertainty(law))
    if points !== nothing
        set_attribute(demand, BreakPoints(), points)
    end
    @constraint(model, sell + ret <= buy)
    @constraint(model, sell <= demand)
    @objective(model, Max, -10buy + 8ret + 15sell)
    return model
end

function newsvendor_random_price(model = ldr_model())
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, ret >= 0)
    @variable(model, demand in Uncertainty(Uniform(80, 120)))
    @variable(model, price in Uncertainty(Uniform(13, 17)))
    @constraint(model, sell + ret <= buy)
    @constraint(model, sell <= demand)
    @objective(model, Max, -10buy + 8ret + price * sell)
    return model
end

function newsvendor_two_products(model = ldr_model())
    @variable(model, buy[1:2] >= 0, FirstStage)
    @variable(model, sell[1:2] >= 0)
    @variable(model, ret[1:2] >= 0)
    law = Distributions.product_distribution([Uniform(80, 120) for _ in 1:2])
    @variable(model, demand[1:2] in Uncertainty(law))
    @constraint(model, [i in 1:2], sell[i] + ret[i] <= buy[i])
    @constraint(model, [i in 1:2], sell[i] <= demand[i])
    @objective(model, Max, sum(-10buy[i] + 8ret[i] + 15sell[i] for i in 1:2))
    return model
end

# With `unmarked_ret3 = true`, `ret3` has no `Stage`, and the stage of the model is inferred.
function newsvendor_three_stages(model = ldr_model(); unmarked_ret3 = false)
    @variable(model, buy >= 0, Stage(1))
    @variable(model, sell2 >= 0, Stage(2))
    @variable(model, ret2 >= 0, Stage(2))
    @variable(model, demand2 in Uncertainty(Uniform(80, 120)), Stage(2))
    @variable(model, buy2 >= 0, Stage(2))
    @variable(model, buy_cost2 in Uncertainty(Uniform(8, 14)), Stage(2))
    @variable(model, sell3 >= 0, Stage(3))
    if unmarked_ret3
        @variable(model, ret3 >= 0)
    else
        @variable(model, ret3 >= 0, Stage(3))
    end
    @variable(model, demand3 in Uncertainty(Uniform(80, 120)), Stage(3))
    @constraint(model, sell2 + ret2 <= buy + buy2)
    @constraint(model, sell2 <= demand2)
    @constraint(model, sell2 + ret2 + sell3 + ret3 <= buy + buy2)
    @constraint(model, sell3 <= demand3)
    @objective(model, Max, -10buy - buy_cost2 * buy2 + 8ret2 + 15sell2 + 8ret3 + 15sell3)
    return model
end

# The inventory with backlog of the benchmarks (`benchmark/timing.jl`), whose primal value is
# 90T + 20 for an initial stock `x0 = 0`. With `parameter = true`, `x0` is the value of the JuMP
# parameter `stock`; `points` are the breakpoints of every demand. `combinations = (:pair, k)` puts
# `k` breakpoints on the demand of each two consecutive periods, and `(:cumulative, k)` on the
# cumulative demand up to each period after the first (`Experimental.set_breakpoints`; the moments
# of a cumulative demand integrate over all its periods, so keep T small).
function inventory(
    T::Int,
    model = ldr_model();
    x0 = 0.0,
    parameter = false,
    points = nothing,
    combinations = nothing,
)
    @variable(model, d[t = 1:T] in Uncertainty(Uniform(60, 100)), Stage(t + 1))
    if points !== nothing
        set_attribute.(d, BreakPoints(), points)
    end
    if combinations !== nothing
        kind, k = combinations
        for t in 2:T
            f = kind == :pair ? d[t-1] + d[t] : sum(d[1:t])
            Experimental.set_breakpoints(model, f, k)
        end
    end
    @variable(model, 0 <= x[t = 1:T] <= 150, Stage(t))
    @variable(model, I[t = 1:T], Stage(t + 1))
    @variable(model, y[t = 1:T], Stage(t + 1))
    stock = parameter ? @variable(model, stock in Parameter(x0)) : x0
    @constraint(model, I[1] == stock + x[1] - d[1])
    @constraint(model, [t in 2:T], I[t] == I[t-1] + x[t] - d[t])
    @constraint(model, [t in 1:T], y[t] >= 0.5 * I[t])
    @constraint(model, [t in 1:T], y[t] >= -3 * I[t])
    @objective(model, Min, sum(x[t] + y[t] for t in 1:T))
    return model
end

function hydro_toy(model = ldr_model())
    @variable(model, vi == 0.5)
    @variable(model, 0 <= vf <= 1)
    @variable(model, gh >= 0)
    @variable(model, gt >= 0)
    @variable(model, inflow[i = 1:3] in Uncertainty(Uniform(0, 0.1i)))
    @constraint(model, vf == vi - gh + sum(inflow))
    @constraint(model, gt + gh == 0.3)
    @objective(model, Min, gt^2 + vf^2 / 2 - vf)
    return model
end

# The newsvendor of the 0.3 tutorials (buy 10, sell 15, salvage 5; they now salvage at 8), with a
# discrete demand.
function newsvendor_discrete(law, model = ldr_model())
    k = length(law)
    @variable(model, buy[1:k] >= 0, FirstStage)
    @variable(model, sell[1:k] >= 0)
    @variable(model, salvage[1:k] >= 0)
    @variable(model, demand[1:k] in Uncertainty(law))
    @constraint(model, [i in 1:k], sell[i] + salvage[i] <= buy[i])
    @constraint(model, [i in 1:k], sell[i] <= demand[i])
    @objective(model, Max, sum(-10buy[i] + 15sell[i] + 5salvage[i] for i in 1:k))
    return model
end

function newsvendor_mv_discrete(model = ldr_model())
    scenarios = [[80.0, 60.0], [100.0, 80.0], [120.0, 100.0], [80.0, 100.0], [120.0, 60.0]]
    law = MvDiscreteNonParametric(scenarios, [0.2, 0.3, 0.2, 0.15, 0.15])
    return newsvendor_discrete(law, model)
end

function newsvendor_discrete_demand(model = ldr_model())
    law = Distributions.DiscreteNonParametric([80.0, 90, 100, 110, 120], [0.1, 0.2, 0.4, 0.2, 0.1])
    return newsvendor_discrete(law, model)
end

end # module TaggedProblems
