# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The production planning problem of Georghiou, Kuhn and Wiesemann, "The decision rule approach to
# optimization under uncertainty: methodology and applications", Comput. Manag. Sci. 16 (2019),
# section 6.1, without changeovers: one machine makes five products over T weeks, with backlog. It
# is a test problem (`test_production.jl`) and a benchmark of `benchmark/breakpoints` and
# `benchmark/scaling`.
#
# At the start of week t the demands ξ[i, t] are revealed; the plant sells s[i, t] from the stock
# made in earlier weeks, backlogs b[i, t] and holds I[i, t] ∈ [0, 10⁶], then makes p[i, t] for the
# next week (t < T), at r[i] units per hour on a machine that runs R = 168 hours a week. It
# maximizes the expected sales minus the backlog and holding costs. The demands of week 1 are known;
# those of week t > 1 are uniform on [(1 - θ) ξ̄[i, t], (1 + θ) ξ̄[i, t]], independent, with the
# nominal demand ξ̄[i, t] = (1 + sin(π (t - 2) / 26) / 2) d̄[i]. Stage t holds the demands and the
# decisions of week t. There is no initial stock or backlog. The data are Table 1 of the paper.
#
# The uncertain variables can be the demands themselves (`parametrization = :raw`, of order 10⁴),
# their deviations from ξ̄ (`:centered`), or these deviations divided by θ ξ̄, in [-1, 1] (`:unit`).
# The demand is an affine function of the variable, and a count of breakpoints is equally spaced in
# the support, so the optimum is the same; only the scaling of the inner problem changes, and only
# with `RescaleUncertainties()` set to `false` (by default, the optimizer rescales the uncertainties
# to `[-1, 1]` itself).
#
# This file defines functions only.

using JuMP
import Distributions
import HiGHS
import LinearDecisionRules as LDR

const PRODUCTION_DATA = (
    average_demand = [10_000.0, 25_000.0, 30_000.0, 30_000.0, 30_000.0],
    price = [0.25, 0.40, 0.65, 0.55, 0.45],
    backlog_cost = [0.05, 0.08, 0.13, 0.11, 0.09],
    rate = [800.0, 900.0, 1_000.0, 1_000.0, 1_200.0],
    holding_cost = 3.06e-5,
    max_stock = 1e6,
    uptime = 168.0,
)

production_nominal(t, i) = (1 + sin(π * (t - 2) / 26) / 2) * PRODUCTION_DATA.average_demand[i]

"""
    production_model(T; formulations = [LDR.PrimalLDR()], breakpoints = nothing, θ = 0.2,
                     parametrization = :raw)

The production planning problem over `T ≥ 2` weeks; `breakpoints` (a count, or `nothing`) applies
to every uncertain demand. `parametrization` is `:raw`, `:centered` or `:unit` (see above).
"""
function production_model(
    T;
    formulations = [LDR.PrimalLDR()],
    breakpoints = nothing,
    θ = 0.2,
    parametrization = :raw,
)
    data = PRODUCTION_DATA
    products = eachindex(data.average_demand)
    model = LDR.LDRModel(HiGHS.Optimizer)
    set_silent(model)
    set_attribute(model, LDR.Formulations(), formulations)
    ξ̄ = [production_nominal(t, i) for i in products, t in 1:T]
    # The demand is `center + scale * ξ`, with ξ uniform on [lo, hi].
    center, scale, lo, hi = if parametrization == :raw
        zero(ξ̄), one.(ξ̄), (1 - θ) * ξ̄, (1 + θ) * ξ̄
    elseif parametrization == :centered
        ξ̄, one.(ξ̄), -θ * ξ̄, θ * ξ̄
    elseif parametrization == :unit
        ξ̄, θ * ξ̄, -one.(ξ̄), one.(ξ̄)
    else
        error("Unknown parametrization $parametrization.")
    end
    @variable(
        model,
        ξ[i = products, t = 2:T] in LDR.Uncertainty(Distributions.Uniform(lo[i, t], hi[i, t])),
        LDR.Stage(t)
    )
    if breakpoints !== nothing
        set_attribute.(ξ, LDR.BreakPoints(), breakpoints)
    end
    demand(i, t) = t == 1 ? ξ̄[i, 1] : center[i, t] + scale[i, t] * ξ[i, t]
    @variable(model, p[i = products, t = 1:T-1] >= 0, LDR.Stage(t))
    @variable(model, s[i = products, t = 1:T] >= 0, LDR.Stage(t))
    @variable(model, b[i = products, t = 1:T] >= 0, LDR.Stage(t))
    @variable(model, 0 <= I[i = products, t = 1:T] <= data.max_stock, LDR.Stage(t))
    @constraint(model, [t = 1:T-1], sum(p[i, t] / data.rate[i] for i in products) <= data.uptime)
    @constraint(model, [i = products], b[i, 1] == demand(i, 1) - s[i, 1])
    @constraint(model, [i = products], I[i, 1] == -s[i, 1])
    @constraint(model, [i = products, t = 2:T], b[i, t] == b[i, t-1] + demand(i, t) - s[i, t])
    @constraint(model, [i = products, t = 2:T], I[i, t] == I[i, t-1] + p[i, t-1] - s[i, t])
    @objective(
        model,
        Max,
        sum(
            data.price[i] * s[i, t] - data.backlog_cost[i] * b[i, t] - data.holding_cost * I[i, t]
            for i in products, t in 1:T
        )
    )
    return model
end
