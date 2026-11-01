# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The inventory of Kuhn, Wiesemann and Georghiou, "Primal and dual linear decision rules in
# stochastic and robust optimization", Math. Program. 130 (2011), section 5, with the data of Ben-Tal
# et al. (2004): a warehouse supplied by three factories over T semimonthly periods. It is a test
# problem (`test_kwg.jl`) and a benchmark of `benchmark/compare`.
#
# In period t the demand d[t] is revealed, then each factory i produces x[i, t] ∈ [0, 567] at the
# cost α[i] ς[t], with α = (1, 1.5, 2) and the season ς[t] = 1 + sin(π (t - 1) / 12) / 2. The demand
# is uniform on [(1 - θ) 1000 ς[t], (1 + θ) 1000 ς[t]], independent across periods. The stock
# x0 + (production) - (demand) up to t must stay in [500, 2000] for every t, and each factory
# produces at most 13,600 T / 24 over the horizon. Stage t holds d[t] and x[:, t], so the
# production of period t depends on the demands of periods 1 to t.
#
# The paper does not give the initial stock in its text; x0 = 1000 reproduces its Table 1.
#
# This file defines functions only.

using JuMP
import Distributions
import HiGHS
import LinearDecisionRules as LDR

# The paper's Table 1 (θ = 0.3): the lower bound (dual LDR) and upper bound (primal LDR) for
# T = 1, ..., 10, as printed (one decimal).
const KWG_TABLE_1 = [
    (508.3, 558.3),
    (1972.7, 2032.6),
    (3825.5, 4005.3),
    (6090.7, 6356.0),
    (8665.4, 9064.0),
    (11483.9, 12047.5),
    (14433.5, 15182.7),
    (17434.4, 18329.3),
    (20255.9, 21279.0),
    (22769.3, 23869.9),
]

kwg_season(t) = 1 + sin(π * (t - 1) / 12) / 2

"""
    kwg_model(T; formulations = [LDR.PrimalLDR()], breakpoints = nothing, θ = 0.3, x0 = 1000.0,
              cumulative = true)

The KWG inventory over `T` periods. `breakpoints` (a count, or `nothing`) applies to every demand;
`cumulative = false` drops the capacity of each factory over the horizon, as in Bodur and Luedtke
(2022).
"""
function kwg_model(
    T;
    formulations = [LDR.PrimalLDR()],
    breakpoints = nothing,
    θ = 0.3,
    x0 = 1000.0,
    cumulative = true,
)
    model = LDR.LDRModel(HiGHS.Optimizer)
    set_silent(model)
    set_attribute(model, LDR.Formulations(), formulations)
    ς = kwg_season.(1:T)
    @variable(
        model,
        d[t = 1:T] in
        LDR.Uncertainty(Distributions.Uniform((1 - θ) * 1000 * ς[t], (1 + θ) * 1000 * ς[t])),
        LDR.Stage(t)
    )
    if breakpoints !== nothing
        set_attribute.(d, LDR.BreakPoints(), breakpoints)
    end
    @variable(model, 0 <= x[i = 1:3, t = 1:T] <= 567, LDR.Stage(t))
    if cumulative
        @constraint(model, [i = 1:3], sum(x[i, :]) <= 13600 * T / 24)
    end
    @constraint(
        model,
        [t = 1:T],
        500 <= x0 + sum(x[i, s] for i in 1:3, s in 1:t) - sum(d[1:t]) <= 2000
    )
    @objective(model, Min, sum([1.0, 1.5, 2.0][i] * ς[t] * x[i, t] for i in 1:3, t in 1:T))
    return model
end
