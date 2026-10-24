# Generates `legacy_values.toml`: golden values from the LEGACY implementation of
# LinearDecisionRules.jl (branch `Multistage` of the old repository, version 0.3.1, old API:
# `LDRModel`, `Uncertainty`, `Stage`, `get_decision`, ...). It does NOT run against this repository.
#
# To run it, from the old repository checkout (an environment with the old package developed
# plus HiGHS, Ipopt, Distributions, JuMP, TOML; `spike/legacy_env` there is one):
#
#   julia --project=<old repo>/spike/legacy_env test/reference/generate_legacy_values.jl [out.toml]
#
# The output defaults to `legacy_values.toml` next to this script. Every entry has a `description`
# precise enough to rebuild the model in another API, the `primal` and `dual` objective values
# (user's sense), optionally `sampled`, and `rules`: for a few variables the constant and the
# coefficients on the uncertainties of the primal (and dual) decision rule. Models come from the
# legacy tests (test_core.jl, test_stages.jl, distributions.jl, sampled_mode.jl), the
# tutorials and benchmark/timing.jl. HiGHS solves everything; Ipopt is only used if HiGHS fails.

import TOML
import HiGHS
import Ipopt
import Distributions
import LinearDecisionRules as LDR
using JuMP

const OUT = isempty(ARGS) ? joinpath(@__DIR__, "legacy_values.toml") : ARGS[1]

# `rules` maps a name to (variable, [(uncertainty name, uncertainty variable), ...]).
struct Problem
    description::String
    model::LDR.LDRModel
    rules::Dict{String,Any}
    sampled::Bool
    pieces::Int  # pieces of the piecewise-linear lifting of the uncertainty; 0 if none
end

function _model(; sampled = false, scenarios = 0)
    m = LDR.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    if sampled
        set_attribute(m, LDR.SolveSampled(), true)
        set_attribute(m, LDR.NumScenarios(), scenarios)
    end
    return m
end

# Newsvendor of test_core.jl: buy 10, return 8, sell 15, demand ~ Uniform(80, 120).
function _newsvendor(m; integer = false)
    if integer
        @variable(m, buy >= 0, LDR.FirstStage, integer = true)
    else
        @variable(m, buy >= 0, LDR.FirstStage)
    end
    @variable(m, sell >= 0)
    @variable(m, ret >= 0)
    @variable(m, demand in LDR.Uncertainty(; distribution = Distributions.Uniform(80, 120)))
    @constraint(m, sell + ret <= buy)
    @constraint(m, sell <= demand)
    @objective(m, Max, -10 * buy + 8 * ret + 15 * sell)
    return (; buy, sell, ret, demand)
end

_nv_rules(v) = Dict{String,Any}(
    "buy" => (v.buy, [("demand", v.demand)]),
    "sell" => (v.sell, [("demand", v.demand)]),
    "ret" => (v.ret, [("demand", v.demand)]),
)

const NV = "Newsvendor: first-stage variable buy >= 0, recourse sell >= 0 and ret >= 0, " *
    "uncertainty demand ~ Uniform(80, 120). Constraints sell + ret <= buy and sell <= demand. " *
    "Maximize -10 buy + 8 ret + 15 sell. All decisions except buy may depend on demand " *
    "(single stage after buy)."

function problem_newsvendor()
    m = _model()
    v = _newsvendor(m)
    return Problem(NV, m, _nv_rules(v), false, 0)
end

function problem_newsvendor_integer()
    m = _model()
    v = _newsvendor(m; integer = true)
    d = NV * " The first-stage variable buy is additionally integer."
    return Problem(d, m, _nv_rules(v), false, 0)
end

function problem_newsvendor_random_price()
    m = _model()
    @variable(m, buy >= 0, LDR.FirstStage)
    @variable(m, sell >= 0)
    @variable(m, ret >= 0)
    @variable(m, demand in LDR.Uncertainty(; distribution = Distributions.Uniform(80, 120)))
    @variable(m, price in LDR.Uncertainty(; distribution = Distributions.Uniform(13, 17)))
    @constraint(m, sell + ret <= buy)
    @constraint(m, sell <= demand)
    @objective(m, Max, -10 * buy + 8 * ret + price * sell)
    d = "Newsvendor with random sell price: buy >= 0 first stage, sell >= 0, ret >= 0; " *
        "independent uncertainties demand ~ Uniform(80, 120) and price ~ Uniform(13, 17), each " *
        "declared in its own @variable call. Constraints sell + ret <= buy, sell <= demand. " *
        "Maximize -10 buy + 8 ret + price * sell (the product price * sell is quadratic in the " *
        "decision rule of sell)."
    rules = Dict{String,Any}(
        "buy" => (buy, [("demand", demand), ("price", price)]),
        "sell" => (sell, [("demand", demand), ("price", price)]),
    )
    return Problem(d, m, rules, false, 0)
end

function problem_newsvendor_two_products()
    m = _model()
    @variable(m, buy[1:2] >= 0, LDR.FirstStage)
    @variable(m, sell[1:2] >= 0)
    @variable(m, ret[1:2] >= 0)
    dist = Distributions.product_distribution([Distributions.Uniform(80, 120) for _ in 1:2])
    @variable(m, demand[1:2] in LDR.Uncertainty(; distribution = dist))
    @constraint(m, [i = 1:2], sell[i] + ret[i] <= buy[i])
    @constraint(m, [i = 1:2], sell[i] <= demand[i])
    @objective(m, Max, sum(-10 * buy[i] + 8 * ret[i] + 15 * sell[i] for i in 1:2))
    d = "Two independent newsvendors (test_core.jl test_double_newsvendor): buy[1:2] >= 0 first " *
        "stage, sell[1:2] >= 0, ret[1:2] >= 0; demand[1:2] one vector uncertainty, product of " *
        "two independent Uniform(80, 120). Constraints sell[i] + ret[i] <= buy[i], sell[i] <= " *
        "demand[i]. Maximize sum_i (-10 buy[i] + 8 ret[i] + 15 sell[i]). Value is twice the " *
        "newsvendor value."
    rules = Dict{String,Any}(
        "buy1" => (buy[1], [("demand1", demand[1]), ("demand2", demand[2])]),
        "sell1" => (sell[1], [("demand1", demand[1]), ("demand2", demand[2])]),
    )
    return Problem(d, m, rules, false, 0)
end

function problem_newsvendor_three_stages()
    m = _model()
    S = LDR.Stage
    U = LDR.Uncertainty
    cost2 = Distributions.Uniform(8.0, 14.0)
    dem = Distributions.Uniform(80, 120)
    @variable(m, buy >= 0, S(1))
    @variable(m, sell2 >= 0, S(2))
    @variable(m, ret2 >= 0, S(2))
    @variable(m, demand2 in U(distribution = dem, stage = 2))
    @variable(m, buy2 >= 0, S(2))
    @variable(m, buy_cost2 in U(distribution = cost2, stage = 2))
    @variable(m, sell3 >= 0, S(3))
    @variable(m, ret3 >= 0, S(3))
    @variable(m, demand3 in U(distribution = dem, stage = 3))
    @constraint(m, sell2 + ret2 <= buy + buy2)
    @constraint(m, sell2 <= demand2)
    @constraint(m, sell2 + ret2 + sell3 + ret3 <= buy + buy2)
    @constraint(m, sell3 <= demand3)
    @objective(m, Max, -10 * buy - buy_cost2 * buy2 + 8 * ret2 + 15 * sell2 + 8 * ret3 + 15 * sell3)
    d = "Three-stage newsvendor (test_stages.jl). Stage 1: buy >= 0. Stage 2: sell2, ret2, buy2 " *
        ">= 0, after observing demand2 ~ Uniform(80, 120) and buy_cost2 ~ Uniform(8, 14) " *
        "(both uncertainties of stage 2). Stage 3: sell3, ret3 >= 0 after demand3 ~ Uniform(80, " *
        "120) (stage 3). All uncertainties independent, each in its own @variable call. " *
        "Constraints: sell2 + ret2 <= buy + buy2; sell2 <= demand2; sell2 + ret2 + sell3 + ret3 " *
        "<= buy + buy2; sell3 <= demand3. Maximize -10 buy - buy_cost2 buy2 + 8 ret2 + 15 sell2 " *
        "+ 8 ret3 + 15 sell3. A decision of stage t may depend on the uncertainties of stages " *
        "<= t only."
    rules = Dict{String,Any}(
        "buy" => (buy, [("demand2", demand2), ("demand3", demand3)]),
        "sell2" => (sell2, [("demand2", demand2), ("buy_cost2", buy_cost2), ("demand3", demand3)]),
        "sell3" => (sell3, [("demand2", demand2), ("demand3", demand3)]),
    )
    return Problem(d, m, rules, false, 0)
end

# Inventory with backlog of benchmark/timing.jl.
function problem_inventory(T)
    lo, hi, c, h, b, xmax, I0 = 60.0, 100.0, 1.0, 0.5, 3.0, 150.0, 0.0
    m = _model()
    dist = Distributions.Uniform(lo, hi)
    @variable(m, d[t = 1:T] in LDR.Uncertainty(; distribution = dist, stage = t + 1))
    @variable(m, 0 <= x[t = 1:T] <= xmax, LDR.Stage(t))
    @variable(m, I[t = 1:T], LDR.Stage(t + 1))
    @variable(m, y[t = 1:T], LDR.Stage(t + 1))
    @constraint(m, I[1] == I0 + x[1] - d[1])
    @constraint(m, [t = 2:T], I[t] == I[t-1] + x[t] - d[t])
    @constraint(m, [t = 1:T], y[t] >= h * I[t])
    @constraint(m, [t = 1:T], y[t] >= -b * I[t])
    @objective(m, Min, sum(c * x[t] + y[t] for t in 1:T))
    desc = "Inventory with backlog, T = $T periods (benchmark/timing.jl). Uncertainties d[t], " *
        "t = 1..T, independent Uniform(60, 100), d[t] of stage t + 1 (one @variable call with " *
        "stage = t + 1). Decisions: order x[t] in [0, 150] of stage t; inventory I[t] and cost " *
        "y[t] (both free) of stage t + 1. Constraints I[1] == 0 + x[1] - d[1]; I[t] == I[t-1] + " *
        "x[t] - d[t] for t >= 2; y[t] >= 0.5 I[t]; y[t] >= -3 I[t]. Minimize sum_t (1.0 x[t] + " *
        "y[t])."
    rules = Dict{String,Any}(
        "x1" => (x[1], [("d1", d[1])]),
        "x2" => (x[2], [("d1", d[1]), ("d2", d[2])]),
    )
    return Problem(desc, m, rules, false, 0)
end

# 1-D hydro toy of test_core.jl (test_recursion, second model), quadratic objective.
function problem_hydro_toy_fixed_volume()
    m = _model()
    @variable(m, vi == 0.5)
    @variable(m, 0 <= vf <= 1)
    @variable(m, gh >= 0)
    @variable(m, gt >= 0)
    @variable(
        m,
        inflow[i = 1:3] in LDR.Uncertainty(; distribution = Distributions.Uniform(0, 0.1 * i)),
    )
    @constraint(m, vf == vi - gh + sum(inflow[i] for i in 1:3))
    @constraint(m, gt + gh == 0.3)
    @objective(m, Min, gt^2 + vf^2 / 2 - vf)
    d = "Hydro toy, one reservoir, quadratic objective (test_core.jl test_recursion, the second " *
        "model with the cost-to-go vf^2/2 - vf added). Fixed initial volume vi == 0.5 (a fixed " *
        "variable, not random). Decisions vf in [0, 1], gh >= 0, gt >= 0. Uncertainties " *
        "inflow[i] ~ Uniform(0, 0.1 i), i = 1..3, independent. Constraints vf == vi - gh + " *
        "inflow[1] + inflow[2] + inflow[3]; gt + gh == 0.3. Minimize gt^2 + vf^2/2 - vf " *
        "(convex QP). Single stage: all decisions may depend on all inflows."
    rules = Dict{String,Any}(
        "gh" => (gh, [("inflow$i", inflow[i]) for i in 1:3]),
        "vf" => (vf, [("inflow$i", inflow[i]) for i in 1:3]),
    )
    return Problem(d, m, rules, false, 0)
end

# Tutorial model (docs distributions.jl): buy 10, sell 15, salvage 5.
function _tutorial_inventory(m, dist; products = 1)
    if products == 1
        @variable(m, buy >= 0, LDR.FirstStage)
        @variable(m, sell >= 0)
        @variable(m, salvage >= 0)
        @variable(m, demand in LDR.Uncertainty(; distribution = dist))
        @constraint(m, sell + salvage <= buy)
        @constraint(m, sell <= demand)
        @objective(m, Max, -10 * buy + 15 * sell + 5 * salvage)
        return Dict{String,Any}(
            "buy" => (buy, [("demand", demand)]),
            "sell" => (sell, [("demand", demand)]),
        )
    end
    @variable(m, buy[1:2] >= 0, LDR.FirstStage)
    @variable(m, sell[1:2] >= 0)
    @variable(m, salvage[1:2] >= 0)
    @variable(m, demand[1:2] in LDR.Uncertainty(; distribution = dist))
    for i in 1:2
        @constraint(m, sell[i] + salvage[i] <= buy[i])
        @constraint(m, sell[i] <= demand[i])
    end
    @objective(m, Max, sum(-10 * buy[i] + 15 * sell[i] + 5 * salvage[i] for i in 1:2))
    return Dict{String,Any}(
        "buy1" => (buy[1], [("demand1", demand[1]), ("demand2", demand[2])]),
        "sell1" => (sell[1], [("demand1", demand[1]), ("demand2", demand[2])]),
    )
end

function problem_newsvendor_discrete_demand()
    m = _model()
    values = [80.0, 90.0, 100.0, 110.0, 120.0]
    dist = Distributions.DiscreteNonParametric(values, [0.1, 0.2, 0.4, 0.2, 0.1])
    rules = _tutorial_inventory(m, dist)
    d = "Tutorial newsvendor (docs distributions.jl): buy >= 0 first stage, sell >= 0, salvage " *
        ">= 0; demand ~ DiscreteNonParametric on [80, 90, 100, 110, 120] with probabilities " *
        "[0.1, 0.2, 0.4, 0.2, 0.1]. Constraints sell + salvage <= buy, sell <= demand. Maximize " *
        "-10 buy + 15 sell + 5 salvage. Note: salvage value 5, not the return value 8."
    return Problem(d, m, rules, false, 0)
end

function problem_newsvendor_mv_discrete()
    m = _model()
    scen = [[80.0, 60.0], [100.0, 80.0], [120.0, 100.0], [80.0, 100.0], [120.0, 60.0]]
    dist = LDR.MvDiscreteNonParametric(scen, [0.2, 0.3, 0.2, 0.15, 0.15])
    rules = _tutorial_inventory(m, dist; products = 2)
    d = "Two-product tutorial newsvendor (docs distributions.jl): buy[1:2] >= 0 first stage, " *
        "sell[1:2] >= 0, salvage[1:2] >= 0; demand[1:2] one multivariate discrete distribution " *
        "with scenarios (demand1, demand2) = (80,60), (100,80), (120,100), (80,100), (120,60) " *
        "and probabilities 0.2, 0.3, 0.2, 0.15, 0.15 (correlated). Constraints sell[i] + " *
        "salvage[i] <= buy[i], sell[i] <= demand[i]. Maximize sum_i (-10 buy[i] + 15 sell[i] + " *
        "5 salvage[i])."
    return Problem(d, m, rules, false, 0)
end

# Nested breakpoint sets for the newsvendor (distributions.jl test_newsvendor_piecewise).
const BREAKPOINTS = Dict(
    1 => [100.0],
    2 => [90.0, 100.0, 110.0],
    3 => collect(85.0:5.0:115.0),
    4 => collect(82.5:2.5:117.5),
)

function problem_newsvendor_breakpoints(k)
    m = _model()
    v = _newsvendor(m)
    set_attribute(v.demand, LDR.BreakPoints(), BREAKPOINTS[k])
    d = NV * " Piecewise-linear lifting of demand with $(length(BREAKPOINTS[k])) breakpoints at " *
        "$(BREAKPOINTS[k]) (nested: set 1 inside 2 inside 3 inside 4; the intervals are " *
        "[80, b1], ..., [bn, 120]). Decision rules are piecewise affine in demand."
    return Problem(d, m, _nv_rules(v), false, length(BREAKPOINTS[k]) + 1)
end

function problem_newsvendor_breakpoint_critical()
    m = _model()
    v = _newsvendor(m)
    set_attribute(v.demand, LDR.BreakPoints(), [760 / 7])
    d = NV * " Piecewise-linear lifting of demand with a single breakpoint at 760/7 = " *
        "108.571428..., the critical quantile of the optimal buy; expected primal 3300/7."
    return Problem(d, m, _nv_rules(v), false, 2)
end

function problem_sampled_newsvendor()
    m = _model(; sampled = true, scenarios = 1000)
    v = _newsvendor(m)
    d = NV * " Besides primal and dual, the sampled LDR with 1000 scenarios drawn by the legacy " *
        "package (Random.Xoshiro(1234), its default seed); the sampled value depends on the " *
        "generator and is only reproducible with the legacy sampler."
    return Problem(d, m, _nv_rules(v), true, 0)
end

const PROBLEMS = [
    "newsvendor" => problem_newsvendor,
    "newsvendor_integer" => problem_newsvendor_integer,
    "newsvendor_random_price" => problem_newsvendor_random_price,
    "newsvendor_two_products" => problem_newsvendor_two_products,
    "newsvendor_three_stages" => problem_newsvendor_three_stages,
    "inventory_T4" => () -> problem_inventory(4),
    "inventory_T8" => () -> problem_inventory(8),
    "hydro_toy_fixed_volume" => problem_hydro_toy_fixed_volume,
    "newsvendor_discrete_demand" => problem_newsvendor_discrete_demand,
    "newsvendor_mv_discrete" => problem_newsvendor_mv_discrete,
    [("newsvendor_breakpoints_$k" => () -> problem_newsvendor_breakpoints(k)) for k in 1:4]...,
    "newsvendor_breakpoint_critical" => problem_newsvendor_breakpoint_critical,
    "sampled_newsvendor_N1000_seed1234" => problem_sampled_newsvendor,
]

# With pieces > 0 each coefficient is the vector over the pieces (coordinates of the lifted
# uncertainty).
function _rules(m, rules, dual, pieces)
    out = Dict{String,Any}()
    coefficient(var, uv) =
        pieces == 0 ? LDR.get_decision(m, var, uv; dual) :
        [LDR.get_decision(m, var, uv; dual, piece = k) for k in 1:pieces]
    for (name, (var, uncs)) in rules
        coefficients = Dict{String,Any}(u => coefficient(var, uv) for (u, uv) in uncs)
        out[name] = Dict{String,Any}(
            "constant" => LDR.get_decision(m, var; dual = dual),
            "coefficients" => coefficients,
        )
    end
    return out
end

function solve(p::Problem)
    m = p.model
    optimize!(m)
    entry = Dict{String,Any}(
        "description" => p.description,
        "termination_status" => string(termination_status(m)),
        "primal" => objective_value(m),
        "dual" => objective_value(m; dual = true),
    )
    p.sampled && (entry["sampled"] = objective_value(m; sampled = true))
    entry["rules"] = Dict{String,Any}(
        "primal" => _rules(m, p.rules, false, p.pieces),
        "dual" => _rules(m, p.rules, true, p.pieces),
    )
    return entry
end

function main()
    result = Dict{String,Any}(
        "meta" => Dict{String,Any}(
            "generator" => "test/reference/generate_legacy_values.jl",
            "legacy_package" => "LinearDecisionRules.jl 0.3.1, branch Multistage",
            "solver" => "HiGHS",
        ),
    )
    for (key, build) in PROBLEMS
        entry = solve(build())
        result[key] = entry
        println(
            rpad(key, 36), entry["primal"], "  ", entry["dual"], "  ", get(entry, "sampled", "-"),
        )
    end
    open(OUT, "w") do io
        TOML.print(io, result; sorted = true)
    end
    println("wrote ", OUT)
    return
end

main()
