# Timing benchmark: LDRModel vs. the same robust LP written by hand.
#
# Problem: T-period inventory with backlog.
#   stage t   : order x[t] in [0, xmax]
#   stage t+1 : demand d[t] ~ Uniform(lo, hi) is revealed; inventory I[t],
#               cost y[t] >= max(h I[t], -b I[t])
#   I[t] = I[t-1] + x[t] - d[t],   min E[sum c x[t] + y[t]]
# The LDR model is `inventory` of `test/problems/tagged.jl`, which the tests share; the hand-written
# LP below uses the same data.
#
# The primal LDR and the hand-written LP are exact reformulations of the same robust
# problem (box support): the primal LDR and the hand LP must agree, and equal 90T + 20 for the data
# below; the dual LDR value is a lower bound on it (not an equality).
# The script warns if they do not.
#
# Variants: `hand`, `ldr_primal` (Formulations = [PrimalLDR()]), `ldr_dual` ([DualLDR()]).
# `model_s` is the time to write the JuMP model, `optimize_s` the time of `optimize!` (it includes
# the canonical form, the moments, the reformulation and the inner solve), `solver_s` the inner
# solver's time. The LDR variants also record the phases of `time_breakdown` (see `PHASE_HEADER` in
# `../common.jl`); the hand-written LP leaves them empty. Sizes are those of the reformulated model
# that is solved.
#
# Results are appended to `results/timings.csv` next to this file (one row per T and variant),
# tagged with the commit. They are recorded, not asserted: nothing here fails on a slowdown.
#
# Usage (from the repository root):
#   julia --project=benchmark benchmark/inventory/timing.jl            # T = 4, 8, 16, 32, 64
#   julia --project=benchmark benchmark/inventory/timing.jl 8 16 128   # custom sizes

using Printf
using JuMP
import HiGHS
import LinearDecisionRules as LDR

include("../common.jl")
include("../../test/problems/tagged.jl")

# The data of `TaggedProblems.inventory`, for the hand-written LP.
const LO, HI = 60.0, 100.0
const μ, r = (LO + HI) / 2, (HI - LO) / 2
const C_ORDER, H_HOLD, B_BACK = 1.0, 0.5, 3.0
const XMAX, I0 = 150.0, 0.0
const REPS = 3
const RESULTS = joinpath(results_dir(@__DIR__), "timings.csv")

function build_ldr(T, formulation)
    return TaggedProblems.inventory(T, TaggedProblems.ldr_model(; formulations = [formulation]))
end

# A decision rule is a vector of affine expressions over ξ = [1; d[1:T]], with free coefficients
# only where the information structure allows.
function _rule(model, name, T, last_seen)
    coef = zeros(AffExpr, T + 1)
    for k in 0:last_seen
        coef[k+1] = @variable(model, base_name = "$name[$k]")
    end
    return coef
end

# g(ξ) = g[1] + Σₖ g[k+1] d[k] ≥ 0 for all d in the box
# ⟺ g[1] + Σₖ g[k+1] μ - r Σₖ |g[k+1]| ≥ 0.
function _robust_nonneg!(model, g)
    nz = findall(!iszero, g[2:end])
    u = @variable(model, [1:length(nz)], lower_bound = 0)
    for (i, k) in enumerate(nz)
        @constraint(model, u[i] >= g[k+1])
        @constraint(model, u[i] >= -g[k+1])
    end
    @constraint(model, g[1] + μ * sum(g[2:end]) - r * sum(u; init = 0.0) >= 0)
    return
end

function build_hand(T)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    x = [_rule(model, "x$t", T, t - 1) for t in 1:T]
    I = [_rule(model, "I$t", T, t) for t in 1:T]
    y = [_rule(model, "y$t", T, t) for t in 1:T]
    for t in 1:T
        # I[t] = I[t-1] + x[t] - d[t], coefficient by coefficient
        prev = t == 1 ? [I0; zeros(T)] : I[t-1]
        e = zeros(T + 1)
        e[t+1] = 1.0
        @constraint(model, [k in 1:t+1], I[t][k] == prev[k] + x[t][k] - e[k])
        _robust_nonneg!(model, y[t] .- H_HOLD .* I[t])
        _robust_nonneg!(model, y[t] .+ B_BACK .* I[t])
        _robust_nonneg!(model, x[t])
        _robust_nonneg!(model, [XMAX; zeros(T)] .- x[t])
    end
    mean(g) = g[1] + μ * sum(g[2:end])
    @objective(model, Min, sum(C_ORDER * mean(x[t]) + mean(y[t]) for t in 1:T))
    return model
end

function run_hand(T)
    t_model = @elapsed model = build_hand(T)
    t_opt = @elapsed optimize!(model)
    return (;
        model = t_model,
        optimize = t_opt,
        solver = solve_time(model),
        objective = objective_value(model),
        size = model_size(model),
        phases = NO_PHASES,
    )
end

# For `DualLDR`, `objective_value(m, formulation)` is the bound.
function run_ldr(T, formulation)
    t_model = @elapsed m = build_ldr(T, formulation)
    t_opt = @elapsed optimize!(m)
    return (;
        model = t_model,
        optimize = t_opt,
        solver = solve_time(m, formulation),
        objective = objective_value(m, formulation),
        size = model_size(LDR.inner_model(m, formulation)),
        phases = phase_times(m),
    )
end

const VARIANTS = [
    "hand" => run_hand,
    "ldr_primal" => T -> run_ldr(T, LDR.PrimalLDR()),
    "ldr_dual" => T -> run_ldr(T, LDR.DualLDR()),
]

const HEADER = [
    "timestamp", "commit", "src_dirty", "julia", "T", "variant", "num_variables",
    "num_constraints", "nonzeros", "objective", "model_s", "optimize_s", "solver_s",
    "src_commit", "highs", PHASE_HEADER...,
]

function main(sizes)
    (; commit, src_commit, src_dirty) = git_info()
    highs = highs_version()
    stamp = timestamp()
    for (_, f) in VARIANTS  # compile everything before timing
        f(2)
    end
    for T in sizes
        expected = 90.0 * T + 20.0
        for (name, f) in VARIANTS
            res, _ = best_of(() -> f(T), REPS)
            # The dual is a lower bound on the optimum, which the primal LDR attains here.
            ok = name == "ldr_dual" ? res.objective <= expected + 1e-6 * expected : isapprox(res.objective, expected; rtol = 1e-6)
            if !ok
                @warn "Objective mismatch" T name res.objective expected
            end
            @printf(
                "T=%-4d %-11s vars=%-7d cons=%-7d nnz=%-8d obj=%-10.4f model=%.3fs optimize=%.3fs solver=%.3fs\n",
                T, name, res.size..., res.objective, res.model, res.optimize, res.solver,
            )
            append_csv_row(
                RESULTS,
                HEADER,
                (stamp, commit, src_dirty, VERSION, T, name, res.size..., res.objective,
                 res.model, res.optimize, res.solver, src_commit, highs, res.phases...),
            )
        end
    end
    return
end

main(isempty(ARGS) ? [4, 8, 16, 32, 64] : parse.(Int, ARGS))
