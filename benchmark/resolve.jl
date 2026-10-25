# Re-solve benchmark: model predictive control with a JuMP parameter.
#
# Problem: the T-period inventory of `timing.jl`, with the initial stock `I0` a parameter. It is
# solved once, then re-solved `N` times with `set_parameter_value(I0, value)`, for each
# formulation (`PrimalLDR()`, `DualLDR()`) and two variants:
#
#  * `cached`: the optimizer keeps its inner problem and updates its parameter (the default);
#  * `rebuild`: the cache of inner problems is emptied before each `optimize!`, which is what every
#    re-solve did before the inner problems carried the parameters.
#
# Per variant, over the `N` re-solves: `build_s` is the time of `optimize!` outside the inner
# solver (canonical form, moments, fingerprint, building or updating the inner problem),
# `solver_s` the inner solver's time, `total_s` their sum, the time of the `N` calls to
# `optimize!`. `fingerprint_s` is the time of one comparison of fingerprints (canonical form,
# moments, settings), measured apart. The objectives of the two variants must agree at every
# re-solve; the script warns otherwise. `num_builds` counts the inner problems built during the
# re-solves: 0 when cached, N when rebuilt.
#
# Results are appended to benchmark/results/resolve.csv, one row per formulation and variant, tagged
# with the commit. Run it alone on an idle machine (timings of parallel runs vary by ±30%).
#
# Usage (from the repository root):
#   julia --project=benchmark benchmark/resolve.jl          # T = 64, N = 20
#   julia --project=benchmark benchmark/resolve.jl 16 5     # T = 16, N = 5

using Printf
using JuMP
import HiGHS
import Distributions
import LinearDecisionRules as LDR

include("common.jl")

const LO, HI = 60.0, 100.0
const C_ORDER, H_HOLD, B_BACK = 1.0, 0.5, 3.0
const XMAX = 150.0
const RESULTS = joinpath(@__DIR__, "results", "resolve.csv")

function build(T, formulation)
    m = LDR.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    set_attribute(m, LDR.Formulations(), [formulation])
    dist = Distributions.Uniform(LO, HI)
    @variable(m, I0 in Parameter(0.0))
    @variable(m, d[t = 1:T] in LDR.Uncertainty(dist), LDR.Stage(t + 1))
    @variable(m, 0 <= x[t = 1:T] <= XMAX, LDR.Stage(t))
    @variable(m, I[t = 1:T], LDR.Stage(t + 1))
    @variable(m, y[t = 1:T], LDR.Stage(t + 1))
    @constraint(m, I[1] == I0 + x[1] - d[1])
    @constraint(m, [t = 2:T], I[t] == I[t-1] + x[t] - d[t])
    @constraint(m, [t = 1:T], y[t] >= H_HOLD * I[t])
    @constraint(m, [t = 1:T], y[t] >= -B_BACK * I[t])
    @objective(m, Min, sum(C_ORDER * x[t] + y[t] for t in 1:T))
    return m
end

# The initial stocks of the re-solves: an inventory that drifts around zero.
initial_stocks(N) = [40.0 * sin(k) for k in 1:N]

function run(T, N, formulation, variant)
    m = build(T, formulation)
    optimize!(m)
    total, solver, builds = 0.0, 0.0, 0
    objectives = Float64[]
    for value in initial_stocks(N)
        set_parameter_value(m[:I0], value)
        if variant == "rebuild"
            unsafe_backend(m).inner_cache = LDR._InnerCache()
        end
        before = unsafe_backend(m).inner_cache.num_builds
        start = time_ns()
        optimize!(m)
        total += (time_ns() - start) / 1e9
        solver += only(unsafe_backend(m).results).solve_time
        builds += unsafe_backend(m).inner_cache.num_builds - before
        push!(objectives, objective_value(m, formulation))
    end
    return (; build = total - solver, solver, total, objectives, builds)
end

# The time of one comparison of fingerprints, for the model of `run` after one solve.
function fingerprint_time(T, formulation)
    m = build(T, formulation)
    optimize!(m)
    o = unsafe_backend(m)
    old = o.inner_cache.fingerprint
    set_parameter_value(m[:I0], 10.0)
    optimize!(m)
    new = o.inner_cache.fingerprint
    @assert LDR._same_problem(old, new)
    reps = 100
    start = time_ns()
    for _ in 1:reps
        LDR._same_problem(old, new)
    end
    return (time_ns() - start) / 1e9 / reps
end

const HEADER = [
    "timestamp", "commit", "src_dirty", "julia", "T", "num_resolves", "formulation", "variant",
    "num_builds", "build_s", "solver_s", "total_s", "fingerprint_s", "last_objective",
]

function main(T, N)
    (; commit, src_dirty) = git_info()
    stamp = timestamp()
    for f in (LDR.PrimalLDR(), LDR.DualLDR()), variant in ("cached", "rebuild")
        run(4, 2, f, variant)  # compile before timing
        fingerprint_time(4, f)
    end
    for f in (LDR.PrimalLDR(), LDR.DualLDR())
        name = string(nameof(typeof(f)))
        fingerprint = fingerprint_time(T, f)
        runs = Dict(variant => run(T, N, f, variant) for variant in ("cached", "rebuild"))
        a, b = runs["cached"].objectives, runs["rebuild"].objectives
        if !isapprox(a, b; rtol = 1e-6)
            @warn "Objectives differ between cached and rebuilt re-solves" name a b
        end
        for variant in ("cached", "rebuild")
            r = runs[variant]
            @printf(
                "T=%-3d N=%-3d %-9s %-8s builds=%-3d build=%.3fs solver=%.3fs total=%.3fs fingerprint=%.2es\n",
                T, N, name, variant, r.builds, r.build, r.solver, r.total, fingerprint,
            )
            append_csv_row(
                RESULTS,
                HEADER,
                (stamp, commit, src_dirty, VERSION, T, N, name, variant, r.builds, r.build,
                 r.solver, r.total, fingerprint, last(r.objectives)),
            )
        end
    end
    return
end

main((isempty(ARGS) ? [64, 20] : parse.(Int, ARGS))...)
