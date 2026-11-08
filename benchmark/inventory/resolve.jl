# Re-solve benchmark: model predictive control with a JuMP parameter.
#
# Problem: the T-period inventory of `timing.jl` (`inventory` of `test/problems/tagged.jl`), with
# the initial stock a parameter, `stock`. It is solved once, then re-solved `N` times with
# `set_parameter_value(stock, value)`, for each formulation (`PrimalLDR()`, `DualLDR()`) and two
# variants:
#
#  * `cached`: the optimizer keeps its inner problem and updates its parameter (the default);
#  * `rebuild`: the cache of inner problems is emptied before each `optimize!`, which is what every
#    re-solve did before the inner problems carried the parameters.
#
# Per variant, over the `N` re-solves: `overhead_s` is the time of `optimize!` outside the inner
# solver (canonical form, moments, fingerprint, building or updating the inner problem),
# `solver_s` the inner solver's time, `optimize_s` their sum, the time of the `N` calls to
# `optimize!`, and the phases of `time_breakdown` (see `PHASE_HEADER` in `../common.jl`) summed over
# them. `fingerprint_s` is the time of one comparison of fingerprints (canonical form,
# moments, settings), measured apart. The objectives of the two variants must agree at every
# re-solve; the script warns otherwise (an error in strict mode, see `../common.jl`). `num_builds`
# counts the inner problems built during the re-solves: 0 when cached, N when rebuilt.
#
# Results are appended to `results/resolve.csv` next to this file, one row per formulation and
# variant, tagged with the commit. Run it alone on an idle machine (timings of parallel runs vary by ±30%).
#
# Usage (from the repository root):
#   julia --project=benchmark benchmark/inventory/resolve.jl          # T = 64, N = 20
#   julia --project=benchmark benchmark/inventory/resolve.jl 16 5     # T = 16, N = 5

using Printf
using JuMP
import LinearDecisionRules as LDR

include("../common.jl")
include("../internals.jl")
include("../../test/problems/tagged.jl")

const RESULTS = joinpath(results_dir(@__DIR__), "resolve.csv")

function build(T, formulation)
    model = TaggedProblems.ldr_model(; formulations = [formulation])
    return TaggedProblems.inventory(T, model; parameter = true)
end

# The initial stocks of the re-solves: an inventory that drifts around zero.
initial_stocks(N) = [40.0 * sin(k) for k in 1:N]

function run(T, N, formulation, variant)
    m = build(T, formulation)
    optimize!(m)
    total, solver, builds = 0.0, 0.0, 0
    phases = ntuple(_ -> 0.0, length(PHASE_HEADER))
    objectives = Float64[]
    for value in initial_stocks(N)
        set_parameter_value(m[:stock], value)
        variant == "rebuild" && empty_cache!(m)
        before = num_builds(m)
        start = time_ns()
        optimize!(m)
        total += (time_ns() - start) / 1e9
        solver += solve_time(m, formulation)
        phases = phases .+ phase_times(m)
        builds += num_builds(m) - before
        push!(objectives, objective_value(m, formulation))
    end
    return (; overhead = total - solver, solver, total, objectives, builds, phases)
end

# The time of one comparison of fingerprints, for the model of `run` after one solve.
function fingerprint_time(T, formulation)
    m = build(T, formulation)
    optimize!(m)
    old = fingerprint(m)
    set_parameter_value(m[:stock], 10.0)
    optimize!(m)
    new = fingerprint(m)
    @assert same_problem(old, new)
    reps = 100
    start = time_ns()
    for _ in 1:reps
        same_problem(old, new)
    end
    return (time_ns() - start) / 1e9 / reps
end

const HEADER = [
    "timestamp", "commit", "src_dirty", "julia", "T", "num_resolves", "formulation", "variant",
    "num_builds", "overhead_s", "solver_s", "optimize_s", "fingerprint_s", "last_objective",
    "src_commit", "highs", PHASE_HEADER...,
]

function main(T, N)
    (; commit, src_commit, src_dirty) = git_info()
    highs = highs_version()
    stamp = timestamp()
    for f in (LDR.PrimalLDR(), LDR.DualLDR()), variant in ("cached", "rebuild")
        run(4, 2, f, variant)  # compile before timing
        fingerprint_time(4, f)
    end
    for f in (LDR.PrimalLDR(), LDR.DualLDR())
        name = string(nameof(typeof(f)))
        fingerprint_s = fingerprint_time(T, f)
        runs = Dict(variant => run(T, N, f, variant) for variant in ("cached", "rebuild"))
        a, b = runs["cached"].objectives, runs["rebuild"].objectives
        check(
            isapprox(a, b; rtol = 1e-6),
            "Objectives differ between cached and rebuilt re-solves";
            name,
            cached = a,
            rebuilt = b,
        )
        for variant in ("cached", "rebuild")
            r = runs[variant]
            @printf(
                "T=%-3d N=%-3d %-9s %-8s builds=%-3d overhead=%.3fs solver=%.3fs optimize=%.3fs fingerprint=%.2es\n",
                T, N, name, variant, r.builds, r.overhead, r.solver, r.total, fingerprint_s,
            )
            append_csv_row(
                RESULTS,
                HEADER,
                (stamp, commit, src_dirty, VERSION, T, N, name, variant, r.builds, r.overhead,
                 r.solver, r.total, fingerprint_s, last(r.objectives), src_commit, highs,
                 r.phases...),
            )
        end
    end
    return
end

main((isempty(ARGS) ? [64, 20] : parse.(Int, ARGS))...)
