# Hydrothermal benchmark: the 4-reservoir, 5-subsystem problem (see model.jl) at horizons
# T = 4, 8, 12, with four joint inflows revealed per stage, solved by
#
#   ldr_primal  the monolithic LDR model, Formulations = [PrimalLDR()]
#   ldr_dual    the same model, Formulations = [DualLDR()] (the value is a bound)
#   average     the deterministic model with the inflows at their means
#   tree        a scenario tree, `branches` joint inflow vectors per stage
#
# Columns (as in ../timing.jl): `build_s` writes the JuMP model, `optimize_s` is `optimize!` (for the
# LDR models it includes the canonical form, the moments, the reformulation and the inner solve),
# `solver_s` is the inner solver's time, so the time spent building the LDR reformulation is
# optimize_s - solver_s. Sizes are those of the model that is solved (for the LDR models, the
# reformulated one): variables, constraints without variable bounds, nonzeros of the constraint
# matrix. Extra columns: `branches` (tree only), `reps` (runs the row is the best of; a model
# whose first run takes longer than SLOW_S is run once) and the inner solver's termination status.
#
# Results are appended to ../results/hydro.csv, tagged with the commit. They are recorded, not
# asserted.
#
# Usage (from the repository root):
#   julia --project=benchmark/hydro benchmark/hydro/run.jl                   # T = 4, 8, 12
#   julia --project=benchmark/hydro benchmark/hydro/run.jl 4 8               # custom horizons
#   julia --project=benchmark/hydro benchmark/hydro/run.jl 12 tree=4         # tree branches
#   julia --project=benchmark/hydro benchmark/hydro/run.jl 12 only=ldr_dual  # some variants only

using Printf
using JuMP
import HiGHS
import LinearDecisionRules as LDR

include("../common.jl")
include("data.jl")
include("model.jl")

const REPS = 3
const SLOW_S = 10.0      # a variant slower than this on its first run is not repeated
const LDR_TIME_LIMIT = 900.0
const BRANCHES = Dict(4 => 3, 8 => 2, 12 => 2)
const RESULTS = joinpath(@__DIR__, "..", "results", "hydro.csv")

_nnz(f::AffExpr) = length(f.terms)
_nnz(f::AbstractVector{AffExpr}) = sum(_nnz, f; init = 0)
_nnz(::Any) = 0  # bounds and other variable sets

# Variables, constraints (without bounds on variables) and nonzeros of the constraint matrix.
function model_size(model)
    nonzeros = 0
    for (F, S) in list_of_constraint_types(model)
        for c in all_constraints(model, F, S)
            nonzeros += _nnz(constraint_object(c).func)
        end
    end
    constraints = num_constraints(model; count_variable_in_set_constraints = false)
    return (num_variables(model), constraints, nonzeros)
end

function run_plain(build, args...)
    t_build = @elapsed model = build(args...)
    t_opt = @elapsed optimize!(model)
    return (;
        build = t_build,
        optimize = t_opt,
        solver = solve_time(model),
        objective = objective_value(model),
        status = string(termination_status(model)),
        size = model_size(model),
    )
end

function run_ldr(data, T, formulation)
    t_build = @elapsed m = ldr_model(data, T; formulations = [formulation])
    set_time_limit_sec(m, LDR_TIME_LIMIT)
    t_opt = @elapsed optimize!(m)
    result = only(unsafe_backend(m).results)
    return (;
        build = t_build,
        optimize = t_opt,
        solver = result.solve_time,
        objective = objective_value(m, formulation),
        status = string(result.termination_status),
        size = model_size(result.model),
    )
end

# (name, T, branches) -> a function that runs the variant once
function variants(data, T, branches)
    return [
        "ldr_primal" => () -> run_ldr(data, T, LDR.PrimalLDR()),
        "ldr_dual" => () -> run_ldr(data, T, LDR.DualLDR()),
        "average" => () -> run_plain(average_scenario_model, data, T),
        "tree" => () -> run_plain(scenario_tree_model, data, T, branches),
    ]
end

const HEADER = [
    "timestamp", "commit", "src_dirty", "julia", "T", "variant", "num_variables",
    "num_constraints", "nonzeros", "objective", "build_s", "optimize_s", "solver_s",
    "branches", "reps", "status",
]

# Runs `f` once; if it is fast, twice more, and keeps the run with the smallest build + optimize.
function best_run(f)
    first_run = f()
    GC.gc()
    first_run.build + first_run.optimize > SLOW_S && return first_run, 1
    runs = [first_run; [f() for _ in 2:REPS]]
    return runs[argmin([r.build + r.optimize for r in runs])], REPS
end

function main(args)
    Ts = [parse(Int, a) for a in args if all(isdigit, a)]
    isempty(Ts) && (Ts = [4, 8, 12])
    options = Dict(Pair(split(a, "="; limit = 2)...) for a in args if occursin("=", a))
    only_variants = haskey(options, "only") ? split(options["only"], ",") : nothing
    (; commit, src_dirty) = git_info()
    stamp = timestamp()
    data = hydro_thermal_data()
    for (_, f) in variants(data, 2, 2)  # compile everything before timing
        f()
    end
    for T in Ts
        branches = haskey(options, "tree") ? parse(Int, options["tree"]) : BRANCHES[T]
        for (name, f) in variants(data, T, branches)
            only_variants !== nothing && !(name in only_variants) && continue
            res, reps = best_run(f)
            b = name == "tree" ? branches : 0
            @printf(
                "T=%-3d %-10s vars=%-8d cons=%-8d nnz=%-9d obj=%-12.6f build=%.3fs optimize=%.3fs solver=%.3fs reps=%d %s\n",
                T, name, res.size..., res.objective, res.build, res.optimize, res.solver, reps,
                res.status,
            )
            append_csv_row(
                RESULTS,
                HEADER,
                (stamp, commit, src_dirty, VERSION, T, name, res.size..., res.objective,
                 res.build, res.optimize, res.solver, b, reps, res.status),
            )
        end
    end
    return
end

main(ARGS)
