# Hydrothermal benchmark: the 4-reservoir, 5-subsystem problem (see test/problems/hydro.jl) at
# horizons T = 4, 8, 12, with four joint inflows revealed per stage, solved by
#
#   ldr_primal  the monolithic LDR model, Formulations = [PrimalLDR()]
#   ldr_dual    the same model, Formulations = [DualLDR()] (the value is a bound)
#   ldr_dual_ipm  ldr_dual with the recommended dual mode, `DualLDR(; attributes = ["solver" => "ipm", "run_crossover" => "off"])`
#   average     the deterministic model with the inflows at their means
#   tree        a scenario tree, `branches` joint inflow vectors per stage
#   ldr_primal_bp<k>, ldr_dual_bp<k>
#               (option `bp=k1,k2`) the LDR model with `k` breakpoints on each inflow. BreakPoints
#               need univariate declarations, so the four inflows of a stage are declared as
#               independent marginals (the joint law of the other variants cannot be lifted); k = 0
#               is the same law without breakpoints. The primal value does not depend on the
#               correlation of the inflows (the cost is linear), the dual bound does.
#
# The primal LDR variants (`ldr_primal`, `ldr_primal_bp<k>`) are also simulated: `simulate` applies
# the policy to `OOS_SCENARIOS` scenarios of the true law (the joint 82-scenario law, independent
# across stages) and records the sample mean, standard error and fraction of violating scenarios.
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
#   julia --project=benchmark benchmark/hydro/run.jl                   # T = 4, 8, 12
#   julia --project=benchmark benchmark/hydro/run.jl 4 8               # custom horizons
#   julia --project=benchmark benchmark/hydro/run.jl 12 tree=4         # tree branches
#   julia --project=benchmark benchmark/hydro/run.jl 12 only=ldr_dual  # some variants only
#   julia --project=benchmark benchmark/hydro/run.jl 4 bp=0,1,2        # breakpoints only

using Printf
using JuMP
import HiGHS
import LinearDecisionRules as LDR

include("../common.jl")
include("../../test/problems/hydro.jl")

const REPS = 3
const SLOW_S = 10.0      # a variant slower than this on its first run is not repeated
const LDR_TIME_LIMIT = 900.0
const BP_TIME_LIMIT = 180.0 # the runs with breakpoints stop at 3 minutes (status TIME_LIMIT)
const OOS_SCENARIOS = 2000
const BRANCHES = Dict(4 => 3, 8 => 2, 12 => 2)
const RESULTS = joinpath(RESULTS_DIR, "hydro.csv")

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
        oos = NO_OOS,
    )
end

# `n` scenarios of the true law: for each stage 2:T one of the 82 joint inflow vectors of its month,
# uniformly and independently, as `simulate` takes them.
function true_scenarios(m, data, T, n; seed = 2024)
    rng = Random.Xoshiro(seed)
    return map(1:n) do _
        scenario = Dict{Any,Any}()
        for t in 2:T
            atoms = data.inflow_scenarios
            ω = rand(rng, 1:length(atoms[1][inflow_month(t)]))
            scenario[m[Symbol("inflow$t")]] = [atoms[i][inflow_month(t)][ω] for i in REES]
        end
        return scenario
    end
end

const NO_OOS = (; mean = NaN, stderr = NaN, violated = NaN)

# `breakpoints` is `nothing` (the joint law) or the number of breakpoints of each inflow (see
# `ldr_model`). The primal rule is simulated on the true law.
function run_ldr(data, T, formulation; breakpoints = nothing)
    t_build = @elapsed m = ldr_model(data, T; formulations = [formulation], breakpoints)
    set_time_limit_sec(m, breakpoints === nothing ? LDR_TIME_LIMIT : BP_TIME_LIMIT)
    t_opt = @elapsed optimize!(m)
    objective = try
        objective_value(m, formulation)
    catch
        NaN
    end
    oos = NO_OOS
    if formulation isa LDR.PrimalLDR && !isnan(objective)
        s = LDR.simulate(m, true_scenarios(m, data, T, OOS_SCENARIOS))
        oos = (; mean = s.objective_mean, stderr = s.objective_stderr, violated = s.violation_fraction)
    end
    return (;
        build = t_build,
        optimize = t_opt,
        solver = solve_time(m, formulation),
        objective,
        status = string(termination_status(m, formulation)),
        size = model_size(LDR.inner_model(m, formulation)),
        oos,
    )
end

# (name, T, branches) -> a function that runs the variant once; `bps` are the numbers of
# breakpoints of the variants with breakpoints.
function variants(data, T, branches, bps = Int[])
    list = [
        "ldr_primal" => () -> run_ldr(data, T, LDR.PrimalLDR()),
        "ldr_dual" => () -> run_ldr(data, T, LDR.DualLDR()),
        "ldr_dual_ipm" => () -> run_ldr(data, T, LDR.DualLDR(; attributes = ["solver" => "ipm", "run_crossover" => "off"])),
        "average" => () -> run_plain(average_scenario_model, data, T),
        "tree" => () -> run_plain(scenario_tree_model, data, T, branches),
    ]
    for k in bps
        push!(list, "ldr_primal_bp$k" => () -> run_ldr(data, T, LDR.PrimalLDR(); breakpoints = k))
        push!(list, "ldr_dual_bp$k" => () -> run_ldr(data, T, LDR.DualLDR(); breakpoints = k))
    end
    return list
end

const HEADER = [
    "timestamp", "commit", "src_dirty", "julia", "T", "variant", "num_variables",
    "num_constraints", "nonzeros", "objective", "build_s", "optimize_s", "solver_s",
    "branches", "reps", "status", "breakpoints", "oos_mean", "oos_stderr", "oos_violated",
    "src_commit", "highs",
]

function main(args)
    Ts = [parse(Int, a) for a in args if all(isdigit, a)]
    isempty(Ts) && (Ts = [4, 8, 12])
    options = Dict(Pair(split(a, "="; limit = 2)...) for a in args if occursin("=", a))
    bps = haskey(options, "bp") ? parse.(Int, split(options["bp"], ",")) : Int[]
    only_variants = haskey(options, "only") ? split(options["only"], ",") : nothing
    (; commit, src_commit, src_dirty) = git_info()
    highs = highs_version()
    stamp = timestamp()
    data = hydro_thermal_data()
    for (_, f) in variants(data, 2, 2, [1])  # compile everything before timing
        f()
    end
    for T in Ts
        branches = haskey(options, "tree") ? parse(Int, options["tree"]) : BRANCHES[T]
        for (name, f) in variants(data, T, branches, bps)
            # With `bp`, the variants with breakpoints only, unless `only` says otherwise.
            run_it = if only_variants !== nothing
                name in only_variants
            else
                isempty(bps) || occursin("_bp", name)
            end
            run_it || continue
            res, reps = best_of(f, REPS; slow = SLOW_S)
            b = name == "tree" ? branches : 0
            k = occursin("_bp", name) ? parse(Int, split(name, "_bp")[2]) : ""
            @printf(
                "T=%-3d %-14s vars=%-8d cons=%-8d nnz=%-9d obj=%-12.6f build=%.3fs optimize=%.3fs solver=%.3fs reps=%d %s",
                T, name, res.size..., res.objective, res.build, res.optimize, res.solver, reps,
                res.status,
            )
            isnan(res.oos.mean) || @printf(" oos=%.4f±%.4f violated=%.3f", res.oos.mean, res.oos.stderr, res.oos.violated)
            println()
            oos = isnan(res.oos.mean) ? ("", "", "") : (res.oos.mean, res.oos.stderr, res.oos.violated)
            append_csv_row(
                RESULTS,
                HEADER,
                (stamp, commit, src_dirty, VERSION, T, name, res.size..., res.objective,
                 res.build, res.optimize, res.solver, b, reps, res.status, k, oos..., src_commit, highs),
            )
        end
    end
    return
end

main(ARGS)
