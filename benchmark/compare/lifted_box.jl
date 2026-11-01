# Study: the implied box rows of lifted coordinates (`lifted_box`) across problems and solvers.
#
# For each problem, horizon T and number k of breakpoints on every uncertainty, it solves the primal
# and the dual LDR with `lifted_box = true` and `false`, under each solver:
#
#   simplex        HiGHS's default (the dual simplex on these LPs)
#   ipx            HiGHS's IPX interior point, without crossover
#   ipx_crossover  IPX with crossover to a basic solution
#   hipo           HiGHS's HiPO interior point, without crossover (`"solver" => "ipm"` in HiGHS 1.15)
#   ipopt          Ipopt, with MUMPS
#
# The problems, all LPs, are in `test/problems/`:
#
#   hydro       the hydrothermal problem of `benchmark/hydro` (`hydro.jl`), whose four inflows of a
#               stage are independent marginals once they have breakpoints
#   inventory   the inventory with backlog of `benchmark/timing.jl` (`tagged.jl`)
#   kwg         the inventory of Kuhn, Wiesemann and Georghiou (2011) (`kwg.jl`)
#   production  the production planning of Georghiou, Kuhn and Wiesemann (2019) (`production.jl`)
#
# Each model is solved once, with a time limit, after a run at T = 2 that compiles the code of the
# problem and the solver. `value` is the value of the primal rule or the bound of the dual; `NaN` when
# the formulation has none. Sizes are those of the inner model. Rows are appended to
# `results/lifted_box.csv` next to this file (or in `LDR_BENCHMARK_RESULTS`).
#
# Usage (from the repository root):
#   julia --project=benchmark benchmark/compare/lifted_box.jl hydro T=4,6,8 k=1,3 solver=simplex,ipx
#   julia --project=benchmark benchmark/compare/lifted_box.jl kwg T=24 k=1 solver=ipopt limit=300

using Printf
using JuMP
import HiGHS
import Ipopt
import LinearDecisionRules as LDR

include("../common.jl")
include("../../test/problems/tagged.jl")
include("../../test/problems/hydro.jl")
include("../../test/problems/kwg.jl")
include("../../test/problems/production.jl")

const RESULTS =
    joinpath(get(ENV, "LDR_BENCHMARK_RESULTS", joinpath(@__DIR__, "results")), "lifted_box.csv")

const SOLVERS = Dict(
    "simplex" => (attributes = Pair{String,Any}[], optimizer = nothing),
    "ipx" => (
        attributes = Pair{String,Any}["solver"=>"ipx", "run_crossover"=>"off"],
        optimizer = nothing,
    ),
    "ipx_crossover" => (
        attributes = Pair{String,Any}["solver"=>"ipx", "run_crossover"=>"on"],
        optimizer = nothing,
    ),
    "hipo" => (
        attributes = Pair{String,Any}["solver"=>"ipm", "run_crossover"=>"off"],
        optimizer = nothing,
    ),
    "ipopt" => (attributes = Pair{String,Any}[], optimizer = Ipopt.Optimizer),
)

const HYDRO = hydro_thermal_data()

function build(problem, T, k, formulation)
    formulations = [formulation]
    if problem == "hydro"
        return ldr_model(HYDRO, T; formulations, breakpoints = k)
    elseif problem == "inventory"
        return TaggedProblems.inventory(T, TaggedProblems.ldr_model(; formulations); points = k)
    elseif problem == "kwg"
        return kwg_model(T; formulations, breakpoints = k)
    elseif problem == "production"
        return production_model(T; formulations, breakpoints = k)
    end
    return error("unknown problem $problem")
end

value_or_nan(f) =
    try
        f()
    catch
        NaN
    end

function run_one(problem, T, k, kind, lifted_box, solver, limit)
    (; attributes, optimizer) = SOLVERS[solver]
    f =
        kind == "primal" ? LDR.PrimalLDR(; attributes, lifted_box) :
        LDR.DualLDR(; attributes, lifted_box)
    model = build(problem, T, k, f)
    optimizer === nothing || set_attribute(model, LDR.InnerOptimizer(), optimizer)
    set_time_limit_sec(model, limit)
    time = @elapsed optimize!(model)
    value =
        value_or_nan(() -> kind == "primal" ? objective_value(model, f) : objective_bound(model, f))
    return (;
        size = model_size(LDR.inner_model(model, f)),
        value,
        status = string(termination_status(model, f)),
        raw = replace(raw_status(model, f), "," => ";"),
        solver = value_or_nan(() -> solve_time(model, f)),
        optimize = time,
    )
end

const HEADER = [
    "timestamp",
    "commit",
    "src_commit",
    "src_dirty",
    "julia",
    "highs",
    "problem",
    "T",
    "k",
    "formulation",
    "lifted_box",
    "solver",
    "num_variables",
    "num_constraints",
    "nonzeros",
    "value",
    "status",
    "raw_status",
    "solver_s",
    "optimize_s",
]

function main(args)
    problems = [a for a in args if !occursin("=", a)]
    options = Dict(Pair(split(a, "="; limit = 2)...) for a in args if occursin("=", a))
    Ts = parse.(Int, split(get(options, "T", "4"), ","))
    ks = parse.(Int, split(get(options, "k", "1"), ","))
    solvers = split(get(options, "solver", "simplex"), ",")
    limit = parse(Float64, get(options, "limit", "600"))
    (; commit, src_commit, src_dirty) = git_info()
    highs = highs_version()
    for problem in problems, solver in solvers
        for kind in ("primal", "dual")  # compile
            run_one(problem, 2, 1, kind, true, solver, limit)
        end
        for T in Ts, k in ks, kind in ("primal", "dual"), lifted_box in (true, false)
            r = run_one(problem, T, k, kind, lifted_box, solver, limit)
            @printf(
                "%-10s T=%-3d k=%d %-6s lifted_box=%-5s %-7s vars=%-7d cons=%-7d nnz=%-8d value=%-16.8g solver=%7.1fs %s\n",
                problem,
                T,
                k,
                kind,
                lifted_box,
                solver,
                r.size...,
                r.value,
                r.solver,
                r.status,
            )
            flush(stdout)
            append_csv_row(
                RESULTS,
                HEADER,
                (
                    timestamp(),
                    commit,
                    src_commit,
                    src_dirty,
                    VERSION,
                    highs,
                    problem,
                    T,
                    k,
                    kind,
                    lifted_box,
                    solver,
                    r.size...,
                    r.value,
                    r.status,
                    r.raw,
                    r.solver,
                    r.optimize,
                ),
            )
        end
    end
    return
end

main(ARGS)
