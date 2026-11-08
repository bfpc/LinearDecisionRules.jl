# The grid of runs of the decision studies (`breakpoints/`, `scaling/`, `solvers/`): each study's
# script includes this file and calls `main` with its CSV file and its defaults.
#
# For each problem, horizon T and number k of breakpoints on every uncertainty, it solves the primal
# and/or the dual LDR, for each value of `lifted_box` and of `RescaleUncertainties()`, under each
# solver:
#
#   simplex        HiGHS's default (the dual simplex on these LPs)
#   ipx            HiGHS's IPX interior point, without crossover
#   ipx_crossover  IPX with crossover to a basic solution
#   hipo           HiGHS's HiPO interior point, without crossover (`"solver" => "ipm"` in HiGHS 1.15)
#   ipopt          Ipopt, with MUMPS
#   gurobi         Gurobi's default (concurrent methods on all cores), if Gurobi is installed
#   gurobi_dual1   Gurobi's dual simplex on one thread (`Threads = 1`, `Method = 1`), likewise
#
# Gurobi is not a dependency of the package or of this environment: the script loads it when Julia
# finds it in the load path, for example when it is added to the default environment
# (`julia -e 'import Pkg; Pkg.add("Gurobi")'`), which `--project=benchmark` stacks below the project.
# It needs a license.
#
# The problems, all LPs, are in `test/problems/`:
#
#   hydro                the hydrothermal problem of `benchmark/hydro` (`hydro.jl`), whose four
#                        inflows of a stage are independent marginals once they have breakpoints
#   hydro_joint          the same with the joint law of the inflows and no breakpoints (`k` is
#                        ignored), the model of `benchmark/hydro/run.jl`
#   inventory            the inventory with backlog of `benchmark/inventory/timing.jl` (`tagged.jl`)
#   kwg                  the inventory of Kuhn, Wiesemann and Georghiou (2011) (`kwg.jl`)
#   production           the production planning of Georghiou, Kuhn and Wiesemann (2019)
#                        (`production.jl`), with the demands as uncertainties
#   production_centered  the same with the deviations of the demands from their nominal values
#   production_unit      the same with these deviations divided by their half-width, in [-1, 1]
#
# Each model is solved once, with a time limit, after runs at T = 3 that compile the code of the
# problem and the solver. `value` is the value of the primal rule or the bound of the dual; `NaN`
# when the formulation has none; the optimal values of one instance and formulation must agree
# across `lifted_box` and `rescale` (a warning, or an error with `LDR_BENCHMARK_STRICT=true`, see
# `common.jl`). Sizes are those of the inner model. `solver_s` is the inner
# solver's time, `optimize_s` the time of `optimize!`, and the last columns are the phases of
# `time_breakdown` (see `PHASE_HEADER` in `common.jl`), empty in the rows recorded before them.
#
# The options, `name=value` with comma-separated values: `T`, `k`, `solver`, `kind` (`primal`,
# `dual`), `lifted_box`, `rescale`, and `limit` (seconds). `rescale` is `true`
# (`RescaleUncertainties()`, each piece of a lifting on its own box), `false`, or, for the study of
# `scaling/`, `before_lifting` or `hybrid`: rescaled, with the pieces of the liftings of breakpoints
# as `set_lifted_rescaling!` of `internals.jl` says.

using Printf
using JuMP
import HiGHS
import Ipopt
import LinearDecisionRules as LDR

include("common.jl")
include("internals.jl")
include("../test/problems/tagged.jl")
include("../test/problems/hydro.jl")
include("../test/problems/kwg.jl")
include("../test/problems/production.jl")

const HAS_GUROBI = Base.find_package("Gurobi") !== nothing

if HAS_GUROBI
    import Gurobi
end

# One environment for all the models, so that the license is checked once.
const GUROBI_ENV = Ref{Any}(nothing)

function gurobi_optimizer()
    if GUROBI_ENV[] === nothing
        GUROBI_ENV[] = Gurobi.Env()
    end
    return Gurobi.Optimizer(GUROBI_ENV[])
end

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
    "gurobi" => (attributes = Pair{String,Any}[], optimizer = gurobi_optimizer),
    "gurobi_dual1" => (
        attributes = Pair{String,Any}["Threads"=>1, "Method"=>1],
        optimizer = gurobi_optimizer,
    ),
)

const HYDRO = hydro_thermal_data()

function build(problem, T, k, formulation)
    formulations = [formulation]
    if problem == "hydro"
        return ldr_model(HYDRO, T; formulations, breakpoints = k)
    elseif problem == "hydro_joint"
        return ldr_model(HYDRO, T; formulations)
    elseif problem == "inventory"
        return TaggedProblems.inventory(T, TaggedProblems.ldr_model(; formulations); points = k)
    elseif problem == "kwg"
        return kwg_model(T; formulations, breakpoints = k)
    elseif problem == "production"
        return production_model(T; formulations, breakpoints = k)
    elseif problem == "production_centered"
        return production_model(T; formulations, breakpoints = k, parametrization = :centered)
    elseif problem == "production_unit"
        return production_model(T; formulations, breakpoints = k, parametrization = :unit)
    end
    return error("unknown problem $problem")
end

value_or_nan(f) =
    try
        f()
    catch
        NaN
    end

function run_one(problem, T, k, kind, lifted_box, rescale, solver, limit)
    (; attributes, optimizer) = SOLVERS[solver]
    f =
        kind == "primal" ? LDR.PrimalLDR(; attributes, lifted_box) :
        LDR.DualLDR(; attributes, lifted_box)
    model = build(problem, T, k, f)
    optimizer === nothing || set_attribute(model, LDR.InnerOptimizer(), optimizer)
    set_attribute(model, LDR.RescaleUncertainties(), rescale != "false")
    set_lifted_rescaling!(rescale in ("before_lifting", "hybrid") ? Symbol(rescale) : :per_piece)
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
        phases = phase_times(model),
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
    "rescale",
    "solver",
    "num_variables",
    "num_constraints",
    "nonzeros",
    "value",
    "status",
    "raw_status",
    "solver_s",
    "optimize_s",
    PHASE_HEADER...,
]

# Run the grid of `args` (problems, then options), appending to `results`; `defaults` gives the
# options of the study that `args` leaves out.
function main(args, results; defaults = Dict{String,String}())
    problems = [a for a in args if !occursin("=", a)]
    options = merge(
        Dict(
            "T" => "4",
            "k" => "1",
            "solver" => "simplex",
            "limit" => "600",
            "kind" => "primal,dual",
            "lifted_box" => "true,false",
            "rescale" => "true",
        ),
        defaults,
        Dict(Pair(split(a, "="; limit = 2)...) for a in args if occursin("=", a)),
    )
    Ts = parse.(Int, split(options["T"], ","))
    ks = parse.(Int, split(options["k"], ","))
    solvers = split(options["solver"], ",")
    limit = parse(Float64, options["limit"])
    kinds = split(options["kind"], ",")
    boxes = parse.(Bool, split(options["lifted_box"], ","))
    rescales = split(options["rescale"], ",")
    for rescale in rescales
        if !(rescale in ("true", "false", "before_lifting", "hybrid"))
            error("unknown value rescale=$rescale")
        end
    end
    for solver in solvers
        if !haskey(SOLVERS, solver)
            error("unknown solver $solver")
        elseif startswith(solver, "gurobi") && !HAS_GUROBI
            error("the solver $solver needs Gurobi in the load path (see `benchmark/grid.jl`)")
        end
    end
    (; commit, src_commit, src_dirty) = git_info()
    highs = highs_version()
    # The first optimal value of each instance and formulation: `lifted_box` and the rescaling
    # change the inner problem, not its value, so the others must agree with it.
    values = Dict{Any,Float64}()
    for problem in problems, solver in solvers
        for kind in kinds, lifted_box in boxes, rescale in rescales  # compile
            run_one(problem, 3, 1, kind, lifted_box, rescale, solver, limit)
        end
        for T in Ts, k in ks, kind in kinds, lifted_box in boxes, rescale in rescales
            r = run_one(problem, T, k, kind, lifted_box, rescale, solver, limit)
            if r.status == "OPTIMAL"
                value = get!(values, (problem, solver, T, k, kind), r.value)
                check(
                    isapprox(r.value, value; rtol = 1e-6),
                    "The value depends on lifted_box or rescale";
                    problem,
                    solver,
                    T,
                    k,
                    kind,
                    lifted_box,
                    rescale,
                    value = r.value,
                    first = value,
                )
            end
            @printf(
                "%-10s T=%-3d k=%d %-6s lifted_box=%-5s rescale=%-14s %-7s vars=%-7d cons=%-7d nnz=%-8d value=%-16.8g solver=%7.1fs %s\n",
                problem,
                T,
                k,
                kind,
                lifted_box,
                rescale,
                solver,
                r.size...,
                r.value,
                r.solver,
                r.status,
            )
            flush(stdout)
            append_csv_row(
                results,
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
                    rescale,
                    solver,
                    r.size...,
                    r.value,
                    r.status,
                    r.raw,
                    r.solver,
                    r.optimize,
                    r.phases...,
                ),
            )
        end
    end
    return
end
