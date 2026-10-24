# Solver modes on the exported MPS files: one fresh process per solve, sequentially, a time limit of
# 120 s per solve, appending to ../results/compare_modes.csv (raw rows; `modes_table.jl` adds the
# relative error against the HiGHS optimum and prints the table).
#
#   julia --project=benchmark/compare benchmark/compare/run_modes.jl modes=dsimplex,ipm_nox [reps=3] stem ...
#   julia --project=benchmark/compare benchmark/compare/run_modes.jl modes=dsimplex set=main
#
# `set=main` is the 25 files of the comparison (inventory T = 32, 64, 128; hydro T = 4, 8; the
# formulations and packages of compare.csv). A mode that reached the time limit on a file of a
# chain (problem, package, formulation) is not run on the larger T of the chain (row `SKIPPED`).
# Ipopt is not run on files over 100 MB (reading them with JuMP alone takes minutes).
# A solve of at most 5 s is repeated up to `reps` times in total and the best time is kept.
using Dates

const DIR = @__DIR__
const MPS = joinpath(DIR, "mps")
const LOGS = joinpath(DIR, "logs")
const RESULTS = joinpath(DIR, "..", "results", "compare_modes.csv")
const TIME_LIMIT = 120.0
const HARD_LIMIT = parse(Float64, get(ENV, "HARD_LIMIT", "400"))   # process kill (reading, start-up included)
const IPOPT_MAX_BYTES = 100e6

# name => (program, options)
const MODES = Dict(
    "dsimplex" => (:highs, ["solver=simplex", "simplex_strategy=1"]),
    "psimplex" => (:highs, ["solver=simplex", "simplex_strategy=4"]),
    "ipm" => (:highs, ["solver=ipm"]),                                # HiPO, with crossover
    "ipm_nox" => (:highs, ["solver=ipm", "run_crossover=off"]),
    "ipx_nox" => (:highs, ["solver=ipx", "run_crossover=off"]),
    # time_limit=120 stops cuPDLP-C after 0.1 s (it seems to read the limit in ms): pass 120000
    "pdlp" => (:highs, ["solver=pdlp", "time_limit=120000"]),
    "hipdlp" => (:highs, ["solver=hipdlp"]),
    "ipopt" => (:ipopt, String[]),                                    # tol = 1e-8, the default of Ipopt
    "ipopt_tol6" => (:ipopt, ["tol=1e-6"]),
    "ipopt_tol10" => (:ipopt, ["tol=1e-10"]),
    "ipopt_br0" => (:ipopt, ["bound_relax_factor=0.0"]),                # no relaxation of the bounds
)
const HIGHS_STATUS = Dict(
    0 => "NOTSET", 1 => "LOAD_ERROR", 2 => "MODEL_ERROR", 3 => "PRESOLVE_ERROR", 4 => "SOLVE_ERROR",
    5 => "POSTSOLVE_ERROR", 6 => "MODEL_EMPTY", 7 => "OPTIMAL", 8 => "INFEASIBLE",
    9 => "UNBOUNDED_OR_INFEASIBLE", 10 => "UNBOUNDED", 11 => "OBJECTIVE_BOUND", 12 => "OBJECTIVE_TARGET",
    13 => "TIME_LIMIT", 14 => "ITERATION_LIMIT", 15 => "UNKNOWN",
)
const HEADER = [
    "timestamp", "problem", "T", "package", "formulation", "mode", "run_s", "status", "objective",
    "iterations", "reps", "read_s", "note",
]

function parse_stem(stem)
    m = match(r"^(\w+?)_T(\d+)_(new|legacy)_(\w+)$", stem)
    m === nothing && error("cannot parse $stem")
    return (m[1], parse(Int, m[2]), m[3], m[4])
end

function main_set()
    stems = String[]
    for (problem, Ts) in (("inventory", [32, 64, 128]), ("hydro", [4, 8]))
        for (pkg, f) in (("new", "primal"), ("legacy", "primal"), ("legacy", "primal_sae"), ("new", "dual"), ("legacy", "dual"))
            for T in Ts
                push!(stems, "$(problem)_T$(T)_$(pkg)_$(f)")
            end
        end
    end
    return stems
end

# Run `cmd` with the output in `logfile`, kill after HARD_LIMIT seconds. Returns the text and a flag.
function run_child(cmd, logfile)
    p = run(pipeline(cmd; stdout = logfile, stderr = devnull); wait = false)
    t0 = time()
    while process_running(p)
        if time() - t0 > HARD_LIMIT
            kill(p)
            sleep(1)
            return read(logfile, String), true
        end
        sleep(0.1)
    end
    return read(logfile, String), false
end

function solve_once(stem, kind, options)
    file = joinpath(MPS, stem * ".mps")
    logfile = joinpath(LOGS, "$(stem)__current.log")
    if kind == :highs
        cmd = `julia --project=$DIR $(joinpath(DIR, "solve_mps.jl")) $file time_limit=$TIME_LIMIT $options`
    else
        cmd = `julia --project=$DIR $(joinpath(DIR, "solve_ipopt.jl")) $file $options`
    end
    text, killed = run_child(cmd, logfile)
    w = match(r"WALL read=(\S+) run=(\S+) status=(\S+) obj=(\S+) (?:simplex_iter=(\d+) ipm_iter=(\d+)|iter=(-?\d+))", text)
    if w === nothing
        return (; read = NaN, run = HARD_LIMIT, status = killed ? "KILLED" : "ERROR", obj = NaN, iter = -1)
    end
    iters = kind == :highs ? parse(Int, w[5]) + parse(Int, w[6]) : parse(Int, w[7])
    status = kind == :highs ? get(HIGHS_STATUS, parse(Int, w[3]), String(w[3])) : String(w[3])
    return (; read = parse(Float64, w[1]), run = parse(Float64, w[2]), status,
        obj = parse(Float64, w[4]), iter = iters)
end

function append_row(row)
    mkpath(dirname(RESULTS))
    new_file = !isfile(RESULTS)
    open(RESULTS, "a") do io
        new_file && println(io, join(HEADER, ","))
        println(io, join(row, ","))
    end
end

# (problem, pkg, formulation, mode) => smallest T where the mode reached the time limit
const TL = Dict{NTuple{4,String},Int}()

function load_tl()
    isfile(RESULTS) || return
    for line in readlines(RESULTS)[2:end]
        f = split(line, ",")
        if f[8] in ("TIME_LIMIT", "KILLED", "ITERATION_LIMIT") || occursin("TIME", f[8])
            key = (f[2], f[4], f[5], f[6])
            TL[key] = min(get(TL, key, typemax(Int)), parse(Int, f[3]))
        end
    end
end

function main(args)
    modes = String[]
    reps = 3
    stems = String[]
    for a in args
        if startswith(a, "modes=")
            append!(modes, split(a[7:end], ","))
        elseif startswith(a, "reps=")
            reps = parse(Int, a[6:end])
        elseif a == "set=main"
            append!(stems, main_set())
        else
            push!(stems, a)
        end
    end
    mkpath(LOGS)
    load_tl()
    t_start = time()
    for mode in modes, stem in stems
        kind, options = MODES[mode]
        problem, T, pkg, f = parse_stem(stem)
        stamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")
        key = (problem, pkg, f, mode)
        base = (stamp, problem, T, pkg, f, mode)
        size_bytes = filesize(joinpath(MPS, stem * ".mps"))
        if haskey(TL, key) && TL[key] < T
            println("$stem [$mode] SKIPPED (time limit at T=$(TL[key]))")
            append_row((base..., "", "SKIPPED", "", "", 0, "", "time limit at T=$(TL[key])"))
            continue
        elseif kind == :ipopt && size_bytes > IPOPT_MAX_BYTES
            println("$stem [$mode] SKIPPED (file of $(round(Int, size_bytes / 1e6)) MB)")
            append_row((base..., "", "SKIPPED", "", "", 0, "", "file of $(round(Int, size_bytes / 1e6)) MB"))
            continue
        end
        best = solve_once(stem, kind, options)
        n = 1
        if best.run <= 5.0 && best.status != "ERROR"
            for _ in 2:reps
                r = solve_once(stem, kind, options)
                n += 1
                r.run < best.run && (best = r)
            end
        end
        cp(joinpath(LOGS, "$(stem)__current.log"), joinpath(LOGS, "$(stem)__$(mode).log"); force = true)
        if best.status in ("TIME_LIMIT", "KILLED", "ITERATION_LIMIT") || occursin("TIME", best.status)
            TL[key] = min(get(TL, key, typemax(Int)), T)
        end
        note = best.status == "KILLED" ? "process killed after $(HARD_LIMIT) s" : ""
        println(
            "$stem [$mode] run=$(round(best.run; digits = 3))s status=$(best.status) obj=$(best.obj) iter=$(best.iter) ",
            "reps=$n | wall $(round((time() - t_start) / 60; digits = 1)) min",
        )
        append_row((base..., best.run, best.status, best.obj, best.iter, n, best.read, note))
    end
end

main(ARGS)
