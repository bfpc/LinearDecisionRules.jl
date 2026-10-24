# Solve exported MPS files with HiGHS, each in a fresh process (solve_mps.jl), best of 3 (a single
# run when the first takes more than 60 s), and append one row per file and setting to
# ../results/compare.csv. The HiGHS log of the kept run goes to logs/<stem>__<setting>.log.
#
#   julia --project=benchmark/compare benchmark/compare/run_solves.jl [setting=NAME] [reps=N] stem ...
#
# `stem` is a file name in mps/ without `.mps` (e.g. inventory_T128_new_primal); `setting` is one of
# SETTINGS (default `default`). `presolve_s` is the run time with `simplex_iteration_limit=0`, which
# is the presolve plus the set-up of the simplex on the presolved model (the 0.1 s resolution of the
# log does not allow reading it there); it is measured with the `default` setting only.
using Dates

const DIR = @__DIR__
const MPS = joinpath(DIR, "mps")
const LOGS = joinpath(DIR, "logs")
const RESULTS = joinpath(DIR, "..", "results", "compare.csv")
const SLOW_S = 60.0

# name => HiGHS options
const SETTINGS = Dict(
    "default" => String[],
    "simplex" => ["solver=simplex"],
    "ipm" => ["solver=ipm"],
    "ipm_nocross" => ["solver=ipm", "run_crossover=off"],
    "primal_simplex" => ["solver=simplex", "simplex_strategy=4"],
    "nopresolve" => ["presolve=off"],
    "nopresolve_ipm" => ["presolve=off", "solver=ipm"],
    "presolve_only" => ["simplex_iteration_limit=0"],
)

const HEADER = [
    "timestamp", "problem", "T", "package", "formulation", "setting", "rows", "cols", "nonzeros",
    "pre_rows", "pre_cols", "pre_nonzeros", "algorithm", "simplex_iter", "ipm_iter", "presolve_s",
    "run_s", "status", "objective", "reps", "highs",
]

# The stem `inventory_T128_new_primal` is problem, T, package, formulation (which may contain `_`).
function parse_stem(stem)
    m = match(r"^(\w+?)_T(\d+)_(new|legacy)_(\w+)$", stem)
    m === nothing && error("cannot parse $stem")
    return (m[1], parse(Int, m[2]), m[3], m[4])
end

function parse_log(log)
    num(r) = (m = match(r, log); m === nothing ? missing : parse(Int, m[1]))
    rows = num(r"LP \S+ has (\d+) rows")
    cols = num(r"; (\d+) cols")
    nnz = num(r"; (\d+) nonzeros")
    # "Presolve reductions: rows 17356(-24306); columns 66254(-24306); nonzeros 200881(-5132)",
    # or "- Reduced to empty" when presolve solves the model.
    pre = match(r"Presolve reductions: rows (\d+)\(-?\d+\); columns (\d+)\(-?\d+\); nonzeros (\d+)", log)
    pre_rows, pre_cols, pre_nnz = pre === nothing ? (rows, cols, nnz) : parse.(Int, pre.captures)
    if occursin("Reduced to empty", log)
        pre_rows = pre_cols = pre_nnz = 0
    end
    algorithm = if (m = match(r"^(Using .*solver.*)$"m, log)) !== nothing
        strip(m[1])
    elseif occursin(r"HiPO|Hipo"i, log)
        "hipo ipm"
    elseif occursin("IPX", log)
        "ipx ipm"
    else
        "unknown"
    end
    if occursin("IPX reports", log) || occursin("Hipo: Solved", log) || occursin("HiPO runtime", log)
        algorithm = occursin("Hipo", log) || occursin("HiPO", log) ? "hipo ipm + crossover" : "ipx ipm + crossover"
    end
    w = match(r"WALL read=(\S+) run=(\S+) status=(\d+) obj=(\S+) simplex_iter=(\d+) ipm_iter=(\d+)", log)
    w === nothing && error("no WALL line in the log:\n$(last(log, 2000))")
    return (; rows, cols, nnz, pre_rows, pre_cols, pre_nnz, algorithm,
        run = parse(Float64, w[2]), status = parse(Int, w[3]), obj = parse(Float64, w[4]),
        simplex_iter = parse(Int, w[5]), ipm_iter = parse(Int, w[6]))
end

function solve_once(stem, options)
    file = joinpath(MPS, stem * ".mps")
    cmd = `julia --project=$DIR $(joinpath(DIR, "solve_mps.jl")) $file $options`
    log = read(pipeline(cmd; stderr = devnull), String)
    return log, parse_log(log)
end

function best_run(stem, options, reps)
    log, res = solve_once(stem, options)
    runs = [(log, res)]
    if res.run <= SLOW_S
        for _ in 2:reps
            push!(runs, solve_once(stem, options))
        end
    end
    best = argmin([r[2].run for r in runs])
    return runs[best]..., length(runs)
end

function highs_version(log)
    m = match(r"Running HiGHS (\S+)", log)
    return m === nothing ? "unknown" : m[1]
end

function append_row(row)
    mkpath(dirname(RESULTS))
    new_file = !isfile(RESULTS)
    open(RESULTS, "a") do io
        new_file && println(io, join(HEADER, ","))
        println(io, join(row, ","))
    end
end

function main(args)
    setting = "default"
    reps = 3
    stems = String[]
    for a in args
        if startswith(a, "setting=")
            setting = a[9:end]
        elseif startswith(a, "reps=")
            reps = parse(Int, a[6:end])
        else
            push!(stems, a)
        end
    end
    options = SETTINGS[setting]
    mkpath(LOGS)
    stamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")
    for stem in stems
        problem, T, package, formulation = parse_stem(stem)
        log, res, n = best_run(stem, options, reps)
        write(joinpath(LOGS, "$(stem)__$(setting).log"), log)
        presolve_s = ""
        if setting == "default"
            runs = [solve_once(stem, SETTINGS["presolve_only"])[2].run for _ in 1:(res.run <= SLOW_S ? reps : 1)]
            presolve_s = minimum(runs)
        end
        println(
            "$stem [$setting] rows=$(res.rows) cols=$(res.cols) -> $(res.pre_rows) x $(res.pre_cols) ",
            "nnz=$(res.pre_nnz) $(res.algorithm) simplex=$(res.simplex_iter) ipm=$(res.ipm_iter) ",
            "presolve=$presolve_s run=$(res.run)s status=$(res.status) obj=$(res.obj) reps=$n",
        )
        append_row((
            stamp, problem, T, package, formulation, setting, res.rows, res.cols, res.nnz,
            res.pre_rows, res.pre_cols, res.pre_nnz, replace(res.algorithm, "," => ";"),
            res.simplex_iter, res.ipm_iter, presolve_s, res.run, res.status, res.obj, n, highs_version(log),
        ))
    end
end

main(ARGS)
