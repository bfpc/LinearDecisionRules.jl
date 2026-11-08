# Helpers shared by the benchmark scripts of the folders below this one.

using Dates
import HiGHS
import LinearDecisionRules

include(joinpath(@__DIR__, "..", "test", "common.jl"))

import .TestCommon: model_size

timestamp() = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")

# The folder of the CSV files of the scripts in the folder `dir`: `results/` in it, unless the
# environment variable `LDR_BENCHMARK_RESULTS` names another one (the smoke run of the CI writes to a
# temporary folder). The names of the CSV files differ across folders, so one folder can hold them
# all.
results_dir(dir) = get(ENV, "LDR_BENCHMARK_RESULTS", joinpath(dir, "results"))

# The measured code: the package and the benchmarked problems, which the tests share.
const MEASURED = ["../src", "../Project.toml", "../test/problems"]

# The commit of the repository; the last commit that changed the measured code, so that rows of
# one state of the code have one `src_commit` whatever else changed; and whether tracked files of
# the measured code have uncommitted changes (untracked scratch files are ignored).
function git_info()
    git(args...) =
        try
            readchomp(`git -C $(@__DIR__) $args`)
        catch
            "unknown"
        end
    commit = git("rev-parse", "--short", "HEAD")
    src_commit = git("log", "-1", "--format=%h", "--", MEASURED...)
    src_dirty = !isempty(git("status", "--porcelain", "--untracked-files=no", "--", MEASURED...))
    return (; commit, src_commit, src_dirty)
end

# Whether a failed check of the values is an error, with `LDR_BENCHMARK_STRICT=true` (the smoke run
# of the CI sets it), or a warning, so that a long recording is not lost to one bad value. In strict
# mode the error comes before the row is written.
const STRICT = get(ENV, "LDR_BENCHMARK_STRICT", "false") == "true"

function check(ok::Bool, message::String; values...)
    ok && return
    if STRICT
        details = join(("$k = $v" for (k, v) in values), ", ")
        error(isempty(details) ? message : "$message ($details)")
    end
    @warn(message, values...)
    return
end

# The version of the HiGHS library (not of the HiGHS.jl wrapper).
function highs_version()
    version = (HiGHS.Highs_versionMajor(), HiGHS.Highs_versionMinor(), HiGHS.Highs_versionPatch())
    return join(version, ".")
end

# Append one row, writing the header first if the file is new. Rows are appended one at a time so
# that partial results survive an interrupted run.
function append_csv_row(path, header, row)
    mkpath(dirname(path))
    new_file = !isfile(path)
    open(path, "a") do io
        new_file && println(io, join(header, ","))
        println(io, join(row, ","))
        return
    end
    return
end

# The phases of `time_breakdown(model)` (see the docstring of `TimeBreakdown`), as the last columns
# of every CSV file: reading the model with its lifting and rescaling, the moments, the checks, the
# build, solve and results of the inner problems, and the whole `optimize!` of the optimizer. Rows
# recorded before these columns, and rows of models without the optimizer (a hand-written LP, the
# deterministic model, the scenario tree), leave them empty.
const PHASE_HEADER =
    ["canonicalize_s", "moments_s", "checks_s", "build_s", "solve_s", "results_s", "total_s"]

const NO_PHASES = ntuple(_ -> "", length(PHASE_HEADER))

# The phases of the last `optimize!` of `model`, in the order of `PHASE_HEADER`. The model solves
# one formulation in the benchmarks, so `build`, `solve` and `results` are those of the formulation.
function phase_times(model)
    t = LinearDecisionRules.time_breakdown(model)
    return (t.canonicalize, t.moments, t.checks, t.build, t.solve, t.results, t.total)
end

# Run `f()` `reps` times and keep the run with the smallest model + optimize time; also return the
# number of runs. A first run slower than `slow` seconds is not repeated. The garbage of each run is
# collected before the next one, so that it is not timed there.
function best_of(f, reps; slow = Inf)
    runs = [f()]
    while length(runs) < reps && runs[1].model + runs[1].optimize <= slow
        GC.gc()
        push!(runs, f())
    end
    return runs[argmin([r.model + r.optimize for r in runs])], length(runs)
end
