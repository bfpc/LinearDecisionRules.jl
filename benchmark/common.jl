# Helpers shared by the timing scripts in this folder.

using Dates
import HiGHS

include(joinpath(@__DIR__, "..", "test", "common.jl"))

import .TestCommon: model_size

timestamp() = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")

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

# Run `f()` `reps` times and keep the run with the smallest build + optimize time; also return the
# number of runs. A first run slower than `slow` seconds is not repeated. The garbage of each run is
# collected before the next one, so that it is not timed there.
function best_of(f, reps; slow = Inf)
    runs = [f()]
    while length(runs) < reps && runs[1].build + runs[1].optimize <= slow
        GC.gc()
        push!(runs, f())
    end
    return runs[argmin([r.build + r.optimize for r in runs])], length(runs)
end
