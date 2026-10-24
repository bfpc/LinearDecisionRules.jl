# Helpers shared by the timing scripts in this folder.

using Dates

timestamp() = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")

# Commit of the repository, and whether tracked files under src/ have uncommitted changes
# (untracked scratch files are ignored).
function git_info()
    git(args...) =
        try
            readchomp(`git -C $(@__DIR__) $args`)
        catch
            "unknown"
        end
    commit = git("rev-parse", "--short", "HEAD")
    src_dirty = !isempty(git("status", "--porcelain", "--untracked-files=no", "--", "../src"))
    return (; commit, src_dirty)
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

# Run `f()` `reps` times and keep the run with the smallest build + optimize time.
function best_of(f, reps)
    runs = [f() for _ in 1:reps]
    return runs[argmin([r.build + r.optimize for r in runs])]
end
