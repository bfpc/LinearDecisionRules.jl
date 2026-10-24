# Solve one MPS file with HiGHS in this (fresh) process (HiGHS's own reader, no JuMP) and print
# HiGHS's log.
#   julia --project=benchmark/compare benchmark/compare/solve_mps.jl file.mps [name=value ...]
# Options are HiGHS options (booleans, ints, floats and strings are parsed). The last line is
#   WALL read=<s> run=<s> status=<model status> obj=<value> simplex_iter=<n> ipm_iter=<n>
# which run_solves.jl parses together with the log.
import HiGHS

function set_option(h, name, s)
    if s in ("true", "false")
        return HiGHS.Highs_setBoolOptionValue(h, name, s == "true" ? 1 : 0)
    elseif tryparse(Int, s) !== nothing
        return HiGHS.Highs_setIntOptionValue(h, name, parse(Int, s))
    elseif tryparse(Float64, s) !== nothing
        return HiGHS.Highs_setDoubleOptionValue(h, name, parse(Float64, s))
    end
    return HiGHS.Highs_setStringOptionValue(h, name, s)
end

function int_info(h, name)
    r = Ref{Cint}(0)
    HiGHS.Highs_getIntInfoValue(h, name, r)
    return r[]
end

function main(args)
    h = HiGHS.Highs_create()
    for a in args[2:end]
        k, v = split(a, "="; limit = 2)
        set_option(h, String(k), String(v)) == 0 || error("bad option $a")
    end
    t_read = @elapsed HiGHS.Highs_readModel(h, args[1])
    t_run = @elapsed HiGHS.Highs_run(h)
    status = HiGHS.Highs_getModelStatus(h)
    obj = HiGHS.Highs_getObjectiveValue(h)
    println(
        "WALL read=$t_read run=$t_run status=$status obj=$obj ",
        "simplex_iter=$(int_info(h, "simplex_iteration_count")) ",
        "ipm_iter=$(int_info(h, "ipm_iteration_count"))",
    )
    HiGHS.Highs_destroy(h)
end

main(ARGS)
