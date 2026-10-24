# The same inner models, solved through JuMP as the packages do (rows and columns in the order of
# their construction, not that of the MPS file). Prints one line per run:
#   package problem T formulation solve_time_s resolve_s simplex_iter objective
# `solve_time_s` is JuMP's `solve_time`, `resolve_s` a second `Highs_run` after `Highs_clearSolver` on the
# same HiGHS object (the model in the order of construction): the solver time without the copy of
# the model into HiGHS, which HiGHS's run clock counts in `solve_time_s`.
#
#   julia --project=benchmark/compare benchmark/compare/solve_jump.jl new inventory 128 primal [reps]
#   julia --project=<legacy_env>      benchmark/compare/solve_jump.jl legacy inventory 128 primal [reps]
#
# `primal` for the legacy package is its default model (explicit slacks), `primal_sae` the
# SlackAsExpr one. Each run builds the whole LDRModel again; the time is HiGHS's own (JuMP
# `solve_time`) on the inner model.
using JuMP
import HiGHS, Distributions
import LinearDecisionRules as LDR

include("inventory.jl")
include("../hydro/data.jl")
include("../hydro/model.jl")

function _times(model)
    t = solve_time(model)
    it, obj = simplex_iterations(model), objective_value(model)
    o = unsafe_backend(model)
    HiGHS.Highs_clearSolver(o)
    t_resolve = @elapsed HiGHS.Highs_run(o)
    return t, t_resolve, it, obj
end

function run_new(problem, T, tag, data)
    f = tag == "dual" ? LDR.DualLDR() : LDR.PrimalLDR()
    m = problem == "hydro" ? ldr_model(data, T; formulations = [f]) : build_new(T, HiGHS.Optimizer)
    problem == "hydro" || set_attribute(m, LDR.Formulations(), [f])
    set_silent(m)
    optimize!(m)
    result = only(unsafe_backend(m).results)
    return _times(result.model)..., NaN
end

function run_legacy(problem, T, tag, data)
    problem == "inventory" || error("the legacy hydro model is solved from its MPS files")
    m = build_legacy(T, HiGHS.Optimizer)
    set_silent(m)
    dual = tag == "dual"
    set_attribute(m, LDR.SolvePrimal(), !dual)
    set_attribute(m, LDR.SolveDual(), dual)
    set_attribute(m, LDR.SlackAsExpr(), tag == "primal_sae")
    optimize!(m)
    inner = dual ? m.dual_model : m.primal_model
    return _times(inner)..., NaN
end

function main(args)
    package, problem, T, tag = args[1], args[2], parse(Int, args[3]), args[4]
    reps = length(args) >= 5 ? parse(Int, args[5]) : 3
    data = problem == "hydro" ? hydro_thermal_data() : nothing
    run = package == "new" ? run_new : run_legacy
    run(problem, 2, tag, data)  # compile
    for _ in 1:reps
        t, t_resolve, it, obj, _ = run(problem, T, tag, data)
        println("$package $problem $T $tag $t $t_resolve $it $obj")
        GC.gc()
    end
end

main(ARGS)
