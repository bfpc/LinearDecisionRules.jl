# Export the inner LPs of the previous version (branch Multistage, v0.3.1) to MPS.
#   julia --project=<legacy_env> benchmark/compare/export_legacy.jl inventory 32 64 128
#   julia --project=<legacy_env> benchmark/compare/export_legacy.jl hydro 4 8
# <legacy_env> is an environment with the old repository developed (spike/legacy_env there).
# Writes mps/<problem>_T<T>_legacy_<primal|primal_sae|dual>.mps. `primal` is the default primal
# model (explicit slacks), `primal_sae` the one with `SlackAsExpr(true)` (what the hydro runs of
# the old repository used), `dual` the dense dual. The old model is built by `optimize!` with a
# HiGHS time limit of zero seconds; the inner model is then written from `primal_model` /
# `dual_model`.
using JuMP
import HiGHS, Distributions
import LinearDecisionRules as LDR

include("inventory.jl")
include("../hydro/data.jl")
include("../hydro/model.jl")   # stage_variables / stage_constraints / stage_cost are plain JuMP

const MPS = joinpath(@__DIR__, "mps")

function ldr_model_legacy(data, T, optimizer)
    m = LDR.LDRModel(optimizer)
    cost = AffExpr(0.0)
    stored_in = data.stored_energy_initial
    for t in 1:T
        v = stage_variables(m, data, t; name = "$t")
        for x in (v.spill, v.hydro, v.thermal, v.exchange, v.deficit, v.stored)
            for xi in x
                set_attribute(xi, LDR.Stage(t))
            end
        end
        if t == 1
            inflow = data.inflow_initial
        else
            law = inflow_law(data, t)
            inflow = @variable(m, [1:4], set = LDR.Uncertainty(; distribution = law, stage = t), base_name = "inflow$t")
        end
        stage_constraints(m, data, t, v, stored_in, inflow)
        add_to_expression!(cost, stage_cost(data, v))
        stored_in = v.stored
    end
    @objective(m, Min, cost)
    return m
end

function export_one(problem, T, tag, data)
    optimizer = optimizer_with_attributes(HiGHS.Optimizer, "time_limit" => 0.0)
    m = problem == "hydro" ? ldr_model_legacy(data, T, optimizer) : build_legacy(T, optimizer)
    set_silent(m)
    dual = tag == "dual"
    set_attribute(m, LDR.SolvePrimal(), !dual)
    set_attribute(m, LDR.SolveDual(), dual)
    set_attribute(m, LDR.SlackAsExpr(), tag == "primal_sae")
    t = @elapsed optimize!(m)
    inner = dual ? m.dual_model : m.primal_model
    path = joinpath(MPS, "$(problem)_T$(T)_legacy_$(tag).mps")
    write_to_file(inner, path)
    println("$path vars=$(num_variables(inner)) optimize(0s)=$(round(t; digits = 2))")
    return
end

function main(args)
    mkpath(MPS)
    problem = args[1]
    data = problem == "hydro" ? hydro_thermal_data() : nothing
    tags = problem == "hydro" ? ["primal_sae", "primal", "dual"] : ["primal", "primal_sae", "dual"]
    skip = [a for a in args[2:end] if startswith(a, "skip=")]
    skip = isempty(skip) ? String[] : split(only(skip)[6:end], ",")
    for T in parse.(Int, filter(a -> !startswith(a, "skip="), args[2:end])), tag in tags
        tag in skip && continue
        export_one(problem, T, tag, data)
        GC.gc()
    end
end

main(ARGS)
