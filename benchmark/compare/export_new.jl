# Export the inner LPs of the 0.4 package to MPS.
#   julia --project=benchmark/compare benchmark/compare/export_new.jl inventory 32 64 128
#   julia --project=benchmark/compare benchmark/compare/export_new.jl hydro 4 8
# Writes benchmark/compare/mps/<problem>_T<T>_new_<primal|dual>.mps and appends the build times to
# mps/export_new.log. The model is built by `LDR._build!` (names on), after one `optimize!` of the
# LDRModel with a zero HiGHS time limit, which only provides the canonical form and the moments.
using JuMP
import HiGHS, Distributions
import LinearDecisionRules as LDR

include("inventory.jl")
include("../hydro/data.jl")
include("../hydro/model.jl")

const MPS = joinpath(@__DIR__, "mps")

function export_one(problem, T, formulation, tag, data)
    m = problem == "hydro" ? ldr_model(data, T; formulations = [formulation]) :
        build_new(T, HiGHS.Optimizer)
    problem == "hydro" || set_attribute(m, LDR.Formulations(), [formulation])
    set_silent(m)
    set_attribute(m, "time_limit", 0.0)
    t_canon = @elapsed optimize!(m)
    opt = unsafe_backend(m)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, true)
    t_build = @elapsed LDR._build!(model, formulation, opt.form, opt.moments)
    path = joinpath(MPS, "$(problem)_T$(T)_new_$(tag).mps")
    write_to_file(model, path)
    println("$path vars=$(num_variables(model)) optimize(0s)=$(round(t_canon; digits = 2)) build=$(round(t_build; digits = 2))")
    return
end

function main(args)
    mkpath(MPS)
    problem = args[1]
    data = problem == "hydro" ? hydro_thermal_data() : nothing
    for T in parse.(Int, args[2:end])
        export_one(problem, T, LDR.PrimalLDR(), "primal", data)
        export_one(problem, T, LDR.DualLDR(), "dual", data)
    end
end

main(ARGS)
