# Study: the scaling of the uncertainties, written by the user (`production`, `production_centered`,
# `production_unit`) or done by the optimizer (`RescaleUncertainties()`).
#
# For each problem, horizon T and number k of breakpoints on every uncertainty, it solves the primal
# (without the implied box rows of lifted coordinates) and the dual (with them), the defaults of
# `PrimalLDR` and `DualLDR`, with `RescaleUncertainties()` true and false. The problems, the solvers
# and the options are those of `../grid.jl`. Rows are appended to `results/rescale.csv` next to this
# file (or in `LDR_BENCHMARK_RESULTS`).
#
# Usage (from the repository root):
#   julia --project=benchmark benchmark/scaling/rescale.jl production_unit T=10,15,20 k=1,3 kind=dual rescale=false
#   julia --project=benchmark benchmark/scaling/rescale.jl kwg T=24,48 k=1,3 kind=primal solver=simplex,gurobi

include("../grid.jl")

# The primal and the dual run separately, each with its default `lifted_box`.
function main_scaling(args)
    results = joinpath(results_dir(@__DIR__), "rescale.csv")
    defaults = Dict("rescale" => "true,false")
    kinds = [split(a, "="; limit = 2)[2] for a in args if startswith(a, "kind=")]
    others = [a for a in args if !startswith(a, "kind=")]
    for kind in split(isempty(kinds) ? "primal,dual" : only(kinds), ",")
        box = kind == "primal" ? "false" : "true"
        main([others; "kind=$kind"], results; defaults = merge(defaults, Dict("lifted_box" => box)))
    end
    return
end

main_scaling(ARGS)
