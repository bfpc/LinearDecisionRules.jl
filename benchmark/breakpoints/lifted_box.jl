# Study: the implied box rows of lifted coordinates (`lifted_box`) across problems and solvers.
#
# For each problem, horizon T and number k of breakpoints on every uncertainty, it solves the primal
# and the dual LDR with `lifted_box = true` and `false`. The problems, the solvers and the options
# are those of `../grid.jl`. Rows are appended to `results/lifted_box.csv` next to this file (or in
# `LDR_BENCHMARK_RESULTS`).
#
# Usage (from the repository root):
#   julia --project=benchmark benchmark/breakpoints/lifted_box.jl hydro T=4,6,8 k=1,3 solver=simplex,ipx
#   julia --project=benchmark benchmark/breakpoints/lifted_box.jl kwg T=24 k=1 solver=ipopt limit=300
#   julia --project=benchmark benchmark/breakpoints/lifted_box.jl kwg T=48 k=3 kind=dual lifted_box=false

include("../grid.jl")

main(ARGS, joinpath(results_dir(@__DIR__), "lifted_box.csv"))
