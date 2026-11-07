# Study: other solvers than HiGHS on the inner problems, here Gurobi (default, and the dual simplex on
# one thread), against the HiGHS runs of `../breakpoints/results/lifted_box.csv`.
#
# For each problem, horizon T and number k of breakpoints on every uncertainty, it solves the primal
# and the dual LDR with `lifted_box = true` and `false`. The problems, the solvers and the options
# are those of `../grid.jl`; the solvers of Gurobi need it in the load path (see there). Rows are
# appended to `results/solvers.csv` next to this file (or in `LDR_BENCHMARK_RESULTS`).
#
# Usage (from the repository root):
#   julia --project=benchmark benchmark/solvers/solvers.jl production T=10,15,20 k=1,3 solver=gurobi,gurobi_dual1
#   julia --project=benchmark benchmark/solvers/solvers.jl hydro_joint T=4,8,12 kind=dual solver=gurobi

include("../grid.jl")

main(ARGS, joinpath(results_dir(@__DIR__), "solvers.csv"))
