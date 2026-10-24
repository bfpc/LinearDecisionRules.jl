# Is the iteration count of a model a property of its formulation, or of the order of its rows and
# columns? Solve the LP as exported (seed 0) and with random row and column permutations, with the
# default HiGHS options, and append the iterations and times to ../results/compare_permute.csv.
#   julia --project=benchmark/compare benchmark/compare/permute_test.jl stem [nperm=5] [setting=default]
using SparseArrays, Statistics
include("lpdata.jl")

const OUT = joinpath(@__DIR__, "..", "results", "compare_permute.csv")

function main(args)
    stem = args[1]
    nperm = 5
    options = Pair{String,Any}[]
    label = "default"
    for a in args[2:end]
        if startswith(a, "nperm=")
            nperm = parse(Int, a[7:end])
        elseif startswith(a, "setting=")
            label = a[9:end]
            label == "primal_simplex" && push!(options, "simplex_strategy" => 4)
            label == "nopresolve" && push!(options, "presolve" => "off")
        end
    end
    lp = read_lp(joinpath(@__DIR__, "mps", stem * ".mps"))
    rows = NamedTuple[]
    for seed in 0:nperm
        m = seed == 0 ? lp : permute_lp(lp, Random.Xoshiro(seed))
        r = solve_lp(m; options)
        push!(rows, (; seed, r...))
        println("$stem [$label] seed=$seed iterations=$(r.iterations) time=$(round(r.time; digits = 3)) obj=$(r.objective)")
        new_file = !isfile(OUT)
        open(OUT, "a") do io
            new_file && println(io, "stem,setting,seed,simplex_iter,run_s,objective,status")
            println(io, join((stem, label, seed, r.iterations, r.time, r.objective, r.status), ","))
        end
    end
    its = [r.iterations for r in rows[2:end]]
    println("permuted: iterations min/median/max = $(minimum(its))/$(median(its))/$(maximum(its)); as exported $(rows[1].iterations)")
end

main(ARGS)
