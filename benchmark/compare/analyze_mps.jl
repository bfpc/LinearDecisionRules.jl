# Structure of exported LPs before and after HiGHS's presolve: rows by type, free columns, singleton
# rows and columns, nonzeros per row, coefficient range, columns by name prefix, presolve time.
# Appends to ../results/compare_structure.csv (`stage` is `original` or `presolved`).
#   julia --project=benchmark/compare benchmark/compare/analyze_mps.jl stem ...
using SparseArrays, Statistics
include("lpdata.jl")

const OUT = joinpath(@__DIR__, "..", "results", "compare_structure.csv")

_fmt(x::Integer) = x
_fmt(x::Real) = round(x; sigdigits = 4)

function record(stem, stage, s, extra)
    keys_ = (:rows, :cols, :nonzeros, :eq_rows, :le_ge_rows, :range_rows, :free_cols, :boxed_cols,
        :lower_only_cols, :singleton_rows, :singleton_cols, :max_col_nnz, :max_row_nnz,
        :mean_row_nnz, :mean_col_nnz, :abs_min, :abs_max, :abs_ratio)
    header = ["stem", "stage", keys_..., "presolve_s", "col_groups"]
    row = (stem, stage, (_fmt(getfield(s, k)) for k in keys_)..., extra.presolve_s, extra.groups)
    new_file = !isfile(OUT)
    open(OUT, "a") do io
        new_file && println(io, join(header, ","))
        println(io, join(row, ","))
    end
    println(rpad("$stem $stage", 42), join(("$k=$(_fmt(getfield(s, k)))" for k in keys_[1:13]), " "))
end

function analyze(stem)
    path = joinpath(@__DIR__, "mps", stem * ".mps")
    lp = read_lp(path; names = true)
    prefixes = Dict{String,Int}()
    for name in lp.colnames
        p = match(r"^[^\[\d]*", name).match
        prefixes[p] = get(prefixes, p, 0) + 1
    end
    groups = join(("$k:$n" for (k, n) in sort(collect(prefixes); by = last, rev = true)), " ")
    record(stem, "original", structure(lp), (; presolve_s = "", groups))
    pre, t = presolved_lp(path)
    record(stem, "presolved", structure(pre), (; presolve_s = round(t; digits = 3), groups = ""))
end

foreach(analyze, ARGS)
