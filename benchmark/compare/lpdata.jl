# Read an MPS file into arrays with HiGHS (no JuMP: the names of the models may be non-ASCII).
import HiGHS, Random
using SparseArrays

function read_lp(path; names = false)
    h = HiGHS.Highs_create()
    HiGHS.Highs_setBoolOptionValue(h, "output_flag", 0)
    HiGHS.Highs_readModel(h, path) == 0 || error("cannot read $path")
    nc, nr = HiGHS.Highs_getNumCol(h), HiGHS.Highs_getNumRow(h)
    nz = HiGHS.Highs_getNumNz(h)
    sense, offset = Ref{Cint}(0), Ref{Cdouble}(0)
    c, cl, cu, rl, ru = (zeros(n) for n in (nc, nc, nc, nr, nr))
    start, index, value = zeros(Cint, nc + 1), zeros(Cint, nz), zeros(nz)
    q = Ref{Cint}(0)
    HiGHS.Highs_getModel(
        h, HiGHS.kHighsMatrixFormatColwise, 0, Ref(Cint(nc)), Ref(Cint(nr)), Ref(Cint(nz)), q,
        sense, offset, c, cl, cu, rl, ru, start, index, value, C_NULL, C_NULL, C_NULL, C_NULL,
    ) == 0 || error("getModel")
    start[end] = nz
    A = SparseMatrixCSC(nr, nc, Int.(start) .+ 1, Int.(index) .+ 1, value)
    colnames = String[]
    if names
        buf = zeros(UInt8, 256)
        for j in 0:nc-1
            HiGHS.Highs_getColName(h, j, buf)
            push!(colnames, unsafe_string(pointer(buf)))
        end
    end
    HiGHS.Highs_destroy(h)
    return (; c, cl, cu, rl, ru, A, sense = sense[], offset = offset[], colnames)
end

# Solve an LP given as arrays, silently; return (iterations, run time, objective, status).
function solve_lp(lp; options = Pair{String,Any}[])
    h = HiGHS.Highs_create()
    HiGHS.Highs_setBoolOptionValue(h, "output_flag", 0)
    for (k, v) in options
        v isa Bool ? HiGHS.Highs_setBoolOptionValue(h, k, v) :
        v isa Integer ? HiGHS.Highs_setIntOptionValue(h, k, v) :
        v isa Real ? HiGHS.Highs_setDoubleOptionValue(h, k, v) :
        HiGHS.Highs_setStringOptionValue(h, k, v)
    end
    A = lp.A
    HiGHS.Highs_passLp(
        h, size(A, 2), size(A, 1), nnz(A), HiGHS.kHighsMatrixFormatColwise, lp.sense, lp.offset,
        lp.c, lp.cl, lp.cu, lp.rl, lp.ru, Cint.(A.colptr .- 1)[1:end-1], Cint.(A.rowval .- 1), A.nzval,
    )
    t = @elapsed HiGHS.Highs_run(h)
    it = Ref{Cint}(0)
    HiGHS.Highs_getIntInfoValue(h, "simplex_iteration_count", it)
    ipm = Ref{Cint}(0)
    HiGHS.Highs_getIntInfoValue(h, "ipm_iteration_count", ipm)
    out = (; iterations = it[], ipm_iterations = ipm[], time = t,
        objective = HiGHS.Highs_getObjectiveValue(h), status = HiGHS.Highs_getModelStatus(h))
    HiGHS.Highs_destroy(h)
    return out
end

function permute_lp(lp, rng)
    nr, nc = size(lp.A)
    pr, pc = Random.randperm(rng, nr), Random.randperm(rng, nc)
    return (; lp..., c = lp.c[pc], cl = lp.cl[pc], cu = lp.cu[pc], rl = lp.rl[pr], ru = lp.ru[pr],
        A = lp.A[pr, pc], colnames = String[])
end

# The LP after HiGHS's presolve (default options), and the time presolve took.
function presolved_lp(path)
    h = HiGHS.Highs_create()
    HiGHS.Highs_setBoolOptionValue(h, "output_flag", 0)
    HiGHS.Highs_readModel(h, path) == 0 || error("cannot read $path")
    t = @elapsed HiGHS.Highs_presolve(h)
    nc, nr, nz = HiGHS.Highs_getPresolvedNumCol(h), HiGHS.Highs_getPresolvedNumRow(h), HiGHS.Highs_getPresolvedNumNz(h)
    sense, offset = Ref{Cint}(0), Ref{Cdouble}(0)
    c, cl, cu, rl, ru = (zeros(n) for n in (nc, nc, nc, nr, nr))
    start, index, value = zeros(Cint, nc + 1), zeros(Cint, nz), zeros(nz)
    HiGHS.Highs_getPresolvedLp(
        h, HiGHS.kHighsMatrixFormatColwise, Ref(Cint(nc)), Ref(Cint(nr)), Ref(Cint(nz)),
        sense, offset, c, cl, cu, rl, ru, start, index, value, C_NULL,
    ) == 0 || error("getPresolvedLp")
    start[end] = nz
    A = SparseMatrixCSC(nr, nc, Int.(start) .+ 1, Int.(index) .+ 1, value)
    HiGHS.Highs_destroy(h)
    return (; c, cl, cu, rl, ru, A, sense = sense[], offset = offset[], colnames = String[]), t
end

# Structure of an LP: counts of row and column types, nonzeros per row and column, coefficient range.
function structure(lp)
    inf(x) = abs(x) >= 1e20  # HiGHS's infinity is 1e30
    A = lp.A
    nr, nc = size(A)
    rownnz = diff(sparse(transpose(A)).colptr)
    colnnz = diff(A.colptr)
    v = abs.(nonzeros(A))
    eq = count(lp.rl .== lp.ru)
    le_ge = count(i -> inf(lp.rl[i]) != inf(lp.ru[i]), 1:nr)
    return (;
        rows = nr, cols = nc, nonzeros = nnz(A), eq_rows = eq, le_ge_rows = le_ge,
        range_rows = nr - eq - le_ge,
        free_cols = count(j -> inf(lp.cl[j]) && inf(lp.cu[j]), 1:nc),
        boxed_cols = count(j -> !inf(lp.cl[j]) && !inf(lp.cu[j]), 1:nc),
        lower_only_cols = count(j -> !inf(lp.cl[j]) && inf(lp.cu[j]), 1:nc),
        singleton_rows = count(==(1), rownnz), singleton_cols = count(==(1), colnnz),
        max_col_nnz = maximum(colnnz), max_row_nnz = maximum(rownnz),
        mean_row_nnz = sum(rownnz) / nr, mean_col_nnz = sum(colnnz) / nc,
        abs_min = minimum(v), abs_max = maximum(v), abs_ratio = maximum(v) / minimum(v),
    )
end
