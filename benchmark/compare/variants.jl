# Variants of the exported 0.4 dual LPs, made by post-processing the MPS file (src/ is not touched).
#
#   julia --project=benchmark/compare benchmark/compare/variants.jl nou <stem of a new dual>
#
# nou: substitute the auxiliary u[r] = S[r,:] mu out. `u` is a variable with the equality
#      `u - sum_j mu_j S[r,j] = 0` and the bound u >= 0; every cone row has `Wmu_k u`. The variant
#      deletes the column u, adds `Wmu_k mu_j` to the coefficient of S[r,j] in each cone row (as the
#      previous version's dense cone `S M W'ᵀ >= 0` does) and turns the equality into the row
#      `sum_j mu_j S[r,j] >= 0`. It is the same LP with fewer columns and denser rows.
#
# Writes mps/<stem>_nou.mps (formulation name `dual_nou`), solvable by run_solves.jl.
using SparseArrays
include("lpdata.jl")

function write_lp(lp, path)
    h = HiGHS.Highs_create()
    HiGHS.Highs_setBoolOptionValue(h, "output_flag", 0)
    A = lp.A
    HiGHS.Highs_passLp(
        h, size(A, 2), size(A, 1), nnz(A), HiGHS.kHighsMatrixFormatColwise, lp.sense, lp.offset,
        lp.c, lp.cl, lp.cu, lp.rl, lp.ru, Cint.(A.colptr .- 1)[1:end-1], Cint.(A.rowval .- 1), A.nzval,
    ) == 0 || error("passLp")
    HiGHS.Highs_writeModel(h, path)
    HiGHS.Highs_destroy(h)
end

function substitute_u(lp)
    names = lp.colnames
    ucols = findall(startswith("u["), names)
    isempty(ucols) && error("no u variables")
    A = lp.A
    nr, nc = size(A)
    keep = setdiff(1:nc, ucols)
    mean_rows = Int[]
    for k in ucols
        rows = A.rowval[nzrange(A, k)]
        eqs = [i for i in rows if lp.rl[i] == lp.ru[i] && lp.rl[i] == 0.0]
        length(eqs) == 1 || error("column $(names[k]) has $(length(eqs)) equality rows")
        i = only(eqs)
        A[i, k] == 1.0 || error("coefficient of $(names[k]) is $(A[i, k])")
        push!(mean_rows, i)
    end
    @assert all(lp.c[ucols] .== 0) && all(lp.cl[ucols] .== 0) && all(lp.cu[ucols] .>= 1e20)
    P = -A[mean_rows, keep]                       # u_k = P[k, :] * x_keep
    B = A[:, keep] + A[:, ucols] * P              # the mean rows become zero
    E = sparse(mean_rows, 1:length(ucols), ones(length(ucols)), nr, length(ucols))
    B = dropzeros(B + E * P)                      # the mean rows are now `mu . S >= 0`
    rl, ru = copy(lp.rl), copy(lp.ru)
    rl[mean_rows] .= 0.0
    ru[mean_rows] .= 1e30
    return (; lp..., A = B, c = lp.c[keep], cl = lp.cl[keep], cu = lp.cu[keep], rl, ru, colnames = String[])
end

function main(args)
    variant, stem = args[1], args[2]
    variant == "nou" || error("unknown variant $variant")
    lp = read_lp(joinpath(@__DIR__, "mps", stem * ".mps"); names = true)
    out = substitute_u(lp)
    path = joinpath(@__DIR__, "mps", stem * "_nou.mps")
    write_lp(out, path)
    println("$path rows=$(size(out.A, 1)) cols=$(size(out.A, 2)) nnz=$(nnz(out.A)) (was $(size(lp.A, 1)) x $(size(lp.A, 2)), $(nnz(lp.A)))")
end

main(ARGS)
