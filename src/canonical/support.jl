# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    SupportBlock

One factor of the support Ξ, built by [`support_blocks`](@ref): the coordinates `η_b = ξ[columns]`
range over `{η_b : W η_b - h ∈ cone}`.

## Fields

 * `columns::Vector{Int}`: the columns of ξ in the block, sorted (never the constant column 1).
 * `W::SparseMatrixCSC{Float64,Int}`: one row per inequality, one column per entry of `columns`.
 * `h::Vector{Float64}`
 * `cone::MOI.Nonnegatives`: the cone of `W η_b - h`; for now always the nonnegative orthant.
 * `roles::Vector{Symbol}`: the role of each row, `:box` (a bound from the distribution), `:valid`
   (a constraint tagged `ValidInequality()`) or `:hull` (a convex-hull row of a lifting).
 * `constraints::Vector{Union{Nothing,MOI.ConstraintIndex}}`: the constraint each `:valid` row
   comes from; `nothing` for the other rows.

A block without rows is unbounded: `η_b ∈ ℝ^k`.
"""
struct SupportBlock
    columns::Vector{Int}
    W::_SparseMatrix
    h::Vector{Float64}
    cone::MOI.Nonnegatives
    roles::Vector{Symbol}
    constraints::Vector{Union{Nothing,MOI.ConstraintIndex}}
end

# The bounds `(lower, upper)` of the support of a law, or `nothing` if the law does not define
# them. `applicable(minimum, d)` cannot tell: every multivariate law falls back on
# `Base.minimum(itr)`, which throws a `MethodError` for laws without bounds, such as `Dirichlet`.
function _support_bounds(d::Distributions.UnivariateDistribution)
    return Float64[minimum(d)], Float64[maximum(d)]
end

function _support_bounds(d::Distributions.MultivariateDistribution)
    try
        return Vector{Float64}(minimum(d)), Vector{Float64}(maximum(d))
    catch err
        err isa MethodError || rethrow()
        return nothing
    end
end

# The box of the support of a law; a bound that the law does not define is infinite, which keeps
# Ξ a superset of the support.
function _support_box(d::Distributions.Distribution)
    bounds = _support_bounds(d)
    if bounds === nothing
        n = _dimension(d)
        return fill(-Inf, n), fill(Inf, n)
    end
    return bounds
end

function _find_root!(parent::Vector{Int}, a::Int)
    while parent[a] != a
        parent[a] = parent[parent[a]]
        a = parent[a]
    end
    return a
end

"""
    support_blocks(form::CanonicalForm) -> Vector{SupportBlock}

Return the support Ξ of the uncertainties of `form` as a product of [`SupportBlock`](@ref)s.

The rows of Ξ are the finite bounds of the distributions (`minimum` and `maximum`), the rows
tagged `ValidInequality()` and the convex-hull rows of the liftings. The blocks are the connected
components of the coordinates coupled by some row, so every uncertainty column is in exactly one
block; blocks are sorted by their first column. Every block contains the mean of its coordinates
(checked by [`canonicalize`](@ref) when the mean is finite), so it is nonempty.
"""
function support_blocks(form::CanonicalForm)
    m = length(form.uncertainties)
    lo, hi = fill(-Inf, m), fill(Inf, m)
    for law in form.laws
        l, u = _support_box(law.distribution)
        for (k, j) in enumerate(law.columns)
            if l[k] > u[k]
                error("The support of the uncertainty $(form.uncertainty_names[j-1]) is empty.")
            end
            lo[j-1], hi[j-1] = l[k], u[k]
        end
    end
    # Union–find of the η coordinates coupled by a valid row.
    parent = collect(1:m)
    Wt = SparseArrays.sparse(transpose(form.W_valid))
    rv, nz = SparseArrays.rowvals(Wt), SparseArrays.nonzeros(Wt)
    for r in axes(Wt, 2), k in SparseArrays.nzrange(Wt, r)
        a = _find_root!(parent, rv[first(SparseArrays.nzrange(Wt, r))])
        b = _find_root!(parent, rv[k])
        parent[b] = a
    end
    block_of_root = Dict{Int,Int}()
    block = zeros(Int, m)
    local_index = zeros(Int, m)
    columns = Vector{Int}[]
    for j in 1:m
        b = get!(block_of_root, _find_root!(parent, j), length(columns) + 1)
        b > length(columns) && push!(columns, Int[])
        push!(columns[b], 1 + j)
        block[j], local_index[j] = b, length(columns[b])
    end
    triplets = [_Triplets() for _ in columns]
    h = [Float64[] for _ in columns]
    roles = [Symbol[] for _ in columns]
    constraints = [Union{Nothing,MOI.ConstraintIndex}[] for _ in columns]
    function add_row!(b, role, rhs, ci)
        push!(h[b], rhs)
        push!(roles[b], role)
        push!(constraints[b], ci)
        return length(h[b])
    end
    for j in 1:m
        b = block[j]
        isfinite(lo[j]) &&
            _push!(triplets[b], add_row!(b, :box, lo[j], nothing), local_index[j], 1.0)
        isfinite(hi[j]) &&
            _push!(triplets[b], add_row!(b, :box, -hi[j], nothing), local_index[j], -1.0)
    end
    for r in axes(Wt, 2)
        range = SparseArrays.nzrange(Wt, r)
        isempty(range) && continue
        b = block[rv[first(range)]]
        ci = form.valid_constraints[r]
        row = add_row!(b, ci === nothing ? :hull : :valid, form.h_valid[r], ci)
        for k in range
            _push!(triplets[b], row, local_index[rv[k]], nz[k])
        end
    end
    return map(eachindex(columns)) do b
        W = _sparse(triplets[b], length(h[b]), length(columns[b]))
        return SupportBlock(
            columns[b],
            W,
            h[b],
            MOI.Nonnegatives(length(h[b])),
            roles[b],
            constraints[b],
        )
    end
end
