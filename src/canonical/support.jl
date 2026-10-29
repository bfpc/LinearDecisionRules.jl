# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    SupportBlock

One factor of the support Ξ, built by [`support_blocks`](@ref): the coordinates `η_b = ξ[columns]`
range over `{η_b : W η_b - h ∈ K}`, where `K = cones[1] × cones[2] × …`.

## Fields

 * `columns::Vector{Int}`: the columns of ξ in the block, sorted (never the constant column 1).
 * `W::SparseMatrixCSC{Float64,Int}`: one row per inequality, one column per entry of `columns`.
 * `h::Vector{Float64}`
 * `cones::Vector{Union{MOI.Nonnegatives,MOI.SecondOrderCone}}`: the pieces of the cone `K` of
   `W η_b - h`, in the order of the rows: first `MOI.Nonnegatives` for the linear rows (possibly
   of dimension zero), then one `MOI.SecondOrderCone` per second-order cone row.
 * `roles::Vector{Symbol}`: the role of each row, `:box` (a bound from the distribution), `:law`
   (a row of [`support_constraints`](@ref) of the distribution), `:valid` (a constraint tagged
   `ValidInequality()`), `:truncate` (a constraint tagged `Truncate()`, which also conditions the
   law) or `:hull` (a convex-hull row of a lifting). The rows of a second-order cone have one
   role.
 * `constraints::Vector{Union{Nothing,MOI.ConstraintIndex}}`: the constraint each `:valid` or
   `:truncate` row comes from; `nothing` for the other rows.

A block without rows is unbounded: `η_b ∈ ℝ^k`.
"""
struct SupportBlock
    columns::Vector{Int}
    W::_SparseMatrix
    h::Vector{Float64}
    cones::Vector{_SupportCone}
    roles::Vector{Symbol}
    constraints::Vector{Union{Nothing,MOI.ConstraintIndex}}
end

# The rows of each piece of the cone of a block.
function _cone_rows(block::SupportBlock)
    offsets = cumsum([0; [MOI.dimension(cone) for cone in block.cones]])
    return [offsets[k]+1:offsets[k+1] for k in eachindex(block.cones)]
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

The rows of Ξ are the finite bounds of the distributions (`minimum` and `maximum`) and their
[`support_constraints`](@ref), the rows tagged `ValidInequality()` or `Truncate()`, linear or in a
second-order cone, and the convex-hull rows of the liftings. The blocks are the connected
components of the coordinates coupled by some row, so every uncertainty column is in exactly one
block; blocks are sorted by their first column. Every block contains the mean of its coordinates
(checked by [`canonicalize`](@ref) when the mean is finite; the estimated mean of a conditioned
law is that of draws in the conditioning set), so it is nonempty.
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
    # The rows of the laws, one piece of their cone at a time, over the η columns that they use.
    law_pieces = Tuple{Vector{Int},Matrix{Float64},Vector{Float64},_SupportCone}[]
    for law in form.laws
        rows = support_constraints(law.distribution)
        offset = 0
        for cone in rows.cones
            r = offset .+ (1:MOI.dimension(cone))
            offset += MOI.dimension(cone)
            W = rows.W[r, :]
            used = [k for k in axes(W, 2) if any(!iszero, view(W, :, k))]
            isempty(used) && continue
            push!(law_pieces, (law.columns[used] .- 1, W[:, used], rows.h[r], cone))
        end
    end
    # Union–find of the η coordinates coupled by a row.
    parent = collect(1:m)
    function join!(coordinates)
        isempty(coordinates) && return
        a = _find_root!(parent, first(coordinates))
        for j in coordinates
            parent[_find_root!(parent, j)] = a
        end
        return
    end
    Wt = SparseArrays.sparse(transpose(form.W_valid))
    rv, nz = SparseArrays.rowvals(Wt), SparseArrays.nonzeros(Wt)
    for r in axes(Wt, 2)
        join!(view(rv, SparseArrays.nzrange(Wt, r)))
    end
    conic_columns = [_nonzero_columns(row.W) for row in form.conic_rows]
    foreach(join!, conic_columns)
    foreach(piece -> join!(piece[1]), law_pieces)
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
    cones = [_SupportCone[] for _ in columns]
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
    function add_piece!(η, W, rhs, role, ci)
        b = block[first(η)]
        first_row = length(h[b]) + 1
        foreach(r -> add_row!(b, role, r, ci), rhs)
        for i in axes(W, 1), (k, j) in enumerate(η)
            _push!(triplets[b], first_row + i - 1, local_index[j], W[i, k])
        end
        return b
    end
    for (η, W, rhs, cone) in law_pieces
        cone isa MOI.Nonnegatives && add_piece!(η, W, rhs, :law, nothing)
    end
    for r in axes(Wt, 2)
        range = SparseArrays.nzrange(Wt, r)
        isempty(range) && continue
        b = block[rv[first(range)]]
        ci = form.valid_constraints[r]
        role = ci === nothing ? :hull : form.conditioning[r] ? :truncate : :valid
        row = add_row!(b, role, form.h_valid[r], ci)
        for k in range
            _push!(triplets[b], row, local_index[rv[k]], nz[k])
        end
    end
    for b in eachindex(columns)
        push!(cones[b], MOI.Nonnegatives(length(h[b])))
    end
    # The second-order cones come after the linear rows, one piece of the cone each.
    for (η, W, rhs, cone) in law_pieces
        cone isa MOI.SecondOrderCone || continue
        push!(cones[add_piece!(η, W, rhs, :law, nothing)], cone)
    end
    for (c, row) in enumerate(form.conic_rows)
        η = conic_columns[c]
        isempty(η) && continue
        role = row.conditioning ? :truncate : :valid
        push!(cones[add_piece!(η, Matrix(row.W[:, η]), row.h, role, row.constraint)], row.cone)
    end
    return map(eachindex(columns)) do b
        W = _sparse(triplets[b], length(h[b]), length(columns[b]))
        return SupportBlock(columns[b], W, h[b], cones[b], roles[b], constraints[b])
    end
end

# The laws with the representation of their support chosen: second-order cones if the inner
# optimizer supports them, unless a decision is integer, since a mixed-integer second-order cone
# program is much harder than a mixed-integer linear one.
function _resolve_supports(form::CanonicalForm, conic_support::Bool)
    any(law -> _has_auto_support(law.distribution), form.laws) || return form
    conic = conic_support && !any(form.integer)
    laws = map(form.laws) do law
        d = _resolve_support(law.distribution, conic)
        return UncertaintyBlock(d, law.columns, law.stages)
    end
    return _replace(form; laws)
end
