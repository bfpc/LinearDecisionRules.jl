# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Generic helpers, with no type of the package: sparse matrices, timing and lists of names.

const _SparseMatrix = SparseArrays.SparseMatrixCSC{Float64,Int}

# The columns where some row of `W` is nonzero.
function _nonzero_columns(W::_SparseMatrix)
    return [j for j in axes(W, 2) if !isempty(SparseArrays.nzrange(W, j))]
end

# Coordinate lists of a sparse matrix.
struct _Triplets
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{Float64}
end

_Triplets() = _Triplets(Int[], Int[], Float64[])

function _push!(t::_Triplets, i::Int, j::Int, v::Float64)
    if !iszero(v)
        push!(t.I, i)
        push!(t.J, j)
        push!(t.V, v)
    end
    return
end

_sparse(t::_Triplets, m::Int, n::Int) = SparseArrays.sparse(t.I, t.J, t.V, m, n)

_symmetric(A::Matrix{Float64}) = (A + A') ./ 2

# A partition of `1:n` into groups, which `_join!` merges (union–find).
struct _UnionFind
    parent::Vector{Int}
end

_UnionFind(n::Int) = _UnionFind(collect(1:n))

function _root!(u::_UnionFind, a::Int)
    while u.parent[a] != a
        u.parent[a] = u.parent[u.parent[a]]
        a = u.parent[a]
    end
    return a
end

# Merge the groups of `elements` into one.
function _join!(u::_UnionFind, elements)
    isempty(elements) && return
    a = _root!(u, first(elements))
    for e in elements
        u.parent[_root!(u, e)] = a
    end
    return
end

# The groups, each sorted, in the order of their smallest element, and the group of each element.
function _groups(u::_UnionFind)
    group_of_root = Dict{Int,Int}()
    groups, group = Vector{Int}[], zeros(Int, length(u.parent))
    for e in eachindex(u.parent)
        g = get!(group_of_root, _root!(u, e), length(groups) + 1)
        g > length(groups) && push!(groups, Int[])
        push!(groups[g], e)
        group[e] = g
    end
    return groups, group
end

# `time_ns` rather than `time`, whose resolution can be 16 ms on Windows.
_seconds_since(start::UInt64) = (time_ns() - start) / 1e9

# "a, b, c and 2 more", for at most `n` names.
function _name_list(names::Vector{String}, n::Int = 5)
    length(names) <= n && return join(names, ", ")
    return "$(join(names[1:n], ", ")) and $(length(names) - n) more"
end
