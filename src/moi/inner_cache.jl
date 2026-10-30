# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The optimizer keeps the inner problems of its last `optimize!` and solves them again, with their
# parameter variables fixed to the new values, when nothing but the values of the parameters
# changed (model predictive control: solve, change an initial state, solve again). Everything else
# that an inner problem depends on is its fingerprint, compared at each `optimize!`: the canonical
# form without the values of the parameters, the moments, the inner optimizer and its attributes,
# and the formulation with its own attributes. The canonical form is rebuilt at every `optimize!`
# (it is cheap), so a change made anywhere in the model, including one that the form only shows as
# a different value (a parameter multiplied by a decision, a coefficient), rebuilds the problems.

# Structural equality: values of the same type whose fields, or elements, are equal recursively.
# Objects that are not plain data (modules, pointers) compare by identity; a false `false` only
# costs a rebuild. Distributions are compared by value, field by field at every depth (unlike
# `_same_distribution`, which tries `==` first and compares only the top-level fields), and treated
# as immutable: a distribution changed in place, rather than replaced, is not seen.
function _same_value(a, b, depth::Int = 0)
    a === b && return true
    typeof(a) === typeof(b) || return false
    depth > 32 && return false
    return _same_fields(a, b, depth + 1)
end

_same_fields(a::Number, b::Number, ::Int) = isequal(a, b)

_same_fields(a::Union{AbstractString,Symbol}, b, ::Int) = a == b

_same_fields(a::Type, b::Type, ::Int) = a == b

_same_fields(::Module, ::Module, ::Int) = false

# `==` is fast for sparse matrices; a `NaN` makes it `false`, which only costs a rebuild.
function _same_fields(a::AbstractArray{T}, b::AbstractArray{T}, depth::Int) where {T}
    size(a) == size(b) || return false
    isbitstype(T) && return a == b
    return all(_same_value(x, y, depth) for (x, y) in zip(a, b))
end

function _same_fields(a::AbstractDict, b::AbstractDict, depth::Int)
    length(a) == length(b) || return false
    return all(haskey(b, k) && _same_value(v, b[k], depth) for (k, v) in a)
end

# `a` and `b` have the same type (see `_same_value`).
function _same_fields(a, b, depth::Int)
    T = typeof(a)
    isstructtype(T) || return false
    for i in 1:fieldcount(T)
        isdefined(a, i) == isdefined(b, i) || return false
        isdefined(a, i) || continue
        _same_value(getfield(a, i), getfield(b, i), depth) || return false
    end
    return true
end

# Whether `A` and `B` are equal except in their first column.
function _same_except_constant(A::_SparseMatrix, B::_SparseMatrix)
    size(A) == size(B) || return false
    a, b = SparseArrays.getcolptr(A), SparseArrays.getcolptr(B)
    # The entries of columns 2:end are `a[2]:a[end]-1` in `A`, `b[2]:b[end]-1` in `B`.
    all(k -> a[k] - a[2] == b[k] - b[2], 2:length(a)) || return false
    ka, kb = a[2]:a[end]-1, b[2]:b[end]-1
    return view(SparseArrays.rowvals(A), ka) == view(SparseArrays.rowvals(B), kb) &&
           view(SparseArrays.nonzeros(A), ka) == view(SparseArrays.nonzeros(B), kb)
end

# Whether two canonical forms differ at most in the values of their parameters: `B_*[:, 1]` and
# `f` are then equal up to the parameter terms, which are compared without the values.
function _same_form(a::CanonicalForm, b::CanonicalForm)
    for name in fieldnames(CanonicalForm)
        if name in (:B_e, :B_u, :B_l)
            _same_except_constant(getfield(a, name), getfield(b, name)) || return false
        elseif name == :parameters
            for field in fieldnames(ParameterTerms)
                field == :values && continue
                pa, pb = getfield(a.parameters, field), getfield(b.parameters, field)
                _same_value(pa, pb) || return false
            end
        elseif name != :f
            _same_value(getfield(a, name), getfield(b, name)) || return false
        end
    end
    return true
end

# What the inner problems of an `optimize!` depend on, except the values of the parameters and the
# formulations (compared one by one).
struct _Fingerprint
    form::CanonicalForm
    moments::Moments
    factory::Any
    silent::Bool
    raw_attributes::Dict{String,Any}
end

function _same_problem(a::_Fingerprint, b::_Fingerprint)
    return a.silent == b.silent &&
           _same_value(a.factory, b.factory) &&
           _same_value(a.raw_attributes, b.raw_attributes) &&
           _same_value(a.moments, b.moments) &&
           _same_form(a.form, b.form)
end

# The inner problems of the last `optimize!`, with their formulations, and the fingerprint they were
# built for. `num_builds` counts the inner problems built, not reused.
mutable struct _InnerCache
    fingerprint::Union{Nothing,_Fingerprint}
    problems::Vector{Pair{AbstractFormulation,_InnerProblem}}
    num_builds::Int
end

_InnerCache() = _InnerCache(nothing, Pair{AbstractFormulation,_InnerProblem}[], 0)

# Start an `optimize!` with `fingerprint`: the problems are dropped unless it is the same.
function _update!(cache::_InnerCache, fingerprint::_Fingerprint)
    if cache.fingerprint === nothing || !_same_problem(cache.fingerprint, fingerprint)
        empty!(cache.problems)
    end
    cache.fingerprint = fingerprint
    return
end

# The inner problem of `f` for the fingerprint of the cache, if it was built.
function _cached_problem(cache::_InnerCache, f::AbstractFormulation)
    k = findfirst(p -> _same_value(p.first, f), cache.problems)
    return k === nothing ? nothing : cache.problems[k].second
end
