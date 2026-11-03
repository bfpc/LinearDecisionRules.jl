# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# `CombinationPiecewise`: the map and the outer approximation of the convex hull of its lifted
# support. Its law, which merges the laws of its inputs, is `_CombinationLaw`
# (`probability/distributions/combination_law.jl`).

"""
    CombinationPiecewise(
        coefficients::AbstractVector{<:Real},
        breakpoints::AbstractVector{<:Real};
        constant::Real = 0.0,
    )

The piecewise-linear lifting of the combination `s = constant + coefficients ⋅ η` of several
uncertainties `η`, with the interior `breakpoints` `b₁ < … < bₙ` of `s` (sorted on construction).
Its lifted coordinates are the pieces of `s` after the first,
```math
p_k = \\min(\\max(s - b_{k-1}, 0), b_k - b_{k-1}), \\quad k = 2, \\dots, n, \\qquad
p_{n+1} = \\max(s - b_n, 0),
```
as for [`AxisPiecewise`](@ref), and they come *in addition to* `η`: the uncertainties stay
coordinates of the rules, so that a rule linear in `(η, p)` is affine in `η` plus piecewise affine
in `s`. The first piece, `min(s, b₁) = s - p₂ - … - pₙ₊₁`, is a combination of the others and of
`η`, so it is not a coordinate, and the retraction is zero: `η` is read from its own columns.

The support of the lifted coordinates is described by the outer approximation of Georghiou,
Wiesemann and Kuhn (2015): the support of `η`, and the convex hull of the lifted segment of `s` on
its range `[s_min, s_max]` over the support (the rows of [`AxisPiecewise`](@ref), written with the
first piece above). The primal rule is then feasible and the dual bound valid, possibly looser
than with the exact convex hull, which is intractable in general.

A lifting in `Experimental`: set it on a model with [`Experimental.set_breakpoints`](@ref).

## Fields

 * `coefficients::Vector{Float64}`: the coefficient of each input in `s`.
 * `constant::Float64`
 * `breakpoints::Vector{Float64}`: the interior breakpoints of `s`, sorted.
"""
struct CombinationPiecewise <: AbstractLifting
    coefficients::Vector{Float64}
    constant::Float64
    breakpoints::Vector{Float64}

    function CombinationPiecewise(
        coefficients::AbstractVector{<:Real},
        breakpoints::AbstractVector{<:Real};
        constant::Real = 0.0,
    )
        a = Vector{Float64}(coefficients)
        points = sort!(Vector{Float64}(breakpoints))
        if isempty(a) || all(iszero, a)
            throw(ArgumentError("CombinationPiecewise needs a nonzero coefficient."))
        elseif !all(isfinite, a) || !isfinite(constant)
            throw(ArgumentError("The coefficients and the constant must be finite."))
        elseif isempty(points)
            throw(ArgumentError("CombinationPiecewise needs at least one breakpoint."))
        elseif !all(isfinite, points)
            throw(ArgumentError("The breakpoints must be finite; got $breakpoints."))
        elseif !allunique(points)
            throw(ArgumentError("The breakpoints must be distinct; got $breakpoints."))
        end
        return new(a, Float64(constant), points)
    end
end

function Base.:(==)(a::CombinationPiecewise, b::CombinationPiecewise)
    return a.coefficients == b.coefficients &&
           a.constant == b.constant &&
           a.breakpoints == b.breakpoints
end

function Base.hash(l::CombinationPiecewise, h::UInt)
    return hash((l.coefficients, l.constant, l.breakpoints), hash(CombinationPiecewise, h))
end

_num_inputs(l::CombinationPiecewise) = length(l.coefficients)

_num_lifted(l::CombinationPiecewise) = length(l.breakpoints)

_keeps_inputs(::CombinationPiecewise) = true

function _combination(l::CombinationPiecewise, η::AbstractVector{<:Real})
    return l.constant + LinearAlgebra.dot(l.coefficients, η)
end

function lift(l::CombinationPiecewise, η::AbstractVector{<:Real})
    if length(η) != _num_inputs(l)
        throw(
            DimensionMismatch("The lifting has $(_num_inputs(l)) inputs; got $(length(η)) values."),
        )
    end
    s, b = _combination(l, η), l.breakpoints
    p = zeros(length(b))
    for k in 2:length(b)
        p[k-1] = clamp(s - b[k-1], 0.0, b[k] - b[k-1])
    end
    p[end] = max(s - b[end], 0.0)
    return p
end

# The inputs are read from their own columns of ξ̃: the pieces do not enter the retraction.
retraction(l::CombinationPiecewise) = SparseArrays.spzeros(_num_inputs(l), _num_lifted(l))

# The vertices `s_min`, `b₁`, …, `bₙ`, `s_max` of the lifted segment of `s`, for the range
# `[s_min, s_max]` of `s` on the support; the breakpoints must be inside it.
function _vertices(l::CombinationPiecewise, s_min::Float64, s_max::Float64)
    if !(s_min < l.breakpoints[1] && l.breakpoints[end] < s_max)
        throw(
            ArgumentError(
                "The breakpoints $(l.breakpoints) must lie strictly inside the range " *
                "[$s_min, $s_max] of the combination on the support of its uncertainties.",
            ),
        )
    end
    return [s_min; l.breakpoints; s_max]
end

# The box of the pieces, `0 ≤ pₖ ≤ Δₖ`.
function _lifted_bounds(l::CombinationPiecewise, s_min::Float64, s_max::Float64)
    Δ = diff(_vertices(l, s_min, s_max))
    return zeros(length(Δ) - 1), Δ[2:end]
end

# The rows `W_η η + W_p p ≥ h` of the outer approximation of the hull: the rows of `AxisPiecewise`
# for the pieces `q₁ = s - Σₖ pₖ`, `qₖ = pₖ` of `s`, that is `Δₖ₊₁ (qₖ - [k = 1] s_min) ≥ Δₖ qₖ₊₁` for
# `k = 1, …, n`, and the bound `q₁ ≤ b₁`. With the lower bound of the last piece, they imply the
# rest of the box of the pieces and `q₁ ≥ s_min`.
function _hull_rows(l::CombinationPiecewise, s_min::Float64, s_max::Float64)
    v = _vertices(l, s_min, s_max)
    Δ = diff(v)
    n, a, c = length(l.breakpoints), l.coefficients, l.constant
    W_η = zeros(n + 1, length(a))
    W_p = zeros(n + 1, n)
    h = zeros(n + 1)
    # Row 1: Δ₂ (c + a ⋅ η - Σ p - s_min) - Δ₁ p₂ ≥ 0.
    W_η[1, :] .= Δ[2] .* a
    W_p[1, :] .= -Δ[2]
    W_p[1, 1] -= Δ[1]
    h[1] = Δ[2] * (s_min - c)
    # Rows k = 2, …, n: Δₖ₊₁ pₖ - Δₖ pₖ₊₁ ≥ 0, with `pₖ` in column `k - 1`.
    for k in 2:n
        W_p[k, k-1] = Δ[k+1]
        W_p[k, k] = -Δ[k]
    end
    # Row n + 1: b₁ - (c + a ⋅ η - Σ p) ≥ 0.
    W_η[n+1, :] .= -a
    W_p[n+1, :] .= 1.0
    h[n+1] = c - v[2]
    return W_η, W_p, h
end

function Base.show(io::IO, l::CombinationPiecewise)
    print(io, "CombinationPiecewise(", l.coefficients, ", ", l.breakpoints)
    iszero(l.constant) || print(io, "; constant = ", l.constant)
    print(io, ")")
    return
end
