# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    AbstractLifting

A lifting of some uncertainties `η` into more coordinates `η̃ = L(η)` (Georghiou, Wiesemann and
Kuhn, 2015): a decision rule that is linear in `η̃` is nonlinear in `η`.

A lifting `l` implements:

 * [`lift`](@ref)`(l, η)`, the nonlinear map `L`, applied to the outcomes;
 * [`retraction`](@ref)`(l)`, the sparse matrix `R` of the linear retraction, with `R L(η) = η`;
   the data of the model are lifted with it (`B ← BR`, `C ← CR`, `W ← WR`).

A lifted coordinate is in the largest stage of the uncertainties that it depends on.

See [`AxisPiecewise`](@ref).
"""
abstract type AbstractLifting end

"""
    lift(l::AbstractLifting, η::AbstractVector{<:Real}) -> Vector{Float64}
    lift(l::AbstractLifting, η::Real) -> Vector{Float64}

Return the lifted coordinates `L(η)` of the outcome `η` of the uncertainties of `l`; a lifting of
one uncertainty also takes a scalar.
"""
function lift end

lift(l::AbstractLifting, η::Real) = lift(l, [η])

"""
    retraction(l::AbstractLifting) -> SparseMatrixCSC{Float64,Int}

Return the matrix `R` of the linear retraction of `l`, one row per uncertainty and one column per
lifted coordinate, such that `R * lift(l, η) == η`.
"""
function retraction end

"""
    AxisPiecewise(breakpoints::AbstractVector{<:Real})

The piecewise-linear lifting of one uncertainty `η` along its axis, with the interior
`breakpoints` `b₁ < … < bₙ` (sorted on construction), into `n + 1` pieces:
```math
\\tilde\\eta_1 = \\min(\\eta, b_1), \\quad
\\tilde\\eta_k = \\min(\\max(\\eta - b_{k-1}, 0), b_k - b_{k-1}), \\quad
\\tilde\\eta_{n+1} = \\max(\\eta - b_n, 0),
```
so that `η = η̃₁ + … + η̃ₙ₊₁` (the retraction is a row of ones). The first piece includes the
minimum of the support, as in the previous versions of the package. A rule that is linear in `η̃`
is piecewise affine in `η`, and its coefficient on `η̃ₖ` is its slope on the `k`-th piece.

With the support `[η_min, η_max]` and the lengths `Δ₁ = b₁ - η_min`, `Δₖ = bₖ - bₖ₋₁` and
`Δₙ₊₁ = η_max - bₙ`, the convex hull of the lifted support is
```math
0 \\le \\frac{\\tilde\\eta_{n+1}}{\\Delta_{n+1}} \\le \\dots \\le \\frac{\\tilde\\eta_2}{\\Delta_2}
\\le \\frac{\\tilde\\eta_1 - \\eta_{\\min}}{\\Delta_1} \\le 1.
```

Set it on a model with `set_attribute(d, BreakPoints(), breakpoints)`.

## Example

```julia
l = AxisPiecewise([90, 100, 110])
lift(l, 104.0)       # [90.0, 10.0, 4.0, 0.0]
retraction(l) * ans  # [104.0]
```
"""
struct AxisPiecewise <: AbstractLifting
    breakpoints::Vector{Float64}

    function AxisPiecewise(breakpoints::AbstractVector{<:Real})
        points = sort!(Vector{Float64}(breakpoints))
        if isempty(points)
            throw(ArgumentError("AxisPiecewise needs at least one breakpoint."))
        elseif !all(isfinite, points)
            throw(ArgumentError("The breakpoints must be finite; got $breakpoints."))
        elseif !allunique(points)
            throw(ArgumentError("The breakpoints must be distinct; got $breakpoints."))
        end
        return new(points)
    end
end

Base.:(==)(a::AxisPiecewise, b::AxisPiecewise) = a.breakpoints == b.breakpoints

Base.hash(l::AxisPiecewise, h::UInt) = hash(l.breakpoints, hash(AxisPiecewise, h))

_num_inputs(::AxisPiecewise) = 1

_num_lifted(l::AxisPiecewise) = length(l.breakpoints) + 1

function lift(l::AxisPiecewise, η::AbstractVector{<:Real})
    if length(η) != 1
        throw(DimensionMismatch("AxisPiecewise lifts one uncertainty; got $(length(η)) values."))
    end
    x, b = Float64(η[1]), l.breakpoints
    η̃ = zeros(length(b) + 1)
    η̃[1] = min(x, b[1])
    for k in 2:length(b)
        η̃[k] = clamp(x - b[k-1], 0.0, b[k] - b[k-1])
    end
    η̃[end] = max(x - b[end], 0.0)
    return η̃
end

retraction(l::AxisPiecewise) = SparseArrays.sparse(ones(1, _num_lifted(l)))

# The vertices of the lifted support, `η_min`, `b₁`, …, `bₙ`, `η_max`, for the support
# `[lower[1], upper[1]]`; the breakpoints must be inside it.
function _vertices(l::AxisPiecewise, lower::Vector{Float64}, upper::Vector{Float64})
    lo, hi = lower[1], upper[1]
    if !(lo < l.breakpoints[1] && l.breakpoints[end] < hi)
        throw(
            ArgumentError(
                "The breakpoints $(l.breakpoints) must lie strictly inside the support " *
                "[$lo, $hi] of the uncertainty.",
            ),
        )
    end
    return [lo; l.breakpoints; hi]
end

# The box of the lifted coordinates, for the support box `[lower, upper]` of the uncertainties.
function _lifted_bounds(l::AxisPiecewise, lower::Vector{Float64}, upper::Vector{Float64})
    v = _vertices(l, lower, upper)
    Δ = diff(v)
    return [v[1]; zeros(length(Δ) - 1)], [v[2]; Δ[2:end]]
end

# The rows `W η̃ ≥ h` of the convex hull of the lifted support, beyond its box. With `Δ` as in the
# docstring of `AxisPiecewise`, row `k` is `(η̃ₖ - [k = 1] η_min) / Δₖ ≥ η̃ₖ₊₁ / Δₖ₊₁`, scaled by
# `Δₖ Δₖ₊₁` as in the previous versions. The scaling is immaterial: with the row `e₁` kept, the dual
# LDR depends only on the support Ξ, not on the rows that describe it (Farkas; Proposition 5.4 of
# `docs/math`).
function _hull_rows(l::AxisPiecewise, lower::Vector{Float64}, upper::Vector{Float64})
    v = _vertices(l, lower, upper)
    Δ = diff(v)
    K = length(Δ)
    I, J, V = Int[], Int[], Float64[]
    for k in 1:K-1
        append!(I, (k, k))
        append!(J, (k, k + 1))
        append!(V, (Δ[k+1], -Δ[k]))
    end
    h = zeros(K - 1)
    h[1] = Δ[2] * v[1]
    return SparseArrays.sparse(I, J, V, K - 1, K), h
end

function Base.show(io::IO, l::AxisPiecewise)
    print(io, "AxisPiecewise(", l.breakpoints, ")")
    return
end

"""
    LiftingBlock(lifting::AbstractLifting, inputs::Vector{Int}, columns::Vector{Int})

One lifting of a [`LiftingMap`](@ref): the columns `columns` of the lifted `ξ̃` are
`lift(lifting, ξ[inputs])`.

## Fields

 * `lifting::AbstractLifting`
 * `inputs::Vector{Int}`: the columns of the original `ξ` that it lifts.
 * `columns::Vector{Int}`: the columns of `ξ̃` of the lifted coordinates, in the order of
   `lift(lifting, ξ[inputs])`.
"""
struct LiftingBlock
    lifting::AbstractLifting
    inputs::Vector{Int}
    columns::Vector{Int}
end

"""
    LiftingMap

The map between the columns `ξ = (1, η)` of a [`CanonicalForm`](@ref) and the columns
`ξ̃ = (1, η̃)` of its lifted form, built by [`lift`](@ref)`(form, liftings)` and kept in the
field `lifting` of the lifted form.

Apply the lifting `L` to an outcome of the original uncertainties with `lift(map, ξ)`, and the
retraction with `retraction(map) * ξ̃`. A decision rule `x(ξ̃) = X̃ ξ̃` of the lifted form is the
rule `x(ξ) = X̃ lift(map, ξ)` on the original uncertainties.

## Fields

 * `uncertainties::Vector{MOI.VariableIndex}`: the original uncertainties, in the columns
   `2:1+m` of `ξ`.
 * `uncertainty_names::Vector{String}`
 * `uncertainty_stages::Vector{Int}`
 * `kept::Vector{Int}`: the column of `ξ̃` of each column of `ξ` that is not lifted (the constant
   column 1 is column 1), and 0 for a lifted column.
 * `blocks::Vector{LiftingBlock}`: the liftings, each with its columns in `ξ` and in `ξ̃`.
 * `R::SparseMatrixCSC{Float64,Int}`: the retraction, `(1 + m) × (1 + m̃)`, with `ξ = R ξ̃`,
   assembled from the identity on the kept columns and the [`retraction`](@ref) of each block.
"""
struct LiftingMap
    uncertainties::Vector{MOI.VariableIndex}
    uncertainty_names::Vector{String}
    uncertainty_stages::Vector{Int}
    kept::Vector{Int}
    blocks::Vector{LiftingBlock}
    R::SparseArrays.SparseMatrixCSC{Float64,Int}
end

retraction(map::LiftingMap) = map.R

"""
    lift(map::LiftingMap, ξ::AbstractVector{<:Real}) -> Vector{Float64}
    lift(map::LiftingMap, ξ::AbstractMatrix{<:Real}) -> Matrix{Float64}

Return the lifted outcome `ξ̃ = L(ξ)` of the outcome `ξ = (1, η)` of the original uncertainties,
in the columns of the lifted form; with a matrix, each column is an outcome.
"""
function lift(map::LiftingMap, ξ::AbstractVector{<:Real})
    m, m̃ = size(map.R)
    if length(ξ) != m
        throw(DimensionMismatch("The outcome has $(length(ξ)) entries; expected $m, ξ = (1, η)."))
    end
    ξ̃ = zeros(m̃)
    for (j, k) in enumerate(map.kept)
        k > 0 && (ξ̃[k] = ξ[j])
    end
    for block in map.blocks
        ξ̃[block.columns] = lift(block.lifting, view(ξ, block.inputs))
    end
    return ξ̃
end

function lift(map::LiftingMap, ξ::AbstractMatrix{<:Real})
    ξ̃ = zeros(size(map.R, 2), size(ξ, 2))
    for (s, outcome) in enumerate(eachcol(ξ))
        ξ̃[:, s] = lift(map, outcome)
    end
    return ξ̃
end
