# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# `AxisPiecewise`: the map, the box and the convex hull of its lifted support, and the moments
# of its pieces under a continuous law.

"""
    AxisPiecewise(breakpoints::AbstractVector{<:Real})

The piecewise-linear lifting of one uncertainty `η` along its axis, with the interior
`breakpoints` `b₁ < … < bₙ` (sorted on construction), into `n + 1` pieces:
```math
\\tilde\\eta_1 = \\min(\\eta, b_1), \\quad
\\tilde\\eta_k = \\min(\\max(\\eta - b_{k-1}, 0), b_k - b_{k-1}), \\quad
\\tilde\\eta_{n+1} = \\max(\\eta - b_n, 0),
```
so that `η = η̃₁ + … + η̃ₙ₊₁` (the retraction is a row of ones): the first piece includes the
minimum of the support. A rule that is linear in `η̃` is piecewise affine in `η`, and its
coefficient on `η̃ₖ` is its slope on the `k`-th piece.

With the support `[η_min, η_max]` and the lengths `Δ₁ = b₁ - η_min`, `Δₖ = bₖ - bₖ₋₁` and
`Δₙ₊₁ = η_max - bₙ`, the convex hull of the lifted support is
```math
0 \\le \\frac{\\tilde\\eta_{n+1}}{\\Delta_{n+1}} \\le \\dots \\le \\frac{\\tilde\\eta_2}{\\Delta_2}
\\le \\frac{\\tilde\\eta_1 - \\eta_{\\min}}{\\Delta_1} \\le 1.
```

Set it on a model with `set_attribute(d, BreakPoints(), breakpoints)`.

## Fields

 * `breakpoints::Vector{Float64}`: the interior breakpoints, sorted.

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
# `Δₖ Δₖ₊₁`. The scaling is immaterial: with the row `e₁` kept, the dual
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

# The pieces of `AxisPiecewise` with `u₁ = η̃₁ - η_min` and `uₖ = η̃ₖ` otherwise: on piece `i`,
# `(vᵢ₋₁, vᵢ]`, `uₖ = Δₖ` for `k < i`, `uᵢ = η - vᵢ₋₁` and `uₖ = 0` for `k > i`. With `pᵢ`, `mᵢ` and
# `sᵢ` the probability and the moments `E[(η - vᵢ₋₁)^q 1{piece i}]`, `q = 1, 2`, and `tₖ` the
# probability of the pieces after `k`:
#   E[uₖ] = mₖ + Δₖ tₖ,   E[uₖ²] = sₖ + Δₖ² tₖ,   E[uₖ uₗ] = Δₖ E[uₗ] for k < l,
# since `uₗ > 0` implies `uₖ = Δₖ`.
function _lifted_moments(
    d::Distributions.ContinuousUnivariateDistribution,
    l::AxisPiecewise,
    lo::Vector{Float64},
    hi::Vector{Float64},
)
    v = _vertices(l, lo, hi)
    Δ = diff(v)
    K = length(Δ)
    p, m, s = zeros(K), zeros(K), zeros(K)
    for i in 1:K
        p[i], m[i], s[i] = _piece_moments(d, v[i], v[i+1])
    end
    tail = [sum(p[k+1:end]; init = 0.0) for k in 1:K]
    Eu = m .+ Δ .* tail
    second = zeros(K, K)
    for k in 1:K
        second[k, k] = s[k] + Δ[k]^2 * tail[k]
        for j in k+1:K
            second[k, j] = second[j, k] = Δ[k] * Eu[j]
        end
    end
    mean = copy(Eu)
    mean[1] += v[1]
    return mean, _symmetric(second - Eu * Eu')
end

# `(P(a < η ≤ b), E[(η - a) 1{a < η ≤ b}], E[(η - a)² 1{a < η ≤ b}])` for `[a, b]` inside the
# support of `d`.
function _piece_moments(d::Distributions.Uniform, a::Float64, b::Float64)
    w, Δ = 1 / (d.b - d.a), b - a
    return Δ * w, Δ^2 * w / 2, Δ^3 * w / 3
end

function _piece_moments(d::Distributions.ContinuousUnivariateDistribution, a::Float64, b::Float64)
    p = Distributions.cdf(d, b) - Distributions.cdf(d, a)
    integrand(x) = Distributions.pdf(d, x) .* [x - a, (x - a)^2]
    ms, _ = QuadGK.quadgk(integrand, a, b; rtol = 1e-12)
    return p, ms[1], ms[2]
end
