# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    LiftedDistribution(original::Distributions.UnivariateDistribution, lifting::AbstractLifting)

The law of the lifted coordinates `lift(lifting, η)` of an uncertainty `η ~ original`, whose
support must be bounded. `optimize!` uses it in the place of `original` for an uncertainty with
[`BreakPoints`](@ref).

Its mean and covariance are computed once, on construction:

 * for a discrete law, exactly, by summing over the support of `original` (which must be finite);
 * for a continuous law and an [`AxisPiecewise`](@ref) lifting, from the probability and the
   first two moments of `η` on each piece: exact formulas for `Uniform`, quadrature (QuadGK) of
   the density otherwise.

`minimum` and `maximum` are the box of the lifted support, and `rand` lifts a draw of `original`,
so a sample of the lifted law uses one draw of `original` per outcome.

It is declared a `ContinuousMultivariateDistribution` even when `original` is discrete, and this
tag is nominal: the lifted law of a discrete `η` is discrete, with one atom `lift(lifting, x)` per
atom `x` of `η` (its moments above are exact sums over them), but its draws are vectors of
`Float64` either way, it defines no density or mass function, and the package reads the atoms from
`original`, never from this tag.

## Fields

 * `original`: the law of `η`.
 * `lifting`: the lifting.
 * `lower::Vector{Float64}`, `upper::Vector{Float64}`: the box of the lifted support.
 * `mean::Vector{Float64}`, `covariance::Matrix{Float64}`: the moments of `lift(lifting, η)`.

## Example

```julia
d = LiftedDistribution(Uniform(80, 120), AxisPiecewise([100]))
mean(d)  # [95.0, 5.0]
```
"""
struct LiftedDistribution{D<:Distributions.UnivariateDistribution,L<:AbstractLifting} <:
       Distributions.ContinuousMultivariateDistribution
    original::D
    lifting::L
    lower::Vector{Float64}
    upper::Vector{Float64}
    mean::Vector{Float64}
    covariance::Matrix{Float64}
end

function LiftedDistribution(
    original::Distributions.UnivariateDistribution,
    lifting::AbstractLifting,
)
    if _num_inputs(lifting) != 1
        throw(ArgumentError("The lifting $lifting of a univariate law must have one input."))
    end
    lo, hi = _support_bounds(original)
    if !(isfinite(lo[1]) && isfinite(hi[1]))
        throw(ArgumentError("The lifted law $original must have a bounded support."))
    end
    lower, upper = _lifted_bounds(lifting, lo, hi)
    mean, covariance = _lifted_moments(original, lifting, lo, hi)
    return LiftedDistribution(original, lifting, lower, upper, mean, covariance)
end

Base.length(d::LiftedDistribution) = length(d.mean)

# Lifted coordinates are real even for a discrete law, such as the pieces of an integer demand
# at a fractional breakpoint.
Base.eltype(::Type{<:LiftedDistribution}) = Float64

Distributions.params(d::LiftedDistribution) = (d.original, d.lifting)

Base.minimum(d::LiftedDistribution) = copy(d.lower)

Base.maximum(d::LiftedDistribution) = copy(d.upper)

# The lifted support of an `AxisPiecewise` lifting is a simplex whose facets are its hull rows and
# the bounds `η̃₁ ≤ upper[1]` and `η̃ₙ₊₁ ≥ lower[n + 1]` (`lem:hull` of `docs/math`): without
# `lifted_box`, its other bounds, which are implied, are not rows of Ξ.
function _box_rows(d::LiftedDistribution{<:Any,AxisPiecewise}, lifted_box::Bool)
    lifted_box && return _support_box(d)
    lower, upper = fill(-Inf, length(d.lower)), fill(Inf, length(d.upper))
    lower[end], upper[1] = d.lower[end], d.upper[1]
    return lower, upper
end

Distributions.mean(d::LiftedDistribution) = copy(d.mean)

Distributions.cov(d::LiftedDistribution) = copy(d.covariance)

Distributions.var(d::LiftedDistribution) = LinearAlgebra.diag(d.covariance)

function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::LiftedDistribution,
    x::AbstractVector{<:Real},
)
    x .= lift(d.lifting, rand(rng, d.original))
    return x
end

function Base.show(io::IO, d::LiftedDistribution)
    print(io, "LiftedDistribution(", d.original, ", ", d.lifting, ")")
    return
end

# The mean and the covariance of `lift(l, η)` for `η ~ d` with support `[lo, hi]`, exactly, by
# summing over the support of a discrete law.
function _lifted_moments(
    d::Distributions.DiscreteUnivariateDistribution,
    l::AbstractLifting,
    ::Vector{Float64},
    ::Vector{Float64},
)
    n = _num_lifted(l)
    mean, second = zeros(n), zeros(n, n)
    for x in Distributions.support(d)
        p = Distributions.pdf(d, x)
        iszero(p) && continue
        η̃ = lift(l, x)
        mean .+= p .* η̃
        second .+= p .* (η̃ * η̃')
    end
    return mean, _symmetric(second - mean * mean')
end
