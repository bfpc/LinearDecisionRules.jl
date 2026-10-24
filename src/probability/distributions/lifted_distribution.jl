# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    LiftedDistribution(original::Distributions.UnivariateDistribution, lifting::AbstractLifting)

The law of the lifted coordinates `lift(lifting, η)` of an uncertainty `η ~ original`, whose
support must be bounded. It replaces `original` in a lifted [`CanonicalForm`](@ref).

Its mean and covariance are computed once, on construction:

 * for a discrete law, exactly, by summing over the support of `original` (which must be finite);
 * for a continuous law and an [`AxisPiecewise`](@ref) lifting, from the probability and the
   first two moments of `η` on each piece: exact formulas for `Uniform`, quadrature (QuadGK) of
   the density otherwise.

`minimum` and `maximum` are the box of the lifted support, and `rand` lifts a draw of `original`,
so a sample of the lifted law uses one draw of `original` per outcome.

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

_symmetric(A::Matrix{Float64}) = (A + A') ./ 2

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
