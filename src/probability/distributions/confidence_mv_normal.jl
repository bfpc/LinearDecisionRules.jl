# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    ConfidenceMvNormal(
        μ::AbstractVector{<:Real},
        Σ::AbstractMatrix{<:Real},
        α::Real;
        support::Symbol = :auto,
    )

The normal law `N(μ, Σ)` conditioned on its `α`-confidence ellipsoid
`E = {η : (η - μ)ᵀ Σ⁻¹ (η - μ) ≤ ρ²}`, where `ρ² = quantile(Chisq(d), α)` in dimension `d`, so that
`P(η ∈ E) = α` for `η ~ N(μ, Σ)`.

Its mean is `μ` and its covariance `cdf(Chisq(d + 2), ρ²) / α ⋅ Σ`: with `Σ = L Lᵀ` and
`z = L⁻¹ (η - μ) ~ N(0, I)`, `‖z‖²` is `χ²(d)`, the direction of `z` is uniform and independent
of `‖z‖`, and `x f_d(x) = d f_{d+2}(x)` for the densities `f_k` of `χ²(k)`, so
`E[z zᵀ | ‖z‖² ≤ ρ²] = E[‖z‖² | ‖z‖² ≤ ρ²] / d ⋅ I = cdf(Chisq(d + 2), ρ²) / α ⋅ I`.
Its components can be in different stages: the conditional expectation of some given the others
is affine, as for the normal law, so the bound of [`DualLDR`](@ref) is certified.
`minimum` and `maximum` are the bounding box of `E`, `μ ± ρ √diag(Σ)`.

Draws are by rejection in the standardized space: `z ~ N(0, I)` until `‖z‖ ≤ ρ`, then
`η = μ + L z`. A candidate is accepted if and only if `‖z‖² ≤ ρ² = quantile(Chisq(d), α)`, and
`‖z‖²` follows `χ²(d)`, so the acceptance rate is exactly `α`, in any dimension `d`: `1 / α`
attempts per draw on average (about 1.05 for `α = 0.95`), so only a small `α` is costly. This is
cheaper, for the usual levels, than the exact radial method, which needs a quantile of `χ²(d)`
per draw.

`support` chooses how [`support_constraints`](@ref) describes `E` beyond its bounding box, so that
the formulations see it:

 * `:ellipsoid`: `E` itself, as one second-order cone row `(ρ, L⁻¹ (η - μ))`. The inner problems
   of [`PrimalLDR`](@ref) and [`DualLDR`](@ref) are second-order cone programs.
 * `:boxes`: the box along the principal axes of `Σ`, a polyhedron that contains `E`: the inner
   problems stay linear, the primal rule stays feasible on `E` and the dual bound stays valid, but
   both are looser than with `:ellipsoid`.
 * `:auto` (the default): `:ellipsoid` if the inner optimizer supports second-order cone
   constraints and the model has no integer decisions, and `:boxes` otherwise (a mixed-integer
   second-order cone program is much harder than a mixed-integer linear one). `optimize!` decides,
   and asks the inner optimizer only when some law has `support = :auto`; outside of `optimize!`,
   [`canonicalize`](@ref) takes the choice of the optimizer as its keyword `conic_support`, and
   `support_constraints` of a law with `support = :auto` is the ellipsoid.

## Fields

 * `μ::Vector{Float64}`, `Σ::Matrix{Float64}`, `α::Float64`, `support::Symbol`: as given.
 * `L::LowerTriangular{Float64,Matrix{Float64}}`: the Cholesky factor, `Σ = L Lᵀ`.
 * `ρ::Float64`: the radius of `E` in the standardized space, `ρ = √quantile(Chisq(d), α)`.
 * `scaling::Float64`: the factor `cdf(Chisq(d + 2), ρ²) / α` of the covariance.

## Example

```julia
d = ConfidenceMvNormal([100.0, 80.0], [100.0 20.0; 20.0 64.0], 0.95)
@variable(model, demand[1:2] in Uncertainty(d))
```
"""
struct ConfidenceMvNormal <: Distributions.ContinuousMultivariateDistribution
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    α::Float64
    support::Symbol
    # `Σ = L Lᵀ`, the radius `ρ` of the ellipsoid in the standardized space, and the factor of the
    # covariance.
    L::LinearAlgebra.LowerTriangular{Float64,Matrix{Float64}}
    ρ::Float64
    scaling::Float64

    function ConfidenceMvNormal(
        μ::AbstractVector{<:Real},
        Σ::AbstractMatrix{<:Real},
        α::Real;
        support::Symbol = :auto,
    )
        d = length(μ)
        if d == 0 || size(Σ) != (d, d)
            throw(ArgumentError("Σ must be a $d × $d matrix, for a mean μ of length $d ≥ 1."))
        elseif !(0 < α < 1)
            throw(ArgumentError("The confidence level α must be in (0, 1); got $α."))
        elseif !LinearAlgebra.issymmetric(Σ)
            throw(ArgumentError("The covariance matrix Σ must be symmetric."))
        elseif !(support in (:auto, :ellipsoid, :boxes))
            throw(ArgumentError("`support` is `:auto`, `:ellipsoid` or `:boxes`; got `:$support`."))
        end
        factor = LinearAlgebra.cholesky(Matrix{Float64}(Σ); check = false)
        if !LinearAlgebra.issuccess(factor)
            throw(ArgumentError("The covariance matrix Σ must be positive definite."))
        end
        ρ² = Distributions.quantile(Distributions.Chisq(d), α)
        scaling = Distributions.cdf(Distributions.Chisq(d + 2), ρ²) / α
        return new(Vector{Float64}(μ), Matrix{Float64}(Σ), α, support, factor.L, sqrt(ρ²), scaling)
    end
end

Base.length(d::ConfidenceMvNormal) = length(d.μ)

Base.eltype(::Type{ConfidenceMvNormal}) = Float64

Distributions.params(d::ConfidenceMvNormal) = (d.μ, d.Σ, d.α)

Distributions.mean(d::ConfidenceMvNormal) = copy(d.μ)

Distributions.cov(d::ConfidenceMvNormal) = d.scaling * d.Σ

Distributions.var(d::ConfidenceMvNormal) = d.scaling * LinearAlgebra.diag(d.Σ)

Base.minimum(d::ConfidenceMvNormal) = d.μ .- d.ρ .* sqrt.(LinearAlgebra.diag(d.Σ))

Base.maximum(d::ConfidenceMvNormal) = d.μ .+ d.ρ .* sqrt.(LinearAlgebra.diag(d.Σ))

# The representation of the support is part of the law, since the formulations depend on it.
function Base.:(==)(a::ConfidenceMvNormal, b::ConfidenceMvNormal)
    return a.μ == b.μ && a.Σ == b.Σ && a.α == b.α && a.support == b.support
end

function Base.hash(d::ConfidenceMvNormal, h::UInt)
    return hash(d.μ, hash(d.Σ, hash(d.α, hash(d.support, hash(:ConfidenceMvNormal, h)))))
end

function Base.show(io::IO, d::ConfidenceMvNormal)
    print(io, "ConfidenceMvNormal(μ = ", d.μ, ", Σ = ", d.Σ, ", α = ", d.α)
    print(io, "; support = :", d.support, ")")
    return
end

# Whether `x` is in the ellipsoid, up to a relative tolerance of `1e-9` for the points of its
# boundary.
function Distributions.insupport(d::ConfidenceMvNormal, x::AbstractVector{<:Real})
    length(x) == length(d) || return false
    z = d.L \ (x .- d.μ)
    return LinearAlgebra.dot(z, z) <= d.ρ^2 * (1 + 1e-9)
end

function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::ConfidenceMvNormal,
    x::AbstractVector{<:Real},
)
    z = Vector{Float64}(undef, length(d))
    while true
        Random.randn!(rng, z)
        if LinearAlgebra.dot(z, z) <= d.ρ^2
            LinearAlgebra.mul!(x, d.L, z)
            x .+= d.μ
            return x
        end
    end
end

function support_constraints(d::ConfidenceMvNormal)
    n = length(d)
    if d.support == :boxes
        # The principal axes `Σ = V diag(λ) Vᵀ`: `|vₖ ⋅ (η - μ)| ≤ ρ √λₖ` on the ellipsoid. With a
        # diagonal Σ, it is the bounding box, which the box of the law already gives.
        LinearAlgebra.isdiag(d.Σ) && return _no_support_constraints(n)
        λ, V = LinearAlgebra.eigen(LinearAlgebra.Symmetric(d.Σ))
        r = d.ρ .* sqrt.(max.(λ, 0.0))
        Vμ = V' * d.μ
        W, h = Matrix([V'; -V']), [Vμ .- r; .-Vμ .- r]
        return (W = W, h = h, cones = _SupportCone[MOI.Nonnegatives(2n)])
    end
    # `(ρ, L⁻¹ (η - μ))` in the second-order cone.
    Linv = Matrix(inv(d.L))
    return (
        W = [zeros(1, n); Linv],
        h = [-d.ρ; Linv * d.μ],
        cones = _SupportCone[MOI.SecondOrderCone(n + 1)],
    )
end

# Given some components `η_h`, the others `η_f` follow the conditional normal law, centered at the
# affine `μ_f + Σ_fh Σ_hh⁻¹ (η_h - μ_h)`, restricted to an ellipsoid with the same center (a Schur
# complement): a symmetric law about that center, whose mean is the center (`docs/math`,
# Proposition "Moments of ConfidenceMvNormal").
_linear_conditional_expectations(::ConfidenceMvNormal, ::AbstractVector) = true

_has_auto_representation(d::ConfidenceMvNormal) = d.support == :auto

function _resolve_representation(d::ConfidenceMvNormal, conic::Bool)
    d.support == :auto || return d
    return ConfidenceMvNormal(d.μ, d.Σ, d.α; support = conic ? :ellipsoid : :boxes)
end
