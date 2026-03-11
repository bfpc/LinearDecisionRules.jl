"""
    ConfidenceMvNormal(μ, Σ, α)

A multivariate normal distribution with mean `μ` and covariance `Σ`, truncated
to the ellipsoid that contains probability mass `α`.

Formally, this is the distribution of `X | X ∈ E_{μ,Σ,ρ}`, where
`E_{μ,Σ,ρ} = {x : (x-μ)' Σ⁻¹ (x-μ) ≤ ρ²}` and `ρ` is chosen so that
`P(X ∈ E_{μ,Σ,ρ}) = α` for `X ~ MvNormal(μ, Σ)`.

## Arguments

- `μ`: mean vector of the underlying normal distribution
- `Σ`: positive-definite covariance matrix of the underlying normal
- `α`: confidence level (`0 < α < 1`); fraction of probability mass retained

## Example

```julia
μ = [0.0, 0.0]
Σ = [1.0 0.3; 0.3 1.0]
dist = LinearDecisionRules.ConfidenceMvNormal(μ, Σ, 0.95)

@variable(ldr, demand[1:2] in LinearDecisionRules.Uncertainty(
    distribution = dist,
))
```
"""
struct ConfidenceMvNormal <: Distributions.ContinuousMultivariateDistribution
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    α::Float64
    # derived
    d::Int
    L::LinearAlgebra.LowerTriangular{Float64,Matrix{Float64}}
    ρ²::Float64
    ρ::Float64
    _cov::Matrix{Float64}
    _lb::Vector{Float64}
    _ub::Vector{Float64}

    function ConfidenceMvNormal(
        μ::AbstractVector{<:Real},
        Σ::AbstractMatrix{<:Real},
        α::Real,
    )
        d = length(μ)
        @assert size(Σ) == (d, d) "Σ must be $d × $d"
        @assert 0 < α < 1 "α must satisfy 0 < α < 1, got $α"
        @assert LinearAlgebra.issymmetric(Σ) "Σ must be symmetric"

        L = LinearAlgebra.cholesky(Σ).L

        # ρ² such that P(||z||² ≤ ρ²) = α for z ~ N(0, Iₐ)
        # This is the α-quantile of χ²(d)
        ρ² = Distributions.quantile(Distributions.Chisq(d), α)
        ρ = sqrt(ρ²)

        # Covariance of the truncated distribution:
        # C_{d,α} = (1/α) · P(d/2+1, ρ²/2) · Σ
        #          = cdf(Chisq(d+2), ρ²) / α  · Σ
        scaling = Distributions.cdf(Distributions.Chisq(d + 2), ρ²) / α
        _cov = scaling * Matrix{Float64}(Σ)

        # Component-wise bounds from the ellipsoid projection:
        # x_k ∈ [μ_k ± ρ·√Σ_{kk}]
        half_widths = ρ * sqrt.(LinearAlgebra.diag(Σ))
        _lb = Vector{Float64}(μ) - half_widths
        _ub = Vector{Float64}(μ) + half_widths

        return new(
            Vector{Float64}(μ),
            Matrix{Float64}(Σ),
            Float64(α),
            d,
            L,
            ρ²,
            ρ,
            _cov,
            _lb,
            _ub,
        )
    end
end

Distributions.params(d::ConfidenceMvNormal) = (d.μ, d.Σ, d.α)

Distributions.length(d::ConfidenceMvNormal) = d.d

Distributions.mean(d::ConfidenceMvNormal) = copy(d.μ)

Distributions.cov(d::ConfidenceMvNormal) = copy(d._cov)

Distributions.var(d::ConfidenceMvNormal) = LinearAlgebra.diag(d._cov)

Distributions.minimum(d::ConfidenceMvNormal) = copy(d._lb)

Distributions.maximum(d::ConfidenceMvNormal) = copy(d._ub)

function Distributions.insupport(d::ConfidenceMvNormal, x::AbstractVector)
    length(x) == d.d || return false
    z = d.L \ (x - d.μ)
    return LinearAlgebra.dot(z, z) <= d.ρ² + sqrt(eps(d.ρ²))
end

function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::ConfidenceMvNormal,
    x::AbstractVector,
)
    # Rejection sampling in the standardized space:
    # draw z ~ N(0, Iₐ) until ||z||² ≤ ρ², then x = μ + L·z.
    # Expected number of draws = 1/α.
    z = similar(x)
    while true
        Random.randn!(rng, z)
        if LinearAlgebra.dot(z, z) <= d.ρ²
            LinearAlgebra.mul!(x, d.L, z)
            x .+= d.μ
            return x
        end
    end
end
