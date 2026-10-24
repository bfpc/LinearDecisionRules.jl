# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    CovarianceBlock(columns::Vector{Int}, covariance::Matrix{Float64})

The covariance matrix of the columns `columns` of ξ: `covariance[a, b] = Cov(ξ[columns[a]],
ξ[columns[b]])`.
"""
struct CovarianceBlock
    columns::Vector{Int}
    covariance::Matrix{Float64}
end

"""
    Moments(mean::Vector{Float64}, blocks::Vector{CovarianceBlock})

The first two moments of ξ = (1, η), with their structure:
`E[ξ ξᵀ] = mean meanᵀ + blockdiag(blocks)`.

## Fields

 * `mean::Vector{Float64}`: `E[ξ]`, with `mean[1] == 1`.
 * `blocks::Vector{CovarianceBlock}`: one per uncertainty declaration; columns in different blocks
   are independent, and the constant column 1 is in no block.

See [`second_moment`](@ref) for the dense matrix.
"""
struct Moments
    mean::Vector{Float64}
    blocks::Vector{CovarianceBlock}
end

_law_mean(d::Distributions.UnivariateDistribution) = [Float64(Distributions.mean(d))]
_law_mean(d::Distributions.MultivariateDistribution) = Vector{Float64}(Distributions.mean(d))

_law_covariance(d::Distributions.UnivariateDistribution) = fill(Float64(Distributions.var(d)), 1, 1)
_law_covariance(d::Distributions.MultivariateDistribution) = Matrix{Float64}(Distributions.cov(d))

"""
    moments(form::CanonicalForm) -> Moments

Return the [`Moments`](@ref) of ξ = (1, η) for the laws of `form`.

Separate declarations are independent; within one declaration, the covariance is
`Distributions.cov` (or `Distributions.var` for a univariate law).
"""
function moments(form::CanonicalForm)
    μ = zeros(1 + length(form.uncertainties))
    μ[1] = 1.0
    blocks = CovarianceBlock[]
    for law in form.laws
        mean, cov = _law_mean(law.distribution), _law_covariance(law.distribution)
        if !(all(isfinite, mean) && all(isfinite, cov))
            names = join((form.uncertainty_names[j-1] for j in law.columns), ", ")
            error("The distribution of the uncertainty $names has no finite mean or covariance.")
        end
        μ[law.columns] = mean
        push!(blocks, CovarianceBlock(copy(law.columns), cov))
    end
    return Moments(μ, blocks)
end

"""
    second_moment(mom::Moments) -> Matrix{Float64}

Return the dense second-moment matrix `M = E[ξ ξᵀ]` of `mom`.
"""
function second_moment(mom::Moments)
    M = mom.mean * mom.mean'
    for block in mom.blocks
        M[block.columns, block.columns] .+= block.covariance
    end
    return M
end
