# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    CovarianceBlock(
        columns::Vector{Int},
        covariance::Matrix{Float64},
        num_samples::Int = 0,
    )

The covariance matrix of the columns `columns` of ξ: `covariance[a, b] = Cov(ξ[columns[a]],
ξ[columns[b]])`.

`num_samples` is `0` when the moments of the block are exact, and otherwise the size of the sample
that estimates them (by rejection sampling, for a [`ConditionedDistribution`](@ref)).

## Fields

 * `columns::Vector{Int}`
 * `covariance::Matrix{Float64}`
 * `num_samples::Int`
"""
struct CovarianceBlock
    columns::Vector{Int}
    covariance::Matrix{Float64}
    num_samples::Int
end

function CovarianceBlock(columns::Vector{Int}, covariance::Matrix{Float64})
    return CovarianceBlock(columns, covariance, 0)
end

"""
    Moments(mean::Vector{Float64}, blocks::Vector{CovarianceBlock})

The first two moments of ξ = (1, η), with their structure:
`E[ξ ξᵀ] = mean meanᵀ + blockdiag(blocks)`.

## Fields

 * `mean::Vector{Float64}`: `E[ξ]`, with `mean[1] == 1`.
 * `blocks::Vector{CovarianceBlock}`: one per law of the form (an uncertainty declaration, or the
   declarations coupled by constraints tagged `Truncate()`); columns in different blocks are
   independent, and the constant column 1 is in no block.

See [`second_moment`](@ref) for the dense matrix.
"""
struct Moments
    mean::Vector{Float64}
    blocks::Vector{CovarianceBlock}
end

# Whether some moments are estimated by Monte Carlo, so that values computed with them are
# estimates too.
_is_estimated(mom::Moments) = any(block -> block.num_samples > 0, mom.blocks)

"""
    moments(
        form::CanonicalForm;
        seed::Integer = 1234,
        time_limit::Real = 10.0,
        max_samples::Integer = 10_000,
    ) -> Moments

Return the [`Moments`](@ref) of ξ = (1, η) for the laws of `form`.

Separate laws are independent; within one law, the covariance is `Distributions.cov` (or
`Distributions.var` for a univariate law).

The moments of a [`ConditionedDistribution`](@ref) (the law of declarations coupled by constraints
tagged `Truncate()`) are those of a sample drawn by rejection: up to `max_samples` draws in the
conditioning set, within `time_limit` seconds for each such law. The `k`-th conditioned law of
`form.laws` draws from `Random.Xoshiro(s[k])`, where `s = rand(Random.Xoshiro(seed), UInt64, K)`
for the `K` conditioned laws, so that laws do not share their draws. Fewer than `max_samples` draws
within the time limit give a warning, listing every such law; fewer than `min(max_samples, 100)`
are an error, which names the constraints. Their [`CovarianceBlock`](@ref) has the size of the
sample in `num_samples`.

The standard error of the estimated moments decreases as `1 / √max_samples`: with the default
`10_000` draws, the standard error of a mean is 1% of the standard deviation, and 3% with `1_000`.
"""
function moments(
    form::CanonicalForm;
    seed::Integer = 1234,
    time_limit::Real = _DEFAULT_SAMPLING_TIME_LIMIT,
    max_samples::Integer = 10_000,
)
    return _moments(form, _RejectionSampling(seed, time_limit, max_samples), nothing)
end

# With `budget` seconds, the conditioned laws to sample share them (see `_sampling_time_limit`);
# those found in `cache` cost nothing.
function _moments(
    form::CanonicalForm,
    s::_RejectionSampling,
    cache::Union{Nothing,_MomentsCache};
    budget::Union{Nothing,Float64} = nothing,
)
    μ = zeros(1 + length(form.uncertainties))
    μ[1] = 1.0
    blocks = CovarianceBlock[]
    distributions = [law.distribution for law in form.laws]
    conditioned = filter(d -> d isa ConditionedDistribution, distributions)
    seeds = rand(Random.Xoshiro(s.seed % UInt64), UInt64, length(conditioned))
    cache_keys = [(d, seed, s.max_samples) for (d, seed) in zip(conditioned, seeds)]
    pending = count(key -> cache === nothing || !haskey(cache.estimates, key), cache_keys)
    spent = 0.0
    k = 0
    small = String[]
    for law in form.laws
        names = join((form.uncertainty_names[j-1] for j in law.columns), ", ")
        d = law.distribution
        if d isa ConditionedDistribution
            k += 1
            key = cache_keys[k]
            estimate = cache === nothing ? nothing : get(cache.estimates, key, nothing)
            if estimate === nothing
                limit = _sampling_time_limit(s, budget, spent, pending)
                estimate = _estimate(d, Random.Xoshiro(key[2]), s.max_samples, limit)
                source = _sampling_time_limit_source(s, budget)
                _check_estimate(estimate, d, names, s.max_samples, limit, source)
                spent += estimate.seconds
                pending -= 1
                if cache !== nothing
                    cache.num_estimates += 1
                    if estimate.num_samples == s.max_samples
                        cache.estimates[key] = estimate
                    end
                end
            end
            if estimate.num_samples < s.max_samples
                push!(small, "$names ($(estimate.num_samples) draws)")
            end
            mean, cov, num_samples = estimate.mean, copy(estimate.covariance), estimate.num_samples
        else
            mean, cov, num_samples = _law_mean(d), _law_covariance(d), 0
        end
        if !(all(isfinite, mean) && all(isfinite, cov))
            error("The distribution of the uncertainty $names has no finite mean or covariance.")
        end
        μ[law.columns] = mean
        push!(blocks, CovarianceBlock(copy(law.columns), cov, num_samples))
    end
    if cache !== nothing
        used = Set(cache_keys)
        filter!(entry -> entry.first in used, cache.estimates)
    end
    if !isempty(small)
        @warn(
            "Rejection sampling kept fewer than $(s.max_samples) draws " *
            "(`RejectionSamplingMaxSamples()`) within its time limit " *
            "($(_sampling_time_limit_source(s, budget))) for the conditioned laws of " *
            "$(join(small, "; ")): the estimates of their moments are less accurate.",
        )
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

# `E[a ⋅ ξ  b ⋅ ξ]` with `E[ξ ξᵀ] = μ μᵀ + blockdiag(Σ)`.
function _product_moment(mom::Moments, a::Vector{Float64}, b::Vector{Float64})
    value = LinearAlgebra.dot(a, mom.mean) * LinearAlgebra.dot(b, mom.mean)
    for block in mom.blocks
        value += LinearAlgebra.dot(a[block.columns], block.covariance, b[block.columns])
    end
    return value
end

function _covariance_matrix(mom::Moments)
    k = length(mom.mean)
    I, J, V = Int[], Int[], Float64[]
    for block in mom.blocks, (b, j) in enumerate(block.columns), (a, i) in enumerate(block.columns)
        if !iszero(block.covariance[a, b])
            push!(I, i)
            push!(J, j)
            push!(V, block.covariance[a, b])
        end
    end
    return SparseArrays.sparse(I, J, V, k, k)
end

# The covariance blocks of `mom` that are singular, as `(block, rank)`: the coordinates of a block
# lie almost surely in an affine subspace, so that `M = E[ξ ξᵀ]` is not positive definite (a
# Schur complement on the constant column: `M ≻ 0` if and only if every block is). The test is on
# the correlation matrix, to not depend on the units of the uncertainties.
function _singular_blocks(mom::Moments; tolerance::Float64 = 1e-10)
    out = Tuple{CovarianceBlock,Int}[]
    for block in mom.blocks
        C = block.covariance
        d = LinearAlgebra.diag(C)
        if all(>(0), d)
            R = C ./ sqrt.(d) ./ sqrt.(d)'
            shifted = LinearAlgebra.Symmetric(R - tolerance * LinearAlgebra.I)
            LinearAlgebra.issuccess(LinearAlgebra.cholesky(shifted; check = false)) && continue
        else
            R = Matrix(C)
        end
        λ = LinearAlgebra.eigvals(LinearAlgebra.Symmetric(R))
        rank = count(>(tolerance * max(maximum(λ), 0.0)), λ)
        push!(out, (block, rank))
    end
    return out
end

# The singular blocks as `"a, b (rank 1 of 2)"`.
function _singular_laws(form::CanonicalForm, mom::Moments)
    return map(_singular_blocks(mom)) do (block, rank)
        names = _name_list([form.uncertainty_names[j-1] for j in block.columns])
        return "$names (rank $rank of $(length(block.columns)))"
    end
end
