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

_law_mean(d::Distributions.UnivariateDistribution) = [Float64(Distributions.mean(d))]
_law_mean(d::Distributions.MultivariateDistribution) = Vector{Float64}(Distributions.mean(d))

_law_covariance(d::Distributions.UnivariateDistribution) = fill(Float64(Distributions.var(d)), 1, 1)
_law_covariance(d::Distributions.MultivariateDistribution) = Matrix{Float64}(Distributions.cov(d))

const _ContinuousTruncated = Distributions.Truncated{<:Distributions.ContinuousUnivariateDistribution}

_law_mean(d::_ContinuousTruncated) = [_truncated_moments(d)[1]]

_law_covariance(d::_ContinuousTruncated) = fill(_truncated_moments(d)[2], 1, 1)

# Distributions has the moments of few truncated laws, such as the normal one; the others are by
# quadrature of the density of the truncated law.
function _truncated_moments(d::_ContinuousTruncated)
    try
        return Float64(Distributions.mean(d)), Float64(Distributions.var(d))
    catch err
        err isa MethodError || rethrow()
    end
    lo, hi = minimum(d), maximum(d)
    mean, _ = QuadGK.quadgk(x -> x * Distributions.pdf(d, x), lo, hi; rtol = 1e-12)
    var, _ = QuadGK.quadgk(x -> (x - mean)^2 * Distributions.pdf(d, x), lo, hi; rtol = 1e-12)
    return mean, var
end

# The settings of the rejection sampling that estimates the moments of conditioned laws.
struct _RejectionSampling
    seed::Int
    time_limit::Float64
    max_samples::Int

    function _RejectionSampling(seed::Integer, time_limit::Real, max_samples::Integer)
        if !(time_limit > 0)
            throw(ArgumentError("The time limit of rejection sampling must be positive."))
        elseif max_samples < 1
            throw(ArgumentError("The number of samples of rejection sampling must be positive."))
        end
        return new(seed, time_limit, max_samples)
    end
end

# The estimates of the moments of a conditioned law.
struct _Estimate
    mean::Vector{Float64}
    covariance::Matrix{Float64}
    num_samples::Int
    num_attempts::Int
    seconds::Float64
end

# The estimates of the conditioned laws, by `(law, seed, max_samples)`, so that a re-solve whose
# laws and conditioning rows did not change does not sample again. Only complete estimates (with
# `max_samples` draws) are kept, since a time limit makes the others depend on the machine.
mutable struct _MomentsCache
    estimates::Dict{Tuple{ConditionedDistribution,UInt64,Int},_Estimate}
    # The number of estimates computed, not found in the cache.
    num_estimates::Int
end

_MomentsCache() = _MomentsCache(Dict{Tuple{ConditionedDistribution,UInt64,Int},_Estimate}(), 0)

# Below this number of draws in the time limit, the estimates are an error, not a warning.
const _MIN_SAMPLES = 100

# The moments of the empirical law of the draws of `d` kept by rejection, up to `max_samples`
# draws or the time limit. The empirical law is supported in the conditioning set, so its mean is in
# Ξ and its moments are consistent. Sums are shifted by the first draw for accuracy, and draws come
# in batches sized by the acceptance rate. The sampling stops early when, even at the upper bound
# (by the rule of three) of the acceptance rate, it cannot keep `_MIN_SAMPLES` draws in time.
function _estimate(d::ConditionedDistribution, rng::Random.AbstractRNG, s::_RejectionSampling)
    k = length(d)
    n, attempts = 0, 0
    shift, s1, s2 = zeros(k), zeros(k), zeros(k, k)
    least = min(s.max_samples, _MIN_SAMPLES)
    batch = 256
    start = time_ns()
    seconds = 0.0
    while n < s.max_samples
        C = _draw_candidates!(rng, d, Matrix{Float64}(undef, k, batch))
        attempts += batch
        kept = findall(_in_set(d, C))
        kept = kept[1:min(end, s.max_samples - n)]
        if !isempty(kept)
            n == 0 && (shift .= view(C, :, first(kept)))
            D = C[:, kept] .- shift
            s1 .+= vec(sum(D; dims = 2))
            LinearAlgebra.mul!(s2, D, transpose(D), 1.0, 1.0)
            n += length(kept)
        end
        seconds = _seconds_since(start)
        if n >= s.max_samples || seconds >= s.time_limit || (n + 3) * s.time_limit < least * seconds
            break
        end
        rate = max(n, 1) / attempts
        batch = clamp(ceil(Int, 1.2 * (s.max_samples - n) / rate), 256, 65_536)
    end
    m = s1 ./ max(n, 1)
    covariance = s2 ./ max(n, 1) .- m * m'
    return _Estimate(shift .+ m, (covariance + covariance') ./ 2, n, attempts, seconds)
end

function _check_estimate(estimate::_Estimate, d::ConditionedDistribution, names, s)
    least = min(s.max_samples, _MIN_SAMPLES)
    estimate.num_samples >= least && return
    rate = round(estimate.num_samples / estimate.num_attempts; sigdigits = 3)
    seconds = round(estimate.seconds; sigdigits = 3)
    return error(
        "The law of the uncertainties $names, conditioned on the constraints " *
        "$(_constraint_names(d)) tagged `Truncate()`, cannot be estimated: rejection sampling " *
        "kept $(estimate.num_samples) of $(estimate.num_attempts) draws (acceptance rate " *
        "$rate) in $seconds seconds, and needs $least within the time limit of " *
        "$(s.time_limit) seconds (`RejectionSamplingTimeLimit()`). The conditioning set has a " *
        "probability that is zero or too small: check these constraints.",
    )
end

# The estimate of the conditioned law `d`, from `cache` if it has it.
function _conditioned_estimate(d, seed::UInt64, s::_RejectionSampling, cache, names)
    key = (d, seed, s.max_samples)
    if cache !== nothing && haskey(cache.estimates, key)
        return cache.estimates[key]
    end
    estimate = _estimate(d, Random.Xoshiro(seed), s)
    _check_estimate(estimate, d, names, s)
    if cache !== nothing
        cache.num_estimates += 1
        if estimate.num_samples == s.max_samples
            cache.estimates[key] = estimate
        end
    end
    return estimate
end

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
"""
function moments(
    form::CanonicalForm;
    seed::Integer = 1234,
    time_limit::Real = 10.0,
    max_samples::Integer = 10_000,
)
    return _moments(form, _RejectionSampling(seed, time_limit, max_samples), nothing)
end

function _moments(form::CanonicalForm, s::_RejectionSampling, cache::Union{Nothing,_MomentsCache})
    μ = zeros(1 + length(form.uncertainties))
    μ[1] = 1.0
    blocks = CovarianceBlock[]
    num_conditioned = count(law -> law.distribution isa ConditionedDistribution, form.laws)
    seeds = rand(Random.Xoshiro(s.seed % UInt64), UInt64, num_conditioned)
    used = Set{Tuple{ConditionedDistribution,UInt64,Int}}()
    k = 0
    small = String[]
    for law in form.laws
        names = join((form.uncertainty_names[j-1] for j in law.columns), ", ")
        d = law.distribution
        if d isa ConditionedDistribution
            k += 1
            seed = seeds[k]
            push!(used, (d, seed, s.max_samples))
            estimate = _conditioned_estimate(d, seed, s, cache, names)
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
        filter!(entry -> entry.first in used, cache.estimates)
    end
    if !isempty(small)
        @warn(
            "Rejection sampling kept fewer than $(s.max_samples) draws " *
            "(`RejectionSamplingMaxSamples()`) within the time limit of $(s.time_limit) " *
            "seconds (`RejectionSamplingTimeLimit()`) for the conditioned laws of " *
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
