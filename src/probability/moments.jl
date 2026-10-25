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

# Whether the conditional expectation of the components of `d` in later stages, given those in
# earlier stages, is affine in the latter, when its components are in the stages `stages`: `true`,
# `false`, or `nothing` when it is not known. Separate laws are independent, so this is, law by
# law, condition (i) of the dual bound (Kuhn, Wiesemann and Georghiou, 2011, Section 4), which
# independent stages satisfy. A law in one stage has it trivially.
function _linear_conditional_expectations(d::Distributions.Distribution, stages::AbstractVector)
    return allequal(stages) ? true : nothing
end

_linear_conditional_expectations(::Distributions.UnivariateDistribution, ::AbstractVector) = true

# The components of a product of univariate laws are independent.
function _linear_conditional_expectations(
    ::Union{Distributions.Product,Distributions.ProductDistribution{1,0}},
    ::AbstractVector,
)
    return true
end

# The law of some components of a normal vector given the others is normal, with an affine mean.
_linear_conditional_expectations(::Distributions.AbstractMvNormal, ::AbstractVector) = true

const _ContinuousTruncated =
    Distributions.Truncated{<:Distributions.ContinuousUnivariateDistribution}

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

# The settings of the rejection sampling that estimates the moments of conditioned laws. A time
# limit of `nothing` is the default: see `_sampling_time_limit`.
struct _RejectionSampling
    seed::Int
    time_limit::Union{Nothing,Float64}
    max_samples::Int

    function _RejectionSampling(
        seed::Integer,
        time_limit::Union{Nothing,Real},
        max_samples::Integer,
    )
        if time_limit !== nothing && !(time_limit > 0)
            throw(ArgumentError("The time limit of rejection sampling must be positive."))
        elseif max_samples < 1
            throw(ArgumentError("The number of samples of rejection sampling must be positive."))
        end
        return new(seed, time_limit === nothing ? nothing : Float64(time_limit), max_samples)
    end
end

# The time limit of the rejection sampling of each conditioned law, by default and without a time
# limit of the whole `optimize!`.
const _DEFAULT_SAMPLING_TIME_LIMIT = 10.0

# The time limit of the next law to sample, when `spent` seconds of `budget` are spent and `pending`
# laws are left, `pending` included: the explicit limit of each law, or else an equal share of what
# remains of `budget` (the part of `TimeLimitSec()` given to the sampling by `optimize!`), so that
# the time that a law does not use goes to the next ones, or else the default.
function _sampling_time_limit(s::_RejectionSampling, budget, spent::Float64, pending::Int)
    if s.time_limit !== nothing
        return s.time_limit
    elseif budget !== nothing
        return max(0.0, budget - spent) / pending
    end
    return _DEFAULT_SAMPLING_TIME_LIMIT
end

# Where the time limit of the sampling comes from, for the messages.
function _sampling_time_limit_source(s::_RejectionSampling, budget)
    if s.time_limit !== nothing
        return "$(s.time_limit) seconds per law, `RejectionSamplingTimeLimit()`"
    elseif budget !== nothing
        total = round(budget; sigdigits = 3)
        return "$total seconds shared by the laws, half of the time that remained of " *
               "`TimeLimitSec()`; set `RejectionSamplingTimeLimit()` to choose it"
    end
    return "$_DEFAULT_SAMPLING_TIME_LIMIT seconds per law, the default of " *
           "`RejectionSamplingTimeLimit()`"
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
# draws or `time_limit` seconds. The empirical law is supported in the conditioning set, so its
# mean is in Ξ and its moments are consistent. Sums are shifted by the first draw for accuracy, and
# draws come in batches sized by the acceptance rate. The sampling stops early when, even at the
# upper bound (by the rule of three) of the acceptance rate, it cannot keep `_MIN_SAMPLES` draws in
# time.
function _estimate(
    d::ConditionedDistribution,
    rng::Random.AbstractRNG,
    max_samples::Int,
    time_limit::Float64,
)
    k = length(d)
    n, attempts = 0, 0
    shift, s1, s2 = zeros(k), zeros(k), zeros(k, k)
    least = min(max_samples, _MIN_SAMPLES)
    batch = 256
    start = time_ns()
    seconds = 0.0
    while n < max_samples
        C = _draw_candidates!(rng, d, Matrix{Float64}(undef, k, batch))
        attempts += batch
        kept = findall(_in_set(d, C))
        kept = kept[1:min(end, max_samples - n)]
        if !isempty(kept)
            n == 0 && (shift .= view(C, :, first(kept)))
            D = C[:, kept] .- shift
            s1 .+= vec(sum(D; dims = 2))
            LinearAlgebra.mul!(s2, D, transpose(D), 1.0, 1.0)
            n += length(kept)
        end
        seconds = _seconds_since(start)
        if n >= max_samples || seconds >= time_limit || (n + 3) * time_limit < least * seconds
            break
        end
        rate = max(n, 1) / attempts
        batch = clamp(ceil(Int, 1.2 * (max_samples - n) / rate), 256, 65_536)
    end
    m = s1 ./ max(n, 1)
    covariance = s2 ./ max(n, 1) .- m * m'
    return _Estimate(shift .+ m, (covariance + covariance') ./ 2, n, attempts, seconds)
end

function _check_estimate(estimate::_Estimate, d, names, max_samples, time_limit, source)
    least = min(max_samples, _MIN_SAMPLES)
    estimate.num_samples >= least && return
    rate = round(estimate.num_samples / estimate.num_attempts; sigdigits = 3)
    seconds = round(estimate.seconds; sigdigits = 3)
    return error(
        "The law of the uncertainties $names, conditioned on the constraints " *
        "$(_constraint_names(d)) tagged `Truncate()`, cannot be estimated: rejection sampling " *
        "kept $(estimate.num_samples) of $(estimate.num_attempts) draws (acceptance rate " *
        "$rate) in $seconds seconds, and needs $least within its time limit of " *
        "$(round(time_limit; sigdigits = 3)) seconds ($source). The conditioning set has a " *
        "probability that is zero or too small, or the time limit is too short: check these " *
        "constraints and the time limits.",
    )
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
