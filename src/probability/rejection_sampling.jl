# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The rejection sampling that estimates the moments of the conditioned laws: its settings, the
# time limit of each law, the cache of the estimates for re-solves, and the estimator.

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
