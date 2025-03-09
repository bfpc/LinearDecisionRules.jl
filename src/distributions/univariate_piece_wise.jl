import Expectations

struct UnivariatePieceWise{
    D<:Distributions.UnivariateDistribution,
    S<:Distributions.ValueSupport,
    T<:Real,
} <: Distributions.MultivariateDistribution{S}
    original::D
    break_points::AbstractVector{T}
    function UnivariatePieceWise(
        d::D,
        break_points::Vector{T},
    ) where {D<:Distributions.UnivariateDistribution,T<:Real}
        @assert eltype(d) == T
        @assert minimum(d) > -Inf
        @assert maximum(d) < Inf
        @assert length(break_points) > 0
        @assert all(break_points .!= NaN)
        sorted_break_points = sort(break_points)
        @assert sorted_break_points[begin] > -Inf
        @assert sorted_break_points[end] < Inf
        @assert sorted_break_points[begin] > minimum(d)
        @assert sorted_break_points[end] < maximum(d)

        return new{typeof(d),Distributions.value_support(typeof(d)),T}(
            d,
            sorted_break_points,
        )
    end
end

Base.eltype(::Type{<:UnivariatePieceWise{D,S,T}}) where {D,S,T} = T

function Distributions.params(d::UnivariatePieceWise)
    return tuple(params(d.original)..., d.break_points)
end

_original(d::UnivariatePieceWise) = d.original

_break_points(d::UnivariatePieceWise) = d.break_points

# Base.isapprox

Distributions.length(d::UnivariatePieceWise) = length(_break_points(d)) + 1

function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::UnivariatePieceWise,
    x::AbstractVector,
)
    draw = rand(rng, d.original)
    @assert length(d) == length(x)
    break_points = _break_points(d)
    pos = findlast(x -> x < draw, break_points)
    fill!(x, zero(eltype(x)))

    if pos === nothing
        x[1] = draw
        return x
    end
    x[1] = break_points[1]
    for i in 1:pos-1
        x[i+1] = break_points[i+1] - break_points[i]
    end
    x[pos+1] = draw - break_points[pos]
    return x
end

function Distributions.minimum(d::UnivariatePieceWise{D,S,T}) where {D,S,T}
    ret = zeros(T, length(d))
    ret[1] = minimum(d.original)
    for i in 2:length(d)
        ret[i] = 0.0
    end
    return ret
end

function Distributions.maximum(d::UnivariatePieceWise{D,S,T}) where {D,S,T}
    ret = zeros(T, length(d))
    ret[end] = maximum(d.original) - d.break_points[end]
    ret[1] = d.break_points[1]
    for i in 2:length(d)-1
        ret[i] = d.break_points[i] - d.break_points[i-1]
    end
    return ret
end

# TODO: do better for normal and uniform
# TODO: overload integration correctly so we dont have too many points
function Distributions.mean(d::UnivariatePieceWise)
    vals = [minimum(d.original), d.break_points..., maximum(d.original)]
    ret = zeros(eltype(d), length(d))
    for i in 1:length(d)
        Δi = vals[i+1] - vals[i]
        E = Expectations.expectation(
            Distributions.truncated(d.original, vals[i], vals[i+1]),
        )
        ret[i] =
            E(x -> x - vals[i]) * (
                Distributions.cdf(d.original, vals[i+1]) -
                Distributions.cdf(d.original, vals[i])
            )
        ret[i] += Δi * (1 - Distributions.cdf(d.original, vals[i+1]))
    end
    ret[1] += vals[1]
    return ret
end

function Distributions.cov(d::UnivariatePieceWise)
    _mean = Distributions.mean(d)
    vals = [minimum(d.original), d.break_points..., maximum(d.original)]
    _mean[1] -= vals[1]
    ret = zeros(eltype(eltype(d)), length(d), length(d))
    for j in 1:length(d)
        for i in 1:(j-1)
            Δi = vals[i+1] - vals[i]
            Δj = vals[j+1] - vals[j]
            E = Expectations.expectation(
                Distributions.truncated(d.original, vals[j], vals[j+1]),
            )
            part1 =
                Δi *
                E(x -> x - vals[j]) *
                (
                    Distributions.cdf(d.original, vals[j+1]) -
                    Distributions.cdf(d.original, vals[j])
                )
            part2 = Δi * Δj * (1 - Distributions.cdf(d.original, vals[j+1]))
            ret[i, j] = part1 + part2 - _mean[i] * _mean[j]
            ret[j, i] = part1 + part2 - _mean[j] * _mean[i]
        end
    end
    _var = Distributions.var(d)
    for i in 1:length(d)
        ret[i, i] = _var[i]
    end
    return ret
end

function Distributions.var(d::UnivariatePieceWise)
    _mean = Distributions.mean(d)
    vals = [minimum(d.original), d.break_points..., maximum(d.original)]
    _mean[1] -= vals[1]
    ret = zeros(eltype(eltype(d)), length(d))
    for i in 1:length(d)
        Δi = vals[i+1] - vals[i]
        E = Expectations.expectation(
            Distributions.truncated(d.original, vals[i], vals[i+1]),
        )
        part1 =
            E(x -> (x - vals[i])^2) * (
                Distributions.cdf(d.original, vals[i+1]) -
                Distributions.cdf(d.original, vals[i])
            )
        part2 = Δi^2 * (1 - Distributions.cdf(d.original, vals[i+1]))
        ret[i] = part1 + part2 - _mean[i]^2
    end
    return ret
end
