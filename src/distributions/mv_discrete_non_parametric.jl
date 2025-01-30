struct MvDiscreteNonParametric{
    T<:Real,
    P<:Real,
    Ts<:AbstractVector{T},
    Tss<:AbstractVector{Ts},
    Ps<:AbstractVector{P},
} <: Distributions.DiscreteMultivariateDistribution
    support::Tss
    p::Ps
    function MvDiscreteNonParametric{T,P,Ts,Tss,Ps}(xs::Tss, ps::Ps; check_args::Bool=true) where {
        T<:Real,P<:Real,Ts<:AbstractVector{T},Tss<:AbstractVector{Ts},Ps<:AbstractVector{P}}
        check_args || return new{T,P,Ts,Tss,Ps}(xs, ps)
        Distributions.@check_args(
            MvDiscreteNonParametric,
            (length(xs) == length(ps), "length of support and probability vector must be equal"),
            (ps, Distributions.isprobvec(ps), "vector is not a probability vector"),
            # (xs, Distributions.allunique(xs), "support must contain only unique elements"),
        )
        return new{T,P,Ts,Tss,Ps}(xs, ps)
    end
end

function MvDiscreteNonParametric(
    xs::AbstractVector{Ts},
    ps::AbstractVector{P};
    check_args::Bool=true,
) where {T<:Real,Ts<:AbstractVector{T},P<:Real}
    return MvDiscreteNonParametric{T,P,eltype(xs),typeof(xs),typeof(ps)}(xs, ps; check_args)
end

Base.eltype(::Type{<:MvDiscreteNonParametric{T,P,Ts}}) where {T,P,Ts} = Ts

Distributions.params(d::MvDiscreteNonParametric) = (d.support, d.p)

_support(d::MvDiscreteNonParametric) = d.support

_probs(d::MvDiscreteNonParametric) = d.p

# Base.isapprox

Distributions.length(d::MvDiscreteNonParametric) = length(_support(d)[1])

function Distributions._rand!(rng::Random.AbstractRNG, d::MvDiscreteNonParametric, x::AbstractVector)
    _x = _support(d)
    p = _probs(d)
    n = length(p)
    draw = rand(rng, float(eltype(p)))
    cp = p[1]
    i = 1
    while cp <= draw && i < n
        @inbounds cp += p[i +=1]
    end
    x .+= _x[i]
    return x
end

function Distributions.minimum(d::MvDiscreteNonParametric)
    support = _support(d)
    ret = copy(support[1])
    for x in support
        for i in eachindex(x)
            if x[i] < ret[i]
                ret[i] = x[i]
            end
        end
    end
    return ret
end

function Distributions.maximum(d::MvDiscreteNonParametric)
    support = _support(d)
    ret = copy(support[begin])
    for x in support
        for i in eachindex(x)
            if x[i] > ret[i]
                ret[i] = x[i]
            end
        end
    end
    return ret
end

function Distributions.mean(d::MvDiscreteNonParametric)
    support = _support(d)
    p = _probs(d)
    ret = zeros(eltype(support[1]), length(support[1]))
    for i in eachindex(support)
        ret .+= p[i] .* support[i]
    end
    return ret
end

function Distributions.cov(d::MvDiscreteNonParametric)
    support = _support(d)
    p = _probs(d)
    mean = Distributions.mean(d)
    ret = zeros(eltype(support[1]), length(support[1]), length(support[1]))
    val = zeros(eltype(support[1]), length(support[1]))
    for i in eachindex(support)
        val .= support[i] .- mean
        ret .+= p[i] .* (val * val')
    end
    return ret
end

function Distributions.var(d::MvDiscreteNonParametric)
    support = _support(d)
    p = _probs(d)
    mean = Distributions.mean(d)
    ret = zeros(eltype(support[1]), length(support[1]))
    val = zeros(eltype(support[1]), length(support[1]))
    for i in eachindex(support)
        val .= support[i] .- mean
        ret .+= p[i] .* (val .* val)
    end
    return ret
end

function Distributions.insupport(d::MvDiscreteNonParametric, x)
    support = _support(d)
    for y in support
        if x == y
            return true
        end
    end
    return false
end
