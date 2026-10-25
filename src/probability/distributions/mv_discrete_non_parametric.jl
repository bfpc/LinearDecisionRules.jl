# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    MvDiscreteNonParametric(support::AbstractVector{<:AbstractVector{<:Real}}, p::AbstractVector{<:Real})

A multivariate discrete distribution with support points `support` and probabilities `p`.

Use it for scenario-based uncertainty with joint realizations of several parameters. Constraints
tagged [`Truncate`](@ref) condition it exactly: the scenarios that satisfy them are kept, with
their probabilities renormalized.

## Example

```julia
scenarios = [[80.0, 60.0], [100.0, 80.0], [120.0, 100.0]]
d = MvDiscreteNonParametric(scenarios, [0.3, 0.4, 0.3])
```
"""
struct MvDiscreteNonParametric{
    T<:Real,
    P<:Real,
    Ts<:AbstractVector{T},
    Tss<:AbstractVector{Ts},
    Ps<:AbstractVector{P},
} <: Distributions.DiscreteMultivariateDistribution
    support::Tss
    p::Ps

    function MvDiscreteNonParametric{T,P,Ts,Tss,Ps}(
        xs::Tss,
        ps::Ps;
        check_args::Bool = true,
    ) where {T<:Real,P<:Real,Ts<:AbstractVector{T},Tss<:AbstractVector{Ts},Ps<:AbstractVector{P}}
        check_args || return new{T,P,Ts,Tss,Ps}(xs, ps)
        Distributions.@check_args(
            MvDiscreteNonParametric,
            (
                (; xs, ps),
                length(xs) == length(ps),
                "length of support and probability vector must be equal",
            ),
            (ps, Distributions.isprobvec(ps), "vector is not a probability vector"),
            (xs, !isempty(xs), "support must not be empty"),
            (xs, allequal(length.(xs)), "support points must have the same length"),
        )
        return new{T,P,Ts,Tss,Ps}(xs, ps)
    end
end

function MvDiscreteNonParametric(
    xs::AbstractVector{Ts},
    ps::AbstractVector{P};
    check_args::Bool = true,
) where {T<:Real,Ts<:AbstractVector{T},P<:Real}
    return MvDiscreteNonParametric{T,P,Ts,typeof(xs),typeof(ps)}(xs, ps; check_args)
end

Base.eltype(::Type{<:MvDiscreteNonParametric{T}}) where {T} = T

Distributions.params(d::MvDiscreteNonParametric) = (d.support, d.p)

Base.length(d::MvDiscreteNonParametric) = length(first(d.support))

function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::MvDiscreteNonParametric,
    x::AbstractVector{<:Real},
)
    p = d.p
    draw = rand(rng, float(eltype(p)))
    cp, i = p[1], 1
    while cp <= draw && i < length(p)
        i += 1
        cp += p[i]
    end
    x .= d.support[i]
    return x
end

function Base.minimum(d::MvDiscreteNonParametric)
    ret = copy(first(d.support))
    for x in d.support
        ret .= min.(ret, x)
    end
    return ret
end

function Base.maximum(d::MvDiscreteNonParametric)
    ret = copy(first(d.support))
    for x in d.support
        ret .= max.(ret, x)
    end
    return ret
end

function Distributions.mean(d::MvDiscreteNonParametric)
    ret = zeros(float(eltype(d)), length(d))
    for (x, p) in zip(d.support, d.p)
        ret .+= p .* x
    end
    return ret
end

function Distributions.cov(d::MvDiscreteNonParametric)
    μ = Distributions.mean(d)
    ret = zeros(eltype(μ), length(d), length(d))
    for (x, p) in zip(d.support, d.p)
        v = x .- μ
        ret .+= p .* (v * v')
    end
    return ret
end

function Distributions.var(d::MvDiscreteNonParametric)
    μ = Distributions.mean(d)
    ret = zeros(eltype(μ), length(d))
    for (x, p) in zip(d.support, d.p)
        ret .+= p .* (x .- μ) .^ 2
    end
    return ret
end

Distributions.insupport(d::MvDiscreteNonParametric, x::AbstractVector) = any(==(x), d.support)

# Its atoms of positive probability, for an exact conditioning (see `_discrete_conditioned_law`).
function _finite_atoms(d::MvDiscreteNonParametric)
    positive = findall(>(0), d.p)
    atoms = reduce(hcat, [Vector{Float64}(d.support[i]) for i in positive])
    return atoms, Vector{Float64}(d.p[positive])
end
