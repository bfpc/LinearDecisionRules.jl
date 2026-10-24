# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    ConditionedDistribution(
        laws::AbstractVector{<:Distributions.Distribution},
        W::AbstractMatrix{<:Real},
        h::AbstractVector{<:Real};
        names::AbstractVector{<:AbstractString} = fill("", length(h)),
    )

The law of `η = (η₁, …, η_L)`, with independent `ηₗ ~ laws[l]`, conditioned on the set
`C = {η : W η ≥ h}`.

The components of `η` are those of the laws, in order: `W` has one column per component. `names`
names the constraint of each row, for the messages. [`canonicalize`](@ref) builds it for the
declarations coupled by constraints tagged [`Truncate`](@ref).

Draws are by rejection: a draw of the product of the laws is kept if it is in `C` (up to a relative
tolerance of `1e-9`). Its moments have no closed form: [`moments`](@ref) estimates them by
rejection sampling. `minimum` and `maximum` are the boxes of the laws.

## Example

```julia
d = ConditionedDistribution([Uniform(40, 80), Uniform(40, 80)], [-1.0 -1.0], [-120.0])
rand(d)  # a draw with d[1] + d[2] ≤ 120
```
"""
struct ConditionedDistribution <: Distributions.ContinuousMultivariateDistribution
    laws::Vector{Distributions.Distribution}
    W::Matrix{Float64}
    h::Vector{Float64}
    names::Vector{String}
    # The first component of each law, and `length + 1` after the last.
    offsets::Vector{Int}

    function ConditionedDistribution(
        laws::AbstractVector{<:Distributions.Distribution},
        W::AbstractMatrix{<:Real},
        h::AbstractVector{<:Real};
        names::AbstractVector{<:AbstractString} = fill("", length(h)),
    )
        dims = map(_dimension, laws)
        if any(iszero, dims)
            throw(ArgumentError("The laws of a ConditionedDistribution must be distributions."))
        elseif any(d -> d isa ConditionedDistribution, laws)
            throw(ArgumentError("The laws of a ConditionedDistribution cannot be conditioned."))
        elseif size(W) != (length(h), sum(dims; init = 0))
            throw(
                ArgumentError(
                    "W must have one row per entry of h ($(length(h))) and one column per " *
                    "component of the laws ($(sum(dims; init = 0))); got size $(size(W)).",
                ),
            )
        elseif length(names) != length(h)
            throw(ArgumentError("There must be one name per row of W."))
        end
        offsets = cumsum([1; dims])
        return new(
            collect(Distributions.Distribution, laws),
            Matrix{Float64}(W),
            Vector{Float64}(h),
            Vector{String}(names),
            offsets,
        )
    end
end

Base.length(d::ConditionedDistribution) = d.offsets[end] - 1

Base.eltype(::Type{ConditionedDistribution}) = Float64

Distributions.params(d::ConditionedDistribution) = (d.laws, d.W, d.h)

function Base.minimum(d::ConditionedDistribution)
    return reduce(vcat, (_support_box(law)[1] for law in d.laws); init = Float64[])
end

function Base.maximum(d::ConditionedDistribution)
    return reduce(vcat, (_support_box(law)[2] for law in d.laws); init = Float64[])
end

# The estimates of the moments are cached by the law, so equality is by value. The names of the
# constraints do not change the law.
function Base.:(==)(a::ConditionedDistribution, b::ConditionedDistribution)
    return isequal(a.laws, b.laws) && a.W == b.W && a.h == b.h
end

function Base.hash(d::ConditionedDistribution, h::UInt)
    return hash(d.laws, hash(d.W, hash(d.h, hash(:ConditionedDistribution, h))))
end

function Base.show(io::IO, d::ConditionedDistribution)
    print(io, "ConditionedDistribution(", join(d.laws, ", "), "; ", length(d.h), " rows)")
    return
end

# The constraints of the rows, for the messages: their names, without repeats.
function _constraint_names(d::ConditionedDistribution)
    names = unique(filter(!isempty, d.names))
    return isempty(names) ? "W η ≥ h" : join(names, ", ")
end

# A draw of each law in the columns of `C`, which are candidates for the rejection.
function _draw_candidates!(rng::Random.AbstractRNG, d::ConditionedDistribution, C::AbstractMatrix)
    for (l, law) in enumerate(d.laws)
        rows = d.offsets[l]:d.offsets[l+1]-1
        if law isa Distributions.UnivariateDistribution
            C[only(rows), :] .= rand(rng, law, size(C, 2))
        else
            C[rows, :] .= rand(rng, law, size(C, 2))
        end
    end
    return C
end

# Whether each column of `C` is in the set `W η ≥ h`, with a relative tolerance, so that the atoms
# of discrete laws on the boundary are kept despite rounding.
function _in_set(d::ConditionedDistribution, C::AbstractMatrix)
    WC = d.W * C
    accepted = trues(size(C, 2))
    for s in axes(C, 2), r in eachindex(d.h)
        if WC[r, s] < d.h[r] - 1e-9 * (1 + abs(d.h[r]))
            accepted[s] = false
        end
    end
    return accepted
end

# One draw costs about a hundred nanoseconds, so this is about ten seconds: far beyond what a law
# whose moments could be estimated needs (see `moments`).
const _MAX_ATTEMPTS_PER_DRAW = 10^8

function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::ConditionedDistribution,
    x::AbstractVector{<:Real},
)
    C = zeros(length(d), 1)
    for _ in 1:_MAX_ATTEMPTS_PER_DRAW
        _draw_candidates!(rng, d, C)
        if only(_in_set(d, C))
            x .= view(C, :, 1)
            return x
        end
    end
    return error(
        "Rejection sampling drew $_MAX_ATTEMPTS_PER_DRAW candidates without one in the set of " *
        "the constraints $(_constraint_names(d)): its probability is zero or too small.",
    )
end
