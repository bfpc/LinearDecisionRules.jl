# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    ConditionedDistribution(
        laws::AbstractVector{<:Distributions.Distribution},
        W::AbstractMatrix{<:Real},
        h::AbstractVector{<:Real};
        cones::AbstractVector = [MOI.Nonnegatives(length(h))],
        names::AbstractVector{<:AbstractString} = fill("", length(h)),
    )

The law of `η = (η₁, …, η_L)`, with independent `ηₗ ~ laws[l]`, conditioned on the set
`C = {η : W η - h ∈ K}`, where `K = cones[1] × cones[2] × …` is a product of `MOI.Nonnegatives`
and `MOI.SecondOrderCone`s whose dimensions add up to `length(h)`: by default, `C = {η : W η ≥ h}`.

The components of `η` are those of the laws, in order: `W` has one column per component. `names`
names the constraint of each row, for the messages. [`canonicalize`](@ref) builds it for the
declarations coupled by constraints tagged [`Truncate`](@ref).

Draws are by rejection: a draw of the product of the laws is kept if it is in `C` (up to a relative
tolerance of `1e-9`). Its moments have no closed form: [`moments`](@ref) estimates them by
rejection sampling. `minimum` and `maximum` are the boxes of the laws. When all the laws have
finitely many atoms, [`canonicalize`](@ref) replaces it by the exact law of the atoms of their
product in `C`, a [`MvDiscreteNonParametric`](@ref).

## Fields

 * `laws::Vector{Distribution}`: the independent laws, in the order of their components.
 * `W::Matrix{Float64}`, `h::Vector{Float64}`, `cones::Vector`: the set `C`.
 * `names::Vector{String}`: the name of the constraint of each row.
 * `offsets::Vector{Int}`: the first component of each law, and `length + 1` after the last.

## Example

```julia
d = ConditionedDistribution([Uniform(40, 80), Uniform(40, 80)], [-1.0 -1.0], [-120.0])
rand(d)  # a draw with d[1] + d[2] ≤ 120
# The disc ‖η - (60, 60)‖ ≤ 20, as (20, η₁ - 60, η₂ - 60) in a second-order cone:
disc = ConditionedDistribution(
    [Uniform(40, 80), Uniform(40, 80)],
    [0 0; 1 0; 0 1],
    [-20, 60, 60];
    cones = [MOI.SecondOrderCone(3)],
)
```
"""
struct ConditionedDistribution <: Distributions.ContinuousMultivariateDistribution
    laws::Vector{Distributions.Distribution}
    W::Matrix{Float64}
    h::Vector{Float64}
    cones::Vector{_SupportCone}
    names::Vector{String}
    # The first component of each law, and `length + 1` after the last.
    offsets::Vector{Int}

    function ConditionedDistribution(
        laws::AbstractVector{<:Distributions.Distribution},
        W::AbstractMatrix{<:Real},
        h::AbstractVector{<:Real};
        cones::AbstractVector = [MOI.Nonnegatives(length(h))],
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
        elseif !all(c -> c isa _SupportCone, cones)
            throw(ArgumentError("The cones must be `MOI.Nonnegatives` or `MOI.SecondOrderCone`."))
        elseif sum(MOI.dimension, cones; init = 0) != length(h)
            throw(ArgumentError("The dimensions of the cones must add up to the rows of W."))
        end
        offsets = cumsum([1; dims])
        return new(
            collect(Distributions.Distribution, laws),
            Matrix{Float64}(W),
            Vector{Float64}(h),
            collect(_SupportCone, cones),
            Vector{String}(names),
            offsets,
        )
    end
end

Base.length(d::ConditionedDistribution) = d.offsets[end] - 1

Base.eltype(::Type{ConditionedDistribution}) = Float64

Distributions.params(d::ConditionedDistribution) = (d.laws, d.W, d.h, d.cones)

function Base.minimum(d::ConditionedDistribution)
    return reduce(vcat, (_support_box(law)[1] for law in d.laws); init = Float64[])
end

function Base.maximum(d::ConditionedDistribution)
    return reduce(vcat, (_support_box(law)[2] for law in d.laws); init = Float64[])
end

# The estimates of the moments are cached by the law, so equality is by value. The names of the
# constraints do not change the law.
function Base.:(==)(a::ConditionedDistribution, b::ConditionedDistribution)
    return isequal(a.laws, b.laws) && a.W == b.W && a.h == b.h && a.cones == b.cones
end

function Base.hash(d::ConditionedDistribution, h::UInt)
    return hash(d.laws, hash(d.W, hash(d.h, hash(d.cones, hash(:ConditionedDistribution, h)))))
end

function Base.show(io::IO, d::ConditionedDistribution)
    print(io, "ConditionedDistribution(", join(d.laws, ", "), "; ", length(d.h), " rows)")
    return
end

# The constraints of the rows, for the messages: their names, without repeats.
function _constraint_names(d::ConditionedDistribution)
    names = unique(filter(!isempty, d.names))
    return isempty(names) ? "W η - h ∈ K" : join(names, ", ")
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

# Whether each column of `C` is in the set `W η - h ∈ K`, with a relative tolerance, so that the
# atoms of discrete laws on the boundary are kept despite rounding.
function _in_set(d::ConditionedDistribution, C::AbstractMatrix)
    WC = d.W * C
    accepted = trues(size(C, 2))
    offset = 0
    for cone in d.cones
        rows = offset .+ (1:MOI.dimension(cone))
        offset += MOI.dimension(cone)
        if cone isa MOI.Nonnegatives
            for s in axes(C, 2), r in rows
                if WC[r, s] < d.h[r] - 1e-9 * (1 + abs(d.h[r]))
                    accepted[s] = false
                end
            end
        else
            tolerance = 1e-9 * (1 + LinearAlgebra.norm(view(d.h, rows)))
            for s in axes(C, 2)
                v = view(WC, rows, s) .- view(d.h, rows)
                if _cone_violation(cone, v) > tolerance
                    accepted[s] = false
                end
            end
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

# The rows of the supports of its laws, each over its own components: its own rows condition the
# law, and they are rows of the canonical form already.
function support_constraints(d::ConditionedDistribution)
    pieces = [support_constraints(law) for law in d.laws]
    W = zeros(sum(p -> length(p.h), pieces; init = 0), length(d))
    row = 0
    for (l, p) in enumerate(pieces)
        W[row.+(1:length(p.h)), d.offsets[l]:d.offsets[l+1]-1] = p.W
        row += length(p.h)
    end
    h = reduce(vcat, (p.h for p in pieces); init = Float64[])
    return (W = W, h = h, cones = reduce(vcat, (p.cones for p in pieces); init = _SupportCone[]))
end

_has_auto_support(d::ConditionedDistribution) = any(_has_auto_support, d.laws)

function _resolve_support(d::ConditionedDistribution, conic::Bool)
    _has_auto_support(d) || return d
    laws = [_resolve_support(law, conic) for law in d.laws]
    return ConditionedDistribution(laws, d.W, d.h; d.cones, d.names)
end
