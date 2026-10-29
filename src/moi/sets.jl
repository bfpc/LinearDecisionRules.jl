# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    UnivariateUncertainty(distribution::Distributions.UnivariateDistribution)

The MOI set of a scalar uncertainty. Build it with [`Uncertainty`](@ref).
"""
struct UnivariateUncertainty{D<:Distributions.UnivariateDistribution} <: MOI.AbstractScalarSet
    distribution::D
end

"""
    MultivariateUncertainty(distribution::Distributions.MultivariateDistribution)

The MOI set of a vector of uncertainties with a joint law. Build it with [`Uncertainty`](@ref).
"""
struct MultivariateUncertainty{D<:Distributions.MultivariateDistribution} <: MOI.AbstractVectorSet
    distribution::D
end

"""
    Uncertainty(distribution)

The set of the uncertainties with law `distribution`.

A univariate distribution gives a [`UnivariateUncertainty`](@ref), a multivariate one a
[`MultivariateUncertainty`](@ref) whose dimension is `length(distribution)`. Separate
declarations are independent. The distribution must have a bounded support, since the support
Ξ of the uncertainties comes from it.

## Example

```julia
@variable(model, d in Uncertainty(Uniform(80, 120)))
@variable(model, e[1:2] in Uncertainty(product_distribution(Uniform(0, 1), Uniform(0, 2))))
```
"""
function Uncertainty(distribution::Distributions.UnivariateDistribution)
    return UnivariateUncertainty(distribution)
end

function Uncertainty(distribution::Distributions.MultivariateDistribution)
    return MultivariateUncertainty(distribution)
end

function Uncertainty(distribution::Distributions.Distribution)
    return throw(
        ArgumentError(
            "Uncertainty(::$(typeof(distribution))): the distribution must be univariate or " *
            "multivariate.",
        ),
    )
end

const _AnyUncertainty = Union{UnivariateUncertainty,MultivariateUncertainty}

MOI.dimension(set::MultivariateUncertainty) = length(set.distribution)

# Distributions are treated as immutable values, so a copy can share them.
Base.copy(set::_AnyUncertainty) = set

# Most distributions have no `==`, which falls back to `===`; compare their fields instead.
function _same_distribution(a, b)
    typeof(a) == typeof(b) || return false
    a == b && return true
    return all(f -> isequal(getfield(a, f), getfield(b, f)), fieldnames(typeof(a)))
end

function Base.:(==)(a::S, b::S) where {S<:_AnyUncertainty}
    return _same_distribution(a.distribution, b.distribution)
end

# Consistent with `==` without hashing the data of the distribution.
Base.hash(set::_AnyUncertainty, h::UInt) = hash(typeof(set.distribution), hash(typeof(set), h))

Base.show(io::IO, set::UnivariateUncertainty) = print(io, "Uncertainty(", set.distribution, ")")

function Base.show(io::IO, set::MultivariateUncertainty)
    name = nameof(typeof(set.distribution))
    return print(io, "Uncertainty(", name, " of dimension ", MOI.dimension(set), ")")
end

"""
    Stage(t::Integer)

The stage of a decision or an uncertainty: the information of stage `t` is revealed before the
decisions of stage `t`, and the decisions of stage 1 are made here-and-now.

It is a `VariableIndex`-in-set constraint, so that stages show in `print(model)` and disappear
with the variable on `delete`. It is also the tag of `@variable(model, x, Stage(t))`. Constraints
have no stage. Stages start at 1.

See also [`FirstStage`](@ref), [`stage`](@ref), [`set_stage`](@ref), [`unset_stage`](@ref).
"""
struct Stage <: MOI.AbstractScalarSet
    t::Int
    function Stage(t::Integer)
        if t < 1
            throw(ArgumentError("Stage($t): stages start at 1."))
        end
        return new(t)
    end
end

Base.show(io::IO, set::Stage) = print(io, "Stage(", set.t, ")")

"""
    FirstStage

An alias of `Stage(1)`, the here-and-now decisions: `@variable(model, x >= 0, FirstStage)`.
"""
const FirstStage = Stage(1)

_law_name(d::Distributions.UnivariateDistribution) = string(d)

_law_name(d::Distributions.MultivariateDistribution) = "of type $(nameof(typeof(d)))"

# The support of an uncertainty must be bounded, since the support Ξ comes from it.
function _unbounded_support_message(distribution::Distributions.Distribution)
    bounds = _support_bounds(distribution)
    if bounds === nothing
        return "The distribution $(_law_name(distribution)) does not define `minimum` and " *
               "`maximum`, so the bounds of its support are unknown."
    elseif all(isfinite, bounds[1]) && all(isfinite, bounds[2])
        return nothing
    end
    return "The distribution $(_law_name(distribution)) has an unbounded support."
end

_is_metadata_constraint(::Type{MOI.VariableIndex}, ::Type{<:UnivariateUncertainty}) = true

_is_metadata_constraint(::Type{MOI.VectorOfVariables}, ::Type{<:MultivariateUncertainty}) = true

_is_metadata_constraint(::Type{MOI.VariableIndex}, ::Type{Stage}) = true

# Writing a model to a MOF file works for inspection. Distributions are written as their fields,
# so uncertainty sets cannot be read back.
MOI.FileFormats.MOF.head_name(::Type{<:UnivariateUncertainty}) = "UnivariateUncertainty"

MOI.FileFormats.MOF.head_name(::Type{<:MultivariateUncertainty}) = "MultivariateUncertainty"

MOI.FileFormats.MOF.head_name(::Type{Stage}) = "Stage"
