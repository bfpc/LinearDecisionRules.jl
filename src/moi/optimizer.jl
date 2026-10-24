# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    Optimizer(inner_factory = nothing)

An MOI optimizer for the linear decision rules of a JuMP model. `inner_factory`, for example
`HiGHS.Optimizer`, builds the optimizer that solves the reformulated models.

`optimize!` reads the model of the user in place, as the cache of JuMP, instead of copying it.
Use it without bridges, since a bridge would add variables that have no stage:
`Model(() -> Optimizer(HiGHS.Optimizer); add_bridges = false)`, or [`LDRModel`](@ref).

It supports exactly what [`canonicalize`](@ref) reads, and the sets of the metadata:
[`Uncertainty`](@ref) and [`Stage`](@ref). Anything else is rejected when the model is built.

## Optimizer attributes

 * [`InnerOptimizer`](@ref)`()`: the factory of the inner optimizer.
 * [`Formulations`](@ref)`()`: the formulations solved by `optimize!`, `[PrimalLDR(), DualLDR()]`
   by default.
 * `MOI.Silent`, `MOI.TimeLimitSec` and `MOI.RawOptimizerAttribute` are kept and forwarded to the
   inner optimizers.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    inner_factory::Any
    formulations::Vector{AbstractFormulation}
    silent::Bool
    time_limit::Union{Nothing,Float64}
    raw_attributes::Dict{String,Any}
    name::String

    function Optimizer(inner_factory = nothing)
        return new(inner_factory, [PrimalLDR(), DualLDR()], false, nothing, Dict{String,Any}(), "")
    end
end

# The optimizer attributes survive `empty!`, as JuMP calls it to reset the optimizer.
MOI.is_empty(::Optimizer) = true

MOI.empty!(::Optimizer) = nothing

MOI.get(::Optimizer, ::MOI.SolverName) = "LinearDecisionRules"

# `optimize!(dest, src)` reads `src` in place, which is JuMP's cache.
MOI.supports_incremental_interface(::Optimizer) = false

const _BoundSets =
    Union{MOI.LessThan{Float64},MOI.GreaterThan{Float64},MOI.EqualTo{Float64},MOI.Interval{Float64}}

# What `canonicalize` reads. Without bridges, anything else is an error when it is added.
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VariableIndex},
    ::Type{
        <:Union{
            _BoundSets,
            MOI.Integer,
            MOI.ZeroOne,
            MOI.Parameter{Float64},
            Stage,
            UnivariateUncertainty,
        },
    },
)
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.ScalarAffineFunction{Float64}},
    ::Type{<:_BoundSets},
)
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorOfVariables},
    ::Type{<:MultivariateUncertainty},
)
    return true
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{
        <:Union{
            MOI.VariableIndex,
            MOI.ScalarAffineFunction{Float64},
            MOI.ScalarQuadraticFunction{Float64},
        },
    },
)
    return true
end

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

MOI.supports(::Optimizer, ::Union{MOI.VariableName,BreakPoints}, ::Type{MOI.VariableIndex}) = true

const _AffineConstraintIndex = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}}

function MOI.supports(
    ::Optimizer,
    ::Union{MOI.ConstraintName,XiRole,ConstraintStage},
    ::Type{<:_AffineConstraintIndex},
)
    return true
end

function MOI.supports(
    ::Optimizer,
    ::Union{
        MOI.Name,
        MOI.Silent,
        MOI.TimeLimitSec,
        MOI.RawOptimizerAttribute,
        InnerOptimizer,
        Formulations,
    },
)
    return true
end

MOI.get(o::Optimizer, ::MOI.Name) = o.name

function MOI.set(o::Optimizer, ::MOI.Name, name::String)
    o.name = name
    return
end

MOI.get(o::Optimizer, ::MOI.Silent) = o.silent

function MOI.set(o::Optimizer, ::MOI.Silent, silent::Bool)
    o.silent = silent
    return
end

MOI.get(o::Optimizer, ::MOI.TimeLimitSec) = o.time_limit

function MOI.set(o::Optimizer, ::MOI.TimeLimitSec, limit::Union{Nothing,Real})
    o.time_limit = limit === nothing ? nothing : Float64(limit)
    return
end

function MOI.get(o::Optimizer, attr::MOI.RawOptimizerAttribute)
    if !haskey(o.raw_attributes, attr.name)
        throw(
            MOI.GetAttributeNotAllowed(
                attr,
                "The raw attribute \"$(attr.name)\" was not set. Raw attributes are kept " *
                "for the inner optimizer, and not read from it.",
            ),
        )
    end
    return o.raw_attributes[attr.name]
end

function MOI.set(o::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    o.raw_attributes[attr.name] = value
    return
end

MOI.get(o::Optimizer, ::InnerOptimizer) = o.inner_factory

function MOI.set(o::Optimizer, ::InnerOptimizer, factory)
    o.inner_factory = factory
    return
end

MOI.get(o::Optimizer, ::Formulations) = copy(o.formulations)

function MOI.set(o::Optimizer, ::Formulations, formulations)
    if isempty(formulations) || !all(f -> f isa AbstractFormulation, formulations)
        throw(ArgumentError("`Formulations()` needs a vector of at least one formulation."))
    end
    o.formulations = collect(AbstractFormulation, formulations)
    return
end

# A new inner optimizer with the settings of `o`.
function _instantiate_inner(o::Optimizer)
    if o.inner_factory === nothing
        error(
            "The LinearDecisionRules.Optimizer has no inner optimizer to solve the " *
            "reformulated models. Build the model with `LDRModel(HiGHS.Optimizer)`, or use " *
            "`set_attribute(model, InnerOptimizer(), HiGHS.Optimizer)`.",
        )
    end
    inner = MOI.instantiate(o.inner_factory)
    if o.silent
        MOI.set(inner, MOI.Silent(), true)
    end
    if o.time_limit !== nothing
        MOI.set(inner, MOI.TimeLimitSec(), o.time_limit)
    end
    for (name, value) in o.raw_attributes
        MOI.set(inner, MOI.RawOptimizerAttribute(name), value)
    end
    return inner
end

function MOI.copy_to(::Optimizer, ::MOI.ModelLike)
    return error("LinearDecisionRules.Optimizer: `copy_to` is not implemented yet (part 2).")
end

function MOI.optimize!(::Optimizer, ::MOI.ModelLike)
    return error("LinearDecisionRules.Optimizer: `optimize!` is not implemented yet (part 2).")
end
