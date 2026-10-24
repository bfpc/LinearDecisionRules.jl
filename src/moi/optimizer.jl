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
    # The state of the last `optimize!`, read by the results.
    form::Union{Nothing,CanonicalForm}
    moments::Union{Nothing,Moments}
    parameters::Dict{MOI.VariableIndex,Float64}
    decision_rows::Dict{MOI.VariableIndex,Int}
    uncertainty_columns::Dict{MOI.VariableIndex,Int}
    results::Vector{FormulationResult}
    solve_time::Float64

    function Optimizer(inner_factory = nothing)
        formulations = AbstractFormulation[PrimalLDR(), DualLDR()]
        parameters = Dict{MOI.VariableIndex,Float64}()
        indices = Dict{MOI.VariableIndex,Int}()
        raw_attributes = Dict{String,Any}()
        results = FormulationResult[]
        return new(
            inner_factory,
            formulations,
            false,
            nothing,
            raw_attributes,
            "",
            nothing,
            nothing,
            parameters,
            indices,
            copy(indices),
            results,
            NaN,
        )
    end
end

# The optimizer attributes survive `empty!`, as JuMP calls it to reset the optimizer.
MOI.is_empty(o::Optimizer) = o.form === nothing

function MOI.empty!(o::Optimizer)
    o.form = nothing
    o.moments = nothing
    empty!(o.parameters)
    empty!(o.decision_rows)
    empty!(o.uncertainty_columns)
    empty!(o.results)
    o.solve_time = NaN
    return
end

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
    # Each type has its result slot or its bound, so a second one would be ambiguous.
    types = map(typeof, formulations)
    if !allunique(types)
        twice = first(T for T in types if count(==(T), types) > 1)
        throw(ArgumentError("`Formulations()` has two formulations of type $(nameof(twice))."))
    end
    o.formulations = collect(AbstractFormulation, formulations)
    return
end

function _inner_factory(o::Optimizer)
    if o.inner_factory === nothing
        error(
            "The LinearDecisionRules.Optimizer has no inner optimizer to solve the " *
            "reformulated models. Build the model with `LDRModel(HiGHS.Optimizer)`, or use " *
            "`set_attribute(model, InnerOptimizer(), HiGHS.Optimizer)`.",
        )
    end
    return o.inner_factory
end

function MOI.copy_to(::Optimizer, ::MOI.ModelLike)
    return error("LinearDecisionRules.Optimizer reads the model in place: use `optimize!`.")
end

# Indices are the same in `src`, which is JuMP's cache, and in the optimizer.
function _identity_map(src::MOI.ModelLike)
    index_map = MOI.Utilities.IndexMap()
    for v in MOI.get(src, MOI.ListOfVariableIndices())
        index_map[v] = v
    end
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        for ci in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
            index_map[ci] = ci
        end
    end
    return index_map
end

# Uncertainties added with `MOI.add_constrained_variable` bypass the checks of `@variable`.
function _check_bounded_support(src::MOI.ModelLike, info::ModelInfo)
    for declaration in info.uncertainties
        message = _unbounded_support_message(declaration.distribution)
        if message !== nothing
            names = join((_variable_name(src, v) for v in declaration.variables), ", ")
            error("The uncertainty $names is not bounded. $message")
        end
    end
    return
end

"""
    MOI.optimize!(dest::Optimizer, src::MOI.ModelLike) -> (MOI.Utilities.IndexMap, Bool)

Read the model `src` in place, and solve each of the [`Formulations`](@ref) of `dest` for it.

The results are kept in `dest`. The index map is the identity, and `false` means that `dest` does
not hold a copy of the model, so that the next call reads `src` again.
"""
function MOI.optimize!(dest::Optimizer, src::MOI.ModelLike)
    MOI.empty!(dest)
    start = time_ns()
    factory = _inner_factory(dest)
    info = _model_info(src)
    _check_bounded_support(src, info)
    form = canonicalize(src, info)
    mom = moments(form)
    results = map(dest.formulations) do f
        return solve_formulation(
            form,
            f,
            mom,
            factory;
            silent = dest.silent,
            time_limit = dest.time_limit,
            raw_attributes = dest.raw_attributes,
        )
    end
    dest.form, dest.moments, dest.results = form, mom, results
    dest.parameters = _parameter_values(src)
    dest.decision_rows = Dict(v => i for (i, v) in enumerate(form.decisions))
    dest.uncertainty_columns = Dict(v => 1 + j for (j, v) in enumerate(form.uncertainties))
    dest.solve_time = (time_ns() - start) / 1e9
    return _identity_map(src), false
end
