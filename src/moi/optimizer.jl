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

It supports what [`canonicalize`](@ref) reads, and the sets of the metadata, [`Uncertainty`](@ref)
and [`Stage`](@ref): affine and quadratic rows in `LessThan`, `GreaterThan`, `EqualTo` and
`Interval`; affine vector rows in `Nonnegatives`, `Nonpositives` and `Zeros`, read one output at a
time; affine vector rows in `SecondOrderCone` over the uncertainties, tagged
[`ValidInequality`](@ref) or [`Truncate`](@ref) (they describe the support Ξ); variable bounds,
`Integer`, `ZeroOne` and `Parameter`; an affine or quadratic objective. Other constraints are
rejected when they are added to the model. A product in a row is read only when one of its factors
is a JuMP `Parameter`; any other product is an error of `optimize!` that names the constraint
(random recourse, or a quadratic constraint).

## Optimizer attributes

 * [`InnerOptimizer`](@ref)`()`: the factory of the inner optimizer.
 * [`Formulations`](@ref)`()`: the formulations solved by `optimize!`, `[PrimalLDR(), DualLDR()]`
   by default.
 * `MOI.Silent` and `MOI.RawOptimizerAttribute` are kept and forwarded to the inner optimizers.
 * `MOI.TimeLimitSec` limits the whole `optimize!`, rejection sampling included: each
   formulation, built and solved in turn, gets the time that remains, and a formulation reached
   after the limit has status `TIME_LIMIT`.
 * [`RejectionSamplingSeed`](@ref)`()`, [`RejectionSamplingTimeLimit`](@ref)`()` and
   [`RejectionSamplingMaxSamples`](@ref)`()`: the rejection sampling of the laws conditioned by
   constraints tagged [`Truncate`](@ref), `1234` and `10_000` draws by default. Without
   `RejectionSamplingTimeLimit()`, the sampling gets half of the time that remains of
   `MOI.TimeLimitSec`, shared by the conditioned laws, or 10 seconds per law without
   `MOI.TimeLimitSec`. Its estimates are kept from one `optimize!` to the next while their inputs
   do not change.

## Re-solving with parameters

The inner problems of the last `optimize!` are kept, and the next `optimize!` solves them again,
from the solution of the inner solver, when nothing changed but the values of JuMP parameters
alone in rows or in the objective, such as `buy <= capacity` or `inventory[1] == x0 - d[1]` with
the parameters `capacity` and `x0` (see [`ParameterTerms`](@ref)): the inner problems carry these
parameters, updated with `set_parameter_value`. Any other change (a coefficient, a bound, a
constraint, a law, a formulation or its attributes, the inner optimizer or its attributes, a
parameter multiplied by a variable) builds them again; the comparison reads the whole canonical
form, which is rebuilt at every `optimize!`. Distributions are compared by value and treated as
immutable: replace a law, rather than changing it in place.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    inner_factory::Any
    formulations::Vector{AbstractFormulation}
    silent::Bool
    time_limit::Union{Nothing,Float64}
    raw_attributes::Dict{String,Any}
    name::String
    rejection_sampling::_RejectionSampling
    # The estimates of the moments of the conditioned laws, kept across `empty!`.
    moments_cache::_MomentsCache
    # The inner problems of the last `optimize!`, kept across `empty!` and solved again when only
    # the values of the parameters change.
    inner_cache::_InnerCache
    # The state of the last `optimize!`, read by the results.
    form::Union{Nothing,CanonicalForm}
    moments::Union{Nothing,Moments}
    parameters::Dict{MOI.VariableIndex,Float64}
    decision_rows::Dict{MOI.VariableIndex,Int}
    # The column of each uncertainty in ξ before lifting (see `CanonicalForm`).
    uncertainty_columns::Dict{MOI.VariableIndex,Int}
    results::Vector{FormulationResult}
    solve_time::Float64

    function Optimizer(inner_factory = nothing)
        return new(
            inner_factory,
            AbstractFormulation[PrimalLDR(), DualLDR()],
            false,                            # silent
            nothing,                          # time_limit
            Dict{String,Any}(),               # raw_attributes
            "",                               # name
            _RejectionSampling(1234, nothing, 10_000),
            _MomentsCache(),
            _InnerCache(),
            nothing,                          # form
            nothing,                          # moments
            Dict{MOI.VariableIndex,Float64}(), # parameters
            Dict{MOI.VariableIndex,Int}(),    # decision_rows
            Dict{MOI.VariableIndex,Int}(),    # uncertainty_columns
            FormulationResult[],
            NaN,                              # solve_time
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

# What `canonicalize` reads. Without bridges, anything else is an error when it is added.
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VariableIndex},
    ::Type{
        <:Union{
            _BoundSet,
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

# Quadratic rows are read so that `canonicalize` can replace parameters in products, and name the
# constraint in its error for any other product.
MOI.supports_constraint(::Optimizer, ::Type{<:_ScalarFunction}, ::Type{<:_BoundSet}) = true

MOI.supports_constraint(::Optimizer, ::Type{<:_VectorFunction}, ::Type{<:_VectorSet}) = true

# Second-order cone rows describe Ξ: `canonicalize` requires them over uncertainties, and tagged.
MOI.supports_constraint(::Optimizer, ::Type{<:_VectorFunction}, ::Type{MOI.SecondOrderCone}) = true

function MOI.supports(
    ::Optimizer,
    ::Union{MOI.ConstraintName,XiRole},
    ::Type{<:MOI.ConstraintIndex{<:_VectorFunction,<:Union{_VectorSet,MOI.SecondOrderCone}}},
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

function MOI.supports(
    ::Optimizer,
    ::Union{MOI.ConstraintName,XiRole,ConstraintStage},
    ::Type{<:MOI.ConstraintIndex{<:_ScalarFunction}},
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
        RejectionSamplingSeed,
        RejectionSamplingTimeLimit,
        RejectionSamplingMaxSamples,
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

MOI.get(o::Optimizer, ::RejectionSamplingSeed) = o.rejection_sampling.seed

function MOI.set(o::Optimizer, ::RejectionSamplingSeed, seed::Integer)
    s = o.rejection_sampling
    o.rejection_sampling = _RejectionSampling(seed, s.time_limit, s.max_samples)
    return
end

MOI.get(o::Optimizer, ::RejectionSamplingTimeLimit) = o.rejection_sampling.time_limit

function MOI.set(o::Optimizer, ::RejectionSamplingTimeLimit, limit::Union{Nothing,Real})
    s = o.rejection_sampling
    o.rejection_sampling = _RejectionSampling(s.seed, limit, s.max_samples)
    return
end

MOI.get(o::Optimizer, ::RejectionSamplingMaxSamples) = o.rejection_sampling.max_samples

function MOI.set(o::Optimizer, ::RejectionSamplingMaxSamples, num::Integer)
    s = o.rejection_sampling
    o.rejection_sampling = _RejectionSampling(s.seed, s.time_limit, num)
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

# Whether the inner optimizer supports second-order cone constraints, with the bridges of the inner
# models. It is asked only when a law chooses the representation of its support (see
# `canonicalize`), since an instance of some optimizers costs a license check.
function _conic_support(factory, info::ModelInfo)
    any(d -> _has_auto_support(d.distribution), info.uncertainties) || return true
    inner = MOI.instantiate(factory; with_bridge_type = Float64)
    return MOI.supports_constraint(inner, MOI.VectorAffineFunction{Float64}, MOI.SecondOrderCone)
end

"""
    MOI.optimize!(dest::Optimizer, src::MOI.ModelLike) -> (MOI.Utilities.IndexMap, Bool)

Read the model `src` in place, and solve each of the [`Formulations`](@ref) of `dest` for it.

A law with a choice of representation for its support, such as a [`ConfidenceMvNormal`](@ref)
with `support = :auto`, gets second-order cone rows when the inner optimizer supports them (with
bridges) and no decision is integer, and linear rows otherwise; the inner optimizer is
instantiated to ask, only for such a law.

The results are kept in `dest`. The index map is the identity, and `false` means that `dest` does
not hold a copy of the model, so that the next call reads `src` again.
"""
function MOI.optimize!(dest::Optimizer, src::MOI.ModelLike)
    MOI.empty!(dest)
    start = time_ns()
    factory = _inner_factory(dest)
    info = _model_info(src)
    _check_bounded_support(src, info)
    form = canonicalize(src, info; conic_support = _conic_support(factory, info))
    # Without `RejectionSamplingTimeLimit()`, the sampling gets half of what remains of
    # `TimeLimitSec`, and the formulations the rest.
    budget = nothing
    if dest.time_limit !== nothing
        budget = max(0.0, dest.time_limit - _seconds_since(start)) / 2
    end
    mom = _moments(form, dest.rejection_sampling, dest.moments_cache; budget)
    _check_convex_objective(form)
    _warn_singular_moments(form, mom)
    cache = dest.inner_cache
    raw_attributes = copy(dest.raw_attributes)
    _update!(cache, _Fingerprint(form, mom, factory, dest.silent, raw_attributes))
    # `TimeLimitSec` limits the whole call: each formulation gets the time that remains.
    results = map(dest.formulations) do f
        remaining = nothing
        if dest.time_limit !== nothing
            remaining = dest.time_limit - _seconds_since(start)
            remaining <= 0 && return _time_limit_result(f, form, mom)
        end
        inner = _cached_problem(cache, f)
        if inner === nothing
            silent = dest.silent
            inner = _build_inner(form, f, mom, factory; silent, raw_attributes)
            push!(cache.problems, f => inner)
            cache.num_builds += 1
            if remaining !== nothing
                remaining = max(0.0, dest.time_limit - _seconds_since(start))
            end
        end
        return _solve_inner!(inner, f, form, mom; time_limit = remaining)
    end
    # Keep the inner problems of these formulations only.
    filter!(p -> any(f -> _same_value(f, p.first), dest.formulations), cache.problems)
    dest.form, dest.moments, dest.results = form, mom, results
    dest.parameters = _parameter_values(src)
    dest.decision_rows = Dict(v => i for (i, v) in enumerate(form.decisions))
    uncertainties = form.lifting === nothing ? form.uncertainties : form.lifting.uncertainties
    dest.uncertainty_columns = Dict(v => 1 + j for (j, v) in enumerate(uncertainties))
    dest.solve_time = _seconds_since(start)
    return _identity_map(src), false
end
