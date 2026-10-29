# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Variables are tagged through JuMP's extension points `build_variable` and `add_variable`, with
# methods that always involve one of our own types (a tag, a wrapper or a set): no type piracy.

# A variable under construction that carries a `Stage`: `@variable(model, x, Stage(t))`. It wraps
# JuMP's own builder, so that the tag composes with bounds, `Int`, `Bin` and `in set`.
struct _StagedVariable{V<:JuMP.AbstractVariable} <: JuMP.AbstractVariable
    variable::V
    stage::Stage
end

# The components of a variable in a vector set, with one stage per component.
struct _StagedVariables{V<:JuMP.AbstractVariable} <: JuMP.AbstractVariable
    variables::V
    stages::Vector{Stage}
end

function JuMP.build_variable(error_fn::Function, info::JuMP.VariableInfo, tag::Stage; kwargs...)
    return _StagedVariable(JuMP.build_variable(error_fn, info; kwargs...), tag)
end

function JuMP.build_variable(error_fn::Function, v::_StagedVariable, set::MOI.AbstractScalarSet)
    return _StagedVariable(JuMP.build_variable(error_fn, v.variable, set), v.stage)
end

function JuMP.build_variable(
    error_fn::Function,
    vs::AbstractArray{<:_StagedVariable},
    set::MOI.AbstractScalarSet,
)
    return JuMP.build_variable.(error_fn, vs, Ref(set))
end

# JuMP has its own sets, converted to MOI ones on creation, such as `Parameter`.
function JuMP.build_variable(error_fn::Function, v::_StagedVariable, set::JuMP.AbstractScalarSet)
    return _StagedVariable(JuMP.build_variable(error_fn, v.variable, set), v.stage)
end

function JuMP.build_variable(
    error_fn::Function,
    vs::AbstractArray{<:_StagedVariable},
    set::JuMP.AbstractScalarSet,
)
    return JuMP.build_variable.(error_fn, vs, Ref(set))
end

function _build_staged_variables(error_fn::Function, vs::Vector{<:_StagedVariable}, set)
    variables = JuMP.build_variable(error_fn, [v.variable for v in vs], set)
    return _StagedVariables(variables, [v.stage for v in vs])
end

function JuMP.build_variable(
    error_fn::Function,
    vs::Vector{<:_StagedVariable},
    set::MOI.AbstractVectorSet,
)
    return _build_staged_variables(error_fn, vs, set)
end

function JuMP.build_variable(
    error_fn::Function,
    vs::Vector{<:_StagedVariable},
    set::JuMP.AbstractVectorSet,
)
    return _build_staged_variables(error_fn, vs, set)
end

function JuMP.add_variable(model::JuMP.GenericModel, v::_StagedVariable, name::String = "")
    x = JuMP.add_variable(model, v.variable, name)
    JuMP.add_constraint(model, JuMP.ScalarConstraint(x, v.stage))
    return x
end

function JuMP.add_variable(
    model::JuMP.GenericModel,
    vs::AbstractArray{<:_StagedVariable},
    names::AbstractArray{<:String},
)
    return JuMP.add_variable.(model, vs, names)
end

function JuMP.add_variable(model::JuMP.GenericModel, v::_StagedVariables, names)
    xs = JuMP.add_variable(model, v.variables, names)
    for (x, tag) in zip(xs, v.stages)
        JuMP.add_constraint(model, JuMP.ScalarConstraint(x, tag))
    end
    return xs
end

# A vector uncertainty in an array that is not a `Vector`, for example `@variable(model, x[R] in
# Uncertainty(law))` with a named range `R` (a `DenseAxisArray`): JuMP builds the variables of a
# vector set from a `Vector` only. The components are those of the array in column-major order,
# and the container keeps the axes of its names.
struct _ArrayOfVariables{V<:JuMP.AbstractVariable} <: JuMP.AbstractVariable
    variables::V
end

function JuMP.build_variable(
    error_fn::Function,
    vs::AbstractArray{<:JuMP.AbstractVariable},
    set::MultivariateUncertainty,
)
    return _ArrayOfVariables(JuMP.build_variable(error_fn, _components(vs), set))
end

function JuMP.build_variable(
    error_fn::Function,
    ::JuMP.Containers.SparseAxisArray,
    ::MultivariateUncertainty,
)
    return error_fn(
        "A vector uncertainty needs an array of variables, such as `x[1:n]` or `x[R]` with a " *
        "range `R`: it cannot be declared over a condition (`x[i in R; cond]`).",
    )
end

_components(a::JuMP.Containers.DenseAxisArray) = vec(a.data)

_components(a::AbstractArray) = vec(a)

function _like(names::JuMP.Containers.DenseAxisArray, xs)
    return JuMP.Containers.DenseAxisArray(reshape(xs, size(names)), axes(names)...)
end

_like(names::AbstractArray, xs) = reshape(xs, size(names))

function JuMP.add_variable(
    model::JuMP.GenericModel,
    v::_ArrayOfVariables,
    names::AbstractArray{<:String},
)
    return _like(names, JuMP.add_variable(model, v.variables, _components(names)))
end

# The stage of a variable is its `VariableIndex`-in-`Stage` constraint, if any. In MOI, the index
# of a `VariableIndex` constraint is the index of its variable.
function _stage_constraint(x::JuMP.GenericVariableRef)
    model = JuMP.owner_model(x)
    ci = MOI.ConstraintIndex{MOI.VariableIndex,Stage}(JuMP.index(x).value)
    if !MOI.is_valid(JuMP.backend(model), ci)
        return nothing
    end
    return JuMP.constraint_ref_with_index(model, ci)
end

"""
    stage(x::JuMP.VariableRef) -> Union{Int,Nothing}

The declared stage of a variable, or `nothing` if it has none (it is unmarked).

See also [`Stage`](@ref), [`set_stage`](@ref), [`unset_stage`](@ref).
"""
function stage(x::JuMP.GenericVariableRef)
    cref = _stage_constraint(x)
    return cref === nothing ? nothing : MOI.get(JuMP.owner_model(x), MOI.ConstraintSet(), cref).t
end

"""
    set_stage(x::JuMP.VariableRef, t::Integer)

Declare or change the stage of a variable.

See also [`Stage`](@ref), [`stage`](@ref), [`unset_stage`](@ref).
"""
function set_stage(x::JuMP.GenericVariableRef, t::Integer)
    model = JuMP.owner_model(x)
    cref = _stage_constraint(x)
    if cref === nothing
        JuMP.add_constraint(model, JuMP.ScalarConstraint(x, Stage(t)))
    else
        MOI.set(model, MOI.ConstraintSet(), cref, Stage(t))
    end
    return
end

"""
    unset_stage(x::JuMP.VariableRef)

Remove the stage of a variable, which becomes unmarked.

See also [`Stage`](@ref), [`stage`](@ref), [`set_stage`](@ref).
"""
function unset_stage(x::JuMP.GenericVariableRef)
    cref = _stage_constraint(x)
    if cref !== nothing
        JuMP.delete(JuMP.owner_model(x), cref)
    end
    return
end
