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

function JuMP.build_variable(
    error_fn::Function,
    vs::Vector{<:_StagedVariable},
    set::MOI.AbstractVectorSet,
)
    variables = JuMP.build_variable(error_fn, [v.variable for v in vs], set)
    return _StagedVariables(variables, [v.stage for v in vs])
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

# The stage of a variable is its `VariableIndex`-in-`Stage` constraint, if any.
function _stage_constraint(x::JuMP.GenericVariableRef)
    model = JuMP.owner_model(x)
    src = JuMP.backend(model)
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex,Stage}())
        if MOI.get(src, MOI.ConstraintFunction(), ci) == JuMP.index(x)
            return JuMP.constraint_ref_with_index(model, ci)
        end
    end
    return nothing
end

"""
    stage(x::JuMP.VariableRef) -> Union{Int,Nothing}
    stage(c::JuMP.ConstraintRef) -> Union{Int,Nothing}

The declared stage of a variable or a constraint, or `nothing` if it has none (it is unmarked).

For a variable, it looks through the `Stage` constraints of the model, so its cost is linear in
their number.

See also [`Stage`](@ref), [`set_stage`](@ref), [`unset_stage`](@ref).
"""
function stage(x::JuMP.GenericVariableRef)
    cref = _stage_constraint(x)
    return cref === nothing ? nothing : MOI.get(JuMP.owner_model(x), MOI.ConstraintSet(), cref).t
end

"""
    set_stage(x::JuMP.VariableRef, t::Integer)
    set_stage(c::JuMP.ConstraintRef, t::Integer)

Declare or change the stage of a variable or a constraint.

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
    unset_stage(c::JuMP.ConstraintRef)

Remove the stage of a variable or a constraint, which becomes unmarked.

See also [`Stage`](@ref), [`stage`](@ref), [`set_stage`](@ref).
"""
function unset_stage(x::JuMP.GenericVariableRef)
    cref = _stage_constraint(x)
    if cref !== nothing
        JuMP.delete(JuMP.owner_model(x), cref)
    end
    return
end
