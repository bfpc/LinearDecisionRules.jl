# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

const _ConstraintTag = Union{AbstractXiRole,Stage}

# A constraint under construction that carries a tag: `@constraint(model, expr, ValidInequality())`.
# JuMP accepts one extra positional argument, so a constraint has one tag.
struct _TaggedConstraint{C<:JuMP.AbstractConstraint,T<:_ConstraintTag} <: JuMP.AbstractConstraint
    constraint::C
    tag::T
end

function _check_tagged_row(error_fn::Function, f, tag)
    if tag isa AbstractXiRole && !(f isa JuMP.GenericAffExpr || f isa JuMP.GenericVariableRef)
        error_fn("A row tagged `$(nameof(typeof(tag)))()` must be affine; got $(typeof(f)).")
    end
    return
end

function JuMP.build_constraint(
    error_fn::Function,
    f::JuMP.AbstractJuMPScalar,
    set::MOI.AbstractScalarSet,
    tag::_ConstraintTag;
    kwargs...,
)
    _check_tagged_row(error_fn, f, tag)
    return _TaggedConstraint(JuMP.build_constraint(error_fn, f, set; kwargs...), tag)
end

# The two-sided syntax `lower <= expr <= upper`.
function JuMP.build_constraint(
    error_fn::Function,
    f::JuMP.AbstractJuMPScalar,
    lower::Real,
    upper::Real,
    tag::_ConstraintTag;
    kwargs...,
)
    _check_tagged_row(error_fn, f, tag)
    return _TaggedConstraint(JuMP.build_constraint(error_fn, f, lower, upper; kwargs...), tag)
end

function JuMP.add_constraint(model::JuMP.GenericModel, c::_TaggedConstraint, name::String = "")
    cref = JuMP.add_constraint(model, c.constraint, name)
    _set_tag(model, cref, c.tag)
    return cref
end

_set_tag(model, cref, tag::AbstractXiRole) = MOI.set(model, XiRole(), cref, tag)

_set_tag(model, cref, tag::Stage) = MOI.set(model, ConstraintStage(), cref, tag.t)

function stage(c::JuMP.ConstraintRef)
    return MOI.get(JuMP.owner_model(c), ConstraintStage(), c)
end

function set_stage(c::JuMP.ConstraintRef, t::Integer)
    MOI.set(JuMP.owner_model(c), ConstraintStage(), c, Stage(t).t)
    return
end

function unset_stage(c::JuMP.ConstraintRef)
    MOI.set(JuMP.owner_model(c), ConstraintStage(), c, nothing)
    return
end
