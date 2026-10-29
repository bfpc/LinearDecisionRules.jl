# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# A constraint under construction that carries a tag: `@constraint(model, expr, ValidInequality())`.
# JuMP accepts one extra positional argument, so a constraint has one tag.
struct _TaggedConstraint{C<:JuMP.AbstractConstraint,T<:AbstractXiRole} <: JuMP.AbstractConstraint
    constraint::C
    tag::T
end

function _check_tagged_row(error_fn::Function, f, tag)
    if !(f isa JuMP.GenericAffExpr || f isa JuMP.GenericVariableRef)
        error_fn("A row tagged `$(nameof(typeof(tag)))()` must be affine; got $(typeof(f)).")
    end
    return
end

function JuMP.build_constraint(
    error_fn::Function,
    f::JuMP.AbstractJuMPScalar,
    set::MOI.AbstractScalarSet,
    tag::AbstractXiRole;
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
    tag::AbstractXiRole;
    kwargs...,
)
    _check_tagged_row(error_fn, f, tag)
    return _TaggedConstraint(JuMP.build_constraint(error_fn, f, lower, upper; kwargs...), tag)
end

# A vector row, such as `[r; d .- μ] in SecondOrderCone()`, describes Ξ only with a role.
function JuMP.build_constraint(
    error_fn::Function,
    f::AbstractVector,
    set::Union{MOI.AbstractVectorSet,JuMP.AbstractVectorSet},
    tag::AbstractXiRole;
    kwargs...,
)
    for fi in f
        fi isa Number || _check_tagged_row(error_fn, fi, tag)
    end
    return _TaggedConstraint(JuMP.build_constraint(error_fn, f, set; kwargs...), tag)
end

function JuMP.add_constraint(model::JuMP.GenericModel, c::_TaggedConstraint, name::String = "")
    cref = JuMP.add_constraint(model, c.constraint, name)
    MOI.set(model, XiRole(), cref, c.tag)
    return cref
end

# A constraint has no stage; without these methods, JuMP would reject the tag less clearly.
function JuMP.build_constraint(
    error_fn::Function,
    f::JuMP.AbstractJuMPScalar,
    set::MOI.AbstractScalarSet,
    tag::Stage;
    kwargs...,
)
    return error_fn(_constraint_stage_message(tag))
end

function JuMP.build_constraint(
    error_fn::Function,
    f::JuMP.AbstractJuMPScalar,
    lower::Real,
    upper::Real,
    tag::Stage;
    kwargs...,
)
    return error_fn(_constraint_stage_message(tag))
end

function JuMP.build_constraint(
    error_fn::Function,
    f::AbstractVector,
    set::Union{MOI.AbstractVectorSet,JuMP.AbstractVectorSet},
    tag::Stage;
    kwargs...,
)
    return error_fn(_constraint_stage_message(tag))
end

function _constraint_stage_message(tag::Stage)
    return "`Stage($(tag.t))` tags variables, not constraints: a constraint holds for every " *
           "outcome, whatever the stages of its variables. Remove the tag."
end
