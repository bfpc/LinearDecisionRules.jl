# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Declaration-time checks: whatever is local to one `@variable` call is checked in `build_variable`,
# whose `error_fn` names the user's macro call.

function _error_two_stages(error_fn::Function)
    return error_fn("A variable has one `Stage`: remove one of the `Stage` tags.")
end

function JuMP.build_variable(error_fn::Function, ::JuMP.VariableInfo, ::Stage, ::Stage, ::Stage...)
    return _error_two_stages(error_fn)
end

# `@variable(model, x in Stage(1), Stage(2))`
function JuMP.build_variable(error_fn::Function, ::_StagedVariable, ::Stage)
    return _error_two_stages(error_fn)
end

# Declaration-time checks of an uncertainty. They run in `build_variable`, whose `error_fn` names
# the user's macro call. What JuMP lets us intercept is the macro itself: `set_lower_bound`, `fix`,
# `set_integer` and `@constraint(model, d >= 0)` on an uncertainty reach the model without any of
# our methods, so `canonicalize` reports them at `optimize!`.
_variable_info(v::JuMP.ScalarVariable) = v.info

_variable_info(v::_StagedVariable) = _variable_info(v.variable)

_variable_info(::JuMP.AbstractVariable) = nothing

function _check_uncertainty_declaration(error_fn::Function, v::JuMP.AbstractVariable, set)
    info = _variable_info(v)
    if info !== nothing
        for (given, what) in (
            (info.has_lb, "a lower bound"),
            (info.has_ub, "an upper bound"),
            (info.integer, "integrality"),
            (info.binary, "integrality (binary)"),
        )
            if given
                error_fn(
                    "An uncertainty cannot have $what: its support comes from its distribution. " *
                    "To tighten the support, use a constraint tagged `ValidInequality()`.",
                )
            end
        end
    end
    message = _unbounded_support_message(set.distribution)
    if message !== nothing
        error_fn(
            "$message The support of an uncertainty must be bounded: use a distribution with " *
            "finite `minimum` and `maximum`, for example a truncated one.",
        )
    end
    return
end

function JuMP.build_variable(error_fn::Function, v::JuMP.ScalarVariable, set::UnivariateUncertainty)
    _check_uncertainty_declaration(error_fn, v, set)
    return JuMP.VariableConstrainedOnCreation(v, set)
end

function JuMP.build_variable(
    error_fn::Function,
    vs::AbstractArray{<:JuMP.ScalarVariable},
    set::UnivariateUncertainty,
)
    return JuMP.build_variable.(error_fn, vs, Ref(set))
end

function JuMP.build_variable(
    error_fn::Function,
    vs::Vector{<:JuMP.AbstractVariable},
    set::MultivariateUncertainty,
)
    for v in vs
        _check_uncertainty_declaration(error_fn, v, set)
    end
    return JuMP.VariablesConstrainedOnCreation(vs, set)
end

# Disambiguates the generic staged method of `variables.jl`.
function JuMP.build_variable(
    error_fn::Function,
    vs::Vector{<:_StagedVariable},
    set::MultivariateUncertainty,
)
    return _build_staged_variables(error_fn, vs, set)
end
