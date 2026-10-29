# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Names of variables and constraints, and printing of MOI functions and sets, for the messages.

# The name of a variable, or `_[i]` as JuMP prints an anonymous variable.
function _variable_name(src::MOI.ModelLike, v::MOI.VariableIndex)
    if MOI.supports(src, MOI.VariableName(), MOI.VariableIndex)
        name = MOI.get(src, MOI.VariableName(), v)
        isempty(name) || return name
    end
    return "_[$(v.value)]"
end

# The name of a constraint, or the constraint itself, such as `d[1] + d[3] <= 190`.
function _constraint_name(src::MOI.ModelLike, ci::MOI.ConstraintIndex{F,S}) where {F,S}
    if F != MOI.VariableIndex && MOI.supports(src, MOI.ConstraintName(), typeof(ci))
        name = MOI.get(src, MOI.ConstraintName(), ci)
        isempty(name) || return name
    end
    f = MOI.get(src, MOI.ConstraintFunction(), ci)
    return "`$(_set_string(_function_string(src, f), MOI.get(src, MOI.ConstraintSet(), ci)))`"
end

_number_string(x::Real) = isinteger(x) && abs(x) < 1e15 ? string(Int(x)) : string(x)

# `c₁ a₁ + c₂ a₂ - c₃ a₃ + constant` for the `(coefficient, name)` pairs of `terms`.
function _terms_string(terms, constant::Float64)
    terms = collect(terms)
    io = IOBuffer()
    for (k, (c, name)) in enumerate(terms)
        if k > 1
            print(io, c < 0 ? " - " : " + ")
        elseif c < 0
            print(io, "-")
        end
        abs(c) == 1 || print(io, _number_string(abs(c)), " ")
        print(io, name)
    end
    if !iszero(constant) || isempty(terms)
        isempty(terms) || print(io, constant < 0 ? " - " : " + ")
        print(io, _number_string(isempty(terms) ? constant : abs(constant)))
    end
    return String(take!(io))
end

_function_string(src, f::MOI.VariableIndex) = _variable_name(src, f)

function _function_string(src, f::MOI.ScalarAffineFunction{Float64})
    terms = ((t.coefficient, _variable_name(src, t.variable)) for t in f.terms)
    return _terms_string(terms, f.constant)
end

function _function_string(src, f::MOI.ScalarQuadraticFunction{Float64})
    quadratic = map(f.quadratic_terms) do t
        a, b = _variable_name(src, t.variable_1), _variable_name(src, t.variable_2)
        # MOI stores `c x²` as the term `(2c, x, x)`.
        if t.variable_1 == t.variable_2
            return (t.coefficient / 2, "$(a)²")
        end
        return (t.coefficient, "$a * $b")
    end
    affine = ((t.coefficient, _variable_name(src, t.variable)) for t in f.affine_terms)
    return _terms_string(Iterators.flatten((quadratic, affine)), f.constant)
end

function _function_string(src, f::Union{MOI.VectorAffineFunction{Float64},MOI.VectorOfVariables})
    rows = MOI.Utilities.scalarize(convert(MOI.VectorAffineFunction{Float64}, f))
    return "[" * join((_function_string(src, row) for row in rows), ", ") * "]"
end

_function_string(::Any, f) = string(f)

_set_string(f::String, s::MOI.LessThan) = "$f <= $(_number_string(s.upper))"
_set_string(f::String, s::MOI.GreaterThan) = "$f >= $(_number_string(s.lower))"
_set_string(f::String, s::MOI.EqualTo) = "$f == $(_number_string(s.value))"

function _set_string(f::String, s::MOI.Interval)
    return "$(_number_string(s.lower)) <= $f <= $(_number_string(s.upper))"
end

function _set_string(f::String, s::MOI.AbstractVectorSet)
    return "$f in $(nameof(typeof(s)))($(MOI.dimension(s)))"
end

_set_string(f::String, s) = "$f in $s"
