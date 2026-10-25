# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The constraints that `canonicalize` reads as rows; the optimizer supports exactly these.
const _BoundSet =
    Union{MOI.LessThan{Float64},MOI.GreaterThan{Float64},MOI.EqualTo{Float64},MOI.Interval{Float64}}

const _ScalarFunction =
    Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}}

# Linear vector rows, read one output at a time; second-order cone rows describe Ξ only.
const _VectorFunction = Union{MOI.VectorAffineFunction{Float64},MOI.VectorOfVariables}

const _VectorSet = Union{MOI.Nonnegatives,MOI.Nonpositives,MOI.Zeros}

const _SparseMatrix = SparseArrays.SparseMatrixCSC{Float64,Int}

# The pieces of the cone of a support block: the nonnegative orthant and second-order cones.
const _SupportCone = Union{MOI.Nonnegatives,MOI.SecondOrderCone}

"""
    ConicRow(
        W::SparseMatrixCSC{Float64,Int},
        h::Vector{Float64},
        cone::MOI.SecondOrderCone,
        constraint::MOI.ConstraintIndex,
        conditioning::Bool,
    )

A second-order cone row of the support Ξ in a [`CanonicalForm`](@ref): the uncertainties `η`
satisfy `W η - h ∈ cone`, that is, `‖(W η - h)[2:end]‖ ≤ (W η - h)[1]`.

## Fields

 * `W::SparseMatrixCSC{Float64,Int}`: `MOI.dimension(cone)` rows, one column per uncertainty.
 * `h::Vector{Float64}`
 * `cone::MOI.SecondOrderCone`
 * `constraint::MOI.ConstraintIndex`: the constraint in `MOI.SecondOrderCone`, tagged
   [`ValidInequality`](@ref) or [`Truncate`](@ref), that the row comes from.
 * `conditioning::Bool`: whether that constraint is tagged `Truncate()`, so that it also conditions
   the law of its uncertainties.
"""
struct ConicRow
    W::_SparseMatrix
    h::Vector{Float64}
    cone::MOI.SecondOrderCone
    constraint::MOI.ConstraintIndex
    conditioning::Bool
end

# The η columns where some row of `W` is nonzero.
function _nonzero_columns(W::_SparseMatrix)
    return [j for j in axes(W, 2) if !isempty(SparseArrays.nzrange(W, j))]
end

# How far `v` is from `cone`, in the units of `v`: 0 if `v ∈ cone`.
function _cone_violation(::MOI.SecondOrderCone, v::AbstractVector)
    return max(0.0, LinearAlgebra.norm(view(v, 2:length(v))) - v[1])
end

_cone_violation(::MOI.Nonnegatives, v::AbstractVector) = max(0.0, -minimum(v; init = 0.0))

"""
    UncertaintyBlock(
        distribution::Distributions.Distribution,
        columns::Vector{Int},
        stages::Vector{Int},
    )

One uncertainty declaration of a [`CanonicalForm`](@ref), placed in the columns of ξ.

## Fields

 * `distribution::Distributions.Distribution`: the law of the block.
 * `columns::Vector{Int}`: the column of ξ of each component, in the order of the components
   (not necessarily sorted nor contiguous, since columns are sorted by stage).
 * `stages::Vector{Int}`: the stage of each component.
"""
struct UncertaintyBlock
    distribution::Distributions.Distribution
    columns::Vector{Int}
    stages::Vector{Int}
end

"""
    ParameterTerms

The JuMP parameters (`VariableIndex`-in-`MOI.Parameter`) of a [`CanonicalForm`](@ref), kept
symbolic where they enter affinely: in the constant column of `B_e`, `B_u` and `B_l` (a parameter
alone in a row, such as `sell <= capacity`), and in the constant `f` of the objective. With `p`
the vector of their values,

```math
B_e[:, 1] = b_e + B^p_e p, \\quad B_u[:, 1] = b_u + B^p_u p, \\quad B_l[:, 1] = b_l + B^p_l p,
\\quad f = f_0 + f_p^\\top p,
```

so that a change of `p` changes nothing else, and the inner problems can carry the parameters
(see [`solve_formulation`](@ref)). A parameter anywhere else (multiplied by a decision, an
uncertainty or a parameter, or in a row tagged `ValidInequality()` or `Truncate()`) is replaced by
its value in the other fields of the form.

## Fields

 * `parameters::Vector{MOI.VariableIndex}`: every parameter of the model, in the order of the
   columns below.
 * `values::Vector{Float64}`: their values `p`.
 * `b_e::Vector{Float64}`, `Bp_e::SparseMatrixCSC{Float64,Int}` (one column per parameter),
   and `b_u`, `Bp_u`, `b_l`, `Bp_l`.
 * `f::Float64`, `fp::Vector{Float64}`: `f_0` and `f_p`.
"""
struct ParameterTerms
    parameters::Vector{MOI.VariableIndex}
    values::Vector{Float64}
    b_e::Vector{Float64}
    Bp_e::_SparseMatrix
    b_u::Vector{Float64}
    Bp_u::_SparseMatrix
    b_l::Vector{Float64}
    Bp_l::_SparseMatrix
    f::Float64
    fp::Vector{Float64}
end

function Base.:(==)(a::ParameterTerms, b::ParameterTerms)
    return all(name -> getfield(a, name) == getfield(b, name), fieldnames(ParameterTerms))
end

function Base.hash(p::ParameterTerms, h::UInt)
    return hash(p.parameters, hash(p.values, hash(:ParameterTerms, h)))
end

# Whether some constant of the form depends on a parameter.
function _has_parameter_terms(p::ParameterTerms)
    return SparseArrays.nnz(p.Bp_e) + SparseArrays.nnz(p.Bp_u) + SparseArrays.nnz(p.Bp_l) > 0 ||
           any(!iszero, p.fp)
end

"""
    CanonicalForm

The canonical form of a linear-decision-rule model, built by [`canonicalize`](@ref).

With `n` decisions `x`, `m` uncertainties `η` and `ξ = (1, η)`, the model is
```math
\\begin{aligned}
\\text{opt}\\; & \\mathbb{E}\\left[x^\\top P x + x^\\top C \\xi + \\eta^\\top Q \\eta + d^\\top \\eta + f\\right] \\\\
\\text{s.t.}\\; & A_e x = B_e \\xi, \\quad A_u x \\le B_u \\xi, \\quad A_l x \\ge B_l \\xi, \\\\
& \\ell \\le x \\le u, \\qquad \\forall \\xi \\in \\Xi,
\\end{aligned}
```
where `x = x(ξ)` follows the information structure (see [`InformationStructure`](@ref)). Interval
rows give one row in `A_u` and one in `A_l`. Ξ is described by [`support_blocks`](@ref), and the law
of `η` by `laws` (see [`moments`](@ref)).

## Fields

Decisions (rows of the decision rule `X`, in the order of the model's variables):

 * `decisions::Vector{MOI.VariableIndex}`
 * `decision_names::Vector{String}`: names, or `_[i]` (as JuMP prints them) when unnamed.
 * `decision_stages::Vector{Int}`
 * `integer::BitVector`: `MOI.Integer` or `MOI.ZeroOne` decisions (all in stage 1).
 * `binary::BitVector`: `MOI.ZeroOne` decisions; their bounds are not tightened to `[0, 1]`.
 * `lower::Vector{Float64}`, `upper::Vector{Float64}`: bounds, `±Inf` when absent.

Uncertainties (`uncertainties[j]` is column `1 + j` of ξ; columns are sorted by stage):

 * `uncertainties::Vector{MOI.VariableIndex}`
 * `uncertainty_names::Vector{String}`
 * `uncertainty_stages::Vector{Int}`: nondecreasing.
 * `laws::Vector{UncertaintyBlock}`: one per declaration, except that the declarations coupled by
   rows tagged `Truncate()` share one; separate laws are independent.
 * `num_stages::Int`: the number of stages `T ≥ 2`.

Rows (`B_*` have `1 + m` columns; `*_constraints[r]` is the constraint that row `r` comes from):

 * `A_e::SparseMatrixCSC{Float64,Int}`, `B_e::SparseMatrixCSC{Float64,Int}`,
   `e_constraints::Vector{MOI.ConstraintIndex}`
 * `A_u`, `B_u`, `u_constraints`
 * `A_l`, `B_l`, `l_constraints`

Rows of Ξ beyond the boxes of the laws, as `W_valid η ≥ h_valid` (`W_valid` has `m` columns, over
η): the rows tagged `ValidInequality()` or `Truncate()`, then, in a lifted form, the convex-hull
rows of each lifting, whose constraint is `nothing`:

 * `W_valid::SparseMatrixCSC{Float64,Int}`, `h_valid::Vector{Float64}`,
   `valid_constraints::Vector{Union{Nothing,MOI.ConstraintIndex}}`
 * `conditioning::BitVector`: whether each row is tagged `Truncate()`. Such a row bounds Ξ and
   conditions the law of its uncertainties: their laws are merged into one
   [`ConditionedDistribution`](@ref) in `laws`, or into an exact `MvDiscreteNonParametric` when
   they have finitely many atoms. The rows tagged `Truncate()` that only bound a
   univariate law are not rows: that law is truncated exactly instead (see [`canonicalize`](@ref)).
 * `conic_rows::Vector{ConicRow}`: the constraints in `MOI.SecondOrderCone` tagged
   `ValidInequality()` or `Truncate()`, as [`ConicRow`](@ref)s.

Objective (`FEASIBILITY_SENSE` when the model has none; then every term is zero):

 * `sense::MOI.OptimizationSense`
 * `P::SparseMatrixCSC{Float64,Int}`: `n × n`, symmetric.
 * `C::SparseMatrixCSC{Float64,Int}`: `n × (1 + m)`; column 1 has the linear costs, the other
   columns the products of decisions and uncertainties.
 * `Q::SparseMatrixCSC{Float64,Int}`: `m × m`, symmetric.
 * `d::Vector{Float64}`, `f::Float64`.

Lifting (see [`lift`](@ref)`(form, liftings)`):

 * `lifting::Union{Nothing,LiftingMap}`: `nothing`, or, in a lifted form, the [`LiftingMap`](@ref)
   from the original uncertainties to the columns of ξ. The columns of a lifted form are the
   lifted coordinates: `uncertainties[j]` is the original uncertainty of column `1 + j`, and its
   name is followed by `[piece k]`.

Parameters:

 * `parameters::ParameterTerms`: the JuMP parameters of the model and how the constant columns of
   `B_e`, `B_u`, `B_l` and the constant `f` depend on them (see [`ParameterTerms`](@ref)). The
   matrices and `f` above hold their values at the current values of the parameters.
"""
struct CanonicalForm
    decisions::Vector{MOI.VariableIndex}
    decision_names::Vector{String}
    decision_stages::Vector{Int}
    integer::BitVector
    binary::BitVector
    lower::Vector{Float64}
    upper::Vector{Float64}
    uncertainties::Vector{MOI.VariableIndex}
    uncertainty_names::Vector{String}
    uncertainty_stages::Vector{Int}
    laws::Vector{UncertaintyBlock}
    num_stages::Int
    A_e::_SparseMatrix
    B_e::_SparseMatrix
    e_constraints::Vector{MOI.ConstraintIndex}
    A_u::_SparseMatrix
    B_u::_SparseMatrix
    u_constraints::Vector{MOI.ConstraintIndex}
    A_l::_SparseMatrix
    B_l::_SparseMatrix
    l_constraints::Vector{MOI.ConstraintIndex}
    W_valid::_SparseMatrix
    h_valid::Vector{Float64}
    valid_constraints::Vector{Union{Nothing,MOI.ConstraintIndex}}
    conditioning::BitVector
    conic_rows::Vector{ConicRow}
    sense::MOI.OptimizationSense
    P::_SparseMatrix
    C::_SparseMatrix
    Q::_SparseMatrix
    d::Vector{Float64}
    f::Float64
    lifting::Union{Nothing,LiftingMap}
    parameters::ParameterTerms
end

# `form` with some fields replaced, such as `_replace(form; laws = laws)`.
function _replace(form::CanonicalForm; kwargs...)
    fields = map(name -> get(kwargs, name, getfield(form, name)), fieldnames(CanonicalForm))
    return CanonicalForm(fields...)
end

function Base.show(io::IO, form::CanonicalForm)
    print(io, "CanonicalForm with $(length(form.decisions)) decisions, ")
    print(io, "$(length(form.uncertainties)) uncertainties, $(form.num_stages) stages and ")
    rows = size(form.A_e, 1) + size(form.A_u, 1) + size(form.A_l, 1)
    print(io, "$rows rows")
    return
end

"""
    _is_metadata_constraint(::Type{F}, ::Type{S}) -> Bool

Return `true` if [`canonicalize`](@ref) must skip the `F`-in-`S` constraints because they carry
metadata (such as stages or uncertainty sets) that the front end already put in the
[`ModelInfo`](@ref).
"""
_is_metadata_constraint(::Type, ::Type) = false

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

_bounds(s::MOI.LessThan{Float64}) = (-Inf, s.upper)
_bounds(s::MOI.GreaterThan{Float64}) = (s.lower, Inf)
_bounds(s::MOI.EqualTo{Float64}) = (s.value, s.value)
_bounds(s::MOI.Interval{Float64}) = (s.lower, s.upper)

_scalar_set(::Type{MOI.Nonnegatives}) = MOI.GreaterThan(0.0)
_scalar_set(::Type{MOI.Nonpositives}) = MOI.LessThan(0.0)
_scalar_set(::Type{MOI.Zeros}) = MOI.EqualTo(0.0)

_dimension(::Distributions.UnivariateDistribution) = 1
_dimension(d::Distributions.MultivariateDistribution) = length(d)
_dimension(::Distributions.Distribution) = 0

# Coordinate lists of a sparse matrix.
struct _Triplets
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{Float64}
end

_Triplets() = _Triplets(Int[], Int[], Float64[])

function _push!(t::_Triplets, i::Int, j::Int, v::Float64)
    if !iszero(v)
        push!(t.I, i)
        push!(t.J, j)
        push!(t.V, v)
    end
    return
end

_sparse(t::_Triplets, m::Int, n::Int) = SparseArrays.sparse(t.I, t.J, t.V, m, n)

# The role of every variable of the model, as `canonicalize` sees it.
struct _Columns
    decision::Dict{MOI.VariableIndex,Int}    # row of the decision rule
    uncertainty::Dict{MOI.VariableIndex,Int} # column of ξ, ≥ 2
    parameter::Dict{MOI.VariableIndex,Float64}
    parameter_column::Dict{MOI.VariableIndex,Int} # column of `ParameterTerms`
    parameter_values::Vector{Float64}
end

# A function is read into an accumulator, a `_Row` or the `_Objective`, term by term, by the role
# of each variable: `_add_decision!`, `_add_uncertainty!` (by column of ξ), `_add_parameter!` (a
# parameter alone, kept symbolic, by column of `ParameterTerms`), `_add_constant!`, and
# `_add_bilinear!` for a product of two variables that are not parameters.

function _add_term!(acc, cols::_Columns, coef::Float64, v::MOI.VariableIndex)
    if haskey(cols.decision, v)
        _add_decision!(acc, cols.decision[v], coef)
    elseif haskey(cols.uncertainty, v)
        _add_uncertainty!(acc, cols.uncertainty[v], coef)
    else
        _add_parameter!(acc, cols.parameter_column[v], coef)
    end
    return
end

# `coef * a * b`, with parameters replaced by their values.
function _add_product!(
    acc,
    cols::_Columns,
    coef::Float64,
    a::MOI.VariableIndex,
    b::MOI.VariableIndex,
)
    if haskey(cols.parameter, a) && haskey(cols.parameter, b)
        _add_constant!(acc, coef * cols.parameter[a] * cols.parameter[b])
    elseif haskey(cols.parameter, a)
        _add_term!(acc, cols, coef * cols.parameter[a], b)
    elseif haskey(cols.parameter, b)
        _add_term!(acc, cols, coef * cols.parameter[b], a)
    else
        _add_bilinear!(acc, cols, coef, a, b)
    end
    return
end

function _read!(acc, cols::_Columns, f::MOI.ScalarAffineFunction{Float64})
    for t in f.terms
        _add_term!(acc, cols, t.coefficient, t.variable)
    end
    _add_constant!(acc, f.constant)
    return acc
end

function _read!(acc, cols::_Columns, f::MOI.ScalarQuadraticFunction{Float64})
    for t in f.affine_terms
        _add_term!(acc, cols, t.coefficient, t.variable)
    end
    for t in f.quadratic_terms
        # MOI stores `c x²` as the term `(2c, x, x)`; `coef` is the coefficient of `a * b`.
        a, b = t.variable_1, t.variable_2
        _add_product!(acc, cols, a == b ? t.coefficient / 2 : t.coefficient, a, b)
    end
    _add_constant!(acc, f.constant)
    return acc
end

# One constraint, read as `x-terms + ξ-terms + parameter-terms + constant`.
mutable struct _Row
    x::Dict{Int,Float64}
    ξ::Dict{Int,Float64}
    parameters::Dict{Int,Float64} # by column of `ParameterTerms`
    constant::Float64
    recourse::Union{Nothing,Tuple{MOI.VariableIndex,MOI.VariableIndex}} # (decision, uncertainty)
    quadratic::Union{Nothing,Tuple{MOI.VariableIndex,MOI.VariableIndex}}
end

function _Row()
    coefficients() = Dict{Int,Float64}()
    return _Row(coefficients(), coefficients(), coefficients(), 0.0, nothing, nothing)
end

function _reset!(row::_Row)
    empty!(row.x)
    empty!(row.ξ)
    empty!(row.parameters)
    row.constant = 0.0
    row.recourse = nothing
    row.quadratic = nothing
    return row
end

function _add_parameter!(row::_Row, k::Int, coef::Float64)
    row.parameters[k] = get(row.parameters, k, 0.0) + coef
    return
end

# The value of the parameter terms of `row`.
function _parameter_value(row::_Row, cols::_Columns)
    value = 0.0
    for (k, c) in row.parameters
        value += c * cols.parameter_values[k]
    end
    return value
end

function _add_decision!(row::_Row, i::Int, coef::Float64)
    row.x[i] = get(row.x, i, 0.0) + coef
    return
end

function _add_uncertainty!(row::_Row, j::Int, coef::Float64)
    row.ξ[j] = get(row.ξ, j, 0.0) + coef
    return
end

function _add_constant!(row::_Row, coef::Float64)
    row.constant += coef
    return
end

# A product in a row is random recourse or a quadratic constraint; `_check_row` reports both.
function _add_bilinear!(row::_Row, cols::_Columns, ::Float64, a, b)
    if haskey(cols.decision, a) && haskey(cols.uncertainty, b)
        row.recourse = (a, b)
    elseif haskey(cols.uncertainty, a) && haskey(cols.decision, b)
        row.recourse = (b, a)
    else
        row.quadratic = (a, b)
    end
    return
end

# The objective `x'Px + x'Cξ + η'Qη + d'η + f + fp'p`, with `P` and `Q` symmetric.
mutable struct _Objective
    P::_Triplets
    C::_Triplets
    Q::_Triplets
    d::Vector{Float64}
    f::Float64
    fp::Vector{Float64}
end

function _Objective(m::Int, num_parameters::Int)
    return _Objective(_Triplets(), _Triplets(), _Triplets(), zeros(m), 0.0, zeros(num_parameters))
end

function _add_parameter!(obj::_Objective, k::Int, coef::Float64)
    obj.fp[k] += coef
    return
end

function _add_decision!(obj::_Objective, i::Int, coef::Float64)
    _push!(obj.C, i, 1, coef)
    return
end

function _add_uncertainty!(obj::_Objective, j::Int, coef::Float64)
    obj.d[j-1] += coef
    return
end

function _add_constant!(obj::_Objective, coef::Float64)
    obj.f += coef
    return
end

function _add_bilinear!(obj::_Objective, cols::_Columns, coef::Float64, a, b)
    if haskey(cols.decision, a) && haskey(cols.decision, b)
        i, k = cols.decision[a], cols.decision[b]
        _push!(obj.P, i, k, coef / 2)
        _push!(obj.P, k, i, coef / 2)
    elseif haskey(cols.uncertainty, a) && haskey(cols.uncertainty, b)
        i, k = cols.uncertainty[a] - 1, cols.uncertainty[b] - 1
        _push!(obj.Q, i, k, coef / 2)
        _push!(obj.Q, k, i, coef / 2)
    elseif haskey(cols.decision, a)
        _push!(obj.C, cols.decision[a], cols.uncertainty[b], coef)
    else
        _push!(obj.C, cols.decision[b], cols.uncertainty[a], coef)
    end
    return
end

# The rows `A x (=, ≤, ≥) B ξ` of one kind, with `B[:, 1] = b + Bp p` (see `ParameterTerms`).
struct _RowBlock
    A::_Triplets
    B::_Triplets
    constraints::Vector{MOI.ConstraintIndex}
    b::Vector{Float64}
    Bp::_Triplets
end

_RowBlock() = _RowBlock(_Triplets(), _Triplets(), MOI.ConstraintIndex[], Float64[], _Triplets())

# Push `row.x ⋅ x + row.ξ ⋅ η + row.parameters ⋅ p (sense) rhs` as `A x (sense) B ξ`, with
# `B ξ = rhs − row.parameters ⋅ p − row.ξ ⋅ η`; `value` is `row.parameters ⋅ p`.
function _push_row!(
    block::_RowBlock,
    row::_Row,
    rhs::Float64,
    value::Float64,
    ci::MOI.ConstraintIndex,
)
    r = length(block.constraints) + 1
    for (i, a) in row.x
        _push!(block.A, r, i, a)
    end
    _push!(block.B, r, 1, rhs - value)
    for (j, b) in row.ξ
        _push!(block.B, r, j, -b)
    end
    push!(block.b, rhs)
    for (k, c) in row.parameters
        _push!(block.Bp, r, k, -c)
    end
    push!(block.constraints, ci)
    return
end

# The rows `W η ≥ h` tagged `ValidInequality()` or `Truncate()` (then `conditioning`).
struct _ValidRows
    W::_Triplets
    h::Vector{Float64}
    constraints::Vector{MOI.ConstraintIndex}
    conditioning::BitVector
end

_ValidRows() = _ValidRows(_Triplets(), Float64[], MOI.ConstraintIndex[], falses(0))

# The decisions and the uncertainties of the model, in the order of the canonical form.
struct _Layout
    decisions::Vector{MOI.VariableIndex}
    decision_stages::Vector{Int}
    uncertainties::Vector{MOI.VariableIndex}
    uncertainty_stages::Vector{Int}
    laws::Vector{UncertaintyBlock}
    num_stages::Int
    columns::_Columns
    parameters::Vector{MOI.VariableIndex}
end

# What the constraints give, read by `_read_constraints`.
struct _Constraints
    e::_RowBlock
    u::_RowBlock
    l::_RowBlock
    valid::_ValidRows
    conic_rows::Vector{ConicRow}
    lower::Vector{Float64}
    upper::Vector{Float64}
    integer::BitVector
    binary::BitVector
    buffer::_Row
end

function _Constraints(n::Int)
    return _Constraints(
        _RowBlock(),
        _RowBlock(),
        _RowBlock(),
        _ValidRows(),
        ConicRow[],
        fill(-Inf, n),
        fill(Inf, n),
        falses(n),
        falses(n),
        _Row(),
    )
end

"""
    canonicalize(
        src::MOI.ModelLike,
        info::ModelInfo;
        conic_support::Bool = true,
    ) -> CanonicalForm

Read the model `src` and its metadata `info` into a [`CanonicalForm`](@ref).

`src` is only read, so it can be `JuMP.backend(model)` or any MOI model. JuMP parameters
(`VariableIndex`-in-`Parameter`) alone in a row or in the objective are kept symbolic in the
constant column of `B` and in `f` (see [`ParameterTerms`](@ref)), and the inner problems carry
them; elsewhere, multiplied by a variable or in a row tagged `ValidInequality()` or `Truncate()`,
they are replaced by their values.

The number of stages is `T = max(2, largest declared stage)`; unmarked variables are in stage `T`,
and a single warning lists them when `T > 2`.

The rows tagged `Truncate()` condition the law on the set `C` where they hold, and bound Ξ:

 * a univariate law whose rows involve it alone, such as `100 ≤ d ≤ 110`, is truncated exactly:
   `Distributions.truncated(d, lo, hi)` for a continuous law, or the atoms in `[lo, hi]`, as a
   `DiscreteNonParametric`, for a discrete law with a finite support. Its rows are then dropped,
   since the box of the truncated law replaces them;
 * otherwise, the declarations coupled by rows tagged `Truncate()` (connected components, by
   union–find) share one law, in the place of the first. When all their laws have finitely many
   atoms, and their product at most a million, it is exact: a [`MvDiscreteNonParametric`](@ref)
   on the atoms of the product that satisfy the rows, with their probabilities renormalized (a
   declared [`ConditionedDistribution`](@ref) of such laws is replaced in the same way). Otherwise
   it is a `ConditionedDistribution`, whose moments are estimated by rejection sampling (see
   [`moments`](@ref)). The rows stay in Ξ.

Constraints in `MOI.SecondOrderCone` describe Ξ only: they must be over uncertainties and tagged
`ValidInequality()` or `Truncate()`, and they become [`ConicRow`](@ref)s. A law conditioned by such
a constraint is estimated by rejection sampling, unless its laws have finitely many atoms.

A law whose representation of the support is chosen automatically, such as a
[`ConfidenceMvNormal`](@ref) with `support = :auto`, gets second-order cone rows (see
[`support_constraints`](@ref)) when `conic_support` is `true` and no decision is integer, and
linear rows otherwise; the optimizer sets `conic_support` to whether its inner optimizer supports
second-order cone constraints.

Errors name the variables and constraints involved, for: unsupported constraint types, uncertainty-
only rows not tagged `ValidInequality()` or `Truncate()`, second-order cone constraints with
decisions or without a tag, tagged equalities (Ξ must be full-dimensional), a truncation of
probability zero, bounds or integrality on an uncertainty, integer decisions after stage 1,
products of uncertainties and decisions in constraints (random recourse), quadratic constraints,
and equality rows with an uncertainty that none of their decisions observe.

## Example

```julia
model = Model()
@variable(model, buy >= 0)
@variable(model, sell >= 0)
@variable(model, d)
@constraint(model, sell <= buy)
@constraint(model, sell <= d)
@objective(model, Max, -10buy + 15sell)
info = ModelInfo(;
    uncertainties = [UncertaintyDeclaration(Uniform(80, 120), [index(d)])],
    decision_stages = Dict(index(buy) => 1),
)
form = canonicalize(backend(model), info)
```
"""
function canonicalize(src::MOI.ModelLike, info::ModelInfo; conic_support::Bool = true)
    variables = MOI.get(src, MOI.ListOfVariableIndices())
    parameters, values = _parameter_list(src)
    parameter = Dict(zip(parameters, values))
    entries = _check_declarations(src, info, variables, parameter)
    layout = _layout(src, info, variables, parameters, values, entries)
    constraints = _read_constraints(src, info, layout)
    sense, objective = _read_objective(src, layout)
    form = _canonical_form(src, layout, constraints, sense, objective)
    form = _condition(src, _resolve_supports(form, conic_support))
    _check_valid_rows_at_mean(src, form)
    _check_observable_equalities(src, form)
    liftings = _breakpoint_liftings(src, form, info.breakpoints)
    return isempty(liftings) ? form : lift(form, liftings)
end

# The laws with the representation of their support chosen: second-order cones if the inner
# optimizer supports them, unless a decision is integer, since a mixed-integer second-order cone
# program is much harder than a mixed-integer linear one.
function _resolve_supports(form::CanonicalForm, conic_support::Bool)
    any(law -> _has_auto_support(law.distribution), form.laws) || return form
    conic = conic_support && !any(form.integer)
    laws = map(form.laws) do law
        d = _resolve_support(law.distribution, conic)
        return UncertaintyBlock(d, law.columns, law.stages)
    end
    return _replace(form; laws)
end

# The `AxisPiecewise` lifting of each uncertainty with breakpoints: a vector of breakpoints, or a
# number `n` of equally spaced breakpoints inside the support. No breakpoint means no lifting.
function _breakpoint_liftings(src, form::CanonicalForm, breakpoints)
    liftings = Dict{MOI.VariableIndex,AbstractLifting}()
    columns = Dict(v => 1 + j for (j, v) in enumerate(form.uncertainties))
    for (v, points) in breakpoints
        if !haskey(columns, v)
            error(
                "The variable $(_variable_name(src, v)) has breakpoints, but it is not an " *
                "uncertainty.",
            )
        elseif points isa Integer
            if points < 0
                error(
                    "The uncertainty $(_variable_name(src, v)) has $points breakpoints; the " *
                    "number of breakpoints must be nonnegative.",
                )
            end
            points == 0 && continue
            law = only(l for l in form.laws if columns[v] in l.columns)
            lo, hi = _support_box(law.distribution)
            k = findfirst(==(columns[v]), law.columns)
            if !(isfinite(lo[k]) && isfinite(hi[k]))
                error(
                    "The uncertainty $(_variable_name(src, v)) has $points breakpoints, but its " *
                    "support is unbounded.",
                )
            end
            points = collect(range(lo[k], hi[k]; length = points + 2)[2:end-1])
        end
        isempty(points) && continue
        liftings[v] = _named_lifting_error(_variable_name(src, v)) do
            return AxisPiecewise(points)
        end
    end
    return liftings
end

# Name the uncertainty in the `ArgumentError` of a lifting.
function _named_lifting_error(f::Function, name::String)
    try
        return f()
    catch err
        err isa ArgumentError || rethrow()
        error("The lifting of the uncertainty $name is invalid: $(err.msg)")
    end
end

# The parameters of the model and their values, in the order of their constraints.
function _parameter_list(src::MOI.ModelLike)
    parameters, values = MOI.VariableIndex[], Float64[]
    S = MOI.Parameter{Float64}
    if (MOI.VariableIndex, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex,S}())
            push!(parameters, MOI.get(src, MOI.ConstraintFunction(), ci))
            push!(values, MOI.get(src, MOI.ConstraintSet(), ci).value)
        end
    end
    return parameters, values
end

_parameter_values(src::MOI.ModelLike) = Dict(zip(_parameter_list(src)...))

# Check the declarations of `info` against the model, and return the components of the
# uncertainty declarations, in order, with their declared stage.
function _check_declarations(src, info::ModelInfo, variables, parameter)
    in_model = Set(variables)
    entries = Tuple{MOI.VariableIndex,Union{Nothing,Int}}[]
    seen = Set{MOI.VariableIndex}()
    for decl in info.uncertainties
        dim = _dimension(decl.distribution)
        who = join((_variable_name(src, v) for v in decl.variables), ", ")
        if dim == 0
            error(
                "The distribution of the uncertainty $who must be univariate or multivariate; " *
                "got $(typeof(decl.distribution)).",
            )
        elseif !(length(decl.variables) == length(decl.stages) == dim)
            error(
                "The distribution of the uncertainty $who has dimension $dim, but " *
                "$(length(decl.variables)) variables and $(length(decl.stages)) stages were given.",
            )
        end
        for (v, s) in zip(decl.variables, decl.stages)
            if !(v in in_model)
                error("The uncertainty with index $(v.value) is not a variable of the model.")
            elseif haskey(parameter, v)
                error("The variable $(_variable_name(src, v)) is a Parameter and an uncertainty.")
            elseif v in seen
                error("The variable $(_variable_name(src, v)) is declared as an uncertainty twice.")
            elseif s !== nothing && s < 1
                error("The uncertainty $(_variable_name(src, v)) has Stage($s); stages start at 1.")
            end
            push!(seen, v)
            push!(entries, (v, s))
        end
    end
    for (v, s) in info.decision_stages
        if !(v in in_model)
            error("The variable with index $(v.value) has a stage, but it is not in the model.")
        elseif v in seen
            error("The uncertainty $(_variable_name(src, v)) has a decision stage.")
        elseif s < 1
            error("The decision $(_variable_name(src, v)) has Stage($s); stages start at 1.")
        end
    end
    return entries
end

# The columns of ξ are the uncertainties in declaration order, stably sorted by stage; the
# decisions are the other variables, in the order of the model.
function _layout(src, info::ModelInfo, variables, parameters, values, entries)
    parameter = Dict(zip(parameters, values))
    T = 2
    for (v, s) in info.decision_stages
        haskey(parameter, v) || (T = max(T, s))
    end
    for (_, s) in entries
        s === nothing || (T = max(T, s))
    end
    stage_of_entry = [something(s, T) for (_, s) in entries]
    order = sortperm(stage_of_entry; alg = Base.Sort.DEFAULT_STABLE)
    uncertainties = [entries[k][1] for k in order]
    uncertainty_stages = stage_of_entry[order]
    ucol = Dict(v => 1 + j for (j, v) in enumerate(uncertainties))
    laws = map(info.uncertainties) do decl
        cols = [ucol[v] for v in decl.variables]
        return UncertaintyBlock(decl.distribution, cols, uncertainty_stages[cols.-1])
    end
    decisions = [v for v in variables if !haskey(ucol, v) && !haskey(parameter, v)]
    decision_stages = [get(info.decision_stages, v, T) for v in decisions]
    if T > 2
        _warn_unmarked(src, T, info, decisions, entries)
    end
    parameter_column = Dict(v => k for (k, v) in enumerate(parameters))
    decision_row = Dict(v => i for (i, v) in enumerate(decisions))
    cols = _Columns(decision_row, ucol, parameter, parameter_column, values)
    return _Layout(
        decisions,
        decision_stages,
        uncertainties,
        uncertainty_stages,
        laws,
        T,
        cols,
        parameters,
    )
end

function _warn_unmarked(src, T, info, decisions, entries)
    unmarked = MOI.VariableIndex[v for v in decisions if !haskey(info.decision_stages, v)]
    append!(unmarked, v for (v, s) in entries if s === nothing)
    isempty(unmarked) && return
    sort!(unmarked; by = v -> v.value)
    shown = join((_variable_name(src, v) for v in first(unmarked, 10)), ", ")
    more = length(unmarked) > 10 ? " and $(length(unmarked) - 10) more" : ""
    @warn(
        "The model has $T stages; the variables without a stage are in the last one, " *
        "Stage($T): $shown$more.",
    )
    return
end

function _read_constraints(src::MOI.ModelLike, info::ModelInfo, layout::_Layout)
    c = _Constraints(length(layout.decisions))
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        if _is_metadata_constraint(F, S) || (F == MOI.VariableIndex && S == MOI.Parameter{Float64})
            continue
        end
        # A function barrier: each method reads the constraints of one concrete type.
        _read_constraints!(c, src, info, layout, F, S)
    end
    return c
end

function _read_constraints!(c, src, info, layout, ::Type{F}, ::Type{S}) where {F,S}
    return error("Constraints of type $F-in-$S are not supported by LinearDecisionRules.")
end

function _read_constraints!(
    c::_Constraints,
    src::MOI.ModelLike,
    info::ModelInfo,
    layout::_Layout,
    ::Type{F},
    ::Type{S},
) where {F<:_ScalarFunction,S<:_BoundSet}
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
        f = MOI.get(src, MOI.ConstraintFunction(), ci)
        _read_row!(c, src, info, layout, ci, f, MOI.get(src, MOI.ConstraintSet(), ci))
    end
    return
end

# Each output of a vector row is one scalar row of the same constraint.
function _read_constraints!(
    c::_Constraints,
    src::MOI.ModelLike,
    info::ModelInfo,
    layout::_Layout,
    ::Type{F},
    ::Type{S},
) where {F<:_VectorFunction,S<:_VectorSet}
    set = _scalar_set(S)
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
        f = MOI.get(src, MOI.ConstraintFunction(), ci)
        for fi in MOI.Utilities.scalarize(convert(MOI.VectorAffineFunction{Float64}, f))
            _read_row!(c, src, info, layout, ci, fi, set)
        end
    end
    return
end

# A second-order cone constraint describes Ξ, so it must be over uncertainties and tagged.
function _read_constraints!(
    c::_Constraints,
    src::MOI.ModelLike,
    info::ModelInfo,
    layout::_Layout,
    ::Type{F},
    ::Type{MOI.SecondOrderCone},
) where {F<:_VectorFunction}
    m = length(layout.uncertainties)
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{F,MOI.SecondOrderCone}())
        f = MOI.get(src, MOI.ConstraintFunction(), ci)
        cone = MOI.get(src, MOI.ConstraintSet(), ci)
        conditioning = ci in info.truncate_rows
        tagged = conditioning || ci in info.valid_rows
        W, h = _Triplets(), Float64[]
        has_decisions = false
        outputs = MOI.Utilities.scalarize(convert(MOI.VectorAffineFunction{Float64}, f))
        for (k, fk) in enumerate(outputs)
            row = _read!(_reset!(c.buffer), layout.columns, fk)
            has_decisions |= any(!iszero, values(row.x))
            for (j, w) in row.ξ
                _push!(W, k, j - 1, w)
            end
            push!(h, -row.constant - _parameter_value(row, layout.columns))
        end
        name = _constraint_name(src, ci)
        if has_decisions
            error(
                "The constraint $name in a second-order cone has decisions; second-order cone " *
                "constraints describe the support of the uncertainties only, and must be tagged " *
                "`ValidInequality()` or `Truncate()`.",
            )
        elseif !tagged
            error(
                "The constraint $name in a second-order cone involves only uncertainties. Tag it " *
                "with `ValidInequality()` if it describes the support Ξ without changing the " *
                "law, or with `Truncate()` if it conditions the law.",
            )
        elseif conditioning && isempty(W.V)
            error("The constraint $name is tagged `Truncate()`, but it has no uncertainty.")
        end
        push!(c.conic_rows, ConicRow(_sparse(W, length(h), m), h, cone, ci, conditioning))
    end
    return
end

# Bounds and integrality of the decisions.
function _read_constraints!(
    c::_Constraints,
    src::MOI.ModelLike,
    info::ModelInfo,
    layout::_Layout,
    ::Type{MOI.VariableIndex},
    ::Type{S},
) where {S<:Union{_BoundSet,MOI.Integer,MOI.ZeroOne}}
    cols = layout.columns
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex,S}())
        v = MOI.get(src, MOI.ConstraintFunction(), ci)
        if S <: _BoundSet && (ci in info.valid_rows || ci in info.truncate_rows)
            # `@constraint(model, d in MOI.LessThan(110.0), ValidInequality())` is a row.
            f = convert(MOI.ScalarAffineFunction{Float64}, v)
            _read_row!(c, src, info, layout, ci, f, MOI.get(src, MOI.ConstraintSet(), ci))
            continue
        elseif haskey(cols.parameter, v)
            error("The Parameter $(_variable_name(src, v)) has a constraint ($S).")
        elseif !haskey(cols.decision, v)
            error(
                "The uncertainty $(_variable_name(src, v)) has a bound or integrality " *
                "constraint ($S). The support of an uncertainty comes from its " *
                "distribution; use a constraint tagged `ValidInequality()` to describe it.",
            )
        end
        i = cols.decision[v]
        if S <: _BoundSet
            lo, hi = _bounds(MOI.get(src, MOI.ConstraintSet(), ci))
            c.lower[i], c.upper[i] = max(c.lower[i], lo), min(c.upper[i], hi)
            continue
        end
        c.integer[i] = true
        c.binary[i] |= S == MOI.ZeroOne
        t = layout.decision_stages[i]
        if t > 1
            unmarked = haskey(info.decision_stages, v) ? "" : " (it is unmarked)"
            error(
                "The integer decision $(_variable_name(src, v)) is in Stage($t)$unmarked; " *
                "integer decisions are supported only in Stage(1).",
            )
        end
    end
    return
end

# Read the constraint `ci`, `f`-in-`set`, into the rows of its kind.
function _read_row!(c::_Constraints, src, info::ModelInfo, layout::_Layout, ci, f, set)
    row = _read!(_reset!(c.buffer), layout.columns, f)
    # `lo` and `hi` without the parameters; only the rows of `A` keep them symbolic.
    lo, hi = _bounds(set) .- row.constant
    value = _parameter_value(row, layout.columns)
    _check_row(src, ci, row)
    filter!(t -> !iszero(t.second), row.x)
    filter!(t -> !iszero(t.second), row.ξ)
    filter!(t -> !iszero(t.second), row.parameters)
    if ci in info.truncate_rows || ci in info.valid_rows
        conditioning = ci in info.truncate_rows
        _push_valid_row!(c.valid, src, ci, row, lo - value, hi - value, conditioning)
    elseif isempty(row.x)
        _check_uncertainty_only_row(src, ci, row, layout.uncertainties, lo - value, hi - value)
    elseif lo == hi
        _push_row!(c.e, row, lo, value, ci)
    else
        isfinite(hi) && _push_row!(c.u, row, hi, value, ci)
        isfinite(lo) && _push_row!(c.l, row, lo, value, ci)
    end
    return
end

function _check_row(src::MOI.ModelLike, ci::MOI.ConstraintIndex, row::_Row)
    if row.recourse !== nothing
        x, η = row.recourse
        error(
            "The constraint $(_constraint_name(src, ci)) multiplies the decision " *
            "$(_variable_name(src, x)) by the uncertainty $(_variable_name(src, η)) " *
            "(random recourse), which is not supported.",
        )
    elseif row.quadratic !== nothing
        a, b = row.quadratic
        error(
            "The constraint $(_constraint_name(src, ci)) has the product of " *
            "$(_variable_name(src, a)) and $(_variable_name(src, b)); quadratic constraints are " *
            "not supported.",
        )
    end
    return
end

function _push_valid_row!(valid::_ValidRows, src, ci, row::_Row, lo, hi, conditioning::Bool)
    name = _constraint_name(src, ci)
    tag = conditioning ? "`Truncate()`" : "`ValidInequality()`"
    if !isempty(row.x)
        error("The constraint $name is tagged $tag, but it has decisions.")
    elseif lo == hi
        error(
            "The constraint $name is tagged $tag, but it is an equality: the support of the " *
            "uncertainty must be full-dimensional.",
        )
    elseif conditioning && isempty(row.ξ)
        error("The constraint $name is tagged $tag, but it has no uncertainty.")
    end
    # `lo ≤ w ⋅ η ≤ hi` gives `w ⋅ η ≥ lo` and `-w ⋅ η ≥ -hi`.
    for (sign, rhs) in ((1.0, lo), (-1.0, -hi))
        isfinite(rhs) || continue
        r = length(valid.h) + 1
        for (j, w) in row.ξ
            _push!(valid.W, r, j - 1, sign * w)
        end
        push!(valid.h, rhs)
        push!(valid.constraints, ci)
        push!(valid.conditioning, conditioning)
    end
    return
end

function _check_uncertainty_only_row(src, ci, row::_Row, uncertainties, lo, hi)
    name = _constraint_name(src, ci)
    if isempty(row.ξ)
        # Only constants and parameters: keep it if it holds.
        if lo > 1e-9 * max(1.0, abs(lo)) || hi < -1e-9 * max(1.0, abs(hi))
            error("The constraint $name has no decisions nor uncertainties, and it is violated.")
        end
        return
    end
    names =
        join((_variable_name(src, uncertainties[j-1]) for j in sort!(collect(keys(row.ξ)))), ", ")
    return error(
        "The constraint $name involves only uncertainties ($names). Tag it with " *
        "`ValidInequality()` if it describes the support Ξ without changing the law, or with " *
        "`Truncate()` if it conditions the law.",
    )
end

# A valid inequality holds on the support, so it holds at the mean: a cheap check of validity,
# which also shows that every support block contains a point (the mean). The mean is that of the
# conditioned law, which is not known before it is estimated for a `ConditionedDistribution`: the
# rows of such a law are not checked, nor the rows tagged `Truncate()`.
function _check_valid_rows_at_mean(src, form::CanonicalForm)
    μ = fill(NaN, length(form.uncertainties))
    for law in form.laws
        law.distribution isa ConditionedDistribution && continue
        μ[law.columns.-1] .= _law_mean(law.distribution)
    end
    Wt = SparseArrays.sparse(transpose(form.W_valid))
    for r in eachindex(form.h_valid)
        form.conditioning[r] && continue
        h = form.h_valid[r]
        value, scale = 0.0, abs(h)
        for k in SparseArrays.nzrange(Wt, r)
            term = SparseArrays.nonzeros(Wt)[k] * μ[SparseArrays.rowvals(Wt)[k]]
            value += term
            scale = max(scale, abs(term))
        end
        if isfinite(value) && value < h - 1e-9 * max(1.0, scale)
            error(
                "The constraint $(_constraint_name(src, form.valid_constraints[r])) is tagged " *
                "`ValidInequality()`, but the mean of the uncertainty violates it, so it does " *
                "not hold on the support of the distribution. Use `Truncate()` to condition " *
                "the law.",
            )
        end
    end
    for row in form.conic_rows
        row.conditioning && continue
        # The sparse product reads only the columns of the row, so the NaN of others do not show.
        Wμ = row.W * μ
        all(isfinite, Wμ) || continue
        scale = max(1.0, maximum(abs, Wμ; init = 0.0), maximum(abs, row.h; init = 0.0))
        if _cone_violation(row.cone, Wμ - row.h) > 1e-9 * scale
            error(
                "The constraint $(_constraint_name(src, row.constraint)) is tagged " *
                "`ValidInequality()`, but the mean of the uncertainty violates it, so it does " *
                "not hold on the support of the distribution. Use `Truncate()` to condition " *
                "the law.",
            )
        end
    end
    return
end

function _read_objective(src::MOI.ModelLike, layout::_Layout)
    objective = _Objective(length(layout.uncertainties), length(layout.parameters))
    sense = MOI.get(src, MOI.ObjectiveSense())
    if sense == MOI.FEASIBILITY_SENSE
        return sense, objective
    end
    F = MOI.get(src, MOI.ObjectiveFunctionType())
    if !(F <: Union{MOI.VariableIndex,_ScalarFunction})
        error("Objective functions of type $F are not supported by LinearDecisionRules.")
    end
    f = convert(MOI.ScalarQuadraticFunction{Float64}, MOI.get(src, MOI.ObjectiveFunction{F}()))
    _read!(objective, layout.columns, f)
    return sense, objective
end

function _canonical_form(src, layout::_Layout, c::_Constraints, sense, objective::_Objective)
    n, m = length(layout.decisions), length(layout.uncertainties)
    A(block) = _sparse(block.A, length(block.constraints), n)
    B(block) = _sparse(block.B, length(block.constraints), 1 + m)
    values = layout.columns.parameter_values
    Bp(block) = _sparse(block.Bp, length(block.constraints), length(values))
    return CanonicalForm(
        layout.decisions,
        String[_variable_name(src, v) for v in layout.decisions],
        layout.decision_stages,
        c.integer,
        c.binary,
        c.lower,
        c.upper,
        layout.uncertainties,
        String[_variable_name(src, v) for v in layout.uncertainties],
        layout.uncertainty_stages,
        layout.laws,
        layout.num_stages,
        A(c.e),
        B(c.e),
        c.e.constraints,
        A(c.u),
        B(c.u),
        c.u.constraints,
        A(c.l),
        B(c.l),
        c.l.constraints,
        _sparse(c.valid.W, length(c.valid.h), m),
        c.valid.h,
        Vector{Union{Nothing,MOI.ConstraintIndex}}(c.valid.constraints),
        c.valid.conditioning,
        c.conic_rows,
        sense,
        _sparse(objective.P, n, n),
        _sparse(objective.C, n, 1 + m),
        _sparse(objective.Q, m, m),
        objective.d,
        objective.f + LinearAlgebra.dot(objective.fp, values),
        nothing,
        ParameterTerms(
            layout.parameters,
            values,
            c.e.b,
            Bp(c.e),
            c.u.b,
            Bp(c.u),
            c.l.b,
            Bp(c.l),
            objective.f,
            objective.fp,
        ),
    )
end

# An equality row holds for every ξ only if, for each uncertainty in it, some decision of the row
# observes that uncertainty.
function _check_observable_equalities(src::MOI.ModelLike, form::CanonicalForm)
    isempty(form.e_constraints) && return
    info = InformationStructure(form)
    At = SparseArrays.sparse(transpose(form.A_e))
    Bt = SparseArrays.sparse(transpose(form.B_e))
    for r in eachindex(form.e_constraints)
        decisions = view(SparseArrays.rowvals(At), SparseArrays.nzrange(At, r))
        visible = maximum((info.num_visible[i] for i in decisions); init = 1)
        for k in SparseArrays.nzrange(Bt, r)
            j = SparseArrays.rowvals(Bt)[k]
            j <= visible && continue
            name = _constraint_name(src, form.e_constraints[r])
            integer = any(i -> form.integer[i], decisions) ? " Integer decisions observe none." : ""
            error(
                "The equality constraint $name has the uncertainty " *
                "$(form.uncertainty_names[j-1]) (Stage($(form.uncertainty_stages[j-1]))), which " *
                "none of its decisions observes; no decision rule satisfies it for every " *
                "outcome.$integer",
            )
        end
    end
    return
end
