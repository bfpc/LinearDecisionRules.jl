# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The canonical form and the types of its parts.

# The constraints that `canonicalize` reads as rows; the optimizer supports exactly these.
const _BoundSet =
    Union{MOI.LessThan{Float64},MOI.GreaterThan{Float64},MOI.EqualTo{Float64},MOI.Interval{Float64}}

const _ScalarFunction =
    Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}}

# Linear vector rows, read one output at a time; second-order cone rows describe Ξ only.
const _VectorFunction = Union{MOI.VectorAffineFunction{Float64},MOI.VectorOfVariables}

const _VectorSet = Union{MOI.Nonnegatives,MOI.Nonpositives,MOI.Zeros}

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
   name is followed by `[piece k]`. The pieces of a combination of uncertainties (see
   `Experimental.set_breakpoints`) come after its inputs, which stay columns; `uncertainties[j]`
   is then the input in the last stage, and the name is that of the combination followed by
   `[piece k]`.

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

# `form` with some fields replaced, such as `_replace(form; laws = laws)`. A keyword that is not a
# field is an error, so that a misspelled one does not leave its field unchanged.
function _replace(form::CanonicalForm; kwargs...)
    unknown = setdiff(keys(kwargs), fieldnames(CanonicalForm))
    if !isempty(unknown)
        throw(ArgumentError("`CanonicalForm` has no fields $(join(unknown, ", "))."))
    end
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
