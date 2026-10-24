# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

const _BoundSet =
    Union{MOI.LessThan{Float64},MOI.GreaterThan{Float64},MOI.EqualTo{Float64},MOI.Interval{Float64}}

const _SparseMatrix = SparseArrays.SparseMatrixCSC{Float64,Int}

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
 * `decision_names::Vector{String}`: names, or the index when unnamed.
 * `decision_stages::Vector{Int}`
 * `integer::BitVector`: `MOI.Integer` or `MOI.ZeroOne` decisions (all in stage 1).
 * `binary::BitVector`: `MOI.ZeroOne` decisions; their bounds are not tightened to `[0, 1]`.
 * `lower::Vector{Float64}`, `upper::Vector{Float64}`: bounds, `±Inf` when absent.

Uncertainties (`uncertainties[j]` is column `1 + j` of ξ; columns are sorted by stage):

 * `uncertainties::Vector{MOI.VariableIndex}`
 * `uncertainty_names::Vector{String}`
 * `uncertainty_stages::Vector{Int}`: nondecreasing.
 * `laws::Vector{UncertaintyBlock}`: one per declaration; separate laws are independent.
 * `num_stages::Int`: the number of stages `T ≥ 2`.

Rows (`B_*` have `1 + m` columns; `*_constraints[r]` is the constraint that row `r` comes from):

 * `A_e::SparseMatrixCSC{Float64,Int}`, `B_e::SparseMatrixCSC{Float64,Int}`,
   `e_constraints::Vector{MOI.ConstraintIndex}`
 * `A_u`, `B_u`, `u_constraints`
 * `A_l`, `B_l`, `l_constraints`

Rows tagged `ValidInequality()`, as `W_valid η ≥ h_valid` (`W_valid` has `m` columns, over η):

 * `W_valid::SparseMatrixCSC{Float64,Int}`, `h_valid::Vector{Float64}`,
   `valid_constraints::Vector{MOI.ConstraintIndex}`

Objective (`FEASIBILITY_SENSE` when the model has none; then every term is zero):

 * `sense::MOI.OptimizationSense`
 * `P::SparseMatrixCSC{Float64,Int}`: `n × n`, symmetric.
 * `C::SparseMatrixCSC{Float64,Int}`: `n × (1 + m)`; column 1 has the linear costs, the other
   columns the products of decisions and uncertainties.
 * `Q::SparseMatrixCSC{Float64,Int}`: `m × m`, symmetric.
 * `d::Vector{Float64}`, `f::Float64`.
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
    valid_constraints::Vector{MOI.ConstraintIndex}
    sense::MOI.OptimizationSense
    P::_SparseMatrix
    C::_SparseMatrix
    Q::_SparseMatrix
    d::Vector{Float64}
    f::Float64
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

function _variable_name(src::MOI.ModelLike, v::MOI.VariableIndex)
    if MOI.supports(src, MOI.VariableName(), MOI.VariableIndex)
        name = MOI.get(src, MOI.VariableName(), v)
        isempty(name) || return name
    end
    return string(v)
end

function _constraint_name(src::MOI.ModelLike, ci::MOI.ConstraintIndex{F,S}) where {F,S}
    if F != MOI.VariableIndex && MOI.supports(src, MOI.ConstraintName(), typeof(ci))
        name = MOI.get(src, MOI.ConstraintName(), ci)
        isempty(name) || return name
    end
    return string(ci)
end

_bounds(s::MOI.LessThan{Float64}) = (-Inf, s.upper)
_bounds(s::MOI.GreaterThan{Float64}) = (s.lower, Inf)
_bounds(s::MOI.EqualTo{Float64}) = (s.value, s.value)
_bounds(s::MOI.Interval{Float64}) = (s.lower, s.upper)

_dimension(::Distributions.UnivariateDistribution) = 1
_dimension(d::Distributions.MultivariateDistribution) = length(d)
_dimension(::Distributions.Distribution) = 0

# The role of every variable of the model, as `canonicalize` sees it.
struct _Columns
    decision::Dict{MOI.VariableIndex,Int}    # row of the decision rule
    uncertainty::Dict{MOI.VariableIndex,Int} # column of ξ, ≥ 2
    parameter::Dict{MOI.VariableIndex,Float64}
end

# One constraint, read as `x-terms + ξ-terms + constant`.
mutable struct _Row
    x::Dict{Int,Float64}
    ξ::Dict{Int,Float64}
    constant::Float64
    recourse::Union{Nothing,Tuple{MOI.VariableIndex,MOI.VariableIndex}} # (decision, uncertainty)
    quadratic::Union{Nothing,Tuple{MOI.VariableIndex,MOI.VariableIndex}}
end

_Row() = _Row(Dict{Int,Float64}(), Dict{Int,Float64}(), 0.0, nothing, nothing)

function _reset!(row::_Row)
    empty!(row.x)
    empty!(row.ξ)
    row.constant = 0.0
    row.recourse = nothing
    row.quadratic = nothing
    return row
end

function _add_term!(row::_Row, cols::_Columns, coef::Float64, v::MOI.VariableIndex)
    if haskey(cols.decision, v)
        i = cols.decision[v]
        row.x[i] = get(row.x, i, 0.0) + coef
    elseif haskey(cols.uncertainty, v)
        j = cols.uncertainty[v]
        row.ξ[j] = get(row.ξ, j, 0.0) + coef
    else
        row.constant += coef * cols.parameter[v]
    end
    return
end

# `coef * a * b`, with parameters replaced by their values.
function _add_product!(
    row::_Row,
    cols::_Columns,
    coef::Float64,
    a::MOI.VariableIndex,
    b::MOI.VariableIndex,
)
    if haskey(cols.parameter, a)
        _add_term!(row, cols, coef * cols.parameter[a], b)
    elseif haskey(cols.parameter, b)
        _add_term!(row, cols, coef * cols.parameter[b], a)
    elseif haskey(cols.decision, a) && haskey(cols.uncertainty, b)
        row.recourse = (a, b)
    elseif haskey(cols.uncertainty, a) && haskey(cols.decision, b)
        row.recourse = (b, a)
    else
        row.quadratic = (a, b)
    end
    return
end

function _read!(row::_Row, cols::_Columns, f::MOI.ScalarAffineFunction{Float64})
    for t in f.terms
        _add_term!(row, cols, t.coefficient, t.variable)
    end
    row.constant += f.constant
    return row
end

function _read!(row::_Row, cols::_Columns, f::MOI.ScalarQuadraticFunction{Float64})
    for t in f.affine_terms
        _add_term!(row, cols, t.coefficient, t.variable)
    end
    for t in f.quadratic_terms
        # MOI stores `c x²` as the term `(2c, x, x)`.
        a, b = t.variable_1, t.variable_2
        _add_product!(row, cols, a == b ? t.coefficient / 2 : t.coefficient, a, b)
    end
    row.constant += f.constant
    return row
end

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

# The rows `A x (=, ≤, ≥) B ξ` of one kind.
struct _RowBlock
    A::_Triplets
    B::_Triplets
    constraints::Vector{MOI.ConstraintIndex}
end

_RowBlock() = _RowBlock(_Triplets(), _Triplets(), MOI.ConstraintIndex[])

# Push `row.x ⋅ x + row.ξ ⋅ η (sense) rhs` as `A x (sense) B ξ`, with `B ξ = rhs − row.ξ ⋅ η`.
function _push_row!(block::_RowBlock, row::_Row, rhs::Float64, ci::MOI.ConstraintIndex)
    r = length(block.constraints) + 1
    for (i, a) in row.x
        _push!(block.A, r, i, a)
    end
    _push!(block.B, r, 1, rhs)
    for (j, b) in row.ξ
        _push!(block.B, r, j, -b)
    end
    push!(block.constraints, ci)
    return
end

"""
    canonicalize(src::MOI.ModelLike, info::ModelInfo) -> CanonicalForm

Read the model `src` and its metadata `info` into a [`CanonicalForm`](@ref).

`src` is only read, so it can be `JuMP.backend(model)` or any MOI model. JuMP parameters
(`VariableIndex`-in-`Parameter`) are replaced by their values.

The number of stages is `T = max(2, largest declared stage)`; unmarked variables are in stage `T`,
and a single warning lists them when `T > 2`.

Errors name the variables and constraints involved, for: unsupported constraint types, uncertainty-
only rows not tagged `ValidInequality()`, rows tagged `Truncate()`, bounds or integrality on an
uncertainty, integer decisions after stage 1, products of uncertainties and decisions in
constraints (random recourse), quadratic constraints, and equality rows with an uncertainty that
none of their decisions observe.

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
function canonicalize(src::MOI.ModelLike, info::ModelInfo)
    if !isempty(info.breakpoints)
        error("Breakpoints (piecewise liftings) are not supported yet.")
    end
    variables = MOI.get(src, MOI.ListOfVariableIndices())
    in_model = Set(variables)
    parameter = _parameter_values(src)
    # Uncertainty columns: declaration order, then stably sorted by stage.
    entries = Tuple{Int,MOI.VariableIndex,Union{Nothing,Int}}[]
    seen = Set{MOI.VariableIndex}()
    for (q, decl) in enumerate(info.uncertainties)
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
        for (k, v) in enumerate(decl.variables)
            if !(v in in_model)
                error("The uncertainty $v is not a variable of the model.")
            elseif haskey(parameter, v)
                error("The variable $(_variable_name(src, v)) is a Parameter and an uncertainty.")
            elseif v in seen
                error("The variable $(_variable_name(src, v)) is declared as an uncertainty twice.")
            end
            s = decl.stages[k]
            if s !== nothing && s < 1
                error("The uncertainty $(_variable_name(src, v)) has Stage($s); stages start at 1.")
            end
            push!(seen, v)
            push!(entries, (q, v, s))
        end
    end
    for (v, s) in info.decision_stages
        if !(v in in_model)
            error("The decision $v, which has a stage, is not a variable of the model.")
        elseif v in seen
            error("The uncertainty $(_variable_name(src, v)) has a decision stage.")
        elseif s < 1
            error("The decision $(_variable_name(src, v)) has Stage($s); stages start at 1.")
        end
    end
    T = 2
    for (v, s) in info.decision_stages
        haskey(parameter, v) || (T = max(T, s))
    end
    for (_, _, s) in entries
        s === nothing || (T = max(T, s))
    end
    ustage_raw = [something(s, T) for (_, _, s) in entries]
    order = sortperm(ustage_raw; alg = Base.Sort.DEFAULT_STABLE)
    uncertainties = [entries[k][2] for k in order]
    uncertainty_stages = ustage_raw[order]
    ucol = Dict(v => 1 + j for (j, v) in enumerate(uncertainties))
    laws = UncertaintyBlock[]
    for decl in info.uncertainties
        cols = [ucol[v] for v in decl.variables]
        push!(laws, UncertaintyBlock(decl.distribution, cols, uncertainty_stages[cols.-1]))
    end
    decisions = [v for v in variables if !(v in seen) && !haskey(parameter, v)]
    decision_stages = [get(info.decision_stages, v, T) for v in decisions]
    cols = _Columns(Dict(v => i for (i, v) in enumerate(decisions)), ucol, parameter)
    if T > 2
        _warn_unmarked(src, T, info, decisions, entries)
    end
    n, m = length(decisions), length(uncertainties)
    # Constraints
    lower, upper = fill(-Inf, n), fill(Inf, n)
    integer, binary = falses(n), falses(n)
    rows = (e = _RowBlock(), u = _RowBlock(), l = _RowBlock())
    valid = (W = _Triplets(), h = Float64[], constraints = MOI.ConstraintIndex[])
    row = _Row()
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        if _is_metadata_constraint(F, S) || (F == MOI.VariableIndex && S == MOI.Parameter{Float64})
            continue
        elseif F == MOI.VariableIndex && (S <: _BoundSet || S == MOI.Integer || S == MOI.ZeroOne)
            for ci in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
                v = MOI.get(src, MOI.ConstraintFunction(), ci)
                if S <: _BoundSet && (ci in info.valid_rows || ci in info.truncate_rows)
                    # `@constraint(model, d <= 110, ValidInequality())` is a row, not a bound.
                    f = convert(MOI.ScalarAffineFunction{Float64}, v)
                    set = MOI.get(src, MOI.ConstraintSet(), ci)
                    _read_row!(rows, valid, row, src, info, cols, uncertainties, ci, f, set)
                    continue
                elseif haskey(parameter, v)
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
                    lower[i], upper[i] = max(lower[i], lo), min(upper[i], hi)
                else
                    integer[i] = true
                    binary[i] |= S == MOI.ZeroOne
                    if decision_stages[i] > 1
                        unmarked = haskey(info.decision_stages, v) ? "" : " (it is unmarked)"
                        error(
                            "The integer decision $(_variable_name(src, v)) is in " *
                            "Stage($(decision_stages[i]))$unmarked; integer decisions are " *
                            "supported only in Stage(1).",
                        )
                    end
                end
            end
        elseif F in (MOI.ScalarAffineFunction{Float64}, MOI.ScalarQuadraticFunction{Float64}) &&
               S <: _BoundSet
            for ci in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
                f = MOI.get(src, MOI.ConstraintFunction(), ci)
                set = MOI.get(src, MOI.ConstraintSet(), ci)
                _read_row!(rows, valid, row, src, info, cols, uncertainties, ci, f, set)
            end
        else
            error("Constraints of type $F-in-$S are not supported by LinearDecisionRules.")
        end
    end
    _check_valid_rows_at_mean(src, laws, valid, m)
    # Objective
    sense = MOI.get(src, MOI.ObjectiveSense())
    P, C, Q = _Triplets(), _Triplets(), _Triplets()
    d, f = zeros(m), 0.0
    if sense != MOI.FEASIBILITY_SENSE
        Fo = MOI.get(src, MOI.ObjectiveFunctionType())
        if !(
            Fo in (
                MOI.VariableIndex,
                MOI.ScalarAffineFunction{Float64},
                MOI.ScalarQuadraticFunction{Float64},
            )
        )
            error("Objective functions of type $Fo are not supported by LinearDecisionRules.")
        end
        obj =
            convert(MOI.ScalarQuadraticFunction{Float64}, MOI.get(src, MOI.ObjectiveFunction{Fo}()))
        f = _read_objective!(P, C, Q, d, cols, obj)
    end
    form = CanonicalForm(
        decisions,
        String[_variable_name(src, v) for v in decisions],
        decision_stages,
        integer,
        binary,
        lower,
        upper,
        uncertainties,
        String[_variable_name(src, v) for v in uncertainties],
        uncertainty_stages,
        laws,
        T,
        _sparse(rows.e.A, length(rows.e.constraints), n),
        _sparse(rows.e.B, length(rows.e.constraints), 1 + m),
        rows.e.constraints,
        _sparse(rows.u.A, length(rows.u.constraints), n),
        _sparse(rows.u.B, length(rows.u.constraints), 1 + m),
        rows.u.constraints,
        _sparse(rows.l.A, length(rows.l.constraints), n),
        _sparse(rows.l.B, length(rows.l.constraints), 1 + m),
        rows.l.constraints,
        _sparse(valid.W, length(valid.h), m),
        valid.h,
        valid.constraints,
        sense,
        _sparse(P, n, n),
        _sparse(C, n, 1 + m),
        _sparse(Q, m, m),
        d,
        f,
    )
    _check_observable_equalities(src, form)
    return form
end

# Read the constraint `ci`, `f`-in-`set`, into the rows of its kind, using `row` as a buffer.
function _read_row!(rows, valid, row::_Row, src, info::ModelInfo, cols, uncertainties, ci, f, set)
    _read!(_reset!(row), cols, f)
    lo, hi = _bounds(set) .- row.constant
    _check_row(src, ci, row)
    filter!(t -> !iszero(t.second), row.x)
    filter!(t -> !iszero(t.second), row.ξ)
    if ci in info.truncate_rows
        error(
            "The constraint $(_constraint_name(src, ci)) is tagged `Truncate()`: " *
            "conditioning the law is not supported yet (Phase 3).",
        )
    elseif ci in info.valid_rows
        _push_valid_row!(valid, src, ci, row, lo, hi)
    elseif isempty(row.x)
        _check_uncertainty_only_row(src, ci, row, uncertainties, lo, hi)
    elseif lo == hi
        _push_row!(rows.e, row, lo, ci)
    else
        isfinite(hi) && _push_row!(rows.u, row, hi, ci)
        isfinite(lo) && _push_row!(rows.l, row, lo, ci)
    end
    return
end

function _parameter_values(src::MOI.ModelLike)
    values = Dict{MOI.VariableIndex,Float64}()
    S = MOI.Parameter{Float64}
    if (MOI.VariableIndex, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex,S}())
            v = MOI.get(src, MOI.ConstraintFunction(), ci)
            values[v] = MOI.get(src, MOI.ConstraintSet(), ci).value
        end
    end
    return values
end

function _warn_unmarked(src, T, info, decisions, entries)
    unmarked = MOI.VariableIndex[v for v in decisions if !haskey(info.decision_stages, v)]
    append!(unmarked, v for (_, v, s) in entries if s === nothing)
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

function _push_valid_row!(valid, src, ci, row::_Row, lo::Float64, hi::Float64)
    name = _constraint_name(src, ci)
    if !isempty(row.x)
        error("The constraint $name is tagged `ValidInequality()`, but it has decisions.")
    elseif lo == hi
        error(
            "The constraint $name is tagged `ValidInequality()`, but it is an equality: the " *
            "support of the uncertainty must be full-dimensional.",
        )
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
# which also shows that every support block contains a point (the mean).
function _check_valid_rows_at_mean(src, laws::Vector{UncertaintyBlock}, valid, m::Int)
    isempty(valid.h) && return
    μ = fill(NaN, m)
    for law in laws
        μ[law.columns.-1] .= _law_mean(law.distribution)
    end
    W = _sparse(valid.W, length(valid.h), m)
    Wt = SparseArrays.sparse(transpose(W))
    for r in eachindex(valid.h)
        value, scale = 0.0, abs(valid.h[r])
        for k in SparseArrays.nzrange(Wt, r)
            term = SparseArrays.nonzeros(Wt)[k] * μ[SparseArrays.rowvals(Wt)[k]]
            value += term
            scale = max(scale, abs(term))
        end
        if isfinite(value) && value < valid.h[r] - 1e-9 * max(1.0, scale)
            error(
                "The constraint $(_constraint_name(src, valid.constraints[r])) is tagged " *
                "`ValidInequality()`, but the mean of the uncertainty violates it, so it does " *
                "not hold on the support of the distribution. Use `Truncate()` to condition the law.",
            )
        end
    end
    return
end

# Fill the objective `x'Px + x'Cξ + η'Qη + d'η + f` and return `f`.
function _read_objective!(P, C, Q, d, cols::_Columns, obj::MOI.ScalarQuadraticFunction{Float64})
    f = obj.constant
    kind(v) = haskey(cols.decision, v) ? :x : haskey(cols.uncertainty, v) ? :η : :p
    function linear!(c, v)
        k = kind(v)
        if k == :x
            _push!(C, cols.decision[v], 1, c)
        elseif k == :η
            d[cols.uncertainty[v]-1] += c
        else
            f += c * cols.parameter[v]
        end
        return
    end
    for t in obj.affine_terms
        linear!(t.coefficient, t.variable)
    end
    for t in obj.quadratic_terms
        a, b = t.variable_1, t.variable_2
        # MOI stores `c x²` as the term `(2c, x, x)`; `c` is then the coefficient of `a * b`.
        c = a == b ? t.coefficient / 2 : t.coefficient
        ka, kb = kind(a), kind(b)
        if ka == :p
            linear!(c * cols.parameter[a], b)
        elseif kb == :p
            linear!(c * cols.parameter[b], a)
        elseif ka == :x && kb == :x
            i, k = cols.decision[a], cols.decision[b]
            _push!(P, i, k, c / 2)
            _push!(P, k, i, c / 2)
        elseif ka == :η && kb == :η
            i, k = cols.uncertainty[a] - 1, cols.uncertainty[b] - 1
            _push!(Q, i, k, c / 2)
            _push!(Q, k, i, c / 2)
        elseif ka == :x
            _push!(C, cols.decision[a], cols.uncertainty[b], c)
        else
            _push!(C, cols.decision[b], cols.uncertainty[a], c)
        end
    end
    return f
end

# An equality row holds for every ξ only if, for each uncertainty in it, some decision of the row
# observes that uncertainty.
function _check_observable_equalities(src::MOI.ModelLike, form::CanonicalForm)
    isempty(form.e_constraints) && return
    info = InformationStructure(form)
    At = SparseArrays.sparse(transpose(form.A_e))
    Bt = SparseArrays.sparse(transpose(form.B_e))
    for r in eachindex(form.e_constraints)
        visible = 1
        for k in SparseArrays.nzrange(At, r)
            visible = max(visible, info.num_visible[SparseArrays.rowvals(At)[k]])
        end
        for k in SparseArrays.nzrange(Bt, r)
            j = SparseArrays.rowvals(Bt)[k]
            j <= visible && continue
            error(
                "The equality constraint $(_constraint_name(src, form.e_constraints[r])) has the " *
                "uncertainty $(form.uncertainty_names[j-1]) (Stage($(form.uncertainty_stages[j-1]))), " *
                "which none of its decisions observes; no decision rule satisfies it for every " *
                "outcome. Integer decisions observe no uncertainty.",
            )
        end
    end
    return
end
