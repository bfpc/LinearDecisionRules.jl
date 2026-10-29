# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The reader of an MOI model: the rows, bounds, objective and parameters of the canonical form.

"""
    _is_metadata_constraint(::Type{F}, ::Type{S}) -> Bool

Return `true` if [`canonicalize`](@ref) must skip the `F`-in-`S` constraints because they carry
metadata (such as stages or uncertainty sets) that the front end already put in the
[`ModelInfo`](@ref).
"""
_is_metadata_constraint(::Type, ::Type) = false

_bounds(s::MOI.LessThan{Float64}) = (-Inf, s.upper)
_bounds(s::MOI.GreaterThan{Float64}) = (s.lower, Inf)
_bounds(s::MOI.EqualTo{Float64}) = (s.value, s.value)
_bounds(s::MOI.Interval{Float64}) = (s.lower, s.upper)

_scalar_set(::Type{MOI.Nonnegatives}) = MOI.GreaterThan(0.0)
_scalar_set(::Type{MOI.Nonpositives}) = MOI.LessThan(0.0)
_scalar_set(::Type{MOI.Zeros}) = MOI.EqualTo(0.0)

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
