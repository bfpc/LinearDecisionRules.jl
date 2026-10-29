# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    AbstractFormulation

A reformulation of a [`CanonicalForm`](@ref) as a deterministic problem over the coefficients of
linear decision rules, solved by [`solve_formulation`](@ref).

The decision rule of decision `i` is `x_i(ξ) = X[i, :] ⋅ ξ` with `ξ = (1, η)`, where `X[i, j]` is
zero for the columns that decision `i` does not observe (see [`InformationStructure`](@ref)).
"""
abstract type AbstractFormulation end

"""
    PrimalLDR(; attributes = [], box_closed_form::Bool = true)

The primal linear decision rule: every constraint holds for every outcome in the support Ξ.

Its rule is a policy, so its objective value is attained (a lower bound of a maximization, an upper
bound of a minimization). Each inequality is replaced by its robust counterpart over the
[`SupportBlock`](@ref)s on which it depends (Kuhn, Wiesemann and Georghiou, 2011), with
multipliers in the dual cone of each block: nonnegative for its linear rows, in a second-order
cone for each of its second-order cone rows, so the inner problem is then a second-order cone
program. On a block that is a box (finite bounds of its coordinates, no other row), the
multipliers are replaced by the closed form `min(lo s_j, hi s_j)` of each coefficient `s_j` of the
slack, with one auxiliary variable per coefficient; `box_closed_form = false` keeps the
multipliers, to compare both forms.

`attributes` are `name => value` raw attributes of the inner optimizer, set after (so over) those
of the optimizer, for this formulation only: `PrimalLDR(; attributes = ["solver" => "ipx"])`.
Formulations are compared by type: the attributes are settings, not part of the identity.

With HiGHS, keep the default dual simplex for this formulation: on inventory and hydro problems
(HiGHS 1.15), it is within 2 times of the best mode almost everywhere.
The exception is a long hydro problem (T = 8), where the interior point method IPX without
crossover, `["solver" => "ipx", "run_crossover" => "off"]`, is 3 times faster; the other modes
cost 3 to 16 times more on the larger inventory problem, or fail on the hydro one.
"""
struct PrimalLDR <: AbstractFormulation
    attributes::Vector{Pair{String,Any}}
    box_closed_form::Bool
end

function PrimalLDR(; attributes = Pair{String,Any}[], box_closed_form::Bool = true)
    return PrimalLDR(_attribute_pairs(attributes), box_closed_form)
end

"""
    DualLDR(; attributes = [])

The dual linear decision rule: a relaxation whose objective value bounds the optimal value of the
stochastic program (an upper bound of a maximization, a lower bound of a minimization).

Its rule is not a policy. Each inequality `s(ξ) ≥ 0` is relaxed to `E[s(ξ) W'ξ] ∈ K`, where
`W'ξ ∈ K` describes Ξ, including the row `ξ₁ ≥ 0` (Kuhn, Wiesemann and Georghiou, 2011): `K` is
the nonnegative orthant of the linear rows of the [`SupportBlock`](@ref)s times a second-order
cone for each of their second-order cone rows, so the inner problem is then a second-order cone
program.

The bound needs the conditional expectations of the uncertainties of later stages, given those of
earlier stages, to be linear (as for independent stages). Separate declarations are independent,
so this is a property of each law whose components are in several stages: it holds for a product
of univariate laws, a normal law and a [`ConfidenceMvNormal`](@ref), and it is checked exactly
for a [`MvDiscreteNonParametric`](@ref); it is unknown for other laws, such as a
[`ConditionedDistribution`](@ref). Building the dual warns, and the bound is not certified, when
such a law does not have it or is not known to.

`attributes` are `name => value` raw attributes of the inner optimizer, set after (so over) those
of the optimizer, for this formulation only. Formulations are compared by type: the attributes are
settings, not part of the identity.

The dual needs its bound only, not a basis. With HiGHS, its interior point method without
crossover is often much faster on large problems:

```julia
DualLDR(; attributes = ["solver" => "ipm", "run_crossover" => "off"])
```

On inventory and hydro problems (HiGHS 1.15), it solved every dual within 7 seconds, where the
default dual simplex took 145 seconds on the hydro problem with T = 8. This is not the default, since the names are HiGHS's
and the best mode depends on the problem. The bound is then the dual objective value of the inner
linear program (see [`FormulationResult`](@ref)). Only a termination status `OPTIMAL` can be
trusted: HiGHS may stop an interior point without crossover at an `unknown` status (its HiPO
stagnates a little above the tolerance), with a wrong objective value or none; the termination
status is then `OTHER_ERROR` and the bound `NaN`. This happens on most of the small problems of the
test library, so keep the default for small models.
"""
struct DualLDR <: AbstractFormulation
    attributes::Vector{Pair{String,Any}}
end

DualLDR(; attributes = Pair{String,Any}[]) = DualLDR(_attribute_pairs(attributes))

_attribute_pairs(attributes) = Pair{String,Any}[String(k) => v for (k, v) in attributes]

# The formulations of a type are the same formulation whatever their attributes.
Base.:(==)(::PrimalLDR, ::PrimalLDR) = true

Base.:(==)(::DualLDR, ::DualLDR) = true

Base.hash(::PrimalLDR, h::UInt) = hash(:PrimalLDR, h)

Base.hash(::DualLDR, h::UInt) = hash(:DualLDR, h)

function Base.show(io::IO, f::Union{PrimalLDR,DualLDR})
    settings = String[]
    if !isempty(f.attributes)
        pairs = ("$(repr(k)) => $(repr(v))" for (k, v) in f.attributes)
        push!(settings, "attributes = [$(join(pairs, ", "))]")
    end
    if f isa PrimalLDR && !f.box_closed_form
        push!(settings, "box_closed_form = false")
    end
    print(io, nameof(typeof(f)), "(", join(settings, ", "), ")")
    return
end

# The raw attributes that a formulation sets on its inner problem, over those of the optimizer.
_formulation_attributes(f::Union{PrimalLDR,DualLDR}) = f.attributes

_formulation_attributes(::AbstractFormulation) = Pair{String,Any}[]

"""
    SampledLDR(num_scenarios::Integer; seed::Integer = 1234)

The sampled linear decision rule: every inequality holds on `num_scenarios` outcomes drawn from the
laws, and the objective uses the empirical moments of the outcomes.

Outcomes are drawn with `Random.Xoshiro(seed)`, one draw per declaration per scenario. Equality rows
hold for every outcome, as in [`PrimalLDR`](@ref). The rule is feasible on the drawn outcomes only.
"""
struct SampledLDR <: AbstractFormulation
    num_scenarios::Int
    seed::Int

    function SampledLDR(num_scenarios::Integer, seed::Integer)
        if num_scenarios < 1
            throw(ArgumentError("The number of scenarios must be positive; got $num_scenarios."))
        end
        return new(num_scenarios, seed)
    end
end

SampledLDR(num_scenarios::Integer; seed::Integer = 1234) = SampledLDR(num_scenarios, seed)

"""
    FormulationCapabilities

What a formulation provides, returned by [`capabilities`](@ref).

## Fields

 * `rule_status::MOI.ResultStatusCode`: the status of its decision rule as a solution of the
   stochastic program when the inner problem is solved: `FEASIBLE_POINT` (feasible for every outcome
   in Ξ), `UNKNOWN_RESULT_STATUS` (feasible on the drawn outcomes only) or `NO_SOLUTION` (the rule is
   not a policy).
 * `bound::Bool`: whether its objective value bounds the optimal value of the stochastic program
   from the side opposite to the sense (an upper bound of a maximization).
"""
struct FormulationCapabilities
    rule_status::MOI.ResultStatusCode
    bound::Bool
end

"""
    capabilities(f::AbstractFormulation) -> FormulationCapabilities

Return the [`FormulationCapabilities`](@ref) of the formulation `f`.
"""
function capabilities(::PrimalLDR)
    return FormulationCapabilities(MOI.FEASIBLE_POINT, false)
end

capabilities(::DualLDR) = FormulationCapabilities(MOI.NO_SOLUTION, true)

function capabilities(::SampledLDR)
    return FormulationCapabilities(MOI.UNKNOWN_RESULT_STATUS, false)
end

"""
    FormulationResult

The result of [`solve_formulation`](@ref).

## Fields

 * `formulation::AbstractFormulation`
 * `termination_status::MOI.TerminationStatusCode`: of the inner problem.
 * `primal_status::MOI.ResultStatusCode`: of the inner problem, except that a feasible point of an
   inner problem that is infeasible or unbounded (by its termination status) is `NO_SOLUTION`.
 * `raw_status::String`: of the inner problem.
 * `objective_value::Float64`: in the sense of the model (zero for a feasibility problem); `NaN`
   unless `primal_status == FEASIBLE_POINT`.
 * `objective_bound::Float64`: the inner solver's bound on `objective_value`, in the sense of the
   model: `MOI.ObjectiveBound` when the inner problem has integer variables; the dual objective
   value when the inner problem is a linear program solved to optimality, if the solver has it
   (by weak duality it is a bound even for an interior point without crossover, whose primal
   point is feasible up to a tolerance only), else the optimal value; the dual objective value
   when the inner problem is not solved to optimality; `Inf` for a maximization (`-Inf` for a
   minimization) whose inner problem is unbounded; `NaN` if unknown. It needs no primal point. For [`DualLDR`](@ref) with integer decisions, this is the bound to report.
 * `certified::Bool`: whether `objective_value` and `objective_bound` are exact for the laws of the
   model, so that the value of [`PrimalLDR`](@ref) is attained and the bound of [`DualLDR`](@ref)
   is a bound. It is `false` when they use moments estimated by rejection sampling (laws
   conditioned by constraints tagged [`Truncate`](@ref)), for a bound when a law spanning several
   stages does not have, or is not known to have, linear conditional expectations of its later
   stages given its earlier ones (see [`DualLDR`](@ref)) or the second-moment matrix is singular,
   and for a rule that is feasible on drawn outcomes only ([`SampledLDR`](@ref)). Except
   for the latter, `raw_status` then ends with `[not certified: <reason>]`.
 * `solve_time::Float64`: wall-clock seconds of the inner `optimize!` (zero if it was not built).
 * `X::Matrix{Float64}`: the decision rules, `decisions × (1 + uncertainties)`, with zeros in the
   columns that a decision does not observe; all `NaN` unless `primal_status == FEASIBLE_POINT`.
 * `model::JuMP.Model`: the inner problem, for inspection and `write_to_file`; an empty model if
   the time limit was reached before it was built.
"""
struct FormulationResult
    formulation::AbstractFormulation
    termination_status::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    raw_status::String
    objective_value::Float64
    objective_bound::Float64
    certified::Bool
    solve_time::Float64
    X::Matrix{Float64}
    model::JuMP.Model
end

function Base.show(io::IO, result::FormulationResult)
    print(io, "FormulationResult of $(result.formulation): $(result.termination_status), ")
    print(io, "$(result.primal_status), objective value $(result.objective_value)")
    return
end

"""
    solve_formulation(
        form::CanonicalForm,
        f::AbstractFormulation,
        mom::Moments,
        optimizer_factory;
        silent::Bool = true,
        time_limit::Union{Nothing,Real} = nothing,
        raw_attributes = Dict{String,Any}(),
        names::Bool = false,
    ) -> FormulationResult

Build the inner problem of the formulation `f` for `form` and the moments `mom`, solve it with
`optimizer_factory` and return a [`FormulationResult`](@ref).

`time_limit` is in seconds, for building and solving the inner problem: the inner optimizer gets
what remains after the build. `raw_attributes` is an iterable of `name => value` pairs passed to
`JuMP.set_attribute` on the inner problem, followed by the `attributes` of `f` (see
[`DualLDR`](@ref)). [`SampledLDR`](@ref) ignores `mom` and uses the empirical moments of its
outcomes.

With `names = true`, the variables of the inner problem are named, so that `write_to_file` is
readable: `X[i,j]` for the rule of decision `i` on column `j` of ξ, and by inequality `r`,
`λ[r,b]` (the multipliers of the linear rows of support block `b`) and `λ[r,b,p]` (those of its
`p`-th piece, a second-order cone), `w[r,j]` (the auxiliary of the closed form on a box, for column
`j`), or `u[r]` and `S[r,j]` (the slack rule). Names slow the build
down, so they are off by default.

## Example

```julia
import LinearDecisionRules as LDR
form = LDR.canonicalize(backend(model), LDR.model_info(backend(model)))
result = LDR.solve_formulation(form, PrimalLDR(), LDR.moments(form), HiGHS.Optimizer)
result.objective_value
```
"""
function solve_formulation(
    form::CanonicalForm,
    f::AbstractFormulation,
    mom::Moments,
    optimizer_factory;
    silent::Bool = true,
    time_limit::Union{Nothing,Real} = nothing,
    raw_attributes = Dict{String,Any}(),
    names::Bool = false,
)
    start = time_ns()
    inner = _build_inner(form, f, mom, optimizer_factory; silent, raw_attributes, names)
    if time_limit !== nothing
        # The limit covers the build too, which can take longer than the solve.
        time_limit = max(0.0, time_limit - _seconds_since(start))
    end
    return _solve_inner!(inner, f, form, mom; time_limit)
end

# The inner problem of a formulation, which the optimizer keeps to solve it again when only the
# values of the parameters change (see `ParameterTerms`).
mutable struct _InnerProblem
    model::JuMP.Model
    # The rule variables of the decisions of the form (see `_rule_variables!`).
    X::Vector{Vector{JuMP.VariableRef}}
    # The variable of `model` fixed to each parameter of `form.parameters`, if it has one.
    parameters::Vector{Union{Nothing,JuMP.VariableRef}}
    # Whether a time limit was set on `model`, to be removed by a solve without one.
    time_limited::Bool
end

function _build_inner(
    form::CanonicalForm,
    f::AbstractFormulation,
    mom::Moments,
    optimizer_factory;
    silent::Bool = true,
    raw_attributes = Dict{String,Any}(),
    names::Bool = false,
)
    model = JuMP.Model(optimizer_factory)
    JuMP.set_string_names_on_creation(model, names)
    if silent
        JuMP.set_silent(model)
    end
    for (name, value) in raw_attributes
        JuMP.set_attribute(model, name, value)
    end
    for (name, value) in _formulation_attributes(f)
        JuMP.set_attribute(model, name, value)
    end
    extended, used = _with_parameter_decisions(form)
    X = _build!(model, f, extended, mom)
    n = length(form.decisions)
    parameters = Vector{Union{Nothing,JuMP.VariableRef}}(nothing, length(form.parameters.values))
    for (q, k) in enumerate(used)
        parameters[k] = only(X[n+q])
    end
    return _InnerProblem(model, X[1:n], parameters, false)
end

# Fix the parameters of `inner` to their values in `form` and solve it. A parameter whose value
# does not change is not set again, so that the solver keeps its solution when nothing changes.
function _solve_inner!(
    inner::_InnerProblem,
    f::AbstractFormulation,
    form::CanonicalForm,
    mom::Moments;
    time_limit::Union{Nothing,Real} = nothing,
)
    model = inner.model
    for (p, value) in zip(inner.parameters, form.parameters.values)
        if p !== nothing && JuMP.fix_value(p) != value
            JuMP.fix(p, value)
        end
    end
    if time_limit !== nothing || inner.time_limited
        JuMP.set_time_limit_sec(model, time_limit)
        inner.time_limited = time_limit !== nothing
    end
    solve_start = time_ns()
    JuMP.optimize!(model)
    return _formulation_result(f, form, mom, model, inner.X, _seconds_since(solve_start))
end

# The form that the formulations build when constants depend on parameters (see `ParameterTerms`):
# each such parameter becomes a decision of stage 0, which observes only the constant column and
# is a fixed variable of the inner problem (see `_rule_variables!`). Its column in `A` is `-Bp`, so
# that `A x - Bp p (=, ≤, ≥) b + (B ξ without its constant)`, and its cost in the constant column
# of `C` is `fp`; `B[:, 1]` and `f` are those without the parameters. Returns the form and the
# columns of `ParameterTerms` of the parameters added, in order.
function _with_parameter_decisions(form::CanonicalForm)
    terms = form.parameters
    _has_parameter_terms(terms) || return form, Int[]
    in_use(Bp) = [!isempty(SparseArrays.nzrange(Bp, k)) for k in axes(Bp, 2)]
    in_rows = in_use(terms.Bp_e) .| in_use(terms.Bp_u) .| in_use(terms.Bp_l)
    used = findall(in_rows .| (terms.fp .!= 0))
    q = length(used)
    extend(A, Bp) = hcat(A, -Bp[:, used])
    cost = SparseArrays.sparse(1:q, ones(Int, q), terms.fp[used], q, size(form.C, 2))
    names = ["p[$k]" for k in used]
    extended = _replace(
        form;
        decisions = vcat(form.decisions, terms.parameters[used]),
        decision_names = vcat(form.decision_names, names),
        decision_stages = vcat(form.decision_stages, zeros(Int, q)),
        integer = vcat(form.integer, falses(q)),
        binary = vcat(form.binary, falses(q)),
        lower = vcat(form.lower, fill(-Inf, q)),
        upper = vcat(form.upper, fill(Inf, q)),
        A_e = extend(form.A_e, terms.Bp_e),
        B_e = _with_constant_column(form.B_e, terms.b_e),
        A_u = extend(form.A_u, terms.Bp_u),
        B_u = _with_constant_column(form.B_u, terms.b_u),
        A_l = extend(form.A_l, terms.Bp_l),
        B_l = _with_constant_column(form.B_l, terms.b_l),
        P = SparseArrays.blockdiag(form.P, SparseArrays.spzeros(q, q)),
        C = vcat(form.C, cost),
        f = terms.f,
    )
    return extended, used
end

# `B` with its first column replaced by `b`.
function _with_constant_column(B::_SparseMatrix, b::Vector{Float64})
    I, J, V = SparseArrays.findnz(B[:, 2:end])
    rows = findall(!iszero, b)
    return SparseArrays.sparse(
        [rows; I],
        [ones(Int, length(rows)); J .+ 1],
        [b[rows]; V],
        size(B)...,
    )
end

# Whether the values of `f` are exact for the laws of `form`, and if not, why (empty for a rule that
# is feasible on drawn outcomes only, whose values are estimates by construction).
function _certification(f::AbstractFormulation, form::CanonicalForm, mom::Moments)
    if capabilities(f).rule_status == MOI.UNKNOWN_RESULT_STATUS
        return false, ""
    end
    reasons = String[]
    if _is_estimated(mom)
        push!(reasons, "moments of conditioned laws estimated by rejection sampling")
    end
    if capabilities(f).bound
        # The dual bound needs `M ≻ 0` (the proof inverts it) and linear conditional expectations.
        isempty(_singular_blocks(mom)) || push!(reasons, "the second-moment matrix is singular")
        nonlinear, unknown = _stage_dependent_laws(form)
        if !isempty(nonlinear)
            push!(reasons, "conditional expectations across stages are not linear")
        elseif !isempty(unknown)
            push!(reasons, "conditional expectations across stages are not known to be linear")
        end
    end
    return isempty(reasons), join(reasons, "; ")
end

function _raw_status(raw::String, reason::String)
    return isempty(reason) ? raw : "$raw [not certified: $reason]"
end

# `time_ns` rather than `time`, whose resolution can be 16 ms on Windows.
_seconds_since(start::UInt64) = (time_ns() - start) / 1e9

# The result of a formulation that was not built, because the time limit was reached before.
function _time_limit_result(f::AbstractFormulation, form::CanonicalForm, mom::Moments)
    k = 1 + length(form.uncertainties)
    certified, reason = _certification(f, form, mom)
    return FormulationResult(
        f,
        MOI.TIME_LIMIT,
        MOI.NO_SOLUTION,
        _raw_status("The time limit was reached before $f was built.", reason),
        NaN,
        NaN,
        certified,
        0.0,
        fill(NaN, length(form.decisions), k),
        JuMP.Model(),
    )
end

# `_build!(model, f, form, mom)` builds the inner problem of the formulation `f` in `model` and
# returns its rule variables `X` (see `_rule_variables!`).
function _build! end

# The rule variables: `X[i][j]` is the coefficient of decision `i` on column `j` of ξ, for its
# observed columns only. A decision that observes only the constant column keeps its bounds as
# variable bounds; the bounds of the other decisions are rows (see `_inequalities`). A decision
# that is a parameter of the form (see `_with_parameter_decisions`) is a variable of `model` fixed
# to its value. It is not a JuMP `Parameter`: MOI's bridge of parameters, used by solvers without
# them (HiGHS), cannot change their value (MOI 1.54), so `set_parameter_value` drops the solver's
# copy of the model and every re-solve starts from scratch, while `fix` changes a bound in place
# and the solver keeps its basis (inventory, T = 64: 0 simplex iterations instead of 4999).
function _rule_variables!(model::JuMP.Model, form::CanonicalForm, info::InformationStructure)
    X = Vector{Vector{JuMP.VariableRef}}(undef, length(form.decisions))
    named = JuMP.set_string_names_on_creation(model)
    terms = form.parameters
    parameter_column = Dict(v => k for (k, v) in enumerate(terms.parameters))
    for i in eachindex(form.decisions)
        k = get(parameter_column, form.decisions[i], 0)
        if k > 0
            name = named ? form.decision_names[i] : ""
            X[i] = [JuMP.@variable(model, base_name = name)]
            JuMP.fix(only(X[i]), terms.values[k])
            continue
        end
        X[i] = map(visible_columns(info, i)) do j
            return JuMP.@variable(model, base_name = named ? "X[$i,$j]" : "")
        end
        if info.num_visible[i] == 1
            isfinite(form.lower[i]) && JuMP.set_lower_bound(X[i][1], form.lower[i])
            isfinite(form.upper[i]) && JuMP.set_upper_bound(X[i][1], form.upper[i])
        end
        if form.binary[i]
            JuMP.set_binary(X[i][1])
        elseif form.integer[i]
            JuMP.set_integer(X[i][1])
        end
    end
    return X
end

# One row `a ⋅ x(ξ) (≤, =, ≥) b ⋅ ξ`, and the columns of ξ on which it depends.
struct _LinearRow
    decisions::Vector{Int}
    coefficients::Vector{Float64}
    b::SparseArrays.SparseVector{Float64,Int}
    # `(a ⋅ X)[j]` can be nonzero for `j in 1:num_visible`.
    num_visible::Int
end

# Columns `j` where the row `a ⋅ X - b` is structurally nonzero: an observed column of one of its
# decisions, or a nonzero of `b`.
function _structural_columns(row::_LinearRow)
    return vcat(1:row.num_visible, filter(>(row.num_visible), SparseArrays.nonzeroinds(row.b)))
end

function _is_deterministic(row::_LinearRow)
    return row.num_visible == 1 && all(<=(1), SparseArrays.nonzeroinds(row.b))
end

function _linear_rows(A::_SparseMatrix, B::_SparseMatrix, info::InformationStructure)
    At, Bt = SparseArrays.sparse(transpose(A)), SparseArrays.sparse(transpose(B))
    return map(axes(A, 1)) do r
        range = SparseArrays.nzrange(At, r)
        decisions = SparseArrays.rowvals(At)[range]
        num_visible = maximum((info.num_visible[i] for i in decisions); init = 1)
        return _LinearRow(decisions, SparseArrays.nonzeros(At)[range], Bt[:, r], num_visible)
    end
end

# One inequality `s(ξ) = sign * (b ⋅ ξ - a ⋅ x(ξ)) ≥ 0`.
struct _Inequality
    sign::Float64
    row::_LinearRow
end

# The inequalities of `form`: the rows of `A_u` (sign 1) and `A_l` (sign -1), then the finite
# bounds of the decisions that observe some uncertainty.
function _inequalities(form::CanonicalForm, info::InformationStructure)
    out = _Inequality[]
    append!(out, _Inequality.(1.0, _linear_rows(form.A_u, form.B_u, info)))
    append!(out, _Inequality.(-1.0, _linear_rows(form.A_l, form.B_l, info)))
    k = 1 + length(form.uncertainties)
    for (sign, bounds) in ((1.0, form.upper), (-1.0, form.lower)), i in eachindex(bounds)
        if isfinite(bounds[i]) && info.num_visible[i] > 1
            b = SparseArrays.sparsevec([1], [bounds[i]], k)
            push!(out, _Inequality(sign, _LinearRow([i], [1.0], b, info.num_visible[i])))
        end
    end
    return out
end

# Add `coef * (a ⋅ X)[j]` to `expr`.
function _add_row_column!(expr::JuMP.AffExpr, X, row::_LinearRow, j::Int, coef::Float64)
    for (i, a) in zip(row.decisions, row.coefficients)
        if j <= length(X[i])
            JuMP.add_to_expression!(expr, coef * a, X[i][j])
        end
    end
    return expr
end

# The equality rows hold for every ξ: match the coefficients of each column on which they depend.
function _add_equalities!(model::JuMP.Model, form::CanonicalForm, info::InformationStructure, X)
    for row in _linear_rows(form.A_e, form.B_e, info), j in _structural_columns(row)
        expr = _add_row_column!(JuMP.AffExpr(0.0), X, row, j, 1.0)
        JuMP.@constraint(model, expr == row.b[j])
    end
    return
end

# A deterministic inequality `sign * (b₁ - a ⋅ X[:, 1]) ≥ 0`.
function _add_deterministic!(model::JuMP.Model, X, ineq::_Inequality)
    expr = _add_row_column!(JuMP.AffExpr(ineq.sign * ineq.row.b[1]), X, ineq.row, 1, -ineq.sign)
    JuMP.@constraint(model, expr >= 0)
    return
end

# `E[η'Qη + d'η + f]`.
function _objective_constant(form::CanonicalForm, mom::Moments)
    μ = mom.mean[2:end]
    value = form.f + LinearAlgebra.dot(form.d, μ) + LinearAlgebra.dot(μ, form.Q, μ)
    for block in mom.blocks
        η = block.columns .- 1
        value += LinearAlgebra.dot(form.Q[η, η], block.covariance)
    end
    return value
end

# The objective `E[x'Px + x'Cξ] + E[η'Qη + d'η + f]` with `x = Xξ` and `E[ξξ'] = μμ' + Σ`, Σ block
# diagonal:
#   E[x'Px]  = (Xμ)'P(Xμ) + Σ_b tr(X_b' P X_b Σ_b),
#   E[x'Cξ] = ⟨C, XM⟩ = (Cμ)'(Xμ) + Σ_b ⟨C_b Σ_b, X_b⟩,
# where `X_b`, `C_b` are the columns of the covariance block `b`.
function _set_objective!(model::JuMP.Model, form::CanonicalForm, mom::Moments, X)
    if form.sense == MOI.FEASIBILITY_SENSE
        JuMP.set_objective_sense(model, MOI.FEASIBILITY_SENSE)
        return
    end
    μ = mom.mean
    obj = JuMP.QuadExpr(JuMP.AffExpr(_objective_constant(form, mom)))
    Cμ = form.C * μ
    for i in eachindex(X)
        iszero(Cμ[i]) && continue
        for j in eachindex(X[i])
            iszero(μ[j]) || JuMP.add_to_expression!(obj, Cμ[i] * μ[j], X[i][j])
        end
    end
    for block in mom.blocks
        Cb = form.C[:, block.columns]
        rows = unique!(sort(SparseArrays.rowvals(Cb)))
        isempty(rows) && continue
        CΣ = Matrix(Cb[rows, :]) * block.covariance
        for (r, i) in enumerate(rows), (c, j) in enumerate(block.columns)
            if j <= length(X[i]) && !iszero(CΣ[r, c])
                JuMP.add_to_expression!(obj, CΣ[r, c], X[i][j])
            end
        end
    end
    _add_quadratic_terms!(obj, model, form, mom, X)
    JuMP.set_objective(model, form.sense, isempty(obj.terms) ? obj.aff : obj)
    return
end

# `E[x'Px] = Σ_ik P_ik (E[x_i] E[x_k] + Σ_b X_ib' Σ_b X_kb)`, term by term: the columns of a block
# are visited in increasing order, so that those beyond what a decision observes are skipped. The
# mean `E[x_i] = X_i ⋅ μ` of a decision is a variable `m_i` with one equality, so that the first
# term has one entry per nonzero of `P` instead of one per pair of columns.
function _add_quadratic_terms!(
    obj::JuMP.QuadExpr,
    model::JuMP.Model,
    form::CanonicalForm,
    mom::Moments,
    X,
)
    μ = mom.mean
    order = [sortperm(block.columns) for block in mom.blocks]
    first_column = [minimum(block.columns; init = typemax(Int)) for block in mom.blocks]
    I, K, V = SparseArrays.findnz(form.P)
    means = _decision_means!(model, X, μ, unique(vcat(I, K)))
    for (i, k, p) in zip(I, K, V)
        Xi, Xk = X[i], X[k]
        JuMP.add_to_expression!(obj, p * means[i], means[k])
        for (b, block) in enumerate(mom.blocks)
            first_column[b] <= min(length(Xi), length(Xk)) || continue
            for a in order[b]
                ja = block.columns[a]
                ja <= length(Xi) || break
                for c in order[b]
                    jc = block.columns[c]
                    jc <= length(Xk) || break
                    s = p * block.covariance[a, c]
                    iszero(s) || JuMP.add_to_expression!(obj, s, Xi[ja], Xk[jc])
                end
            end
        end
    end
    return
end

# `E[x_i]` for the decisions `rows`: the expression `X_i ⋅ μ`, or a variable equal to it when it
# has more than one term.
function _decision_means!(model::JuMP.Model, X, μ::Vector{Float64}, rows)
    means = Dict{Int,JuMP.AffExpr}()
    for i in rows
        expr = JuMP.AffExpr(0.0)
        for j in eachindex(X[i])
            iszero(μ[j]) || JuMP.add_to_expression!(expr, μ[j], X[i][j])
        end
        if length(expr.terms) > 1
            m = JuMP.@variable(model)
            JuMP.@constraint(model, m == expr)
            means[i] = JuMP.AffExpr(0.0, m => 1.0)
        else
            means[i] = expr
        end
    end
    return means
end

function _has_integers(model::JuMP.Model)
    return JuMP.num_constraints(model, JuMP.VariableRef, MOI.Integer) +
           JuMP.num_constraints(model, JuMP.VariableRef, MOI.ZeroOne) > 0
end

# Whether the inner problem has an affine objective and scalar constraints only: for a quadratic
# or conic one, the dual objective of the solver can leave out terms (HiGHS's, for a QP).
function _is_linear_program(model::JuMP.Model)
    if !(JuMP.objective_function_type(model) <: Union{JuMP.VariableRef,JuMP.AffExpr})
        return false
    end
    return all(JuMP.list_of_constraint_types(model)) do (F, S)
        return F <: Union{JuMP.VariableRef,JuMP.AffExpr} && S <: MOI.AbstractScalarSet
    end
end

# The dual objective of an optimal linear program is a bound by weak duality, also when the primal
# point is only feasible up to a tolerance (an interior point without crossover). It is used when
# the solver provides it and it is finite, else the primal objective.
function _dual_objective_or(model::JuMP.Model, objective_value::Float64)
    if JuMP.dual_status(model) == MOI.FEASIBLE_POINT
        try
            bound = JuMP.dual_objective_value(model)
            isfinite(bound) && return bound
        catch err
            err isa Union{MOI.UnsupportedAttribute,MOI.GetAttributeNotAllowed} || rethrow()
        end
    end
    return objective_value
end

# The bound that the inner solver proves, which needs no primal point: a MIP bound, the dual
# objective of an LP, or no bound at all for an unbounded inner problem.
function _objective_bound(form::CanonicalForm, model::JuMP.Model, objective_value::Float64)
    status = JuMP.termination_status(model)
    if form.sense == MOI.FEASIBILITY_SENSE
        return objective_value
    elseif status == MOI.DUAL_INFEASIBLE
        return form.sense == MOI.MAX_SENSE ? Inf : -Inf
    elseif status in _NO_SOLUTION_STATUS
        return NaN
    elseif _has_integers(model)
        return JuMP.objective_bound(model)
    elseif status == MOI.OPTIMAL && _is_linear_program(model)
        return _dual_objective_or(model, objective_value)
    elseif status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
        return objective_value
    elseif status in (MOI.TIME_LIMIT, MOI.ITERATION_LIMIT) &&
           JuMP.dual_status(model) == MOI.FEASIBLE_POINT
        # A solver stopped by a limit keeps its dual feasible point (dual simplex), so its value
        # is still a bound. After an error (e.g. HiGHS's interior point without crossover ending in
        # OTHER_ERROR), the dual point is feasible only up to a tolerance and its value can lie on
        # the wrong side of the optimum, so there is no bound.
        return JuMP.dual_objective_value(model)
    end
    return NaN
end

# A solver can report a feasible point of an unbounded problem (the origin of a ray), or a point
# along with an infeasibility certificate: neither is a solution.
const _NO_SOLUTION_STATUS =
    (MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE, MOI.DUAL_INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)

function _formulation_result(
    f::AbstractFormulation,
    form::CanonicalForm,
    mom::Moments,
    model::JuMP.Model,
    X,
    time::Float64,
)
    termination = JuMP.termination_status(model)
    status = JuMP.primal_status(model)
    if termination in _NO_SOLUTION_STATUS && status == MOI.FEASIBLE_POINT
        status = MOI.NO_SOLUTION
    end
    k = 1 + length(form.uncertainties)
    if status == MOI.FEASIBLE_POINT
        values = zeros(length(X), k)
        for i in eachindex(X)
            values[i, eachindex(X[i])] .= JuMP.value.(X[i])
        end
        objective = form.sense == MOI.FEASIBILITY_SENSE ? 0.0 : JuMP.objective_value(model)
    else
        values = fill(NaN, length(X), k)
        objective = NaN
    end
    certified, reason = _certification(f, form, mom)
    return FormulationResult(
        f,
        termination,
        status,
        _raw_status(JuMP.raw_status(model), reason),
        objective,
        _objective_bound(form, model, objective),
        certified,
        time,
        values,
        model,
    )
end
