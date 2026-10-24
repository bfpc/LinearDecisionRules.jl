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
    PrimalLDR()

The primal linear decision rule: every constraint holds for every outcome in the support Ξ.

Its rule is a policy, so its objective value is attained (a lower bound of a maximization, an upper
bound of a minimization). Each inequality is replaced by its robust counterpart over the
[`SupportBlock`](@ref)s on which it depends (Kuhn, Wiesemann and Georghiou, 2011).
"""
struct PrimalLDR <: AbstractFormulation end

"""
    DualLDR()

The dual linear decision rule: a relaxation whose objective value bounds the optimal value of the
stochastic program (an upper bound of a maximization, a lower bound of a minimization).

Its rule is not a policy. Each inequality `s(ξ) ≥ 0` is relaxed to `E[s(ξ) (W'ξ)ᵀ] ≥ 0`, where
`W'ξ ≥ 0` describes Ξ, including the row `ξ₁ ≥ 0` (Kuhn, Wiesemann and Georghiou, 2011). The bound
is certified when the uncertainties of different stages are independent; building it warns when a
law correlates uncertainties of different stages.
"""
struct DualLDR <: AbstractFormulation end

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

What a formulation provides and supports, returned by [`capabilities`](@ref).

## Fields

 * `rule_status::MOI.ResultStatusCode`: the status of its decision rule as a solution of the
   stochastic program when the inner problem is solved: `FEASIBLE_POINT` (feasible for every outcome
   in Ξ), `UNKNOWN_RESULT_STATUS` (feasible on the drawn outcomes only) or `NO_SOLUTION` (the rule is
   not a policy).
 * `bound::Bool`: whether its objective value bounds the optimal value of the stochastic program
   from the side opposite to the sense (an upper bound of a maximization).
 * `integer::Bool`: whether it supports integer and binary decisions (of stage 1).
 * `quadratic_objective::Bool`: whether it supports a quadratic objective.
"""
struct FormulationCapabilities
    rule_status::MOI.ResultStatusCode
    bound::Bool
    integer::Bool
    quadratic_objective::Bool
end

"""
    capabilities(f::AbstractFormulation) -> FormulationCapabilities

Return the [`FormulationCapabilities`](@ref) of the formulation `f`.
"""
function capabilities(::PrimalLDR)
    return FormulationCapabilities(MOI.FEASIBLE_POINT, false, true, true)
end

capabilities(::DualLDR) = FormulationCapabilities(MOI.NO_SOLUTION, true, true, true)

function capabilities(::SampledLDR)
    return FormulationCapabilities(MOI.UNKNOWN_RESULT_STATUS, false, true, true)
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
   model: `MOI.ObjectiveBound` when the inner problem has integer variables, else the optimal value
   (or the dual objective value when the inner problem is not solved to optimality); `Inf` for a
   maximization (`-Inf` for a minimization) whose inner problem is unbounded; `NaN` if unknown. It
   needs no primal point. For [`DualLDR`](@ref) with integer decisions, this is the bound to report.
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
    ) -> FormulationResult

Build the inner problem of the formulation `f` for `form` and the moments `mom`, solve it with
`optimizer_factory` and return a [`FormulationResult`](@ref).

`time_limit` is in seconds, for building and solving the inner problem: the inner optimizer gets
what remains after the build. `raw_attributes` is an iterable of `name => value` pairs passed to
`JuMP.set_attribute` on the inner problem. [`SampledLDR`](@ref) ignores `mom` and uses the
empirical moments of its outcomes.

## Example

```julia
form = canonicalize(backend(model), info)
result = solve_formulation(form, PrimalLDR(), moments(form), HiGHS.Optimizer)
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
)
    start = time_ns()
    model = JuMP.Model(optimizer_factory)
    if silent
        JuMP.set_silent(model)
    end
    for (name, value) in raw_attributes
        JuMP.set_attribute(model, name, value)
    end
    X = _build!(model, f, form, mom)
    if time_limit !== nothing
        # The limit covers the build too, which can take longer than the solve.
        JuMP.set_time_limit_sec(model, max(0.0, time_limit - _seconds_since(start)))
    end
    solve_start = time_ns()
    JuMP.optimize!(model)
    return _formulation_result(f, form, model, X, _seconds_since(solve_start))
end

# `time_ns` rather than `time`, whose resolution can be 16 ms on Windows.
_seconds_since(start::UInt64) = (time_ns() - start) / 1e9

# The result of a formulation that was not built, because the time limit was reached before.
function _time_limit_result(f::AbstractFormulation, form::CanonicalForm)
    k = 1 + length(form.uncertainties)
    return FormulationResult(
        f,
        MOI.TIME_LIMIT,
        MOI.NO_SOLUTION,
        "The time limit was reached before $f was built.",
        NaN,
        NaN,
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
# variable bounds; the bounds of the other decisions are rows (see `_inequalities`).
function _rule_variables!(model::JuMP.Model, form::CanonicalForm, info::InformationStructure)
    X = Vector{Vector{JuMP.VariableRef}}(undef, length(form.decisions))
    for i in eachindex(form.decisions)
        X[i] = [JuMP.@variable(model, base_name = "X[$i,$j]") for j in visible_columns(info, i)]
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
# bounds of the decisions that observe some uncertainty, as the rows `Sxu` and `Sxl` of the
# previous version.
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
    expr = _add_row_column!(JuMP.AffExpr(0.0), X, ineq.row, 1, -ineq.sign)
    JuMP.@constraint(model, expr >= -ineq.sign * ineq.row.b[1])
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
    I, K, V = SparseArrays.findnz(form.P)
    if !isempty(V)
        Xμ = Dict(i => LinearAlgebra.dot(μ[eachindex(X[i])], X[i]) for i in unique(I))
        for (i, k, p) in zip(I, K, V)
            JuMP.add_to_expression!(obj, p * Xμ[i], Xμ[k])
            for block in mom.blocks
                ci = findall(<=(length(X[i])), block.columns)
                ck = findall(<=(length(X[k])), block.columns)
                if !isempty(ci) && !isempty(ck)
                    Σx = block.covariance[ci, ck] * X[k][block.columns[ck]]
                    JuMP.add_to_expression!(obj, p, LinearAlgebra.dot(X[i][block.columns[ci]], Σx))
                end
            end
        end
    end
    JuMP.set_objective(model, form.sense, isempty(obj.terms) ? obj.aff : obj)
    return
end

function _has_integers(model::JuMP.Model)
    return JuMP.num_constraints(model, JuMP.VariableRef, MOI.Integer) +
           JuMP.num_constraints(model, JuMP.VariableRef, MOI.ZeroOne) > 0
end

# The bound that the inner solver proves, which needs no primal point: a MIP bound, the dual
# objective of an LP stopped early, or no bound at all for an unbounded inner problem.
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
    elseif status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
        return objective_value
    elseif JuMP.dual_status(model) == MOI.FEASIBLE_POINT
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
    return FormulationResult(
        f,
        termination,
        status,
        JuMP.raw_status(model),
        objective,
        _objective_bound(form, model, objective),
        time,
        values,
        model,
    )
end
