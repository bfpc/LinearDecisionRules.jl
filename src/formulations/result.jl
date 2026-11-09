# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The result of a formulation: `FormulationResult`, its statuses, values, bound and certification.

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
   (a bound by weak duality, up to the solver's tolerance on the feasibility of its dual point:
   that of an interior point without crossover can be slightly invalid), else the optimal value;
   the dual objective value
   when the inner problem is not solved to optimality; `Inf` for a maximization (`-Inf` for a
   minimization) whose inner problem is unbounded; `NaN` if unknown. It needs no primal point. For [`DualLDR`](@ref) with integer decisions, this is the bound to report.
 * `certified::Bool`: whether `objective_value` and `objective_bound` are exact for the laws of the
   model, so that the value of [`PrimalLDR`](@ref) is attained and the bound of [`DualLDR`](@ref)
   is a bound. It is `false` when they use moments estimated by rejection sampling (laws
   conditioned by constraints tagged [`Truncate`](@ref)), for a bound when the second-moment
   matrix is singular or the quadratic objective is not convex (a law without linear conditional
   expectations does not prevent it, see [`DualLDR`](@ref)), and for a rule that is feasible on
   drawn outcomes only
   ([`SampledLDR`](@ref)). Except for the latter, `raw_status` then ends with
   `[not certified: <reason>]`.
 * `solve_time::Float64`: wall-clock seconds of the inner `optimize!` (zero if it was not built).
 * `build_time::Float64`: wall-clock seconds to build the inner problem, and to set the values of
   its parameters; only the latter when the optimizer solves again the inner problem of its last
   `optimize!`, and zero if it was not built.
 * `results_time::Float64`: wall-clock seconds to read the statuses, the values and the rules from
   the inner problem after its `optimize!`.
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
    build_time::Float64
    results_time::Float64
    X::Matrix{Float64}
    model::JuMP.Model
end

function Base.show(io::IO, result::FormulationResult)
    print(io, "FormulationResult of $(result.formulation): $(result.termination_status), ")
    print(io, "$(result.primal_status), objective value $(result.objective_value)")
    return
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
        # The dual bound needs a convex objective and `M ≻ 0` (the proof inverts it). It does not
        # need linear conditional expectations: where they fail or are not known, the dual relaxes
        # its information structure (`_dual_information`).
        _nonconvex_decisions(form) === nothing || push!(reasons, "the objective is not convex")
        isempty(_singular_blocks(mom)) || push!(reasons, "the second-moment matrix is singular")
    end
    return isempty(reasons), join(reasons, "; ")
end

function _raw_status(raw::String, reason::String)
    return isempty(reason) ? raw : "$raw [not certified: $reason]"
end

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
        0.0,
        0.0,
        fill(NaN, length(form.decisions), k),
        JuMP.Model(),
    )
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

# The dual objective of an optimal linear program is a bound by weak duality, up to the tolerance on
# the feasibility of the dual point (looser for an interior point without crossover). It is used when
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
    X;
    solve_time::Float64,
    build_time::Float64,
)
    start = time_ns()
    termination = JuMP.termination_status(model)
    status = JuMP.primal_status(model)
    if termination in _NO_SOLUTION_STATUS && status == MOI.FEASIBLE_POINT
        status = MOI.NO_SOLUTION
    end
    k = 1 + length(form.uncertainties)
    if status == MOI.FEASIBLE_POINT
        values = zeros(length(X), k)
        for i in eachindex(X)
            values[i, X[i].columns] .= JuMP.value.(X[i].variables)
        end
        objective = form.sense == MOI.FEASIBILITY_SENSE ? 0.0 : JuMP.objective_value(model)
    else
        values = fill(NaN, length(X), k)
        objective = NaN
    end
    certified, reason = _certification(f, form, mom)
    raw = _raw_status(JuMP.raw_status(model), reason)
    bound = _objective_bound(form, model, objective)
    return FormulationResult(
        f,
        termination,
        status,
        raw,
        objective,
        bound,
        certified,
        solve_time,
        build_time,
        _seconds_since(start),
        values,
        model,
    )
end
