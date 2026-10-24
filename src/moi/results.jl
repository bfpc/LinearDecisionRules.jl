# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Results follow the design of DESIGN.md:
#
#  * result 1 is the rule of `PrimalLDR` (feasible for every outcome in Ξ), result 2 the rule of
#    `SampledLDR` (feasible on the drawn outcomes only, `UNKNOWN_RESULT_STATUS`); a result whose
#    formulation was not solved or has no solution is `NO_SOLUTION`, and keeps its index;
#  * the rule of `DualLDR` is not a policy, so it has no result: its value is `ObjectiveBound`,
#    and `RelativeGap` is the gap between the primal and the dual;
#  * `VariablePrimal` is the expected value `E[x(ξ)]` of a decision, `E[d]` of an uncertainty.

function _find_result(o::Optimizer, f::AbstractFormulation)
    k = findfirst(r -> r.formulation == f, o.results)
    return k === nothing ? nothing : o.results[k]
end

_has_rule(r::FormulationResult) = capabilities(r.formulation).rule_status != MOI.NO_SOLUTION

_is_bound(r::FormulationResult) = capabilities(r.formulation).bound

_has_solution(r::FormulationResult) = r.primal_status == MOI.FEASIBLE_POINT

# The formulation of each result slot. A slot keeps its index whether or not its formulation was
# solved or has a solution, so that result 1 is always the primal rule.
const _RESULT_SLOTS = (PrimalLDR, SampledLDR)

function _slot_result(o::Optimizer, k::Int)
    k in eachindex(_RESULT_SLOTS) || return nothing
    i = findfirst(r -> r.formulation isa _RESULT_SLOTS[k], o.results)
    return i === nothing ? nothing : o.results[i]
end

# Up to the last slot whose formulation was solved: 2 with `SampledLDR`, even without `PrimalLDR`.
function MOI.get(o::Optimizer, ::MOI.ResultCount)
    k = findlast(T -> any(r -> r.formulation isa T, o.results), _RESULT_SLOTS)
    return k === nothing ? 0 : k
end

function MOI.get(o::Optimizer, attr::MOI.PrimalStatus)
    r = _slot_result(o, attr.result_index)
    if r === nothing || !_has_solution(r)
        return MOI.NO_SOLUTION
    end
    return capabilities(r.formulation).rule_status
end

# The result of the slot of `attr`, which must have a solution.
function _solution_of_slot(o::Optimizer, attr::Union{MOI.ObjectiveValue,MOI.VariablePrimal})
    MOI.check_result_index_bounds(o, attr)
    k = attr.result_index
    r = _slot_result(o, k)
    what = "Result $k is the rule of $(nameof(_RESULT_SLOTS[k]))"
    if r === nothing
        msg = "$what, which was not solved: add it to `Formulations()` and solve again."
        throw(MOI.GetAttributeNotAllowed(attr, msg))
    elseif !_has_solution(r)
        msg = "$what, which has no solution (termination status $(r.termination_status))."
        throw(MOI.GetAttributeNotAllowed(attr, msg))
    end
    return r
end

MOI.get(::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION

# Only the dual proves that the model is infeasible: a primal or sampled rule can be infeasible
# in a feasible model.
function _termination_status(o::Optimizer)
    if isempty(o.results)
        return MOI.OPTIMIZE_NOT_CALLED
    end
    if any(r -> _is_bound(r) && r.termination_status == MOI.INFEASIBLE, o.results)
        return MOI.INFEASIBLE
    end
    candidates = filter(_has_rule, o.results)
    principal = isempty(candidates) ? o.results[1] : first(sort(candidates; by = _rank))
    status = principal.termination_status
    if status == MOI.INFEASIBLE && _has_rule(principal)
        return MOI.LOCALLY_INFEASIBLE
    end
    return status
end

_rank(r::FormulationResult) = r.formulation isa PrimalLDR ? 0 : 1

MOI.get(o::Optimizer, ::MOI.TerminationStatus) = _termination_status(o)

function MOI.get(o::Optimizer, ::MOI.RawStatusString)
    return join(("$(r.formulation): $(r.raw_status)" for r in o.results), "; ")
end

MOI.get(o::Optimizer, ::MOI.SolveTimeSec) = o.solve_time

function MOI.get(o::Optimizer, attr::MOI.ObjectiveValue)
    return _solution_of_slot(o, attr).objective_value
end

# The bound of the first formulation that has one, with a value.
function MOI.get(o::Optimizer, attr::MOI.ObjectiveBound)
    k = findfirst(r -> _is_bound(r) && !isnan(r.objective_bound), o.results)
    if k === nothing
        msg = "There is no bound: solve `DualLDR()`, whose inner solver must prove one."
        throw(MOI.GetAttributeNotAllowed(attr, msg))
    end
    return o.results[k].objective_bound
end

function MOI.get(o::Optimizer, ::MOI.RelativeGap)
    value = MOI.get(o, MOI.ObjectiveValue())
    bound = MOI.get(o, MOI.ObjectiveBound())
    return iszero(value) ? (bound == value ? 0.0 : Inf) : abs(bound - value) / abs(value)
end

# The rule of `vi` on ξ = (1, η) in the result `r`.
function _rule_coefficients(o::Optimizer, r::FormulationResult, vi::MOI.VariableIndex)
    k = size(r.X, 2)
    if haskey(o.decision_rows, vi)
        return r.X[o.decision_rows[vi], :]
    end
    rule = zeros(k)
    if haskey(o.uncertainty_columns, vi)
        rule[o.uncertainty_columns[vi]] = 1.0
    elseif haskey(o.parameters, vi)
        rule[1] = o.parameters[vi]
    else
        throw(MOI.InvalidIndex(vi))
    end
    return rule
end

function MOI.get(o::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    result = _solution_of_slot(o, attr)
    return LinearAlgebra.dot(_rule_coefficients(o, result, vi), o.moments.mean)
end

function _require_result(o::Optimizer, attr::FormulationAttribute)
    r = _find_result(o, attr.formulation)
    if r === nothing
        throw(
            MOI.GetAttributeNotAllowed(
                attr,
                "$(attr.formulation) was not solved: add it to `Formulations()` and solve again.",
            ),
        )
    end
    return r
end

# A value that is `NaN` is unknown: it is an error, not a result.
function _require_value(attr::FormulationAttribute, value::Float64, what::String)
    if isnan(value)
        throw(MOI.GetAttributeNotAllowed(attr, "$(attr.formulation) has no $what."))
    end
    return value
end

function MOI.get(o::Optimizer, attr::FormulationAttribute{MOI.TerminationStatus})
    r = _find_result(o, attr.formulation)
    return r === nothing ? MOI.OPTIMIZE_NOT_CALLED : r.termination_status
end

# Statuses never throw: a formulation that was not solved has no solution.
function MOI.get(o::Optimizer, attr::FormulationAttribute{MOI.PrimalStatus})
    r = _find_result(o, attr.formulation)
    if r === nothing || !_has_solution(r)
        return MOI.NO_SOLUTION
    end
    return capabilities(r.formulation).rule_status
end

# The value of a bound needs no primal point: an inner MIP can prove a bound without an incumbent.
function MOI.get(o::Optimizer, attr::FormulationAttribute{MOI.ObjectiveValue})
    r = _require_result(o, attr)
    if _is_bound(r)
        return _require_value(attr, r.objective_bound, "bound")
    end
    return _require_value(attr, _has_solution(r) ? r.objective_value : NaN, "solution")
end

function MOI.get(o::Optimizer, attr::FormulationAttribute{MOI.ObjectiveBound})
    r = _require_result(o, attr)
    return _require_value(attr, r.objective_bound, "bound")
end

function MOI.get(o::Optimizer, attr::FormulationAttribute{MOI.RawStatusString})
    r = _find_result(o, attr.formulation)
    return r === nothing ? "optimize not called" : r.raw_status
end

function MOI.get(o::Optimizer, attr::FormulationAttribute{MOI.SolveTimeSec})
    return _require_result(o, attr).solve_time
end

# Internal attributes, for `decision_rule` and `expected_value`.
function MOI.get(o::Optimizer, attr::RuleCoefficients, vi::MOI.VariableIndex)
    r = _require_result(o, FormulationAttribute(attr.formulation, MOI.PrimalStatus()))
    if !_has_solution(r)
        throw(MOI.GetAttributeNotAllowed(attr, "$(attr.formulation) has no solution."))
    end
    return _rule_coefficients(o, r, vi)
end

MOI.get(o::Optimizer, ::UncertaintyVariables) = copy(o.form.uncertainties)

MOI.get(o::Optimizer, ::XiMoments) = o.moments
