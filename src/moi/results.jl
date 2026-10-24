# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Results follow the design of DESIGN.md:
#
#  * result 1 is the rule of `PrimalLDR` (feasible for every outcome in Ξ), result 2 the rule of
#    `SampledLDR` (feasible on the drawn outcomes only, `UNKNOWN_RESULT_STATUS`);
#  * the rule of `DualLDR` is not a policy, so it has no result: its value is `ObjectiveBound`,
#    and `RelativeGap` is the gap between the primal and the dual;
#  * `VariablePrimal` is the expected value `E[x(ξ)]` of a decision, `E[d]` of an uncertainty.

function _find_result(o::Optimizer, f::AbstractFormulation)
    k = findfirst(r -> r.formulation == f, o.results)
    return k === nothing ? nothing : o.results[k]
end

_has_rule(r::FormulationResult) = capabilities(r.formulation).rule_status != MOI.NO_SOLUTION

_is_bound(r::FormulationResult) = capabilities(r.formulation).bound

# The results with a rule that is a solution, primal rules first: the results of `MOI`.
function _rule_results(o::Optimizer)
    usable = filter(r -> _has_rule(r) && r.primal_status == MOI.FEASIBLE_POINT, o.results)
    return sort(usable; by = r -> !(r.formulation isa PrimalLDR), alg = Base.Sort.DEFAULT_STABLE)
end

MOI.get(o::Optimizer, ::MOI.ResultCount) = length(_rule_results(o))

function MOI.get(o::Optimizer, attr::MOI.PrimalStatus)
    results = _rule_results(o)
    if !(1 <= attr.result_index <= length(results))
        return MOI.NO_SOLUTION
    end
    return capabilities(results[attr.result_index].formulation).rule_status
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
    MOI.check_result_index_bounds(o, attr)
    return _rule_results(o)[attr.result_index].objective_value
end

# The bound of the first formulation that has one, with a value.
function MOI.get(o::Optimizer, attr::MOI.ObjectiveBound)
    k = findfirst(r -> _is_bound(r) && !isnan(r.objective_bound), o.results)
    if k === nothing
        throw(
            MOI.GetAttributeNotAllowed(
                attr,
                "There is no bound: solve `DualLDR()`, which must have a solution.",
            ),
        )
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
    MOI.check_result_index_bounds(o, attr)
    result = _rule_results(o)[attr.result_index]
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
function _require_value(attr::FormulationAttribute, value::Float64)
    if isnan(value)
        throw(MOI.GetAttributeNotAllowed(attr, "$(attr.formulation) has no solution."))
    end
    return value
end

function MOI.get(o::Optimizer, attr::FormulationAttribute{MOI.TerminationStatus})
    r = _find_result(o, attr.formulation)
    return r === nothing ? MOI.OPTIMIZE_NOT_CALLED : r.termination_status
end

function MOI.get(o::Optimizer, attr::FormulationAttribute{MOI.PrimalStatus})
    r = _require_result(o, attr)
    return r.primal_status == MOI.FEASIBLE_POINT ? capabilities(r.formulation).rule_status :
           r.primal_status
end

function MOI.get(o::Optimizer, attr::FormulationAttribute{MOI.ObjectiveValue})
    r = _require_result(o, attr)
    return _require_value(attr, _is_bound(r) ? r.objective_bound : r.objective_value)
end

function MOI.get(o::Optimizer, attr::FormulationAttribute{MOI.ObjectiveBound})
    return _require_value(attr, _require_result(o, attr).objective_bound)
end

function MOI.get(o::Optimizer, attr::FormulationAttribute{MOI.RawStatusString})
    return _require_result(o, attr).raw_status
end

function MOI.get(o::Optimizer, attr::FormulationAttribute{MOI.SolveTimeSec})
    return _require_result(o, attr).solve_time
end

# Internal attributes, for `decision_rule` and `expected_value`.
function MOI.get(o::Optimizer, attr::RuleCoefficients, vi::MOI.VariableIndex)
    r = _require_result(o, FormulationAttribute(attr.formulation, MOI.PrimalStatus()))
    if any(isnan, r.X)
        throw(MOI.GetAttributeNotAllowed(attr, "$(attr.formulation) has no solution."))
    end
    return _rule_coefficients(o, r, vi)
end

MOI.get(o::Optimizer, ::UncertaintyVariables) = copy(o.form.uncertainties)

MOI.get(o::Optimizer, ::XiMoments) = o.moments
