# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The results of one formulation, `termination_status(model, DualLDR())` and the like. The
# methods take one of our types, `AbstractFormulation`, so they extend JuMP without piracy.

function _formulation_attribute(model::JuMP.GenericModel, f::AbstractFormulation, attribute)
    return MOI.get(model, FormulationAttribute(f, attribute))
end

# As in JuMP, statuses do not throw before `optimize!` or after a change to the model, which
# JuMP reports as `OPTIMIZE_NOT_CALLED`; values throw `OptimizeNotCalled`.
function _optimize_not_called(model::JuMP.GenericModel)
    return JuMP.termination_status(model) == MOI.OPTIMIZE_NOT_CALLED
end

"""
    JuMP.termination_status(model::JuMP.Model, formulation::AbstractFormulation)

The termination status of the inner problem of `formulation`, or `OPTIMIZE_NOT_CALLED` if it was
not solved, or if the model changed since the last `optimize!`.
"""
function JuMP.termination_status(model::JuMP.GenericModel, f::AbstractFormulation)
    _optimize_not_called(model) && return MOI.OPTIMIZE_NOT_CALLED
    return _formulation_attribute(model, f, MOI.TerminationStatus())
end

"""
    JuMP.primal_status(model::JuMP.Model, formulation::AbstractFormulation)

The status of the rule of `formulation`: `FEASIBLE_POINT` for [`PrimalLDR`](@ref),
`UNKNOWN_RESULT_STATUS` for [`SampledLDR`](@ref), `NO_SOLUTION` for [`DualLDR`](@ref), whose rule is
not a policy. If the inner problem has no solution, the status of the inner problem.
"""
function JuMP.primal_status(model::JuMP.GenericModel, f::AbstractFormulation)
    _optimize_not_called(model) && return MOI.NO_SOLUTION
    return _formulation_attribute(model, f, MOI.PrimalStatus())
end

"""
    JuMP.objective_value(model::JuMP.Model, formulation::AbstractFormulation)

The objective value of `formulation`. For [`DualLDR`](@ref), it is the bound on the optimal value,
which is the inner solver's `MOI.ObjectiveBound` when there are integer decisions.

It is an error if the inner problem has no solution.
"""
function JuMP.objective_value(model::JuMP.GenericModel, f::AbstractFormulation)
    return _formulation_attribute(model, f, MOI.ObjectiveValue())
end

"""
    JuMP.objective_bound(model::JuMP.Model, formulation::AbstractFormulation)

The inner solver's bound on the objective value of `formulation`.
"""
function JuMP.objective_bound(model::JuMP.GenericModel, f::AbstractFormulation)
    return _formulation_attribute(model, f, MOI.ObjectiveBound())
end

"""
    JuMP.raw_status(model::JuMP.Model, formulation::AbstractFormulation)

The raw status of the inner problem of `formulation`, or `"optimize not called"`.
"""
function JuMP.raw_status(model::JuMP.GenericModel, f::AbstractFormulation)
    _optimize_not_called(model) && return "optimize not called"
    return _formulation_attribute(model, f, MOI.RawStatusString())
end

"""
    JuMP.solve_time(model::JuMP.Model, formulation::AbstractFormulation)

The wall-clock seconds of the inner `optimize!` of `formulation`.
"""
function JuMP.solve_time(model::JuMP.GenericModel, f::AbstractFormulation)
    return _formulation_attribute(model, f, MOI.SolveTimeSec())
end
