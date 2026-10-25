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
not a policy. It is `NO_SOLUTION` if the inner problem has no solution (see
`termination_status(model, formulation)`), if `formulation` was not solved, or if the model
changed since the last `optimize!`.
"""
function JuMP.primal_status(model::JuMP.GenericModel, f::AbstractFormulation)
    _optimize_not_called(model) && return MOI.NO_SOLUTION
    return _formulation_attribute(model, f, MOI.PrimalStatus())
end

"""
    JuMP.objective_value(model::JuMP.Model, formulation::AbstractFormulation)

The objective value of `formulation`. For [`DualLDR`](@ref), it is the bound on the optimal value,
which is the inner solver's `MOI.ObjectiveBound` when there are integer decisions; it needs no
primal point, and it is `Inf` for a maximization (`-Inf` for a minimization) whose dual LDR is
unbounded.

It is an error if the inner problem has no solution (no bound, for `DualLDR`).
"""
function JuMP.objective_value(model::JuMP.GenericModel, f::AbstractFormulation)
    return _formulation_attribute(model, f, MOI.ObjectiveValue())
end

"""
    JuMP.objective_bound(model::JuMP.Model, formulation::AbstractFormulation)

The inner solver's bound on the objective value of `formulation`, even without a primal point (see
[`FormulationResult`](@ref)).
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

"""
    is_certified(model::JuMP.Model, formulation::AbstractFormulation = DualLDR()) -> Bool

Whether the values of `formulation` are exact for the laws of the model: for [`DualLDR`](@ref),
whether `objective_bound(model)` is a bound on the optimal value; for [`PrimalLDR`](@ref), whether
its objective value is the expected value of its rule. It is the attribute
`FormulationAttribute(formulation, Certified())` (see [`Certified`](@ref)).

The values are not certified when they use moments estimated by rejection sampling (a law
conditioned by constraints tagged [`Truncate`](@ref) that is not exact), and, for the bound of
`DualLDR`, when the second-moment matrix is singular or a law spanning several stages is not known
to have linear conditional expectations (see [`DualLDR`](@ref)). `raw_status(model, formulation)`
then ends with the reason, `[not certified: <reason>]`. The values of [`SampledLDR`](@ref) are
estimates by construction: they are never certified, and have no such reason.

It is an error if `formulation` was not solved, or before `optimize!`.

## Example

```julia
optimize!(model)
is_certified(model)               # the bound of DualLDR()
is_certified(model, PrimalLDR())
```
"""
function is_certified(model::JuMP.GenericModel, f::AbstractFormulation = DualLDR())
    return _formulation_attribute(model, f, Certified())
end
