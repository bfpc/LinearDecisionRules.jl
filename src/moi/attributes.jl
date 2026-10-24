# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    AbstractXiRole

The role of an uncertainty-only row in the description of the uncertainties: see
[`ValidInequality`](@ref) and [`Truncate`](@ref).
"""
abstract type AbstractXiRole end

# So that `@constraint(model, x .<= 1, ValidInequality())` broadcasts.
Base.broadcastable(role::AbstractXiRole) = Ref(role)

"""
    ValidInequality()

The tag of a constraint over the uncertainties only that tightens the support Ξ without changing
their law.

## Example

```julia
@constraint(model, d[1] + d[2] <= 150, ValidInequality())
```
"""
struct ValidInequality <: AbstractXiRole end

"""
    Truncate()

The tag of a constraint over the uncertainties only that conditions their law on the constraint
(rejection sampling). Not supported yet.
"""
struct Truncate <: AbstractXiRole end

"""
    XiRole()

A constraint attribute: the [`AbstractXiRole`](@ref) of an uncertainty-only row, or `nothing`.
It is set by `@constraint(model, expr, ValidInequality())` and read by `ModelInfo`.
"""
struct XiRole <: MOI.AbstractConstraintAttribute end

"""
    ConstraintStage()

A constraint attribute: the stage of a constraint, as an `Int`, or `nothing`. It is set by
`@constraint(model, expr, Stage(t))`. It is stored and not read yet, since it is reserved for
expectation constraints at time `t`.
"""
struct ConstraintStage <: MOI.AbstractConstraintAttribute end

"""
    BreakPoints()

A variable attribute: the breakpoints of the [`AxisPiecewise`](@ref) lifting of an uncertainty,
as a vector of interior breakpoints, or as an integer `n` for `n` equally spaced breakpoints inside
the support. `nothing` (the default), an empty vector or `0` mean no lifting.

The uncertainty must be declared alone, with a univariate law, and the breakpoints must lie
strictly inside its support. At `optimize!`, the uncertainty is replaced by its pieces (see
[`lift`](@ref)), so that the decision rules are piecewise affine in it: [`decision_rule`](@ref)
gives the slope of a rule on each piece.

## Example

```julia
set_attribute(d, BreakPoints(), [90.0, 100.0, 110.0])
set_attribute(d, BreakPoints(), 3)        # the same, for d ~ Uniform(80, 120)
set_attribute(d, BreakPoints(), nothing)  # no lifting
```
"""
struct BreakPoints <: MOI.AbstractVariableAttribute end

"""
    InnerOptimizer()

An optimizer attribute: the factory of the optimizer that solves the reformulated models, for
example `HiGHS.Optimizer`. [`LDRModel`](@ref)`(factory)` sets it.

## Example

```julia
set_attribute(model, InnerOptimizer(), HiGHS.Optimizer)
```
"""
struct InnerOptimizer <: MOI.AbstractOptimizerAttribute end

"""
    Formulations()

An optimizer attribute: the vector of formulations that `optimize!` solves, at most one of each
type.

The results do not depend on the order: result 1 is always the rule of [`PrimalLDR`](@ref) and
result 2 the rule of [`SampledLDR`](@ref), with status `NO_SOLUTION` when their formulation was not
solved or has no solution. `result_count` is 2 with `SampledLDR`, else 1 with `PrimalLDR`, else 0.
The bound of [`DualLDR`](@ref) is `objective_bound`.

## Example

```julia
set_attribute(model, Formulations(), [PrimalLDR(), DualLDR()])
```
"""
struct Formulations <: MOI.AbstractOptimizerAttribute end

"""
    FormulationAttribute(formulation::AbstractFormulation, attribute)

A model attribute: the `attribute` of one formulation, for example
`FormulationAttribute(DualLDR(), MOI.TerminationStatus())`.

`attribute` is one of `MOI.TerminationStatus()`, `MOI.PrimalStatus()`, `MOI.ObjectiveValue()`,
`MOI.ObjectiveBound()`, `MOI.RawStatusString()` and `MOI.SolveTimeSec()`. The objective value of a
formulation whose rule is not a policy, [`DualLDR`](@ref), is its bound.

See also `JuMP.termination_status(model, formulation)` and the other short helpers.
"""
struct FormulationAttribute{A<:MOI.AbstractModelAttribute} <: MOI.AbstractModelAttribute
    formulation::AbstractFormulation
    attribute::A
end

MOI.is_set_by_optimize(::FormulationAttribute) = true

# The attributes below are internal: the JuMP functions of `jump/rules.jl` read them. The
# columns ξ are those of the canonical form, so the lifted coordinates when there are breakpoints.

# The rule of a variable on ξ = (1, η), as a `Vector{Float64}`.
struct RuleCoefficients <: MOI.AbstractVariableAttribute
    formulation::AbstractFormulation
end

# The uncertainties of the model, before lifting: those of the columns of ξ after the constant
# when nothing is lifted.
struct UncertaintyVariables <: MOI.AbstractModelAttribute end

# The decisions of the model, in the rows of `FormulationResult.X`.
struct DecisionVariables <: MOI.AbstractModelAttribute end

# The `LiftingMap` of the last solve, or `nothing` if nothing was lifted.
struct UncertaintyLifting <: MOI.AbstractModelAttribute end

# The `Moments` of ξ of the last solve.
struct UncertaintyMoments <: MOI.AbstractModelAttribute end

# What `evaluate` and `simulate` read: the form and the rule of `formulation` of the last solve.
struct SimulationData <: MOI.AbstractModelAttribute
    formulation::AbstractFormulation
end

function MOI.is_set_by_optimize(
    ::Union{
        RuleCoefficients,
        UncertaintyVariables,
        DecisionVariables,
        UncertaintyLifting,
        UncertaintyMoments,
        SimulationData,
    },
)
    return true
end

# The values of these attributes hold no indices, except `UncertaintyVariables` and
# `UncertaintyLifting`, which the identity map of `optimize!` leaves as they are.
const _AttributesWithoutIndices = Union{
    XiRole,
    ConstraintStage,
    BreakPoints,
    InnerOptimizer,
    Formulations,
    FormulationAttribute,
    RuleCoefficients,
    UncertaintyVariables,
    DecisionVariables,
    UncertaintyLifting,
    UncertaintyMoments,
    SimulationData,
}

MOI.Utilities.map_indices(::Any, ::_AttributesWithoutIndices, x) = x
