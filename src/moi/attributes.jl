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

A variable attribute: the breakpoints of the piecewise lifting of an uncertainty, as a
`Vector{Float64}`. It is stored and read back, but the lifting is not supported yet.

## Example

```julia
set_attribute(d, BreakPoints(), [90.0, 100.0, 110.0])
```
"""
struct BreakPoints <: MOI.AbstractVariableAttribute end

# The values of these attributes hold no indices.
function MOI.Utilities.map_indices(::Any, ::Union{XiRole,ConstraintStage,BreakPoints}, x)
    return x
end

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

An optimizer attribute: the vector of formulations that `optimize!` solves.

## Example

```julia
set_attribute(model, Formulations(), [PrimalLDR(), DualLDR()])
```
"""
struct Formulations <: MOI.AbstractOptimizerAttribute end
