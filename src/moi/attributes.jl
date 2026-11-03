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

The tag of a constraint over the uncertainties only that conditions their law on the constraint:
the uncertainties follow their declared laws conditioned on the set `C` where the constraints
tagged `Truncate()` hold, and `C` bounds the support Ξ.

A univariate law conditioned by constraints on it alone, such as `100 <= d <= 110`, is truncated
exactly (`Distributions.truncated`). Otherwise, the declarations coupled by such constraints share
one law. When all their laws have finitely many atoms (such as [`MvDiscreteNonParametric`](@ref)
or `DiscreteNonParametric`), it is exact: the atoms of their product that satisfy the constraints,
with their probabilities renormalized. Otherwise it is a [`ConditionedDistribution`](@ref), whose
moments are estimated by rejection sampling (see [`RejectionSamplingMaxSamples`](@ref)); the values that use them,
such as the bound of [`DualLDR`](@ref), are then estimates. [`SampledLDR`](@ref) and
[`simulate`](@ref) draw from the conditioned law. The constraint must not be an equality, since Ξ
must be full-dimensional.

## Example

```julia
@variable(model, d in Uncertainty(Uniform(80, 120)))
@constraint(model, 100 <= d <= 110, Truncate())   # d ~ Uniform(100, 110)
@variable(model, e[1:2] in Uncertainty(product_distribution([Uniform(40, 80), Uniform(40, 80)])))
@constraint(model, e[1] + e[2] <= 120, Truncate())  # uniform on a triangle
```
"""
struct Truncate <: AbstractXiRole end

"""
    XiRole()

A constraint attribute: the [`AbstractXiRole`](@ref) of an uncertainty-only row, or `nothing`.
It is set by `@constraint(model, expr, ValidInequality())` or `Truncate()`, and read by
`model_info`.
"""
struct XiRole <: MOI.AbstractConstraintAttribute end

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
    CombinationBreakPoints()

A model attribute of `Experimental`: the combinations of uncertainties with breakpoints, as a
vector of pairs `f => points`, where `f` is a `MOI.ScalarAffineFunction{Float64}` of uncertainties
and `points` its interior breakpoints or their number (see
[`Experimental.set_breakpoints`](@ref), which sets it).
"""
struct CombinationBreakPoints <: MOI.AbstractModelAttribute end

function MOI.Utilities.map_indices(f::Function, ::CombinationBreakPoints, x)
    return [MOI.Utilities.map_indices(f, g) => points for (g, points) in x]
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

An optimizer attribute: the vector of formulations that `optimize!` solves, at most one of each
type; `[PrimalLDR(), DualLDR()]` by default, the rule and its bound.

The results do not depend on the order: result 1 is always the rule of [`PrimalLDR`](@ref) and
result 2 the rule of [`SampledLDR`](@ref), with status `NO_SOLUTION` when their formulation was not
solved or has no solution. `result_count` is 2 with `SampledLDR`, else 1 with `PrimalLDR`, else 0.
The bound of [`DualLDR`](@ref) is `objective_bound`.

## Example

```julia
set_attribute(model, Formulations(), [PrimalLDR(), DualLDR(), SampledLDR(100)])
```
"""
struct Formulations <: MOI.AbstractOptimizerAttribute end

"""
    RejectionSamplingSeed()

An optimizer attribute: the seed, an `Int`, of the rejection sampling that estimates the moments of
the laws conditioned by constraints tagged [`Truncate`](@ref); `1234` by default.

Each conditioned law draws from its own stream, whose seed is derived from this one and from its
rank among the conditioned laws, so that two laws do not share their
draws. The estimates are cached: a new `optimize!` samples again only the laws whose distributions,
conditioning rows, seed or sample size changed. [`SampledLDR`](@ref) and [`simulate`](@ref) draw
their scenarios with their own seeds.
"""
struct RejectionSamplingSeed <: MOI.AbstractOptimizerAttribute end

"""
    RejectionSamplingTimeLimit()

An optimizer attribute: the time limit in seconds, a `Float64`, of the rejection sampling of each
law conditioned by constraints tagged [`Truncate`](@ref), or `nothing` (the default).

With `nothing`, the time limit depends on `MOI.TimeLimitSec`, which limits the whole `optimize!`
(sampling included): when it is set, the sampling gets half of the time that remains of it,
shared by the conditioned laws to sample (a law gets an equal share of what the laws before it
left), so that the formulations get the rest; when it is not, each law gets 10 seconds. A time
limit set with this attribute wins: the sampling of each law can take that long, and the time
that it takes still counts against `MOI.TimeLimitSec`.

If fewer than [`RejectionSamplingMaxSamples`](@ref)`()` draws are kept within the limit, the
moments are estimated from those, with a warning; fewer than `min(RejectionSamplingMaxSamples(),
100)` are an error that names the conditioning constraints. The sampling stops with this error
early when, even at an optimistic acceptance rate, too few draws can be kept in time. Laws whose
estimates are kept from a previous `optimize!` take no time.

## Example

```julia
set_time_limit_sec(model, 60.0)  # up to 30 seconds of sampling, by default
set_attribute(model, RejectionSamplingTimeLimit(), 5.0)  # 5 seconds per conditioned law
```
"""
struct RejectionSamplingTimeLimit <: MOI.AbstractOptimizerAttribute end

"""
    RejectionSamplingMaxSamples()

An optimizer attribute: the number of draws, an `Int`, kept by the rejection sampling of each law
conditioned by constraints tagged [`Truncate`](@ref); `10_000` by default.

It trades accuracy for time: the standard error of the estimated moments is of order
`1 / √RejectionSamplingMaxSamples()`, so that the standard error of a mean is 1% of the
standard deviation with `10_000` draws, and about 3% with `1_000`; the time is proportional to
it, divided by the acceptance rate of the conditioning constraints. The values that use estimated
moments are not certified (see [`is_certified`](@ref)).

The time limit bounds the attempts (see [`RejectionSamplingTimeLimit`](@ref)), and a draw is never
replaced by a default value.

## Example

```julia
set_attribute(model, RejectionSamplingMaxSamples(), 100_000)
```
"""
struct RejectionSamplingMaxSamples <: MOI.AbstractOptimizerAttribute end

"""
    FormulationAttribute(formulation::AbstractFormulation, attribute)

A model attribute: the `attribute` of one formulation, for example
`FormulationAttribute(DualLDR(), MOI.TerminationStatus())`.

`attribute` is one of `MOI.TerminationStatus()`, `MOI.PrimalStatus()`, `MOI.ObjectiveValue()`,
`MOI.ObjectiveBound()`, `MOI.RawStatusString()`, `MOI.SolveTimeSec()`, [`Certified`](@ref)`()`
and [`TimeBreakdown`](@ref)`()`. The objective value of a formulation whose rule is not a policy, [`DualLDR`](@ref), is its bound.

See also `JuMP.termination_status(model, formulation)`, [`is_certified`](@ref) and the other short
helpers.
"""
struct FormulationAttribute{A<:MOI.AbstractModelAttribute} <: MOI.AbstractModelAttribute
    formulation::AbstractFormulation
    attribute::A
end

MOI.is_set_by_optimize(::FormulationAttribute) = true

"""
    Certified()

A model attribute of one formulation, read through [`FormulationAttribute`](@ref): whether its
objective value and bound are exact for the laws of the model, as a `Bool`. See
[`is_certified`](@ref), its short helper.

## Example

```julia
get_attribute(model, FormulationAttribute(DualLDR(), Certified()))  # is_certified(model)
```
"""
struct Certified <: MOI.AbstractModelAttribute end

"""
    TimeBreakdown()

A model attribute set by `optimize!`: where its wall-clock time went, in seconds, as a
`NamedTuple` with the fields

 * `canonicalize`: reading the model into its canonical form, with the checks of its declarations
   and the liftings of its breakpoints, except their moments;
 * `moments`: the moments of the laws, including those of lifted laws, and the rejection sampling
   of laws conditioned by constraints tagged [`Truncate`](@ref);
 * `checks`: the checks of the objective and of the moments, and the comparison with the inner
   problems of the last `optimize!`;
 * `build`, `solve` and `results`: over the formulations, building the inner problems (or setting
   their parameters, when they are solved again), their `optimize!`, and reading their results;
 * `total`: the whole `optimize!`, as `MOI.SolveTimeSec()`, which also includes what is left.

Read through [`FormulationAttribute`](@ref), it is `(build, solve, results)` for one formulation.
See [`time_breakdown`](@ref), its short helper.

## Example

```julia
get_attribute(model, TimeBreakdown()).moments  # time_breakdown(model).moments
```
"""
struct TimeBreakdown <: MOI.AbstractModelAttribute end

MOI.is_set_by_optimize(::TimeBreakdown) = true

const _TIME_PHASES = (:canonicalize, :moments, :checks, :build, :solve, :results, :total)

const _TimeBreakdown = NamedTuple{_TIME_PHASES,NTuple{7,Float64}}

_no_time_breakdown() = _TimeBreakdown(ntuple(_ -> NaN, 7))

MOI.is_set_by_optimize(::Certified) = true

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

# The `LiftingMap` of the last solve, the identity if nothing was lifted.
struct UncertaintyLifting <: MOI.AbstractModelAttribute end

# The `Moments` of ξ of the last solve.
struct UncertaintyMoments <: MOI.AbstractModelAttribute end

# What `evaluate` and `simulate` read: the form and the rule of `formulation` of the last solve.
struct SimulationData <: MOI.AbstractModelAttribute
    formulation::AbstractFormulation
end

# The inner `JuMP.Model` of one formulation, read through `FormulationAttribute` by `inner_model`.
struct InnerModel <: MOI.AbstractModelAttribute end

function MOI.is_set_by_optimize(
    ::Union{
        InnerModel,
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
    BreakPoints,
    InnerOptimizer,
    Formulations,
    RejectionSamplingSeed,
    RejectionSamplingTimeLimit,
    RejectionSamplingMaxSamples,
    FormulationAttribute,
    Certified,
    RuleCoefficients,
    UncertaintyVariables,
    DecisionVariables,
    UncertaintyLifting,
    UncertaintyMoments,
    SimulationData,
    InnerModel,
}

MOI.Utilities.map_indices(::Any, ::_AttributesWithoutIndices, x) = x
