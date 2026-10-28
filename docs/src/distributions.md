# Distributions and liftings

```@meta
CurrentModule = LinearDecisionRules
```

Laws of the uncertainties and the liftings (breakpoints) that make the rules piecewise.

## Laws

```@docs
MvDiscreteNonParametric
ConfidenceMvNormal
ConditionedDistribution
LiftedDistribution
```

## The support of a law

A law whose support is not its box adds a method of `support_constraints`.

```@docs
support_constraints
```

## Liftings

```@docs
AbstractLifting
AxisPiecewise
lift
lift(::LiftingMap, ::AbstractVector{<:Real})
retraction
LiftingMap
LiftingBlock
```
