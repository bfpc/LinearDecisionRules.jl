# Internals

```@meta
CurrentModule = LinearDecisionRules
```

This page looks under the hood: what the optimizer reads from a model, and what the formulations work on (the canonical form, the information structure, the support blocks and the moments). These names are not exported, so write them qualified, as in `LinearDecisionRules.canonicalize`. They are documented for reference and for debugging; their design may still change, and they will become stable with a planned `Experimental` submodule.

## Reading a model

The metadata of a model are constraints and attributes of MOI, which the macros of the [Modelling](modelling.md) page add.

```@docs
model_info
ModelInfo
UncertaintyDeclaration
canonicalize
UnivariateUncertainty
MultivariateUncertainty
AbstractXiRole
XiRole
_is_metadata_constraint
```

## The canonical form

```@docs
CanonicalForm
ConicRow
UncertaintyBlock
ParameterTerms
lift(::CanonicalForm, ::AbstractDict{MOI.VariableIndex,<:AbstractLifting})
```

## The information structure

```@docs
InformationStructure
observes
visible_columns
```

## The support

```@docs
SupportBlock
support_blocks
```

## The moments

```@docs
CovarianceBlock
Moments
moments
second_moment
```

## Solving one formulation

```@docs
FormulationCapabilities
capabilities
FormulationResult
solve_formulation
```
