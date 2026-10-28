# Modelling

```@meta
CurrentModule = LinearDecisionRules
```

Declaring a model: the optimizer, the stages, the uncertainties and the attributes that choose the formulations and the inner solver.

```@docs
LinearDecisionRules
LDRModel
Optimizer
MOI.optimize!(::Optimizer, ::MOI.ModelLike)
```

## Stages

```@docs
Stage
FirstStage
stage
set_stage
unset_stage
```

## Uncertainties and their support

```@docs
Uncertainty
ValidInequality
Truncate
BreakPoints
```

## Settings

```@docs
Formulations
InnerOptimizer
RejectionSamplingSeed
RejectionSamplingTimeLimit
RejectionSamplingMaxSamples
```
