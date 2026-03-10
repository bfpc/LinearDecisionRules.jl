# LinearDecisionRules.jl Documentation

```@meta
CurrentModule = LinearDecisionRules
```

[LinearDecisionRules.jl](https://github.com/bfpc/LinearDecisionRules.jl) is a
Julia package for solving stochastic optimization problems using linear decision
rules (LDRs).

## Overview

LinearDecisionRules.jl provides a simple [JuMP](https://jump.dev) abstraction to
represent decision rules as linear functions of random variables. It handles
two-stage stochastic optimization problems where:

 * **First-stage decisions** are made before uncertainty is revealed
 * **Second-stage decisions** can adapt (linearly) to the realized uncertainty

The package reformulates these problems into tractable deterministic equivalent
optimization problems and provides both primal and dual bounds.

## Problem formulation

The problems the package deals with are of the form:
```math
\begin{array}{rl}
\min \ & \mathbb{E}_\xi [ c(\xi)^\top x(\xi) + x(\xi)^\top Q x(\xi) + r ] \\[0.5ex]
\text{s.t.} & A_e x(\xi) = b_e(\xi) \\
&  b_l(\xi) \leq A x(\xi) \leq b_u(\xi) \\
& x_l \leq x(\xi) \leq x_u \\
& x_i(\xi) \text{ is non-anticipative, for } i \in I \\
& \forall \xi \in \Xi
\end{array}
```
where ``\Xi \subset \mathbb{R}^m`` is a polytope described by:
```math
\begin{aligned}
\Xi & = \{\, \xi = (1, \eta) \in \mathbb{R}^m \mid W_u \eta \leq h_u, W_l \eta \geq h_l, lb \leq \eta \leq ub \,\} \\
& = \{\, \xi \in \mathbb{R}^m \mid W \xi \geq h \,\}.
\end{aligned}
```

Non-anticipative variables cannot depend on the random variable ``\xi``. This is
equivalent to fixing their decision rules to have coefficient equal to zero,
except for the constant term.

## Installation

```julia
import Pkg
Pkg.add("LinearDecisionRules")
```

## Quick example

Here's a simple "Newsvendor" problem:

```julia
using JuMP
using LinearDecisionRules
using HiGHS
using Distributions

ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
set_silent(ldr)

@variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
@variable(ldr, sell >= 0)
@variable(ldr, demand in LinearDecisionRules.Uncertainty(distribution = Uniform(80, 120)))

@constraint(ldr, sell <= buy)
@constraint(ldr, sell <= demand)

@objective(ldr, Max, -10 * buy + 15 * sell)

optimize!(ldr)

objective_value(ldr)                              # Primal bound
objective_value(ldr; dual = true)                 # Dual bound
LinearDecisionRules.get_decision(ldr, buy)        # First-stage decision
LinearDecisionRules.get_decision(ldr, sell)       # Decision rule coefficients
```

For a detailed walkthrough, see [Getting started with LinearDecisionRules](@ref).

## Learn more

 * **[Tutorials](@ref tutorials_introduction)**: Step-by-step guides to learn
   LinearDecisionRules
 * **[Manual](@ref math_formulation)**: Technical details and mathematical
   formulations
 * **[API Reference](@ref api_reference)**: Complete reference for all exported
   functions and types


## Related packages

Other Julia packages for stochastic programming include:

 * [SDDP.jl](https://github.com/odow/SDDP.jl): Stochastic Dual Dynamic
   Programming for multi-stage stochastic optimization problems
 * [StochasticPrograms.jl](https://github.com/martinbiel/StochasticPrograms.jl):
   A framework for modeling and solving stochastic programs
