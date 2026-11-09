# [Performance](@id performance)

```@meta
CurrentModule = LinearDecisionRules
```

None of the settings on this page changes the values or the rules. They change only the size of the
inner problems, how well they are scaled, and how they are solved. The defaults were chosen on the
benchmarks of the package, in the folder
[`benchmark`](https://github.com/bfpc/LinearDecisionRules.jl/tree/main/benchmark) of the
repository, where each choice links to the experiments behind it. The times below come from those
benchmarks, with HiGHS 1.15. The best setting depends on the problem, so measure your own.

## Where the time goes

Start by measuring. [`time_breakdown`](@ref) splits the time of `optimize!` into reading the model
(with its lifting and rescaling), the moments, the checks, and the build, solve and results of
each formulation:

```julia
optimize!(model)
time_breakdown(model)               # all the formulations
time_breakdown(model, DualLDR())    # (build = …, solve = …, results = …)
```

The inner solver usually takes most of the time, and the dual often takes longer than the
primal. The moments can take long too, for a law conditioned by constraints
tagged [`Truncate`](@ref) (see [`RejectionSamplingMaxSamples`](@ref)). [`inner_model`](@ref) gives
the inner problem of a formulation, to export it or to read its size.

## The size of the inner problems

* **Solve only what you need.** [`Formulations`](@ref)`()` is `[PrimalLDR(), DualLDR()]` by
  default. Without the bound, `set_attribute(model, Formulations(), [PrimalLDR()])` skips the dual.
* **The horizon.** A decision observes every uncertainty revealed before it, so a rule of stage `t`
  has one coefficient per uncertainty of stages `1` to `t`. With the same uncertainties in every
  stage, the inner problems grow with the square of the number of stages.
* **Breakpoints.** Each breakpoint adds a piece, and the size grows with the number of pieces. On a
  hydrothermal problem, one breakpoint per inflow made the primal 2.6 times larger and 5 to 6 times
  slower to solve. A few breakpoints, well placed, are usually enough.
* **Independent declarations.** Uncertainties declared separately are independent, and their
  moments are stored as separate blocks. Declaring independent uncertainties as one joint law makes
  the moments dense: on the hydrothermal problem, the dual had 42% fewer nonzeros with separate
  declarations.

## Settings

| Setting | Default | What it does |
|---|---|---|
| [`RescaleUncertainties`](@ref)`()` | `true` | writes each uncertainty in `[-1, 1]` in the inner problems |
| `PrimalLDR(; lifted_box)` | `false` | with breakpoints, leaves out the bounds of the pieces that their other rows imply |
| `DualLDR(; lifted_box)` | `false` | the same, for the dual |
| `PrimalLDR(; box_closed_form)` | `true` | writes the robust counterpart over a box in closed form, without multipliers |
| `attributes` of a formulation | none | options of the inner solver, for that formulation only |
| [`InnerOptimizer`](@ref)`()` | the solver of [`LDRModel`](@ref) | the solver of the inner problems |
| `set_time_limit_sec(model, t)` | none | limits the whole `optimize!` |

### Rescaling the uncertainties

Uncertainties of order 10⁴, such as demands, give moments of order 10⁸ against 1 for the constant
term of the rules, and the inner solvers can then fail or slow down, mostly on the dual. So the
optimizer writes each uncertainty as its center plus its half-width times a new uncertainty in
`[-1, 1]`, solves the inner problems in these units, and reports the rules in the units of the
model. On a production planning problem with demands of order 10⁴, HiGHS solved none of the six
duals in the units of the demands, and five with rescaling.

There is no need to rescale the uncertainties yourself. Turn rescaling off to read the inner
problems ([`inner_model`](@ref)) in the units of the model:

```julia
set_attribute(model, RescaleUncertainties(), false)
```

With breakpoints, each piece is rescaled on its own interval. Rescaling the uncertainty before
lifting it, or only scaling the pieces after the first (a hybrid), gives the same values. On the
production problem, with HiGHS's simplex method, rescaling before lifting solved the primals 1.2
to 5 times faster, and the hybrid the duals 1.2 to 3.8 times faster. Which one to use is still
being studied: [`Experimental.RescalingMode`](@ref) has both, for trying them on your problem.

### The bounds of the pieces of an uncertainty with breakpoints (`lifted_box`)

The pieces of an uncertainty with `n` breakpoints are described by `n + 2` rows. The other `2n`
bounds of the pieces follow from them. Leaving these bounds out (`lifted_box = false`) gives the same
values with a smaller problem: 30% to 48% fewer variables in the primal, and as many fewer
constraints in the dual. Both formulations leave them out by default. The primal was faster without
them on every instance measured with rescaling, under both the simplex method and IPX with
crossover. The dual was faster or as fast without them on every instance under IPX with crossover,
and on most under the simplex method, but 2 to 3.3 times slower on two hydrothermal instances
under the simplex method. To keep them in the dual:

```julia
set_attribute(model, Formulations(), [PrimalLDR(), DualLDR(; lifted_box = true)])
```

### The closed form over boxes (`box_closed_form`)

When the support of a group of uncertainties is a box, the primal writes its robust counterpart
with one auxiliary variable per coefficient, instead of two multipliers. This is the default.
`PrimalLDR(; box_closed_form = false)` keeps the multipliers. That model is larger, but neither
form is faster everywhere: 1.3 s against 0.63 s with the multipliers on an inventory problem with
128 stages, but 92 s against 131 s on a hydrothermal problem with 12 stages.

## Choosing the inner solver

The `attributes` of a formulation are passed to its inner solver only, after those of the optimizer.
With HiGHS:

| Mode | `attributes` | Measured |
|---|---|---|
| dual simplex (default) | none | robust; the safest choice for small problems |
| IPX with crossover | `["solver" => "ipx", "run_crossover" => "on"]` | recommended for large duals; often faster on large primals with breakpoints |
| IPX without crossover | `["solver" => "ipx", "run_crossover" => "off"]` | fast when it succeeds, but often stops at an unknown status |
| HiPO without crossover | `["solver" => "ipm", "run_crossover" => "off"]` | the same |

For the dual, IPX with crossover was faster than the dual simplex on every production planning and
hydrothermal dual measured, by 1.4 to 37 times, and solved within 10 minutes the two that the
dual simplex did not: the hydrothermal dual with 8 stages and 3 breakpoints took 63 s. For the
primal, it was 1.2 to 7 times faster on the large problems with breakpoints, but slower on small
ones, which take seconds either way. For example, IPX with crossover for the dual only:

```julia
set_attribute(
    model,
    Formulations(),
    [PrimalLDR(), DualLDR(; attributes = ["solver" => "ipx", "run_crossover" => "on"])],
)
```

The interior points without crossover are not recommended. They stop at an unknown status on
many problems (every production planning problem, every dual of the inventory problem of Kuhn,
Wiesemann and Georghiou without rescaling): the termination status is then `OTHER_ERROR`, and the
bound is `NaN`. And when they succeed, their point is feasible only up to a tolerance, so the
bound of the dual can be slightly invalid: up to 7e-6 relative above the exact bound on the
hydrothermal duals.

The inner solver can be any solver of JuMP that supports linear programs, and second-order cones
when the support has them; [`InnerOptimizer`](@ref)`()` sets it.

## Solving again

[Re-solving with parameters](getting_started.md#Re-solving-with-parameters) keeps the inner problems when only the values of JuMP
parameters change, and the inner solver starts from its last solution. Any other change, such as a
coefficient, a law or a setting on this page, builds the inner problems again.
