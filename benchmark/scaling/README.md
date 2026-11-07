# Scaling

The studies of the scaling of the uncertainties, with or without breakpoints. An uncertain demand
of order 10⁴ gives second moments of order 10⁸ against 1 for the constant column of ξ, and the
inner problems are then badly scaled. The user can write the uncertainty in other units, and the
optimizer rescales each uncertainty to `[-1, 1]` before the formulations (`RescaleUncertainties()`,
true by default since 82c4496). The script runs the grid of [`../grid.jl`](../grid.jl), shared with
[`../breakpoints/`](../breakpoints/README.md) and [`../solvers/`](../solvers/README.md); its results
are in `results/rescale.csv`. The decisions that it informed are in the
[table of decisions](../README.md#decisions-and-the-experiments-behind-them).

## The scaling of the uncertainties (`rescale.jl`)

The questions: does writing the uncertainties in other units help the solvers, and should the
optimizer do it? Each formulation keeps its default `lifted_box` (`false` for the primal, `true` for
the dual). The runs were on 7 November 2026, with HiGHS 1.15.1, Gurobi 13.0.3 (see
[`../solvers/`](../solvers/README.md)) and Julia 1.12.6 on Windows 11, alone on the machine, with a
time limit of 600 s; times are those of the inner solver, in seconds.

```
julia --project=benchmark benchmark/scaling/rescale.jl production production_centered production_unit T=10,15,20 k=1,3 solver=simplex,hipo rescale=false
julia --project=benchmark benchmark/scaling/rescale.jl kwg T=24,48 k=1,3 solver=simplex,gurobi,gurobi_dual1
```

### The production planning, in three units

`production_model(T; parametrization)` writes the demands `d` of week `t > 1` as the uncertainties
(`production`, of order 10⁴), as their deviations from the nominal demands `d̄` (`production_centered`:
`d = d̄ + ξ`), or as these deviations divided by their half-width (`production_unit`:
`d = d̄ + 0.2 d̄ ξ`, `ξ ∈ [-1, 1]`). A count of breakpoints is equally spaced in the support, so the
three give the same values. They ran without the optimizer's rescaling (`rescale = false`, at
b8a7159, from 09:19 to 13:10, CPU load 0 to 34%); the Gurobi rows of the raw demands are those of
[`../solvers/`](../solvers/README.md). The column "rescaled" is the raw demands with
`RescaleUncertainties()`, at 0d2fae4 (16:46 to 17:49, CPU load 0 to 3%, and 20 to 24% at the start
of two runs).

The dual:

| T | k | HiGHS simplex: raw | centered | unit | rescaled | Gurobi: raw | centered | unit | rescaled |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 | 1 | fails 13 | 9.6 | 5.4 | 6.6 | 2.2 | 3.0 | 0.7 | 0.9 |
| 10 | 3 | fails 146 | fails 158 | 61 | 67 | 16 | 70 | 2.5 | 2.3 |
| 15 | 1 | fails 114 | fails 99 | 39 | 44 | num. 84 | 93 | 2.5 | 2.5 |
| 15 | 3 | limit | limit | limit | 537 | 28 | 60 | 9.3 | 8.0 |
| 20 | 1 | limit | fails 555 | 346 | 376 | num. 600 | 25 | 6.7 | 7.4 |
| 20 | 3 | limit | limit | limit | limit | limit | limit | 29 | 22 |

The primal:

| T | k | HiGHS simplex: raw | centered | unit | rescaled | Gurobi: raw | centered | unit | rescaled |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 | 1 | 1.0 | 1.5 | 1.2 | 2.1 | 0.3 | 0.3 | 0.2 | 0.2 |
| 10 | 3 | 25 | 6.1 | 6.2 | 8.7 | 0.8 | 0.9 | 0.6 | 0.6 |
| 15 | 1 | 9.1 | 5.8 | 8.9 | 15 | 1.1 | 0.8 | 0.7 | 0.7 |
| 15 | 3 | 26 | 30 | 44 | 52 | 2.3 | 2.8 | 2.3 | 1.7 |
| 20 | 1 | 24 | limit | 14 | 63 | 3.0 | 2.2 | 1.5 | 1.7 |
| 20 | 3 | limit | limit | 260 | 498 | 6.4 | 7.6 | 5.2 | 3.4 |

"fails" is the status Unknown of HiGHS (`OTHER_ERROR`), and "num." the status `NUMERICAL_ERROR` of
Gurobi, before the limit. Gurobi's dual simplex on one thread (`gurobi_dual1`) solved 4, 5 and 6
of the 6 duals in the raw, centered and unit forms. HiPO (`hipo`) ended with the status Unknown on
every one of the 36 models, in every form, so the other data of the problem (a holding cost of
3e-5 against prices of 0.25 to 0.65, a stock bound of 10⁶, production rates of 800 to 1200) are
the next suspects.

The rescaling of the optimizer first ran at 82c4496 (the rows with that `src_commit`, from 14:20 to
16:26): the values agreed, but Gurobi was 10 to 45 times slower than on the unit form (T = 20, k = 3,
primal: 233 s against 5.2 s), and HiGHS's primal failed or reached the limit on 5 of the 6
instances. The hull rows of the lifted coordinates (`Δₖ₊₁ η̃ₖ - Δₖ η̃ₖ₊₁ ≥ …`, scaled by the lengths
`Δ` of the pieces) kept coefficients of order 10⁶ once rescaled, against 0.1 in the unit form, and
the first one a right-hand side of rounding noise. Since 0d2fae4, each rescaled row is divided by
its largest coefficient, and a right-hand side within rounding of zero is zero.

### The inventory of Kuhn, Wiesemann and Georghiou

`kwg`, with and without `RescaleUncertainties()`: without at 82c4496 (the rows without rescaling
are not affected by 0d2fae4), with at 0d2fae4.

| Form | T | k | HiGHS simplex: without | with | Gurobi: without | with | Gurobi, 1 thread: without | with |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| primal | 24 | 1 | 0.8 | 0.5 | 0.1 | 0.1 | 0.3 | 0.4 |
| primal | 24 | 3 | 3.3 | 2.6 | 0.3 | 0.3 | 1.5 | 1.7 |
| primal | 48 | 1 | 17 | 11 | 1.0 | 1.0 | 7.6 | 7.4 |
| primal | 48 | 3 | 77 | 60 | 3.2 | 2.7 | 81 | 83 |
| dual | 24 | 1 | 11 | 4.7 | 1.5 | 0.4 | 1.2 | 1.8 |
| dual | 24 | 3 | 86 | 61 | 0.7 | 0.8 | 3.6 | 5.3 |
| dual | 48 | 1 | 357 | 141 | 23 | 3.1 | 21 | 11 |
| dual | 48 | 3 | limit | limit | 33 | 9.4 | 12 | 14 |

HiPO ended with the status Unknown on the 8 duals, with and without rescaling. The values agree
to 2e-8 relative across the forms of production, the rescaling and the solvers, except the
rescaled runs at 82c4496 (up to 1.3e-7, on Gurobi's duals), and to 5e-10 on kwg.

### What it found

- **Units in `[-1, 1]` rescue the duals.** HiGHS's simplex solves none of the 6 production duals
  with the raw demands, 1 centered, 4 in the unit form and 5 with the optimizer's rescaling; Gurobi
  solves 3, 5, 6 and 6, and the unit form or the rescaling makes it 3 to 20 times faster than the
  raw demands. Centering alone is not enough, and can be slower (Gurobi, T = 10, k = 3: 70 s against
  16 s raw). On kwg, HiGHS's simplex and Gurobi's default are faster with rescaling on the duals
  (T = 48, k = 1: 141 s against 357 s, and 3.1 s against 23 s), and HiGHS's simplex on the primals;
  Gurobi on one thread is slower on 3 of the 4 duals, by up to 1.5 times (T = 24, k = 3: 5.3 s
  against 3.6 s).
- **The primal changes less.** Gurobi solves every form in about the same time. HiGHS's simplex
  solves production T = 20, k = 3 only in the unit form and with the rescaling, but the rescaling
  is 1.2 to 4.5 times slower than the unit form on the primals of production (T = 20, k = 1: 63 s
  against 14 s).
- **The default:** the optimizer rescales (`RescaleUncertainties()` is true by default).
- **The rescaling and the unit form differ with breakpoints.** The unit form rescales the demand
  and then lifts it: its pieces keep 0 as the value below their breakpoint, and the first piece is
  not centered (in `[-1, -0.5]` with 3 breakpoints). The optimizer rescales each lifted
  coordinate to `[-1, 1]` on its own box. They are the same on Gurobi and on HiGHS's duals,
  but not on HiGHS's primals (above). Which one to keep is open.
