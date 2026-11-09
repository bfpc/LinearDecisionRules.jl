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

### The ways to rescale (`Experimental.RescalingMode`)

`RescaleUncertainties()` takes `true` or `false`, and also a mode of
`Experimental.RescalingMode` (212e41d):
- `PER_PIECE`, the same as `true`, rescales each lifted coordinate to `[-1, 1]` on its own box.
- `BEFORE_LIFTING` rescales the uncertainty and then lifts it, as the unit form of production
  does. The pieces are 0 below their breakpoint, and only the first is centered.
- `HYBRID` centers and scales the first piece on its own box, and only scales the later ones.
- `SCALE_ONLY` divides each uncertainty by the largest magnitude of its box, without centering.

The first three differ only with breakpoints. The values agree to 2e-14 relative across the modes.
The question: which mode should `true` mean, and should `RescaleUncertainties()` stay on by default
for the hydrothermal problem, which the study above did not measure?

The runs used HiGHS's simplex method, with a time limit of 600 s. The machine check (see "Running"
in [`../README.md`](../README.md)) is noted with each round:

| Round | Runs | Commit | When | Machine check |
|---|---|---|---|---|
| 1 | hydro, under `true`, `before_lifting`, `hybrid` and `false` | b3b2861 (`src_commit` 518ece2 with the warnings of f58da4d in the working tree, which do not change the inner problems) | 8 November 2026, 15:34 to 16:30 | 3 to 7% slower than the reference |
| 2 | hydro under `scale_only`; production, kwg and the inventory under every mode | 212e41d | night of 8 to 9 November 2026, 22:13 to 00:38 and 04:05 to 04:17 | the LP from 1.95 to 2.05 s |

The hydro duals of both commits are the same problems: the same sizes and values, and the per-piece
runs of 212e41d in [`../breakpoints/`](../breakpoints/README.md) are 3 to 6% faster than those of
b3b2861, as the machine check predicts.

```
julia --project=benchmark benchmark/scaling/rescale.jl hydro T=4,6 k=1,2,3 rescale=true,before_lifting,hybrid,scale_only,false
julia --project=benchmark benchmark/scaling/rescale.jl production T=10,15 k=1,3 rescale=true,before_lifting,hybrid,scale_only
```

The hydrothermal problem without breakpoints (`hydro_joint`), where only three modes differ:

| Form | T | per piece | scale only | none |
|---|--:|--:|--:|--:|
| primal | 4 | 0.66 | 0.4 | 0.28 |
| primal | 6 | 5.8 | 2.3 | 2.4 |
| primal | 8 | 17 | 14 | 11 |
| dual | 4 | 2.5 | 3.5 | 3.5 |
| dual | 6 | 18 | 26 | 29 |
| dual | 8 | 95 | 107 | 116 |

With `k` breakpoints on every inflow (`hydro`):

| Form | T | k | per piece | before lifting | hybrid | scale only | none |
|---|--:|--:|--:|--:|--:|--:|--:|
| primal | 4 | 1 | 3.5 | 2.2 | 2.3 | 2.8 | 1.6 |
| primal | 4 | 2 | 6.3 | 6.0 | 6.2 | 6.0 | 4.1 |
| primal | 4 | 3 | 12 | 10 | 10 | 11 | 5.8 |
| primal | 6 | 1 | 24 | 19 | 17 | 16 | 13 |
| primal | 6 | 2 | 54 | 45 | 43 | 42 | 29 |
| primal | 6 | 3 | 93 | 81 | 88 | 75 | 58 |
| dual | 4 | 1 | 14 | 11 | 10 | 13 | 6.5 |
| dual | 4 | 2 | 19 | 49 | 57 | 38 | 34 |
| dual | 4 | 3 | 122 | 45 | 29 | 38 | 25 |
| dual | 6 | 1 | 139 | 125 | 108 | 113 | 41 |
| dual | 6 | 2 | 154 | 115 | 120 | 128 | 69 |
| dual | 6 | 3 | 206 | 342 | 234 | 349 | 177 |

Production and kwg ran without `false`, since the production duals fail without rescaling
(above). "limit" is the limit of 600 s.

| Problem | Form | T | k | per piece | before lifting | hybrid | scale only |
|---|---|--:|--:|--:|--:|--:|--:|
| production | primal | 10 | 1 | 2.0 | 1.1 | 1.6 | 1.5 |
| production | primal | 10 | 3 | 8.5 | 6.6 | 7.4 | 6.8 |
| production | primal | 15 | 1 | 15 | 11 | 15 | 18 |
| production | primal | 15 | 3 | 53 | 43 | 46 | 55 |
| production | primal | 20 | 1 | 64 | 13 | 49 | 53 |
| production | primal | 20 | 3 | 479 | 242 | 349 | 265 |
| production | dual | 10 | 1 | 6.7 | 6.7 | 4.1 | 10 |
| production | dual | 10 | 3 | 68 | 95 | 18 | 55 |
| production | dual | 15 | 1 | 45 | 51 | 36 | 134 |
| production | dual | 15 | 3 | 534 | limit | 290 | limit |
| production | dual | 20 | 1 | 379 | 324 | 252 | 583 |
| kwg | primal | 24 | 1 | 0.5 | 0.4 | 0.7 | 0.8 |
| kwg | primal | 24 | 3 | 2.6 | 2.8 | 2.8 | 3.0 |
| kwg | primal | 48 | 1 | 11 | 8.7 | 12 | 15 |
| kwg | primal | 48 | 3 | 56 | 56 | 58 | 70 |
| kwg | dual | 24 | 1 | 4.6 | 3.9 | 4.6 | 7.5 |
| kwg | dual | 24 | 3 | 60 | 49 | 29 | 38 |
| kwg | dual | 48 | 1 | 141 | 120 | 172 | limit |

The inventory takes at most 1.5 s on every primal. Its duals:

| T | k | per piece | before lifting | hybrid | scale only | none |
|--:|--:|--:|--:|--:|--:|--:|
| 32 | 0 | 0.0 | | | 0.1 | 0.1 |
| 64 | 0 | 0.2 | | | 0.7 | 0.9 |
| 128 | 0 | 1.0 | | | 10 | 7.4 |
| 32 | 1 | 0.5 | 0.7 | 0.7 | 1.3 | 1.2 |
| 32 | 3 | 11 | 8.1 | 9.4 | 9.6 | 7.4 |
| 64 | 1 | 6.1 | 5.2 | 5.5 | 9.1 | 8.8 |
| 64 | 3 | 133 | 66 | 75 | 63 | 55 |

Each time is one run, and the simplex method's path can change a lot with the scaling: the hydro
duals range from 0.24 to 3 times per piece under the same mode.

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
  but not on HiGHS's primals (above).
- **Per piece, the meaning of `true` today, is rarely the fastest of the modes.** Here is how each
  mode compares with per piece on the instances with breakpoints. "n of m" counts the instances
  where it is faster; the ratio is the geometric mean of its time over that of per piece.

  | Problem | before lifting: primal | dual | hybrid: primal | dual |
  |---|---|---|---|---|
  | production | 6 of 6, 0.54 | 2 of 5, 1.09, one limit | 6 of 6, 0.83 | 5 of 5, 0.54 |
  | kwg | 2 of 4, 0.94 | 3 of 3, 0.84 | 0 of 4, 1.14 | 1 of 3, 0.84 |
  | hydro | 6 of 6, 0.81 | 4 of 6, 0.97 | 6 of 6, 0.81 | 4 of 6, 0.85 |

  Hybrid is the fastest on every production dual, the instances that need rescaling at all, and
  never reaches the limit. Before lifting is the fastest on every production primal, by up to 5
  times (T = 20, k = 1: 13 s against 64 s), but it reaches the limit on one dual.
- **Scaling without centering is not a compromise.** It is the slowest mode on 3 of the 5
  production duals and reaches the limit on two, and on kwg it is the slowest on every primal and
  reaches the limit on one dual. On the inventory dual without breakpoints it is as
  slow as no rescaling (T = 128: 10 s against 1.0 s per piece). Centering makes the first moments
  zero, which removes nonzeros (240,311 against 273,079 there).
- **The hydrothermal problem is fastest without rescaling,** on every primal and on 5 of the 6
  duals with breakpoints: 0.55 and 0.52 times per piece in geometric mean. Without breakpoints,
  rescaling slows the primal by 1.5 to 2.4 times and speeds the dual by 1.2 to 1.6 times.
  Production cannot do without rescaling, so `false` cannot be the default.
- **Open: the default.** `true` could mean hybrid, or before lifting, instead of per piece.
