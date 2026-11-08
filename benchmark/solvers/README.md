# Solvers

The studies of other solvers than HiGHS on the inner problems. Gurobi is not a dependency of the
package or of the benchmark environment: [`../grid.jl`](../grid.jl) offers its solvers when Julia
finds it in the load path, for example when it is added to the default environment
(`julia -e 'import Pkg; Pkg.add("Gurobi")'`), which `--project=benchmark` stacks below the project.
The decisions that these studies informed are in the
[table of decisions](../README.md#decisions-and-the-experiments-behind-them).

## Gurobi against HiGHS (`solvers.jl`)

The question: is Gurobi more robust or faster than HiGHS on the inner problems of the package, and
is the difference that of the algorithm or that of the cores? Gurobi 13.0.3 ran with its default
options (`gurobi`: concurrent methods on the 28 logical cores, 20 physical, of the machine) on 6 November 2026,
and with the dual simplex on one thread (`gurobi_dual1`: `Threads = 1`, `Method = 1`, one core like
HiGHS's default) on 7 November, at commit 87c03a7 (`src_commit` 92e69a1), with Julia 1.12.6 on
Windows 11, alone on the machine, with a time limit of 600 s. The instances are the 22 of the grid
of [`../breakpoints/`](../breakpoints/README.md) (`hydro`, `inventory`, `kwg`, `production`), the
hydrothermal problem with the joint law of its inflows (`hydro_joint`, the model of
[`../hydro/`](../hydro/README.md)) at T = 4, 8, 12, and the inventory without breakpoints at
T = 64, 128; each formulation with its default `lifted_box`. The runs predate
`RescaleUncertainties()`, so their rows have `rescale = false`.

```
julia --project=benchmark benchmark/solvers/solvers.jl production T=10,15,20 k=1,3 solver=gurobi,gurobi_dual1 rescale=false
julia --project=benchmark benchmark/solvers/solvers.jl hydro_joint T=4,8,12 solver=gurobi rescale=false
```

The HiGHS times come from [`../breakpoints/results/`](../breakpoints/results) (the grid, with the
reruns of one hour), from [`../hydro/results/hydro.csv`](../hydro/results/hydro.csv) (commit
901bd78, `hydro_joint`) and from [`../inventory/results/timings.csv`](../inventory/results/timings.csv)
(commit 2509dbb, the inventory with k = 0). "Best HiGHS mode" is the fastest of the simplex method,
IPX with and without crossover, and HiPO, where they were run; "rel. diff." is the largest relative
difference of the values of the solvers that succeeded. Solver times in seconds:

| Problem | T | k | Form | Gurobi, 1 thread, dual simplex | Gurobi, default | HiGHS, simplex | Best HiGHS mode | Rel. diff. |
|---|--:|--:|---|---|---|---|---|--:|
| inventory | 64 | 0 | primal | 0.1 | 0.08 | 0.15 | simplex 0.15 | 0.0 |
| inventory | 128 | 0 | primal | 0.49 | 0.37 | 1.27 | simplex 1.27 | 0.0 |
| inventory | 64 | 0 | dual | 0.36 | 0.22 | 0.85 | simplex 0.85 | 1.7e-16 |
| inventory | 128 | 0 | dual | 4.81 | 1.01 | 7.39 | simplex 7.39 | 1.7e-16 |
| hydro_joint | 4 | 0 | primal | 0.29 | 0.08 | 0.27 | simplex 0.27 | 1.3e-15 |
| hydro_joint | 8 | 0 | primal | 4.95 | 0.65 | 11 | simplex 11 | 1.3e-15 |
| hydro_joint | 12 | 0 | primal | 56 | 2.34 | 92 | simplex 92 | 3.6e-16 |
| hydro_joint | 4 | 0 | dual | 0.38 | 0.21 | 3.38 | hipo 0.55 | 1.1e-16 |
| hydro_joint | 8 | 0 | dual | 4.23 | 2.73 | 116 | hipo 5.34 | 2.1e-12 |
| hydro_joint | 12 | 0 | dual | 66 | 21 | 142 | simplex 142 | 8.5e-14 |
| inventory | 16 | 1 | primal | 0.01 | 0.01 | 0.02 | simplex 0.02 | 0.0 |
| inventory | 16 | 3 | primal | 0.01 | 0.01 | 0.05 | simplex 0.05 | 0.0 |
| inventory | 32 | 1 | primal | 0.02 | 0.02 | 0.12 | simplex 0.12 | 0.0 |
| inventory | 32 | 3 | primal | 0.04 | 0.05 | 0.29 | simplex 0.29 | 0.0 |
| inventory | 64 | 1 | primal | 0.1 | 0.12 | 0.72 | simplex 0.72 | 0.0 |
| inventory | 64 | 3 | primal | 0.32 | 0.32 | 2.3 | simplex 2.3 | 0.0 |
| inventory | 16 | 1 | dual | 0.03 | 0.03 | 0.13 | simplex 0.13 | 2.0e-15 |
| inventory | 16 | 3 | dual | 0.07 | 0.1 | 0.94 | ipx_crossover 0.28 | 4.0e-15 |
| inventory | 32 | 1 | dual | 0.19 | 0.16 | 2.13 | hipo 0.67 | 2.0e-15 |
| inventory | 32 | 3 | dual | 0.22 | 0.19 | 12 | ipx_crossover 2.2 | 7.1e-15 |
| inventory | 64 | 1 | dual | 0.75 | 0.72 | 15 | hipo 3.46 | 6.9e-15 |
| inventory | 64 | 3 | dual | 1.07 | 1.21 | 103 | ipx_crossover 10 | 3.5e-13 |
| kwg | 24 | 1 | primal | 0.32 | 0.11 | 1.27 | ipx 0.77 | 4.4e-16 |
| kwg | 24 | 3 | primal | 1.46 | 0.26 | 5.22 | ipx 2.09 | 0.0 |
| kwg | 48 | 1 | primal | 7.32 | 1.07 | 27 | ipx 7.98 | 4.4e-16 |
| kwg | 48 | 3 | primal | 75 | 3.28 | 117 | ipx 26 | 4.4e-16 |
| kwg | 24 | 1 | dual | 1.17 | 1.49 | 17 | simplex 17 | 4.4e-16 |
| kwg | 24 | 3 | dual | 3.6 | 0.76 | 138 | simplex 138 | 2.0e-15 |
| kwg | 48 | 1 | dual | 21 | 25 | 625 | simplex 625 | 1.8e-15 |
| kwg | 48 | 3 | dual | 12 | 34 | limit 3600 | none solved | — |
| hydro | 4 | 1 | primal | 2.06 | 0.27 | 2.71 | ipx 1.91 | 1.1e-15 |
| hydro | 4 | 3 | primal | 7.32 | 0.5 | 9.08 | ipx 5.23 | 5.9e-16 |
| hydro | 6 | 1 | primal | 13 | 0.86 | 20 | hipo 7.12 | 8.5e-16 |
| hydro | 6 | 3 | primal | 104 | 1.81 | 86 | hipo 26 | 4.3e-10 |
| hydro | 8 | 1 | primal | 48 | 1.71 | 122 | ipx 31 | 1.1e-15 |
| hydro | 8 | 3 | primal | 234 | 4.58 | 713 | ipx 94 | 1.1e-11 |
| hydro | 4 | 1 | dual | 0.54 | 0.74 | 10 | ipx 2.46 | 1.6e-13 |
| hydro | 4 | 3 | dual | 7.76 | 3.99 | 41 | ipx 8.97 | 3.1e-12 |
| hydro | 6 | 1 | dual | 3.39 | 5.87 | 67 | ipx 9.15 | 5.0e-13 |
| hydro | 6 | 3 | dual | 141 | 22 | 279 | ipx 36 | 6.9e-13 |
| hydro | 8 | 1 | dual | 14 | 14 | 108 | simplex 108 | 2.4e-13 |
| hydro | 8 | 3 | dual | limit 600 | 69 | 1677 | hipo 114 | — |
| production | 10 | 1 | primal | 0.52 | 0.31 | 1.7 | simplex 1.7 | 1.7e-16 |
| production | 10 | 3 | primal | 1.94 | 0.77 | 38 | simplex 38 | 1.7e-16 |
| production | 15 | 1 | primal | 3.11 | 1.1 | 14 | simplex 14 | 1.5e-15 |
| production | 15 | 3 | primal | 18 | 2.27 | 40 | simplex 40 | 6.4e-16 |
| production | 20 | 1 | primal | 15 | 3.03 | 37 | simplex 37 | 2.6e-15 |
| production | 20 | 3 | primal | 58 | 6.38 | limit 3601 | none solved | — |
| production | 10 | 1 | dual | 30 | 2.2 | fails 21 | none solved | — |
| production | 10 | 3 | dual | 253 | 16 | fails 260 | none solved | — |
| production | 15 | 1 | dual | 83 | numerical error 84 | fails 186 | none solved | — |
| production | 15 | 3 | dual | 25 | 28 | limit 3600 | none solved | — |
| production | 20 | 1 | dual | numerical error 205 | numerical error 600 | limit 601 | none solved | — |
| production | 20 | 3 | dual | limit 600 | limit 600 | limit 601 | none solved | — |

"fails" is the status Unknown of HiGHS (`OTHER_ERROR`), and "numerical error" the status
`NUMERICAL_ERROR` of Gurobi, both before the limit.

### What it found

- **Gurobi is more robust.** Each mode solved 51 of the 54 models, with the values of HiGHS to
  1e-10. Both solved kwg T = 48, k = 3 (dual), production T = 20, k = 3 (primal), and production
  duals that no mode of HiGHS solved in an hour. Their failures were the production duals (and
  hydro T = 8, k = 3 for the dual simplex on one thread). The production duals are badly scaled
  without rescaling: with `RescaleUncertainties()`, the default since, Gurobi solves all of them
  (see [`../scaling/`](../scaling/README.md)).
- **The algorithm matters, not only the cores.** On one core, Gurobi's dual simplex was faster than
  HiGHS's simplex on 44 of the 45 instances that both solved (the exception: hydro T = 6, k = 3,
  primal, 104 s against 86 s), by a median of 4 times and up to about 100 times (inventory T = 64,
  k = 3, dual: 1.1 s against 103 s). The gains were largest on the duals with breakpoints; on the
  large primals it was 1.5 to 3 times faster.
- **The cores matter on the large primals.** The concurrent default was faster still there (hydro
  T = 8, k = 3, primal: 4.6 s against 234 s on one thread), but not on every dual (kwg T = 48,
  k = 3: 34 s against 12 s). Against the best mode of HiGHS, it was at least 1.5 times faster on
  every instance (hydro T = 8, k = 3, dual: 69 s, against 114 s with HiPO and 1677 s with the
  simplex method).

The documentation (the docstrings and `docs/src/performance.md`) does not mention commercial
solvers. Open: HiGHS's parallel dual simplex (`"parallel" => "on"` with `"simplex_strategy" => 2`
or `3`), which has not been tried.
