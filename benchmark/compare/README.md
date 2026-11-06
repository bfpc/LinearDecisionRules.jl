# Decision studies

Each script here answers one question with a grid of runs; its results are in `results/`, and
its findings below. When a study is finished and its script can no longer run from the tree, it
moves to "Past studies" in [`../README.md`](../README.md), with a tag that keeps it.

## The implied box rows of lifted coordinates, across problems and solvers (`lifted_box.jl`)

The question: should the support of an uncertainty with breakpoints keep the bounds of the box of
its lifted coordinates that its hull rows imply (`lifted_box = true`, "with" below), or leave them
out (`false`, "without")? A first study (October 2026, in "Past studies" of
[`../README.md`](../README.md)) measured two problems under the dual simplex, and chose `false` for
`PrimalLDR` and `true` for `DualLDR`, because one dual, the hydrothermal problem with T = 8, was
2.6 times slower without them. This one widens the grid to four problems and four solvers of HiGHS.

The problems are those of `test/problems/`, each with `k` breakpoints on every uncertainty:

| Problem | Source | Horizons |
|---|---|---|
| `inventory` | the inventory with backlog of `benchmark/inventory/timing.jl` | T = 16, 32, 64 |
| `kwg` | the inventory of Kuhn, Wiesemann and Georghiou (2011) | T = 24, 48 |
| `hydro` | the hydrothermal problem of `benchmark/hydro` | T = 4, 6, 8 |
| `production` | the production planning of Georghiou, Kuhn and Wiesemann (2019) | T = 10, 15, 20 |

The solvers are HiGHS's dual simplex (its default), IPX with crossover to a basic solution, IPX
without crossover, and HiPO without crossover (`"solver" => "ipm"` in HiGHS 1.15). The grid was run
from 1 to 2 November 2026 at commit c55805e (`src_commit` f4a45fa), with HiGHS 1.15.1 and Julia
1.12.6 on Windows 11 (20 cores), alone on the machine, with a time limit of 600 s. The runs that
reached the limit or came close were run again with a limit of 3600 s at 40ad9a2 (same
`src_commit`), into `results/limit3600/`; tests and documentation builds of another checkout ran
for a few minutes during those reruns. Times are those of the inner solver, in seconds.

```
julia --project=benchmark benchmark/compare/lifted_box.jl hydro T=4,6,8 k=1,3 solver=simplex,ipx,ipx_crossover,hipo
julia --project=benchmark benchmark/compare/lifted_box.jl kwg T=48 k=3 kind=dual solver=simplex limit=3600
```

Every value agrees between "with" and "without", to 1e-10 relative with a basic solution and to
5e-7 without crossover.

### The primal

Solver time; **bold** values come from the reruns with a limit of 3600 s, and "limit" is the limit.

| Problem | T | k | Variables, with → without | Simplex, with | Simplex, without | IPX + crossover, with | IPX + crossover, without |
|---|--:|--:|--:|--:|--:|--:|--:|
| inventory | 16 | 1 | 3,392 → 2,368 | 0.0 | 0.0 | 0.1 | 0.1 |
| inventory | 16 | 3 | 7,248 → 4,176 | 0.0 | 0.1 | 0.2 | 0.1 |
| inventory | 32 | 1 | 13,440 → 9,344 | 0.1 | 0.1 | 0.5 | 0.4 |
| inventory | 32 | 3 | 28,832 → 16,544 | 0.3 | 0.3 | 1.5 | 1.3 |
| inventory | 64 | 1 | 53,504 → 37,120 | 0.6 | 0.7 | 5.1 | 4.3 |
| inventory | 64 | 3 | 115,008 → 65,856 | 1.7 | 2.3 | 17 | 16 |
| kwg | 24 | 1 | 14,232 → 9,288 | 1.3 | 1.3 | 1.3 | 0.8 |
| kwg | 24 | 3 | 30,864 → 16,032 | 7.6 | 5.2 | 5.6 | 2.5 |
| kwg | 48 | 1 | 54,960 → 35,856 | 30 | 27 | 9.3 | 8.3 |
| kwg | 48 | 3 | 119,328 → 62,016 | 157 | 117 | 60 | 27 |
| hydro | 4 | 1 | 42,736 → 28,720 | 3.5 | 2.7 | 3.1 | 1.9 |
| hydro | 4 | 3 | 91,888 → 49,840 | 12 | 9.1 | 19 | 5.4 |
| hydro | 6 | 1 | 106,248 → 71,208 | 33 | 20 | 15 | 11 |
| hydro | 6 | 3 | 229,128 → 124,008 | 216 | 86 | 91 | 30 |
| hydro | 8 | 1 | 197,856 → 132,448 | 197 | 122 | 63 | 37 |
| hydro | 8 | 3 | 427,232 → 231,008 | **1281** | **713** | 292 | 122 |
| production | 10 | 1 | 36,645 → 25,485 | 2.8 | 1.7 | 4.5 | 3.9 |
| production | 10 | 3 | 78,675 → 45,195 | 16 | 38 | 24 | 45 |
| production | 15 | 1 | 86,745 → 60,285 | 34 | 14 | 21 | 23 |
| production | 15 | 3 | 186,425 → 107,045 | 105 | 40 | 162 | 79 |
| production | 20 | 1 | 158,095 → 109,835 | 79 | 37 | 72 | 59 |
| production | 20 | 3 | 339,925 → 195,145 | **596** | **limit** | **839** | **limit** |

### The dual

| Problem | T | k | Constraints, with → without | Simplex, with | Simplex, without | IPX + crossover, with | IPX + crossover, without |
|---|--:|--:|--:|--:|--:|--:|--:|
| inventory | 16 | 1 | 3,996 → 2,972 | 0.1 | 0.1 | 0.1 | 0.2 |
| inventory | 16 | 3 | 8,364 → 5,292 | 0.9 | 0.9 | 0.3 | 0.1 |
| inventory | 32 | 1 | 15,676 → 11,580 | 2.1 | 1.5 | 2.3 | 1.1 |
| inventory | 32 | 3 | 33,116 → 20,828 | 12 | 11 | 2.2 | 1.0 |
| inventory | 64 | 1 | 62,076 → 45,692 | 15 | 13 | 16 | 15 |
| inventory | 64 | 3 | 131,772 → 82,620 | 103 | 82 | 10 | 9.4 |
| kwg | 24 | 1 | 17,694 → 12,750 | 17 | 11 | 21 | 14 |
| kwg | 24 | 3 | 37,470 → 22,638 | 138 | 78 | 144 | 7.2 |
| kwg | 48 | 1 | 67,638 → 48,534 | **625** | 435 | **675** | 452 |
| kwg | 48 | 3 | 144,054 → 86,742 | **limit** | **limit** | **limit** | **limit** |
| hydro | 4 | 1 | 51,276 → 37,260 | 10 | 6.5 | 11 | 7.3 |
| hydro | 4 | 3 | 107,772 → 65,724 | 41 | 58 | 20 | 8.9 |
| hydro | 6 | 1 | 126,694 → 91,654 | 67 | 64 | 71 | 67 |
| hydro | 6 | 3 | 267,934 → 162,814 | 279 | **658** | 390 | 45 |
| hydro | 8 | 1 | 235,104 → 169,696 | 108 | 256 | 122 | 285 |
| hydro | 8 | 3 | 498,752 → 302,528 | **1677** | **1235** | **2022** | 162 |
| production | 10 | 1 | 44,117 → 32,957 | fails | fails | fails | fails |
| production | 10 | 3 | 93,257 → 59,777 | fails | 416 | fails | 414 |
| production | 15 | 1 | 103,977 → 77,517 | fails | fails | fails | fails |
| production | 15 | 3 | 220,317 → 140,937 | **limit** | **limit** | **limit** | **limit** |
| production | 20 | 1 | 189,087 → 140,827 | limit | limit | limit | limit |
| production | 20 | 3 | 401,127 → 256,347 | limit | limit | limit | limit |

"fails" is a status Unknown of HiGHS (`OTHER_ERROR`) before the limit.

### Without crossover

The interior points without crossover often end with the status Unknown (`OTHER_ERROR`), when
their solution is not accurate enough and HiGHS cannot clean it up:

| Problem | Formulation | IPX fails | HiPO fails |
|---|---|--:|--:|
| inventory | primal | 0 of 12 | 0 of 12 |
| inventory | dual | 0 of 12 | 0 of 12 |
| kwg | primal | 0 of 8 | 0 of 8 |
| kwg | dual | 8 of 8 | 8 of 8 |
| hydro | primal | 3 of 12 | 8 of 12 |
| hydro | dual | 1 of 12 | 6 of 12 |
| production | primal | 12 of 12 | 12 of 12 |
| production | dual | 10 of 12 | 10 of 12 |

When they succeed they are often the fastest (the hydrothermal dual with T = 8 and k = 3: 61 s with
IPX, 71 s with HiPO, against 162 s with IPX and crossover), but their solution is not exactly
feasible, so the bound of a dual can be invalid by a little: the hydrothermal duals with k = 1 end
up to 7e-6 relative above the exact bound of the simplex method, for a minimization.

### What it found

- **The primal: leave the implied bounds out.** Without them, the primal has 30% to 48% fewer
  variables. It is faster on every kwg and hydrothermal instance, under both solvers: up to 2.5
  times under the simplex method (hydro T = 6, k = 3) and 3.4 times under IPX with crossover (hydro
  T = 4, k = 3), and 1281 s against 713 s with T = 8 and k = 3. On the inventory, the times are
  below 2.5 s under the simplex method, and differ by noise. The exception is the production
  planning with 3 breakpoints, where the simplex method takes 16 s against 38 s with T = 10, and
  596 s against more than an hour with T = 20; IPX with crossover behaves the same way. The
  default of `PrimalLDR` stays `false`.
- **The dual: leaving them out is faster more often than not, but not always.** Under IPX with
  crossover, the dual without the implied bounds is faster on 13 of the 15 instances solved both
  ways, by up to 20 times (kwg T = 24, k = 3: 144 s against 7.2 s; hydro T = 6, k = 3: 390 s against
  45 s; hydro T = 8, k = 3: 2022 s against 162 s); the exceptions are inventory T = 16, k = 1 (0.1 s
  against 0.2 s) and the hydrothermal problem with T = 8 and k = 1 (122 s against 285 s), the
  instance of the first study. Under the simplex method, it is faster on the inventory and on kwg,
  but slower by 1.4 to 2.4 times on three of the six hydrothermal instances. The only production
  dual solved at all, T = 10 with k = 3, is solved only without them.
- **The solver matters more than the rows.** The largest duals solve much faster with IPX and
  crossover than with the simplex method when the implied bounds are left out (hydro T = 8, k = 3:
  162 s against 1235 s), and the interior points without crossover fail too often to be a default
  (above). The production duals, which no solver solves from T = 15, are the hard instances of the
  grid.

The duals of kwg with T = 48 and k = 3 and of the production planning with T = 15 and k = 3 reach
the limit of one hour too, under both solvers, with and without the implied bounds.
