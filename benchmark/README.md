# Benchmarks

The benchmarks of this folder time the package on problems that the tests share
(`test/problems/`). Their results are recorded, not asserted: nothing here fails on a slowdown.

Each folder holds the scripts of one problem, or the studies of one kind, with a README of what
they measure and of their results, and its CSV files in `results/`:

| Folder | Scripts | What they measure |
|---|---|---|
| [`inventory/`](inventory/README.md) | `timing.jl`, `resolve.jl` | the T-period inventory, one scalar uncertainty per stage: `PrimalLDR`, `DualLDR` and the same robust LP written by hand; re-solves after a change of a JuMP parameter, with and without the cache of inner problems |
| [`hydro/`](hydro/README.md) | `run.jl` | the 4-reservoir hydrothermal problem, four joint uncertainties per stage: the LDR models, the average scenario and a scenario tree, with breakpoints and out-of-sample simulations |
| [`breakpoints/`](breakpoints/README.md) | `lifted_box.jl` | the choices for uncertainties with breakpoints: the implied box rows of lifted coordinates (`lifted_box`), on four problems and four solvers of HiGHS |
| [`scaling/`](scaling/README.md) | `rescale.jl` | the scaling of the uncertainties: the production planning in three units, `RescaleUncertainties()` on and off on production and kwg, and the ways to rescale (`Experimental.RescalingMode`) on four problems |
| [`solvers/`](solvers/README.md) | `solvers.jl` | other solvers than HiGHS: Gurobi, by default and with the dual simplex on one thread, on the grid of `breakpoints/` and the hydrothermal problem |

In this folder, `common.jl` has the helpers that the scripts share, `grid.jl` the grid of runs of
the three studies (problems, solvers, options and CSV rows; it offers Gurobi when Gurobi is
installed, which is not a dependency), and `internals.jl` is the only file of `benchmark/` that
reads internals of the package. [`PROBLEMS.md`](PROBLEMS.md) is the
catalogue of the LDR test problems found in this package, the legacy repository, `CasosLDR` and
the papers (data, known values, solver notes, candidates to port).

The READMEs of `inventory/` and `hydro/` have the same sections: **Current** (the last recording),
**Against 0.3** (the previous version of the package) and **History** (the earlier recordings, kept
for comparison). [Past studies](#past-studies) are finished studies whose scripts are no longer in
the tree.

## Decisions and the experiments behind them

The choices of the package that rest on measurements, with every experiment that informed each
one. An experiment can serve several decisions, and cover several problems.

| Decision | In the code | Experiments: script, problems, solvers | Recorded at | Details |
|---|---|---|---|---|
| The primal writes a box support in closed form: one auxiliary variable per row and box, instead of two multipliers | `PrimalLDR(; box_closed_form = true)`, the default | `inventory/timing.jl`, T = 4 to 128, and `hydro/run.jl`, T = 4, 8, 12; HiGHS's default options | 7dbebf0 against 5f20d11 | [inventory](inventory/README.md#history), [hydro](hydro/README.md#history) |
| The formulations of 0.4 replace those of 0.3: a smaller primal and a structured dual. The dual keeps the auxiliary variables `u = S μ` of its cone rows; substituting them out is left open | no setting | the past study against 0.3: the inner LPs of both versions exported to MPS, inventory T = 32, 64, 128 and hydro T = 4, 8, 12; the modes of HiGHS and Ipopt. Also "Against 0.3" of `inventory/timing.jl` and `hydro/run.jl` | tag `study-0.3-comparison` | [past study](#solve-times-against-03-october-2026-tag-study-03-comparison), [inventory](inventory/README.md#against-03), [hydro](hydro/README.md#against-03) |
| The solver advice: keep HiGHS's dual simplex by default; for `DualLDR`, IPX with crossover is often the fastest on large duals, and the interior points without crossover fail too often to be a default. Gurobi was more robust and faster than every mode of HiGHS, on one thread too; the documentation does not mention commercial solvers | the docstrings of `PrimalLDR` and `DualLDR`, and `docs/src/performance.md` | the past study against 0.3 (above); `hydro/run.jl`, variant `ldr_dual_ipm`, T = 4, 8, 12; `breakpoints/lifted_box.jl`, four problems under the simplex method, IPX with and without crossover, and HiPO; `solvers/solvers.jl`, the same and the hydrothermal problem under Gurobi, by default and on one thread | tag `study-0.3-comparison`; 7dbebf0; c55805e and 40ad9a2; 92e69a1 | [past study](#solve-times-against-03-october-2026-tag-study-03-comparison), [hydro](hydro/README.md#current), [`breakpoints/`](breakpoints/README.md#what-it-found), [`solvers/`](solvers/README.md#what-it-found) |
| The implied bounds of lifted coordinates: left out of the primal's support, kept in the dual's. Open for the dual: the wider study found it faster without them on 13 of 15 instances under IPX with crossover, without `RescaleUncertainties()`; with it, faster or as fast on all 12 production and hydrothermal duals under IPX with crossover, but slower on two hydrothermal duals under the simplex method | `PrimalLDR(; lifted_box = false)` and `DualLDR(; lifted_box = true)`, the defaults | the first study: inventory T = 16 to 64 and hydro T = 4, 8, with 1 to 3 breakpoints, under the dual simplex; then `breakpoints/lifted_box.jl`: inventory, kwg, hydro and production planning, under the simplex method, IPX with and without crossover, and HiPO; then production and hydro with rescaling, under the simplex method and IPX with crossover | e26b0e5; c55805e and 40ad9a2; 212e41d | [past study](#redundant-box-rows-of-liftings-october-2026-lifted_box), [`breakpoints/`](breakpoints/README.md) |
| The optimizer rescales each uncertainty to `[-1, 1]` before the formulations, and divides each rescaled row of the support by its largest coefficient. With breakpoints, each lifted coordinate is rescaled on its own box. It is rarely the fastest of the modes (hybrid is the fastest on every production dual, before lifting on every production primal), but no other mode is more reliable, so it stays the default for now; the hydrothermal problem is fastest without rescaling | `RescaleUncertainties()`, true by default | `scaling/rescale.jl`: the production planning in three units and rescaled, T = 10 to 20, and kwg with and without rescaling, T = 24, 48, with 1 and 3 breakpoints, under HiGHS's simplex and HiPO and Gurobi; then the modes on production, kwg, hydro and the inventory, under HiGHS's simplex | b8a7159; 82c4496; 0d2fae4; b3b2861 and 212e41d | [`scaling/`](scaling/README.md#what-it-found) |
| A re-solve fixes variables of the inner problem to the values of the parameters, instead of holding JuMP `Parameter`s, whose bridges for HiGHS made JuMP copy the whole model again at every re-solve | no setting (`src/formulations/inner.jl`) | `inventory/resolve.jl`, T = 64 and 20 re-solves, `PrimalLDR` and `DualLDR`; HiGHS's default options | f204dc1 against 423d23c | [inventory](inventory/README.md#history-1) |
| The optimizer keeps its inner problems across re-solves when only the values of parameters change | no setting (`src/moi/inner_cache.jl`) | `inventory/resolve.jl`, as above, the cache kept or emptied | 7dbebf0; f204dc1 | [inventory](inventory/README.md#current-1) |

## Running

The scripts share one environment, which uses the package of this repository (it needs Julia 1.11
or later). Run them from the repository root, alone on an idle machine (timings of parallel runs
vary by ±30%):

```
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/inventory/timing.jl 4 8 16 32 64 128
julia --project=benchmark benchmark/inventory/resolve.jl          # T = 64, N = 20
julia --project=benchmark benchmark/hydro/run.jl                  # T = 4, 8, 12
```

The machine itself drifts between days. On 31 October 2026 the same models took about 1.7 times as
long as on 25 October and 5 November, while four rounds in a row on 5 November differed by 2 to 4%.
So check its state before a recording with `julia --project=benchmark
benchmark/inventory/timing.jl 128`, and compare two of its rows with the times of the machine of
these READMEs (Windows 11, Julia 1.12.6, HiGHS 1.15.1):

| Row | Model | Solver time | Measured |
|---|---|--:|---|
| `hand` | the hand-written LP of the inventory, T = 128 | 1.95 to 1.97 s | 8 November 2026 |
| `ldr_primal` | the primal LDR of the same inventory | 1.23 to 1.25 s | 8 November 2026, code of c359166 and e2e212f |

The hand-written LP does not use the package, so its time depends on the machine and HiGHS alone.
The primal LDR depends on the code too: a change of the package can change its time without any
change of the machine, as the rescaling of the uncertainties (82c4496) changed those of the duals.
When its time moves and that of the LP does not, look for the commit; when both move, the machine.
If the LP is more than 10% slower, wait, or restart the machine and wait: on 5 November, a run
soon after a restart was still 10 to 25% slower, and a run 1h45 later was not. A recording on a
slower day is still kept, with a note of its speed, as in the
[`lifted_box` study](#redundant-box-rows-of-liftings-october-2026-lifted_box). Update the table
with each recording, and the commit of the primal LDR when its time changes.

Each script appends its rows to its CSV file, in the folder `results/` next to it, one row at a
time, so that an interrupted run keeps its results. The environment variable
`LDR_BENCHMARK_RESULTS` names another folder for all the CSV files; the CI workflow `Benchmarks`
uses it to run the three scripts above and `scaling/rescale.jl` at tiny sizes into a temporary
folder, to check that they still run. The scripts also check their values: `timing.jl` that the
primal LDR and the hand-written LP equal 90T + 20 and that the dual is not above it, `resolve.jl`
that the cached and rebuilt re-solves agree, `hydro/run.jl` that no dual bound exceeds the primal
value, and `grid.jl` that the values do not depend on `lifted_box` or `rescale`. A failed check
warns, so that a long recording is not lost to one value; with `LDR_BENCHMARK_STRICT=true`, which
the CI workflow sets, it is an error, raised before the row is written. The commands of the
studies are in their READMEs.

Every row is tagged with `commit` (the commit of the repository), `src_commit` (the last commit
that changed the measured code: `src/`, `Project.toml` and `test/problems/`), `src_dirty`
(uncommitted changes to that code), the Julia version and `highs`, the version of the HiGHS library.
The last seven columns of every file are the phases of `time_breakdown(model)`, through
`phase_times` of `common.jl`: `canonicalize_s`, `moments_s`, `checks_s`, `build_s`, `solve_s`,
`results_s` and `total_s`. They are empty in the rows recorded before them, and for models that do
not use the package. Rows with the same `src_commit` measure the same code, whatever else changed,
so the recordings below are named by their `src_commit`:

| `src_commit` | `commit` of its rows | The measured code |
|---|---|---|
| 92889ec | eb80cdf (`timings.csv`) | the first recording of `timing.jl` |
| 5f20d11 | 46c6473 (`timings.csv`); 30bc10c, 6436a06, fcf7815 (`hydro.csv`) | before the closed form of box supports |
| 423d23c | 354a8de (`resolve.csv`) | re-solves with JuMP `Parameter`s in the inner problems |
| f204dc1 | f204dc1 (`resolve.csv`) | re-solves with the parameters as fixed variables |
| 7dbebf0 | 2509dbb (`timings.csv`); 901bd78 (`hydro.csv`); bf91f9c, 682dd11 (`resolve.csv`) | the closed form of box supports and the re-solves with fixed variables: the last recording |

The hydrothermal README has two earlier recordings of its own, 6307590 and 01c5f1c.

## Past studies

These studies answered one question each and are finished. Their scripts and results are no longer
in the tree, because they cannot be re-run from it; the tag given with each study keeps them.

### Solve times against 0.3 (October 2026, tag `study-0.3-comparison`)

The question: is the smaller 0.4 inner LP slower for HiGHS than the one of 0.3? The study exported
the inner LPs of both versions to MPS and solved each file with HiGHS 1.15.1 in a fresh process, on
inventory T = 32, 64, 128 and hydro T = 4, 8, 12. It needed the environment of 0.3 and about 0.5 GB
of MPS files, which were not committed. Its 0.4 primal is the one before the closed form of box
supports. The scripts, the README with every table, and the CSVs are in
[`benchmark/compare/` at the tag](https://github.com/bfpc/LinearDecisionRules.jl/tree/study-0.3-comparison/benchmark/compare).

Inventory, T = 128 (constraints without variable bounds; HiGHS run time on the exported models, and
the solver time through JuMP for the hand-written LP, from `inventory/results/timings.csv`):

| Model | Variables | Constraints | Nonzeros | HiGHS (s) |
|---|---:|---:|---:|---:|
| 0.3 primal | 223,168 | 149,120 | 471,231 | 0.61 |
| 0.3 primal, `SlackAsExpr` | 157,120 | 83,072 | 339,135 | 0.72 |
| hand-written robust LP | 57,792 | 74,432 | 271,807 | 1.9 |
| 0.4 primal | 90,560 | 41,662 | 206,013 | 0.63 |
| 0.3 dual (dense) | 91,072 | 213,632 | 17,049,279 | 1048 |
| 0.4 dual (structured) | 58,812 | 107,708 | 273,079 | 6.9 |

What it found:
- **Primal: no.** After presolve, the 0.4 primal and the 0.3 primal with `SlackAsExpr` are the same
  LP (hydro T = 12: about 42.2k rows, 150.5k columns and 390.9k nonzeros for both, 161 s and
  168 s). The order of rows and columns alone changes the simplex iterations by ±30%. The smaller
  formulation saves the time to build and to read the model, not solver time.
- **Dual: it depends on the instance, and the size is not the reason.** The 0.4 dual is far faster
  on the inventory (6.9 s against 1,048 s at T = 128) but slower on the hydro problem (145 s
  against 22.4 s at T = 8), where presolve reduces the dense 0.3 dual much more than the sparse
  0.4 one. Conditioning and row order were ruled out.
- **Solver modes.** No HiGHS mode is the best everywhere. For `DualLDR`, the interior point method
  without crossover (`"solver" => "ipm", "run_crossover" => "off"`) solved all five 0.4 duals
  within 7 s, with status `OPTIMAL`. For `PrimalLDR`, the default dual simplex is within 2 times of
  the best mode except at hydro T = 8, where IPX without crossover is 3 times faster; HiGHS's
  interior point is not reliable on the 0.4 hydro primal. Ipopt converged but was never the
  fastest, and is off by 5e-5 to 4e-4 on the hydro problems. These are the recommendations of the
  `PrimalLDR` and `DualLDR` docstrings.
- **An idea left open.** Substituting out the auxiliary variables `u = S μ` of the dual cone rows
  gives the presolved model of the dense 0.3 dual: 3.7 times faster at hydro T = 8 with the default
  dual simplex, but 20 times slower at inventory T = 64, and slower with the interior point. A rule
  to substitute `u` only where a row has few columns would need more instances.

### Redundant box rows of liftings (October 2026, `lifted_box`)

The question: the support of an uncertainty with `n` breakpoints is a simplex with `n + 2` facets,
its hull rows and two bounds of the box of the lifted coordinates; should Ξ also keep the other
`2n` bounds, which are implied? They change no value, but give the primal more multipliers and the
dual more rows. Measured on 31 October 2026 at e26b0e5 (HiGHS 1.15.1, default dual simplex,
Julia 1.12.6, Windows 11, a machine about 1.6 times slower that day than for the other tables),
with the whole box ("with") and without the implied bounds ("without"); every value agrees to
1e-12. The scripts were not kept: the setting is `lifted_box` of `PrimalLDR` and `DualLDR`, so the
study can be re-run from the tree.

Hydrothermal problem, `k` breakpoints per inflow (solver time; T = 8 with a limit of 600 s):

| T | k | Primal variables | Primal, with | Primal, without | Dual rows | Dual, with | Dual, without |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 1 | 42,736 → 28,720 | 3.5 s | 2.5 s | 51,276 → 37,260 | 10.3 s | 6.4 s |
| 4 | 2 | 67,312 → 39,280 | 8.5 s | 6.5 s | 79,524 → 51,492 | 54.7 s | 28.9 s |
| 8 | 1 | 197,856 → 132,448 | 187 s | 106 s | 235,104 → 169,696 | 105 s | 276 s |

The T = 8 dual is slower without the implied bounds under every seed of HiGHS tried
(`"random_seed" => 1, 2`: 110 s and 84 s with them, 303 s and 267 s without), and the T = 4 dual
faster (seeds 0 to 3: 9.9 to 12.2 s with them, 6.4 to 7.5 s without).

Inventory (the problem of `inventory/timing.jl`), `k` breakpoints per demand, solver time:

| T | k | Primal variables | Primal, with | Primal, without | Dual rows | Dual, with | Dual, without |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 1 | 3,392 → 2,368 | 0.02 s | 0.02 s | 3,996 → 2,972 | 0.13 s | 0.12 s |
| 16 | 3 | 7,248 → 4,176 | 0.05 s | 0.04 s | 8,364 → 5,292 | 0.85 s | 0.69 s |
| 32 | 1 | 13,440 → 9,344 | 0.10 s | 0.10 s | 15,676 → 11,580 | 1.98 s | 1.37 s |
| 32 | 3 | 28,832 → 16,544 | 0.27 s | 0.22 s | 33,116 → 20,828 | 12.1 s | 11.2 s |
| 64 | 1 | 53,504 → 37,120 | 0.57 s | 0.59 s | 62,076 → 45,692 | 14.7 s | 12.7 s |

What it found: leaving the implied bounds out saved 30% to 43% of the primal variables and 26% to
37% of the dual rows. It never slowed the primal down by more than the noise (0.57 s against
0.59 s at inventory T = 64), and saved up to 43% of its solve time; it made the
dual 8% to 47% faster, except on the hydro problem with T = 8, 2.6 times slower. So `PrimalLDR`
leaves them out by default, and `DualLDR` keeps them (`DualLDR(; lifted_box = false)` leaves them
out). A wider study, on four problems and four solvers, followed in
[`breakpoints/README.md`](breakpoints/README.md).
