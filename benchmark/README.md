# Benchmarks

The benchmarks of this folder time the package on problems that the tests share
(`test/problems/`). Their results are recorded, not asserted: nothing here fails on a slowdown.

| Script | What it measures | Results | Section |
|---|---|---|---|
| `timing.jl` | the T-period inventory: `PrimalLDR`, `DualLDR` and the same robust LP written by hand | `results/timings.csv` | [Inventory](#inventory-timingjl) |
| `resolve.jl` | the inventory re-solved after a change of a JuMP parameter, with and without the cache of inner problems | `results/resolve.csv` | [Re-solves](#re-solves-with-a-parameter-resolvejl) |
| `hydro/run.jl` | the 4-reservoir hydrothermal problem: the LDR models, the average scenario and a scenario tree, with breakpoints and out-of-sample simulations | `results/hydro.csv` | [`hydro/README.md`](hydro/README.md) |
| [`PROBLEMS.md`](PROBLEMS.md) | the catalogue of the LDR test problems found in this package, the legacy repository, `CasosLDR` and the papers (data, known values, solver notes, candidates to port) | | |

Each README has the same sections: **Current** (the last recording), **Against 0.3** (the previous
version of the package) and **History** (the earlier recordings, kept for comparison).
[Past studies](#past-studies) are finished studies whose scripts are no longer in the tree.

## Running

The scripts share one environment, which uses the package of this repository (it needs Julia 1.11
or later). Run them from the repository root, alone on an idle machine (timings of parallel runs
vary by ±30%):

```
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/timing.jl 4 8 16 32 64 128
julia --project=benchmark benchmark/resolve.jl          # T = 64, N = 20
julia --project=benchmark benchmark/hydro/run.jl        # T = 4, 8, 12
```

Each script appends its rows to its CSV file, one row at a time, so that an interrupted run keeps
its results. The environment variable `LDR_BENCHMARK_RESULTS` names another folder for the CSV
files; the CI workflow `Benchmarks` uses it to run the three scripts at tiny sizes into a temporary
folder, which only checks that they still run.

Every row is tagged with `commit` (the commit of the repository), `src_commit` (the last commit
that changed the measured code: `src/`, `Project.toml` and `test/problems/`), `src_dirty`
(uncommitted changes to that code), the Julia version and `highs`, the version of the HiGHS library.
Rows with the same `src_commit` measure the same code, whatever else changed, so the recordings
below are named by their `src_commit`:

| `src_commit` | `commit` of its rows | The measured code |
|---|---|---|
| 92889ec | eb80cdf (`timings.csv`) | the first recording of `timing.jl` |
| 5f20d11 | 46c6473 (`timings.csv`); 30bc10c, 6436a06, fcf7815 (`hydro.csv`) | before the closed form of box supports |
| 423d23c | 354a8de (`resolve.csv`) | re-solves with JuMP `Parameter`s in the inner problems |
| f204dc1 | f204dc1 (`resolve.csv`) | re-solves with the parameters as fixed variables |
| 7dbebf0 | 2509dbb (`timings.csv`); 901bd78 (`hydro.csv`); bf91f9c, 682dd11 (`resolve.csv`) | the closed form of box supports and the re-solves with fixed variables: the last recording |

The hydrothermal README has two earlier recordings of its own, 6307590 and 01c5f1c.

## Inventory (`timing.jl`)

`timing.jl` times the T-period inventory problem with backlog (see the header of the script) for
the primal LDR (`ldr_primal`), the dual LDR (`ldr_dual`) and a hand-written robust LP (`hand`),
and appends one row per T and variant. The objective of the primal LDR and of the hand LP must be
90T + 20; the script warns otherwise. The dual value is a lower bound (for example 10673.33 against
11540 at T = 128).

Columns: `build_s` writes the JuMP model, `optimize_s` is `optimize!` (canonical form, moments,
reformulation, inner solve), `solver_s` is the inner solver's time. The sizes are those of the
reformulated model that is solved (constraints do not count variable bounds; nonzeros count the
coefficients of the constraint matrix). Times are the best of 3 runs.

### Current

Recorded at 7dbebf0: `timing.jl 4 8 16 32 64 128`, Julia 1.12.6, HiGHS 1.15.1, idle machine. T = 128, times in seconds:

| Model | Variables | Constraints | Nonzeros | build | optimize | solver |
|---|---:|---:|---:|---:|---:|---:|
| hand-written LP | 57,792 | 74,432 | 271,807 | 0.047 | 2.105 | 1.977 |
| `PrimalLDR` | 57,792 | 41,662 | 189,757 | 0.0003 | 1.313 | 1.274 |
| `DualLDR` | 58,812 | 107,708 | 273,079 | 0.0003 | 7.442 | 7.391 |

`PrimalLDR` eliminates box support blocks in closed form (`box_closed_form = true`), so the primal
has the hand LP's number of variables at every T (70, 252, 952, 3,696, 14,560 and 57,792 for T = 4
to 128): this was the target of the closed form, and it is met. Writing the LDR model takes under
0.5 ms; `optimize!` spends about 0.04 s outside the inner solve at T = 128.

The smaller primal is not faster for HiGHS: it is equal or slightly faster up to T = 64 and 16%
slower at T = 128 than before the closed form (see History), although it also has 8% fewer
nonzeros. The saving is in memory and in the reformulation, not in the HiGHS time (the run-to-run
spread of the other rows is 1 to 2%). The dual has about as many nonzeros as the hand LP and is the
slowest variant, with a solve time that grows faster with T than the primal's (0.85 s at T = 64,
7.39 s at T = 128).

### Against 0.3

The legacy rows are `results/timings.csv` of the old repository, at commit 17ce1eb of its branch
`Multistage` (version 0.3.1, Julia 1.12.6; `ldr_primal` is its default primal model). The header
of `test/reference/generate_legacy_values.jl` says how to make an environment for that version.
T = 128, times in seconds, best of 3:

| Model | Variables | Constraints | Nonzeros | build | optimize | solver |
|---|---:|---:|---:|---:|---:|---:|
| legacy primal | 223,168 | 149,120 | n/a | 0.0004 | 2.013 | 0.602 |
| legacy primal, `SlackAsExpr` | 157,120 | 83,072 | n/a | 0.0004 | 1.688 | 0.617 |
| hand-written LP (5f20d11) | 57,792 | 74,432 | 271,807 | 0.085 | 2.077 | 1.952 |
| 0.4 `PrimalLDR` (5f20d11) | 90,560 | 41,662 | 206,013 | 0.0004 | 1.137 | 1.095 |
| 0.4 `PrimalLDR` (7dbebf0) | 57,792 | 41,662 | 189,757 | 0.0003 | 1.313 | 1.274 |
| 0.4 `DualLDR` (5f20d11) | 58,812 | 107,708 | 273,079 | 0.0003 | 7.346 | 7.295 |

In the legacy recording the hand LP took 0.115 s to build and 1.686 s to optimize; the same script
gave 0.085 s and 2.077 s at 5f20d11 (0.106 s and 2.032 s at 92889ec), so differences of that size
between runs are noise. Before the closed form, the 0.4 primal model had 41% of the legacy
variables (58% of the legacy `SlackAsExpr` model) and 28% of its constraints; with it, 26% of the
variables. `optimize!` takes 1.14 s against 2.01 s, of which 0.04 s is not the inner solve (1.41 s
in the legacy recording). HiGHS needs more time on the smaller 0.4 primal model (1.10 s against
0.60 s); the [past study of solve times](#solve-times-against-03-october-2026-tag-study-03-comparison)
shows why.

### History

**7dbebf0 against 5f20d11: the closed form of box supports.** Same protocol. The primal lost its
multipliers on boxes: 57,792 variables at T = 128 against 90,560, the same 41,662 constraints, and
189,757 nonzeros against 206,013; its variables at T = 4 to 64 were 102, 380, 1,464, 5,744 and
22,752. The objectives are identical in all 12 LDR rows (90T + 20 for the primal); the dual's
sizes, nonzeros and bounds are identical. Times at T = 128 in seconds, 5f20d11 then 7dbebf0:

| Model | Variables | Nonzeros | build | optimize | solver | solver ratio |
|---|---|---|---|---|---|---:|
| hand | 57,792 | 271,807 | 0.085, 0.047 | 2.077, 2.105 | 1.952, 1.977 | 1.01 |
| `PrimalLDR` | 90,560, 57,792 | 206,013, 189,757 | 0.0004, 0.0003 | 1.137, 1.313 | 1.095, 1.274 | 1.16 |
| `DualLDR` | 58,812 | 273,079 | 0.0003, 0.0003 | 7.346, 7.442 | 7.295, 7.391 | 1.01 |

Primal solver times, 5f20d11 then 7dbebf0, in ms: T = 4 0.76, 0.70; T = 8 1.75, 1.70; T = 16 6.9,
5.3; T = 32 27.8, 26.9; T = 64 143, 146; T = 128 1095, 1274.

**5f20d11 against 92889ec.** `timing.jl 4 8 16 32 64 128`, same protocol, Julia 1.12.6. The
quadratic-objective, dual-bound and attribute changes between them do not touch these LP models:
variables, constraints, nonzeros and objectives are identical in all 18 rows. Optimize times,
92889ec then 5f20d11, in seconds (ratio new/old; differences below about 5% are noise, and the
sub-10 ms rows are dominated by timer resolution):

| T | hand | primal | dual |
|--:|---|---|---|
| 4 | 0.00082, 0.00084 | 0.0018, 0.0019 | 0.0020, 0.0021 |
| 8 | 0.0020, 0.0019 | 0.0028, 0.0027 | 0.0041, 0.0040 |
| 16 | 0.0079, 0.0078 | 0.0094, 0.0085 | 0.0146, 0.0149 |
| 32 | 0.0354, 0.0354 | 0.0331, 0.0317 | 0.0911, 0.0900 |
| 64 | 0.218, 0.212 (0.97) | 0.161, 0.154 (0.96) | 0.838, 0.847 (1.01) |
| 128 | 2.032, 2.077 (1.02) | 1.154, 1.137 (0.99) | 7.449, 7.346 (0.99) |

Writing the LDR models took under 0.5 ms at T = 128 in both recordings.

## Re-solves with a parameter (`resolve.jl`)

`resolve.jl` solves the inventory with its initial stock as a JuMP parameter, then re-solves it
`N` times with new values, keeping the inner problem (`cached`, the default) or emptying the cache
before each `optimize!` (`rebuild`, what every re-solve did before the inner problems carried the
parameters). It appends one row per formulation and variant, and warns if the objectives of the two
variants differ. The columns, over the `N` re-solves: `build_s` is the time of `optimize!` outside
the inner solver, `solver_s` the inner solver's time, `total_s` their sum; `fingerprint_s` is one
comparison of fingerprints (canonical form, moments, settings), measured apart; `num_builds` counts
the inner problems built during the re-solves.

The previous version had no parameters, so there is no comparison against 0.3.

### Current

Recorded at 7dbebf0: inventory T = 64, 20 re-solves, HiGHS 1.15.1, Julia 1.12.6, idle machine, after a warm-up run
(seconds over the 20 re-solves; rows with `commit` 682dd11):

| Formulation | Variant | build | solver | total | fingerprint |
|---|---|---:|---:|---:|---:|
| `PrimalLDR` | cached | 0.044 | 0.036 | 0.080 | 3.1e-5 |
| `PrimalLDR` | rebuild | 0.499 | 3.088 | 3.588 | |
| `DualLDR` | cached | 0.045 | 0.056 | 0.101 | 3.0e-5 |
| `DualLDR` | rebuild | 0.521 | 14.454 | 14.975 | |

The cached re-solves spend 2 ms each outside the solver, and the cached total is 0.080 s against
3.6 s for the primal (45 times faster) and 0.101 s against 15.0 s for the dual (148 times). HiGHS
starts from its basis, which stays optimal after a change of the initial stock (0 simplex
iterations, against 4999 and 15790 from scratch), and nothing is built. The fingerprint costs 30 µs
per `optimize!`, under 2% of what a cached re-solve spends outside the solver (canonical form and
moments).

### History

| `src_commit` (`commit`) | Formulation | Variant | build | solver | total | fingerprint |
|---|---|---|---:|---:|---:|---:|
| f204dc1 (f204dc1) | `PrimalLDR` | cached | 0.079 | 0.051 | 0.130 | 3.0e-5 |
| f204dc1 (f204dc1) | `PrimalLDR` | rebuild | 0.380 | 3.419 | 3.799 | |
| f204dc1 (f204dc1) | `DualLDR` | cached | 0.046 | 0.063 | 0.109 | 3.2e-5 |
| f204dc1 (f204dc1) | `DualLDR` | rebuild | 0.506 | 14.651 | 15.156 | |
| 423d23c (354a8de) | `PrimalLDR` | cached | 0.072 | 3.473 | 3.545 | 3.0e-5 |
| 423d23c (354a8de) | `DualLDR` | cached | 0.112 | 14.701 | 14.813 | 3.2e-5 |
| 7dbebf0 (bf91f9c) | `PrimalLDR` | cached | 0.236 | 0.037 | 0.273 | 3.1e-5 |
| 7dbebf0 (bf91f9c) | `PrimalLDR` | rebuild | 0.289 | 3.124 | 3.413 | |
| 7dbebf0 (bf91f9c) | `DualLDR` | cached | 0.218 | 0.061 | 0.278 | 3.1e-5 |
| 7dbebf0 (bf91f9c) | `DualLDR` | rebuild | 0.405 | 14.144 | 14.549 | |

At 423d23c the inner problems held JuMP `Parameter`s, which MOI bridges for HiGHS without support
for changing their value, so JuMP copied the model to HiGHS again at every re-solve and the solver
time did not improve; f204dc1 fixes variables instead, and its re-solves are 29 times faster for
the primal and 139 times for the dual. The bf91f9c rows ran right after the benchmark environment
was instantiated: their cached `build` (0.22 to 0.24 s) includes compilation, not a per-re-solve
cost. The current rows rerun the same code after a warm-up run.

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
the solver time through JuMP for the hand-written LP, from `results/timings.csv`):

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

Inventory (`timing.jl`'s problem), `k` breakpoints per demand, solver time:

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
out).
