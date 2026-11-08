# Inventory benchmarks: one scalar uncertainty per stage

The T-period inventory problem with backlog: an order `x[t]` in [0, 150] at cost 1 in stage t,
then the demand `d[t] ~ U(60, 100)` in stage t + 1, with a holding cost of 0.5 and a backlog cost
of 3 per unit of the inventory `I[t]`, starting from `I[0] = 0`. Its stage dimension is 1, and its
support a box, so it is the problem where the closed forms of box supports apply in full. The LDR
model is `inventory` of `test/problems/tagged.jl`, which the tests share.

| Script | What it measures | What varies | Results |
|---|---|---|---|
| `timing.jl` | the time to write the model, to reformulate it and to solve it, and the sizes of the solved model, against the same robust LP written by hand | T; `PrimalLDR`, `DualLDR` and the hand-written LP, each with its default settings and HiGHS's default options | `results/timings.csv` |
| `resolve.jl` | the time of re-solves after a change of a JuMP parameter, the initial stock, with and without the cache of inner problems | `PrimalLDR` and `DualLDR`; the cache kept or emptied before each re-solve; T and the number of re-solves, 64 and 20 by default | `results/resolve.csv` |

The decisions that these measurements informed are in the
[table of decisions](../README.md#decisions-and-the-experiments-behind-them). How to run them, and
how their rows are tagged, is in [`../README.md`](../README.md#running):

```
julia --project=benchmark benchmark/inventory/timing.jl 4 8 16 32 64 128
julia --project=benchmark benchmark/inventory/resolve.jl          # T = 64, N = 20
```

## Build and solve times (`timing.jl`)

`timing.jl` times the T-period inventory problem with backlog for the primal LDR (`ldr_primal`),
the dual LDR (`ldr_dual`) and a hand-written robust LP (`hand`), and appends one row per T and
variant. The objective of the primal LDR and of the hand LP must be
90T + 20; the script warns otherwise. The dual value is a lower bound (for example 10673.33 against
11540 at T = 128).

Columns: `model_s` writes the JuMP model, `optimize_s` is `optimize!` (canonical form, moments,
reformulation, inner solve), `solver_s` is the inner solver's time. The last seven columns split
`optimize!` of the LDR variants into the phases of `time_breakdown`: `canonicalize_s` (reading the
model, with its lifting and rescaling), `moments_s`, `checks_s`, `build_s` (building the inner
problem), `solve_s`, `results_s` and `total_s`. They are empty for the hand-written LP and for the
rows recorded before them, whose `model_s` was named `build_s`. The sizes are those of the
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
0.60 s); the [past study of solve times](../README.md#solve-times-against-03-october-2026-tag-study-03-comparison)
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
variants differ. The columns, over the `N` re-solves: `overhead_s` is the time of `optimize!`
outside the inner solver, `solver_s` the inner solver's time, `optimize_s` their sum, and the phases
of `time_breakdown` summed over the re-solves (as in `timings.csv`); `fingerprint_s` is one
comparison of fingerprints (canonical form, moments, settings), measured apart; `num_builds` counts
the inner problems built during the re-solves. In the rows recorded before the phases,
`overhead_s` was named `build_s` and `optimize_s` was named `total_s`.

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

