# Benchmarks

[`PROBLEMS.md`](PROBLEMS.md) is the catalogue of the LDR test problems found in this package, the legacy
repository, `CasosLDR` and the papers (data, known values, solver notes, candidates to port).

`timing.jl` times the T-period inventory problem (see the header of the script) for the primal LDR,
the dual LDR and a hand-written robust LP, and appends one row per T and variant to
`results/timings.csv`, tagged with the commit, a flag for uncommitted changes in `src/`, and the
Julia version. The objective of the primal LDR and of the hand LP must be 90T + 20; the script warns
otherwise. The dual value is a lower bound (for example 10673.33 against 11540 at T = 128).

```
julia --project=benchmark -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
julia --project=benchmark benchmark/timing.jl 4 8 16 32 64 128
```

Columns: `build_s` writes the JuMP model, `optimize_s` is `optimize!` (canonical form, moments,
reformulation, inner solve), `solver_s` is the inner solver's time. The sizes are those of the
reformulated model that is solved (constraints do not count variable bounds; nonzeros count the
coefficients of the constraint matrix).

## T = 128, against the previous version (0.4 rows measured at main #119 46c6473)

Legacy: `results/timings.csv` of the old repository (commit 17ce1eb, Julia 1.12.6;
`ldr_primal` is its default primal model). Times in seconds, best of 3.

| Model | Variables | Constraints | Nonzeros | build | optimize | solver |
|---|---:|---:|---:|---:|---:|---:|
| legacy primal | 223,168 | 149,120 | n/a | 0.0004 | 2.013 | 0.602 |
| legacy primal, `SlackAsExpr` | 157,120 | 83,072 | n/a | 0.0004 | 1.688 | 0.617 |
| hand-written LP | 57,792 | 74,432 | 271,807 | 0.085 | 2.077 | 1.952 |
| 0.4 `PrimalLDR` | 90,560 | 41,662 | 206,013 | 0.0004 | 1.137 | 1.095 |
| 0.4 `DualLDR` | 58,812 | 107,708 | 273,079 | 0.0003 | 7.346 | 7.295 |

In the legacy recording the hand LP had 57,792 variables and took 0.115 s to build and 1.686 s to
optimize; the same script gave 0.085 s and 2.077 s in the 46c6473 run (0.106 s and 2.032 s at eb80cdf), so differences of that size between
runs are noise. The 0.4 primal model has 41% of the legacy variables (58% of the legacy
`SlackAsExpr` model) and 28% of its constraints, and `optimize!` takes 1.14 s against 2.01 s, of
which 0.04 s is not the inner solve (1.41 s in the legacy recording). HiGHS needs more time on the
smaller 0.4 primal model (1.10 s against 0.60 s). The dual has about as many nonzeros as the hand LP
and is the slowest variant, with a solve time that grows faster with T than the primal's
(0.83 s at T = 64, 7.30 s at T = 128).

## Refresh at main #144 (2509dbb), after the box closed form

`timing.jl 4 8 16 32 64 128` (rows tagged 2509dbb), same protocol, Julia 1.12.6, idle machine. `PrimalLDR` now
eliminates box support blocks in closed form (`box_closed_form = true`): the primal has the hand LP's
number of variables (57,792 at T = 128, against 90,560 at 46c6473), 41,662 constraints (unchanged) and
189,757 nonzeros (206,013). The objectives are identical in all 12 LDR rows (90T + 20 for the primal);
the dual's sizes, nonzeros and bounds are identical. Times at T = 128 in seconds, 46c6473 then #144:

| Model | Variables | Nonzeros | build | optimize | solver | solver ratio |
|---|---|---|---|---|---|---:|
| hand | 57,792 | 271,807 | 0.085, 0.047 | 2.077, 2.105 | 1.952, 1.977 | 1.01 |
| `PrimalLDR` | 90,560, 57,792 | 206,013, 189,757 | 0.0004, 0.0004 | 1.137, 1.313 | 1.095, 1.274 | 1.16 |
| `DualLDR` | 58,812 | 273,079 | 0.0003, 0.0003 | 7.346, 7.442 | 7.295, 7.391 | 1.01 |

Primal variables at T = 4, 8, 16, 32, 64: 70, 252, 952, 3,696, 14,560 (were 102, 380, 1,464, 5,744,
22,752), all equal to the hand LP's. Primal solver times, 46c6473 then #144, in ms: T = 4 0.76, 0.70;
T = 8 1.75, 1.70; T = 16 6.9, 5.3; T = 32 27.8, 26.9; T = 64 143, 146; T = 128 1095, 1274. So the smaller primal
is not faster: it is equal or slightly faster up to T = 64 and 16% slower at T = 128 (the hand LP, with the
same variable count, takes 1.98 s), although it also has 8% fewer nonzeros. The saving is in memory and
in the reformulation, not in the HiGHS time (the run-to-run spread of the other rows is 1 to 2%).
The build time of the LDR models is below 0.5 ms.

## Refresh at main #119 (46c6473) against the first recording (eb80cdf)

`timing.jl 4 8 16 32 64 128`, same protocol (best of 3), Julia 1.12.6. The quadratic-objective, dual-bound
and attribute changes since eb80cdf do not touch these LP models: variables, constraints, nonzeros and
objectives are identical in all 18 rows. Optimize times, eb80cdf then 46c6473, in seconds (ratio
new/old; differences below about 5% are noise, and the sub-10 ms rows are dominated by timer resolution):

| T | hand | primal | dual |
|--:|---|---|---|
| 4 | 0.00082, 0.00084 | 0.0018, 0.0019 | 0.0020, 0.0021 |
| 8 | 0.0020, 0.0019 | 0.0028, 0.0027 | 0.0041, 0.0040 |
| 16 | 0.0079, 0.0078 | 0.0094, 0.0085 | 0.0146, 0.0149 |
| 32 | 0.0354, 0.0354 | 0.0331, 0.0317 | 0.0911, 0.0900 |
| 64 | 0.218, 0.212 (0.97) | 0.161, 0.154 (0.96) | 0.838, 0.847 (1.01) |
| 128 | 2.032, 2.077 (1.02) | 1.154, 1.137 (0.99) | 7.449, 7.346 (0.99) |

The build time of the LDR models is below 0.5 ms at T = 128 in both recordings.

## Re-solves with a parameter (`resolve.jl`)

`resolve.jl` solves the inventory with its initial stock as a JuMP parameter, then re-solves it
`N` times with new values, keeping the inner problem (`cached`, the default) or emptying the cache
before each `optimize!` (`rebuild`, what every re-solve did before). It appends one row per
formulation and variant to `results/resolve.csv` (commit, dirty flag, Julia version, as above),
and warns if the objectives of the two variants differ.

```
julia --project=benchmark benchmark/resolve.jl          # T = 64, N = 20
```

Inventory T = 64, 20 re-solves, HiGHS 1.15, Julia 1.12.6, one run on an idle machine
(seconds over the 20 re-solves; `fingerprint` is one comparison):

| Commit | Formulation | Variant | build | solver | total | fingerprint |
|---|---|---|---:|---:|---:|---:|
| f204dc1 | `PrimalLDR` | cached | 0.079 | 0.051 | 0.130 | 3.0e-5 |
| f204dc1 | `PrimalLDR` | rebuild | 0.380 | 3.419 | 3.799 | |
| f204dc1 | `DualLDR` | cached | 0.046 | 0.063 | 0.109 | 3.2e-5 |
| f204dc1 | `DualLDR` | rebuild | 0.506 | 14.651 | 15.156 | |
| 354a8de | `PrimalLDR` | cached | 0.072 | 3.473 | 3.545 | 3.0e-5 |
| 354a8de | `DualLDR` | cached | 0.112 | 14.701 | 14.813 | 3.2e-5 |
| bf91f9c (main #144) | `PrimalLDR` | cached | 0.236 | 0.037 | 0.273 | 3.1e-5 |
| bf91f9c (main #144) | `PrimalLDR` | rebuild | 0.289 | 3.124 | 3.413 | |
| bf91f9c (main #144) | `DualLDR` | cached | 0.218 | 0.061 | 0.278 | 3.1e-5 |
| bf91f9c (main #144) | `DualLDR` | rebuild | 0.405 | 14.144 | 14.549 | |
| main #147 (src of #144) | `PrimalLDR` | cached | 0.044 | 0.036 | 0.080 | 3.1e-5 |
| main #147 (src of #144) | `PrimalLDR` | rebuild | 0.499 | 3.088 | 3.588 | |
| main #147 (src of #144) | `DualLDR` | cached | 0.045 | 0.056 | 0.101 | 3.0e-5 |
| main #147 (src of #144) | `DualLDR` | rebuild | 0.521 | 14.454 | 14.975 | |

The bf91f9c rows (`src/` of main #144 2509dbb) ran right after the benchmark environment was
instantiated: their cached `build` (0.22 to 0.24 s) includes compilation, not a per-re-solve cost.
The main #147 rows rerun the same `src/` on an idle machine, after a warm-up run: the cached re-solves
spend 2 ms each outside the solver, and the cached total is 0.080 s against 3.6 s for the primal
(45 times faster) and 0.101 s against 15.0 s for the dual (148 times).

The f204dc1 re-solves are 29 times faster for the primal and 139 times for the dual: HiGHS starts from its
basis, which stays optimal after a change of the initial stock (0 simplex iterations, against 4999
and 15790 from scratch), and nothing is built. At 354a8de the inner problems held JuMP
`Parameter`s, which MOI bridges for HiGHS without support for changing their value, so JuMP copied
the model to HiGHS again at every re-solve and the solver time did not improve; f204dc1 fixes
variables instead. The fingerprint costs 30 µs per `optimize!`, under 1% of the 4 ms that a cached
re-solve spends outside the solver (canonical form and moments).
