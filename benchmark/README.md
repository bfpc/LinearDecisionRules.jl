# Benchmarks

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
