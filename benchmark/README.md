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

## T = 128, against the previous version

Legacy: `results/timings.csv` of the old repository (commit 17ce1eb, Julia 1.12.6;
`ldr_primal` is its default primal model). Times in seconds, best of 3.

| Model | Variables | Constraints | Nonzeros | build | optimize | solver |
|---|---:|---:|---:|---:|---:|---:|
| legacy primal | 223,168 | 149,120 | n/a | 0.0004 | 2.013 | 0.602 |
| legacy primal, `SlackAsExpr` | 157,120 | 83,072 | n/a | 0.0004 | 1.688 | 0.617 |
| hand-written LP | 57,792 | 74,432 | 271,807 | 0.106 | 2.032 | 1.905 |
| 0.4 `PrimalLDR` | 90,560 | 41,662 | 206,013 | 0.0003 | 1.154 | 1.108 |
| 0.4 `DualLDR` | 58,812 | 107,708 | 273,079 | 0.0003 | 7.449 | 7.359 |

In the legacy recording the hand LP had 57,792 variables and took 0.115 s to build and 1.686 s to
optimize; the same script gave 0.106 s and 2.032 s in this run, so differences of that size between
runs are noise. The 0.4 primal model has 41% of the legacy variables (58% of the legacy
`SlackAsExpr` model) and 28% of its constraints, and `optimize!` takes 1.15 s against 2.01 s, of
which 0.05 s is not the inner solve (1.41 s in the legacy recording). HiGHS needs more time on the
smaller 0.4 primal model (1.11 s against 0.60 s). The dual has about as many nonzeros as the hand LP
and is the slowest variant, with a solve time that grows faster with T than the primal's
(0.83 s at T = 64, 7.36 s at T = 128).
