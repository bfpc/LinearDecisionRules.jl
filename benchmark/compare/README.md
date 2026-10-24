# Solve times: 0.4 against the previous version

Question from the authors (DESIGN.md, phase 4): "a smaller (but equivalent) LP might prove harder
to solve". Is the 0.4 inner LP slower for HiGHS than the previous version's, and why?

Method: export the inner LPs of both packages to MPS, solve each file with HiGHS in a fresh process
(same options, best of 3, a single run above 60 s), record the size before and after presolve, the
algorithm and the iterations, then change the solver settings and post-process the models to test
explanations. Nothing in `src/` is touched; the variants are made from the exported files.

Instances: inventory T = 32, 64, 128 (`benchmark/timing.jl`) and hydro T = 4, 8 (`benchmark/hydro`;
primal at T = 12 as well), primal and dual. Legacy models: `primal` (default, explicit slacks),
`primal_sae` (`SlackAsExpr(true)`, what the previous version's hydro runs used) and its dense `dual`.
The objective values agree between the packages (all rows of `results/compare.csv`; for instance
11540 primal and 10673.33 dual at inventory T = 128, 468.156822 at hydro T = 12).

## Reproduce

```
# environments: this one with the package developed, and the legacy one (spike/legacy_env of the
# previous repository, with Distributions, HiGHS, JuMP and the old package developed)
julia --project=benchmark/compare -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
L=<previous repository>/spike/legacy_env

julia --project=benchmark/compare benchmark/compare/export_new.jl inventory 32 64 128
julia --project=benchmark/compare benchmark/compare/export_new.jl hydro 4 8 12
julia --project=$L benchmark/compare/export_legacy.jl inventory 32 64 128
julia --project=$L benchmark/compare/export_legacy.jl hydro 4 8          # `skip=dual` for T = 12

# one fresh process per run; appends to benchmark/results/compare.csv
julia --project=benchmark/compare benchmark/compare/run_solves.jl inventory_T128_new_primal inventory_T128_legacy_primal
julia --project=benchmark/compare benchmark/compare/run_solves.jl setting=ipm_nocross hydro_T8_new_dual
bash benchmark/compare/run_settings.sh inv128      # inv128 | inv64dual | hydro4 | hydro8

julia --project=benchmark/compare benchmark/compare/analyze_mps.jl hydro_T8_new_dual   # structure, results/compare_structure.csv
julia --project=benchmark/compare benchmark/compare/permute_test.jl hydro_T4_new_dual  # row/column order, results/compare_permute.csv
julia --project=benchmark/compare benchmark/compare/variants.jl nou hydro_T8_new_dual  # u substituted out
julia --project=benchmark/compare benchmark/compare/run_solves.jl hydro_T8_new_dual_nou
julia --project=benchmark/compare benchmark/compare/solve_jump.jl new inventory 128 primal   # through JuMP, as the packages do
julia benchmark/compare/table.jl [default|settings]
```

The MPS files (up to 0.5 GB) and the logs go to `mps/` and `logs/`, which are not committed.
`results/compare.csv` has one row per file and setting: sizes before and after presolve, algorithm,
iterations, `presolve_s` (see below), HiGHS run time, status and objective; `compare_structure.csv`,
`compare_permute.csv` and `compare_jump.csv` have the other tests. HiGHS is the one of HiGHS.jl 1.26.1
(HiGHS 1.15.1, whose `solver = "ipm"` is HiPO), Windows 11, Julia 1.12.6, 28 threads.

Caveats. The times are HiGHS's own `run time` (reading the file excluded). `presolve_s` is the run
time with `simplex_iteration_limit = 0`, that is, presolve plus the set-up of the simplex on the
presolved model; the HiGHS log only has a 0.1 s resolution. To fit the budget, up to four solves
ran at the same time on the machine (single-threaded simplex, so the effect is small, but the IPM
and the settings rows can be 30% off); the hydro T = 12 rows are single runs. The legacy dual of
hydro T = 12 (14M nonzeros, 230 s in the previous repository's recording) was not exported, and the
`nou` variant of inventory T = 128 (5.8M nonzeros) was not solved.

## Result: default HiGHS (dual simplex after presolve)

`Solve s` is `Total s - Presolve s`. All of these use the default options of HiGHS 1.15.1, which
chose the dual simplex everywhere.

| Instance | Formulation | Package | Rows x cols | After presolve | Iterations | Presolve s | Total s | Solve s |
|---|---|---|---|---|--:|--:|--:|--:|
| hydro T=4 | primal | new | 8,136 x 18,160 | 4,419 x 14,058 (36,532 nz) | 8,354 | 0.03 | 0.5 | 0.46 |
| hydro T=4 | primal | legacy | 32,004 x 49,696 | 7,665 x 17,304 (39,717 nz) | 11,427 | 0.07 | 0.9 | 0.83 |
| hydro T=4 | primal_sae | legacy | 16,820 x 34,512 | 4,434 x 14,072 (36,499 nz) | 6,807 | 0.05 | 0.33 | 0.28 |
| hydro T=4 | dual | new | 23,028 x 12,904 | 14,735 x 4,916 (81,346 nz) | 25,691 | 0.05 | 3.68 | 3.63 |
| hydro T=4 | dual_nou | new | 23,028 x 12,028 | 7,800 x 4,104 (82,090 nz) | 9,640 | 0.08 | 1.24 | 1.16 |
| hydro T=4 | dual | legacy | 43,684 x 19,328 | 7,307 x 4,104 (76,641 nz) | 9,033 | 0.11 | 0.85 | 0.74 |
| hydro T=8 | primal | new | 35,828 x 83,168 | 18,587 x 64,242 (166,910 nz) | 40,256 | 0.18 | 12.37 | 12.19 |
| hydro T=8 | primal | legacy | 139,912 x 220,992 | 33,693 x 79,348 (182,082 nz) | 56,186 | 0.31 | 20.97 | 20.66 |
| hydro T=8 | primal_sae | legacy | 72,168 x 153,248 | 18,622 x 64,276 (166,939 nz) | 40,016 | 0.27 | 13.48 | 13.22 |
| hydro T=8 | dual | new | 103,280 x 54,552 | 66,743 x 19,620 (374,006 nz) | 181,835 | 0.22 | 145.2 | 144.98 |
| hydro T=8 | dual_nou | new | 103,280 x 52,508 | 35,268 x 17,720 (734,158 nz) | 48,837 | 0.61 | 39.19 | 38.59 |
| hydro T=8 | dual | legacy | 200,648 x 85,504 | 33,455 x 17,720 (695,021 nz) | 45,563 | 1.58 | 22.42 | 20.84 |
| hydro T=12 | primal | new | 82,784 x 195,024 | 42,227 x 150,522 (390,872 nz) | 108,421 | 0.75 | 161.05 | 160.3 |
| hydro T=12 | primal_sae | legacy | 166,044 x 356,208 | 42,282 x 150,576 (390,933 nz) | 111,936 | 1.17 | 168.0 | 166.84 |
| inventory T=32 | primal | new | 2,734 x 5,744 | 1,183 x 4,193 (12,492 nz) | 1,364 | 0.01 | 0.02 | 0.01 |
| inventory T=32 | primal | legacy | 9,632 x 14,320 | 1,500 x 4,510 (13,006 nz) | 805 | 0.02 | 0.03 | 0.01 |
| inventory T=32 | primal_sae | legacy | 5,408 x 10,096 | 1,236 x 4,246 (12,411 nz) | 1,634 | 0.02 | 0.04 | 0.02 |
| inventory T=32 | dual | new | 6,956 x 3,948 | 4,331 x 1,323 (13,719 nz) | 3,675 | 0.02 | 0.09 | 0.07 |
| inventory T=32 | dual | legacy | 13,472 x 5,872 | 3,720 x 1,611 (74,720 nz) | 5,383 | 0.07 | 0.58 | 0.51 |
| inventory T=64 | primal | new | 10,590 x 22,752 | 4,506 x 16,668 (49,962 nz) | 4,543 | 0.05 | 0.12 | 0.07 |
| inventory T=64 | primal | legacy | 37,696 x 56,288 | 4,886 x 17,048 (55,247 nz) | 1,353 | 0.08 | 0.12 | 0.04 |
| inventory T=64 | primal_sae | legacy | 21,056 x 39,648 | 4,480 x 16,642 (56,183 nz) | 1,502 | 0.06 | 0.1 | 0.04 |
| inventory T=64 | dual | new | 27,228 x 15,068 | 17,246 x 5,086 (56,835 nz) | 16,300 | 0.05 | 0.72 | 0.67 |
| inventory T=64 | dual_nou | new | 27,228 x 14,814 | 14,616 x 6,297 (559,444 nz) | 23,678 | 0.38 | 15.2 | 14.82 |
| inventory T=64 | dual | legacy | 53,568 x 23,008 | 14,625 x 6,308 (560,669 nz) | 26,939 | 0.64 | 12.44 | 11.8 |
| inventory T=128 | primal | new | 41,662 x 90,560 | 17,356 x 66,254 (200,881 nz) | 18,369 | 0.22 | 0.63 | 0.41 |
| inventory T=128 | primal | legacy | 149,120 x 223,168 | 20,409 x 69,307 (212,917 nz) | 7,887 | 0.37 | 0.61 | 0.24 |
| inventory T=128 | primal_sae | legacy | 83,072 x 157,120 | 17,674 x 66,572 (215,188 nz) | 15,829 | 0.31 | 0.72 | 0.41 |
| inventory T=128 | dual | new | 107,708 x 58,812 | 69,156 x 20,264 (223,599 nz) | 65,715 | 0.2 | 6.86 | 6.67 |
| inventory T=128 | dual | legacy | 213,632 x 91,072 | 57,948 x 24,927 (4,335,064 nz) | 384,307 | 7.19 | 1048.35 | 1041.16 |

`dual_nou` is the 0.4 dual with `u` substituted out (hypothesis D1 below), not a package output.

## Answer

**Primal: no.** After presolve the three primal models are the same LP. At hydro T = 12 the 0.4
model (82,784 x 195,024) and the legacy `SlackAsExpr` model (166,044 x 356,208) both end at about
42.2k x 150.5k with 390.9k nonzeros, take 108k and 112k simplex iterations, and 161 s and 168 s; at
T = 8 the presolved sizes agree to 0.2% (18,587 x 64,242 against 18,622 x 64,276) and the times are
12.4 s and 13.5 s (the legacy default, with explicit slacks, takes 21.0 s and has 34k x 79k after
presolve). At inventory T = 128 the times are 0.63 s (0.4), 0.61 s and 0.72 s. The iteration
counts vary by +-30% with the order of rows and columns alone (`compare_permute.csv`: 13.0k to
18.4k iterations for the same 0.4 model at inventory T = 128, 11.1k to 11.4k for the legacy default
one), so differences of that size between two exports of "the same" LP are noise. The 130 s against
103 s of hydro T = 12 came from different runs; in the same conditions they are equal. Presolve
gives the smaller LP to HiGHS whatever the formulation, so for the primal the smaller formulation
only saves the time to build and to read the model.

The observation "1.11 s against 0.60 s" at inventory T = 128 is also partly a measurement issue:
with HiGHS.jl, `solve_time` is HiGHS's run clock, which counts the copy of the model into HiGHS.
Through JuMP (`compare_jump.csv`, models in the order of construction) `solve_time` is 0.96 s (0.4)
and 2.7 s (legacy default, 149k rows to copy), but a second `Highs_run` after `Highs_clearSolver`
on the same objects takes 0.52 s and 0.53 s. I cannot reproduce the legacy 0.60 s with
`solve_time` in this environment (it may be an older HiGHS clock); the pure solve times are equal.

**Dual: it depends on the instance, and the size is not the reason.**

| Dual LDR, default HiGHS | 0.4 | legacy (dense cone) | 0.4 / legacy |
|---|--:|--:|--:|
| inventory T = 64 | 0.72 s | 12.4 s | 0.06 |
| inventory T = 128 | 6.9 s (65.7k iter.) | 1,048 s (384k iter.) | 0.007 |
| hydro T = 4 | 3.68 s (25.7k iter.) | 0.85 s (9.0k iter.) | 4.3 |
| hydro T = 8 | 145 s (182k iter.) | 22.4 s (45.6k iter.) | 6.5 |

On the inventory the 0.4 dual is far faster (4.3M nonzeros after presolve for the legacy one at
T = 128, 224k for the 0.4 one). On the hydro instances the 0.4 dual, which has 8 times fewer
nonzeros before presolve (454k against 3.9M at T = 8), is slower. Here presolve works on the dense
legacy dual and not on the sparse one: at T = 8 the legacy model goes from 200,648 to 33,455 rows
(29,670 of them range rows, merged from pairs) while the 0.4 model goes from 103,280 to 66,743 rows
(no range row). The dual simplex works on a basis of the size of the number of rows, and needs 4
times more iterations. It is not the conditioning: the coefficient range of the 0.4 dual is smaller
(ratio 511 against 2,439 at hydro T = 8, 267 against 3,467 at inventory T = 64), nor the order of
rows (the permuted 0.4 dual at hydro T = 4 needs 22.4k to 24.1k iterations, the legacy one 9.3k to
9.6k).

## Solver settings (`results/compare.csv`, total HiGHS seconds)

| Instance | pkg | default | primal simplex | IPM + crossover | IPM, no crossover | no presolve |
|---|---|--:|--:|--:|--:|--:|
| inventory T=64 dual | 0.4 | 0.72 | **0.18** | 1.54 | 1.22 | 1.82 |
| | legacy | 12.4 | **1.46** | 7.9 | 7.1 | 47.6 |
| hydro T=4 dual | 0.4 | 3.68 | 3.04 | 1.34 | **0.64** | 8.29 |
| | legacy | 0.85 | 1.66 | 1.64 | 1.55 | 19.3 |
| hydro T=8 dual | 0.4 | 145 | 168 | 207 (IPM "no progress", simplex fallback) | **7.99** | |
| | legacy | 22.4 | 94.6 | 25.5 | 21.0 | |
| hydro T=8 dual, `nou` | 0.4 | 39.2 | 118 | 17.3 | | |
| inventory T=128 primal | 0.4 | 0.63 | 4.29 | 1.92 | 1.89 | 1.39 |
| | legacy | 0.61 | 9.8 | 2.55 | 2.28 | 6.79 |
| | legacy SAE | 0.72 | 8.55 | 2.35 | 2.16 | 4.01 |
| hydro T=4 primal | 0.4 | 0.50 | 0.32 | 1.10 (fell back to simplex) | 0.58 (objective 44.7, wrong) | 0.78 |
| | legacy SAE | 0.33 | 0.40 | 1.34 | 1.02 (3.7401, inexact) | 1.72 |
| hydro T=8 primal | 0.4 | 12.4 | | HiGHS presolve error (status 4) | 7.57 (94.8718, inexact) | |
| | legacy SAE | 13.5 | | 6.18 | 4.61 | |

Without presolve the smaller 0.4 models win in every instance where it was run (1.4 s against 6.8 s at inventory T = 128,
0.78 s against 2.1 s at hydro T = 4 primal): the "smaller is harder" effect is not there, and
presolve is what equalizes the primal. Status 15 (unknown) of the IPM without crossover is HiGHS's
status for a solution without a basis; on the 0.4 dual rows its objective agrees with the simplex one to 2e-10 (2e-6 at worst, legacy dual at inventory T = 64). The IPM
rows of the primal are the exception to the parity, see hypothesis P1.

## Hypotheses for the formulation code, ranked by evidence

**D1. The auxiliary `u = S mu` keeps the cone rows sparse but costs presolve half of its work
(DualLDR; strong evidence, mixed effect).** `u[r]` is a variable with the equality
`u = sum_j mu_j S[r,j]`, and every cone row carries `Wmu_k u`; the legacy dense cone has no `u`.
Test (`variants.jl nou`: the same LP with the column `u` eliminated, each cone row getting the
coefficients `Wmu_k mu_j` on `S[r,j]` and the equality becoming `mu . S >= 0`): the presolved model
becomes that of the legacy dual (hydro T = 4: 7,800 x 4,104 against 7,307 x 4,104; inventory
T = 64: 14,616 x 6,297 with 559k nonzeros against 14,625 x 6,308 with 561k), with the legacy's
iteration counts (9.6k against 9.0k at hydro T = 4; 48.8k against 45.6k at T = 8). The effect is a
3.7x speedup at hydro T = 8 (145 s to 39 s; 17 s with IPM) and 3x at T = 4 (3.68 s to 1.24 s), and
a 20x slowdown at inventory T = 64 (0.72 s to 15.2 s), where the denser rows cost more than the
rows saved. So the cone formulation needs a switch or a density rule (substitute `u` when a row
has few `S` columns), to be tested on more instances before it is coded: neither "sparse" nor
"dense" is better in general.

**D2. For the DualLDR, ask HiGHS for what the bound needs: IPM without crossover (DualLDR; strong
evidence, solver level).** The dual value is the objective of the inner LP and its rule is not a
policy (DESIGN: the bound needs no primal point), so a basis is not needed. With
`solver = "ipm", run_crossover = "off"` the 0.4 dual solves hydro T = 8 in 8.0 s against 145 s
(18x) and hydro T = 4 in 0.64 s against 3.68 s, but inventory T = 64 in 1.2 s against 0.72 s;
primal simplex is 4x faster at inventory T = 64 (0.18 s) but not at hydro T = 8 (168 s). The IPM
with crossover failed on the 0.4 dual at hydro T = 8 (HiPO "no progress", then simplex clean-up,
207 s) while solving it in 8 s without crossover, so the crossover is the part to drop. Testable in
`solve_formulation`: pass these raw attributes for `DualLDR` when the inner optimizer is HiGHS and
compare the bound (relative difference 2e-10 on the 0.4 rows here).

**P1. The primal has no size problem, but the 0.4 primal is more fragile for the IPM (PrimalLDR;
weak evidence).** With `solver = "ipm"` the 0.4 hydro primal ends with a HiGHS presolve error
(T = 8) or with a wrong objective without crossover (44.7 against 3.76 at T = 4), where the legacy
`SlackAsExpr` model only loses accuracy (3.7401). The logs have "Dependent equations search" lines
for both. A guess to test: the equalities `W' lambda = s` and the rule equalities are linearly
dependent where the support has repeated structural columns, and the slack-free form has fewer
rows to hide it; check by counting the rows removed by the dependent-equation search, and whether
explicit slacks (the legacy default, 2 to 3x more variables) make the IPM robust.

**Rejected.** "Explicit slacks give presolve more room": for the primal the legacy default (with
explicit slacks) is not better than the 0.4 form (hydro T = 8: 21.0 s against 12.4 s), and for the
dual HiGHS already eliminates the `S` columns (free columns: 52,360 before presolve, 17,538 after,
about the number of rule coefficients). "Worse conditioning": the coefficient ratios of the 0.4
dual are smaller than the legacy's. "Row order": permutations change the iterations by 10% to 40%
but not by the factors seen on the dual.

## Files

`inventory.jl` (the instance, for both packages), `export_new.jl`, `export_legacy.jl`,
`solve_mps.jl`, `run_solves.jl`, `run_settings.sh`, `solve_jump.jl`, `analyze_mps.jl` and
`lpdata.jl` (structure before and after presolve), `permute_test.jl`, `variants.jl`, `table.jl`;
results in `../results/compare*.csv`.
