# dualnnz: size of the structured primal / dual LDR on the hydrothermal instance

Instance: `examples/recursion/monolithic_vs_sddp.jl` (4 reservoirs, one declaration of `4(k+1)` lifted inflows per
stage 2..T, `SlackAsExpr`). Everything is computed from the canonical data of the old package
(`ABC`, `M`, stage vectors). Run: `julia --project=spike/dualnnz spike/dualnnz/measure.jl {validate|check|sizes|all}`
(about 110 s, 30 s, 17 s; `spike/dualnnz/Project.toml` is the `spike` environment minus Aqua plus the old
package under `Pkg.develop` and Expectations). Raw numbers: `sizes.csv`.

## Validation (HiGHS, T = 2, 3, 4; k = 0, 1, 2; 12 cases)

Block primal vs current primal and structured dual vs current dual (`src/solve_primal.jl`, `src/solve_dual.jl`):
objectives agree to a maximum relative difference of 1.2e-12 (primal: <= 2e-15). The partition "one block per
stage" and the partition "one block for all uncertainties" give the same objectives. Example, T = 3, k = 1:
primal 1.0779030511, dual 0.7412856275 with both implementations.

Size formulas (pattern arithmetic) equal the real JuMP models exactly (variables, linear constraints, variable
bounds, nonzeros) for current primal, current dual, block primal, structured dual, at T = 2 k = 0, T = 3 k = 1
and T = 12 k = 0. `ldr_sizes.csv` is reproduced (T = 12 k = 0: 356 208 / 166 044 primal, 198 528 dual variables,
13 875 840 cone nonzeros). JuMP stores only nonzero coefficients, so `nnz` below is structural nonzeros.

## Sizes (variables / linear constraints / nonzeros; Λ >= 0 and u >= 0 bounds not counted as constraints)

`stage`: blocks = declarations (one per stage here). `dense`: worst case, one block that couples all stages with a
dense covariance (here Σ is really block diagonal, so the formula counts entries JuMP would drop as zeros; the check
accounts for them). "x" is nnz(current) / nnz(new).

| T | k | blocks | dim ξ | current primal | block primal | current dual | structured dual | dual nnz x |
|---|---|--------|-------|----------------|--------------|--------------|-----------------|-----------|
| 12 | 0 | stage | 45 | 356k / 166k / 758k | 202k / 89k / 450k | 199k / 471k / 14.2M | 125k / 243k / 1.06M | 13.3 |
| 12 | 1 | stage | 89 | 858k / 325k / 1.81M | 472k / 171k / 1.03M | 392k / 1.09M / 69.2M | 241k / 556k / 4.04M | 17.1 |
| 12 | 2 | stage | 133 | 1.36M / 484k / 2.70M | 743k / 253k / 1.54M | 585k / 1.71M / 165M | 357k / 869k / 8.86M | 18.6 |
| 12 | 3 | stage | 177 | 1.86M / 643k / 3.59M | 1.01M / 335k / 2.05M | 778k / 2.34M / 301M | 473k / 1.18M / 15.5M | 19.4 |
| 24 | 0 | stage | 93 | 1.47M / 679k / 3.13M | 826k / 357k / 1.84M | 819k / 1.96M / 121M | 503k / 1.00M / 4.41M | 27.5 |
| 24 | 1 | stage | 185 | 3.57M / 1.34M / 7.51M | 1.96M / 699k / 4.28M | 1.63M / 4.56M / 599M | 989k / 2.31M / 16.8M | 35.5 |
| 24 | 2 | stage | 277 | 5.67M / 2.01M / 11.2M | 3.09M / 1.04M / 6.41M | 2.43M / 7.16M / 1.43B | 1.47M / 3.62M / 37.0M | 38.7 |
| 24 | 3 | stage | 369 | 7.76M / 2.67M / 15.0M | 4.22M / 1.38M / 8.53M | 3.24M / 9.76M / 2.62B | 1.96M / 4.93M / 64.9M | 40.4 |
| 12 | 3 | dense | 177 | 1.86M / 643k / 3.59M | 1.72M / 591k / 3.33M | 778k / 2.34M / 301M | 473k / 1.89M / 152M | 2.0 |
| 24 | 3 | dense | 369 | 7.76M / 2.67M / 15.0M | 7.47M / 2.57M / 14.4M | 3.24M / 9.76M / 2.62B | 1.96M / 8.18M / 1.32B | 2.0 |

(All 16 rows, with the cone-only counts, are in `sizes.csv`.) The cone nonzeros alone, i.e. the figure of
`ldr_sizes.csv`, go from 13.9M to 0.77M (T = 12, k = 0), 300M to 14.4M (T = 12, k = 3) and 2.62B to 60.3M
(T = 24, k = 3).

## Findings

With one covariance block per stage, the dual shrinks 13x (T = 12, k = 0) to 19x (k = 3) and 27x to 40x at T = 24
in nonzeros, and its growth with k and T goes from quadratic in the number of uncertainties per row to roughly
linear in the number of blocks the row can see: a stage-t row touches only the blocks of stages <= t, so the
per-row cost is the sum over seen blocks of (rows of W' in the block) x (columns of the block), while all the
blocks it does not see collapse into one `u_r >= 0`. S masking alone (dense cone, S columns restricted to the
history) gives only about 2x; the remaining 7x to 20x comes from the collapse and the block-local entries, and it
depends on the covariance being block diagonal over stages. The primal gains less: Λ variables drop 1.8x
(Λ only for the touched blocks) but the constraint matrix is already sparse, so nonzeros drop 1.7-1.8x. If one
declaration couples all stages with a dense covariance (`dense` rows), every row of stages >= 2 touches the single
block, nothing collapses, the primal gain disappears (1.0-1.1x) and the dual keeps only the S masking gain (2.0x,
1.3B nonzeros at T = 24, k = 3), at the price of R extra variables and equalities for u.

## Caveats

- Sizes only; solve times of the smaller models (apart from the T <= 4 validation) were not measured.
- Blocks are the declared stages, and the check `Σ block diagonal` is asserted; W' rows are asserted to lie in one block.
- λ₁, λ₂ (the rows ±e₁ of W) are kept for every row in the block primal, as in the current one (they could be merged
  into one free multiplier). Λ for untouched blocks is dropped; this is exact because Ξ is nonempty (Λ_b W_b = 0, Λ_b >= 0
  implies Λ_b h_b <= 0).
- `u_r >= 0` is added only when a block with (W'μ̄)_k > 0 is untouched by the row; if every block is touched it is implied by the
  bound rows (their sum), so it is not needed. Interval rows of `ABC` are not handled (the instance has none).
- The objective (one row, nonzeros of X) is the same for all models and excluded; P = 0 is assumed by the spike builders.
