# Catalogue of LDR test problems

Written 2026-10-26 at main `c592575` (0.4.0-DEV). A catalogue, not a benchmark: it says which problems exist
in which place, how they differ, which solvers were tried on them, and which ones are worth bringing into
`test/` or `benchmark/`. Nothing here is asserted by the test suite. Extended the same day with a
classification of the families (section 1.1), the SDDP.jl examples (section 9) and LDR test problems from
the literature (section 10), and a proposed benchmark set (section 11.1).

Abbreviations for the sources:

| Tag | Source |
|---|---|
| PT | this package, `test/` (`test/problems/`, `test/reference/legacy_values.toml`, `test/jump/`) |
| PB | this package, `benchmark/` (`timing.jl`, `resolve.jl`, `hydro/`; `compare/` is the past study of solve times, kept at the tag `study-0.3-comparison`) |
| TUT | this package, `docs/src/tutorials/*.jl` and `docs/src/getting_started.md` |
| LEG | the legacy repository `Packages/LinearDecisionRules.jl` (0.3.1, branch `Multistage`, 17ce1eb): `examples/`, `test/` |
| CL | `Projetos/ONS/CasosLDR` (49 commits, 2026-07-14 to 2026-10-17; branches `solvertest`, `moresolvers`, `benders_scenario`) |
| KWG | Kuhn, Wiesemann, Georghiou, Math. Program. 130 (2011) 177-209, section 5 |
| GWK | Georghiou, Wiesemann, Kuhn, Math. Program. 152 (2015) 301-338, sections 4, 5 and 7 |

Stage convention used below: stage 1 is "here and now" (no uncertainty observed); a decision of stage `t`
observes the uncertainties of stages `<= t`. The package inventory has `x_t` in `Stage(t)` and the demand
`d_t` in `Stage(t + 1)`; KWG's inventory has the production `x_it` observe `xi_t` (same stage), so a
reproduction of KWG puts `d_t` in `Stage(t)` (section 6).

Everything marked "(run)" was recomputed for this catalogue in a scratch environment outside both
repositories (0.4.0-DEV from this worktree, JuMP 1.32.1, HiGHS.jl 1.26.1, Ipopt.jl 1.16.0 with Ipopt 3.14.19,
Clarabel.jl 0.11.1, Julia 1.12.6, Windows 11); the scripts are in section 8. CL's manifest has HiGHS.jl
1.26.0, Ipopt.jl 1.16.0, JuMP 1.32.1, SDDP.jl 1.15.0 and the legacy package as a path dependency. Neither
Gurobi nor MOSEK appears in any of the three code bases.

## 1. Problem families

"Known values" are `primal` (value of the primal LDR, an upper bound for a minimization) / `dual` (bound of
the dual LDR); "bp k" is `BreakPoints() = k` (k equispaced interior breakpoints) unless a list is given.

| # | Family | Sources | Stages | Uncertainty per stage | Objective | Known values | In this package today |
|---|---|---|---:|---|---|---|---|
| F1 | Newsvendor, one `Uniform(80, 120)` demand, buy 10, return 8, sell 15 | PT `newsvendor`, `newsvendor_breakpoints_*`; TUT getting_started, piecewise_linear, sampled, advanced_distributions; LEG `test_newsvendor`; CL `Newsvendor/` | 2 | 1 (stage 2) | linear (max) | bp 0: 460 / 486.67 (1460/3); true optimum 3300/7 = 471.43 at the critical breakpoint 760/7 (derived: critical ratio 5/7); more in table 2 | tests + tutorials (ported); CL's price set sell = 20 not ported |
| F2 | Newsvendor extensions: random price, two products, three stages, discrete and multivariate discrete laws, integer `buy` | PT `newsvendor_random_price`, `_two_products`, `_three_stages`, `_mv_discrete`, `_discrete_demand`, `newsvendor(integer = true)`; LEG test_core, test_stages; TUT distributions | 2 (3 for three-stage) | 1 to 3 | linear; bilinear (price x sell) for random price | values in `legacy_values.toml` (table 2) | tests (ported); none in CL |
| F3 | T-period inventory with backlog: `x_t in [0, 150]` at cost 1, `d_t ~ U(60, 100)`, `y_t >= max(0.5 I_t, -3 I_t)`, `I_0 = 0` | PT `inventory(T)`; PB `timing.jl`, `resolve.jl` (initial stock as a `Parameter`); TUT multistage, getting_started (re-solve); LEG `benchmark/timing.jl` | T + 1 | 1 per stage 2..T+1 | linear (min) | primal 90T + 20 for all T; dual 340 (T = 4), 673.33 (T = 8), 10673.33 (T = 128); tutorial T = 2, 6: 200 / 173.33, 560 / 506.67 | tests + benchmark + tutorial (ported) |
| F4 | KWG inventory: 3 factories, one warehouse, seasonal demand and costs (stochastic and robust versions) | KWG section 5 (Ben-Tal et al. 2004 data) | T = 1..72 | 1 per stage | linear (min expected cost; min worst case) | KWG Table 1, T = 1..10 (section 6); run: reproduced to 0.1 for T = 1..6, 8, 10 | not ported |
| F5 | GWK multi-product inventory with backlog and random prices and demands (P = 4, T up to 30, 25 random instances) | GWK section 7, problem (16) | T up to 30 | 4 per stage (uniform on `[-1, 1]^4`) | bilinear in the rules (price x sales), max profit | relative gaps, GWK Table 1 (section 6) | not ported |
| F6 | GWK toy programs: Example 4.5 (`x >= max(abs xi2, abs xi3)`, min `E x`) and Example 5.1 (`x <= xi2, x <= xi3`, min `E -x`) | GWK Examples 4.5, 5.1, 5.11 | 1 | 2 | linear | 4.5: optimum 2/3, best axial 1; 5.1: optimum -1/3, axial primal 0, axial bound -0.377 (10 segments); run: reproduced | not ported |
| F7 | Shared stock and constrained / non-box supports: `Truncate`, `ValidInequality`, discs (SOC rows), `ConfidenceMvNormal` ellipsoid (mu = (100, 80), Sigma = [100 40; 40 64], prices 15 and 12) | PT `shared_stock*`, `law_spanning_stages`, `conic_library`; TUT advanced_distributions, confidence_mv_normal; LEG test_core (rejection sampling), `test/distributions.jl` | 2 (3 for the three-stage disc) | 2 to 3 | linear, SOC support | tutorial values in table 2; PT checks against the dense oracle | tests + tutorials (ported) |
| F8 | One-reservoir hydro toy, fixed `vi = 0.5`, `gh + gt = 0.3`, three inflows `U(0, 0.1 i)` | PT `hydro_toy` (`hydro_toy_fixed_volume`); LEG test_core (`test_0_*`, `test_1`, `test_2`, `test_recursion`), `test/bug_ipopt.jl`, `sampled_mode_ipopt_too_few_dof.jl` | 1 (single stage) | 3 | quadratic: `gt^2 + vf^2/2 - vf` | primal -0.41277778, dual -0.41277778 (QP solved to a tolerance, 1e-6) | test (ported) |
| F9 | Hydrothermal toy, one reservoir, 3 thermal plants (cost 250 / 750 / 1750, capacity 10 / 5 / 5), 5-cut future cost, 2 stages (`HydroToy2`) or 3 stages (`HydroToy3`) | CL `HydroToy/hydro2.jl`, `hydro3.jl`; CL `EVPI/` (same data, discrete inflow) | 2 or 3 | 1 per stage (`truncated(Normal(60, 20), 0, 120)`) | linear (min) | table 2 (T = 2: 119250 / 90500 at bp 0; T = 3: 191500 / 90750); EVPI base instance: EV 90500, WS 92071.07, optimum 92246.31 | not ported; run: reproduced exactly by 0.4 |
| F10 | `HydroQuad` and `HydroQuadPw`: the 2-stage toy with one thermal plant of quadratic (resp. two-piece quadratic) cost | CL `HydroQuad/` | 2 | 1 (same law) | quadratic (min) | table 2 (`res_*.txt`; run) | not ported |
| F11 | 4-reservoir, 5-subsystem hydrothermal problem (Shapiro, Tekaya, da Costa, Soares 2013; MSPPy / SDDP.jl data): 82 equiprobable joint inflow vectors per stage | PB `benchmark/hydro` (T = 4, 8, 12); PT `test_hydro.jl` (T = 3); LEG `examples/recursion/recursion.jl`, `monolithic_vs_sddp.jl`, `results/` | 3 to 12 (legacy sizes up to 24) | 4 joint (82 atoms), stages 2..T | linear (min) | T = 4: 3.759985 / 0.980316; T = 8: 94.874564 / 1.960632; T = 12: 468.156822 / 10.753064; SDDP (legacy, T = 12): bound 15.79, simulated 15.89 +- 0.42; T = 3 (PT): 1.888358847336 / 0.7352371632 | benchmark + test (ported) |
| F12 | Legacy value-function recursion `hydro_thermal_rpwldr` (one LDR per stage, `set_parametric_objective!`, breakpoints on stored energy and inflows) | LEG `examples/recursion/recursion.jl`, `1dtoy_value_function_comparison*.pdf`, `SDDP_vs_LFDR10_24stages.pdf` | 4 to 24 | 4 | linear | none recorded as numbers | not ported (`set_parametric_objective!` is not in 0.4) |
| F13 | Unit commitment, 30 buses, 36 h, deterministic / SAA / LDR with first-stage integer commitment and uncertain extra demand | LEG `examples/unit_commitment/uc-ldr.ipynb`, `30_2017-01-01.json` (UnitCommitment.jl data) | 2 | 36 (`demand_extra[t]`) | linear with binaries (MILP) | none recorded | not ported |
| F14 | Structural and edge models: `deterministic`, `unbounded_support`, `infeasible`, `unbounded`, `stage_one_uncertainty`, `objective_only_uncertainty` (QP), `valid_row_across_declarations`, `quadratic_inventory` (T = 16, `0.01 sum I^2 + 0.001 (sum I)^2`) | PT `edge_cases()`, `test_size.jl` | 1 to 3 | 1 to 4 | linear and quadratic | dense oracle (`test/oracles/dense.jl`) is the reference | tests (package only) |
| F15 | EVPI study of `hydro2.jl`: 25-point discretized N(60, 20) inflow, grid dynamic program, about 20 variants (spread, fleet, shape, future-cost steepness) | CL `EVPI/` (`results/*.md`, `math/evpi_bounds.tex`) | 2 | 1 | linear | base: EV 90500.00, WS 92071.07, Z* = EEV = 92246.31, EVPI 175.25 (0.19 %), VSS 0 | not ported; analysis, not an LDR problem |

### 1.1 Classification

Axes: **data** (toy, stylized = made-up but shaped like a real problem, real = real data or a realistic
instance); **horizon** (fixed, or a parameter `T` with per-stage data that are generated or repeated);
**where the uncertainty enters** (RHS = right-hand side; obj = cost / price coefficients of decisions;
tech = coefficient of a first-stage decision; rec = coefficient of a later-stage decision, i.e. random
recourse; supp = constraints on the uncertainty only, `Truncate` / `ValidInequality` / cones);
**objective / constraints**; **integers**; **dimensions** (decisions / uncertainties per stage); and the
**law across stages** (SWI = stagewise independent).

What 0.4 accepts, from the canonical form (`src/canonical/canonicalize.jl`, docstring of `canonicalize`): RHS,
obj (products of an uncertainty and a decision in the objective become a quadratic objective), supp,
linear and quadratic objectives, SOC rows on uncertainties only, integers in stage 1 only. **Any product
of an uncertainty and a decision in a constraint is rejected as random recourse, also when the decision
is a first-stage one** (run: `y <= xi * x` with `x` in `Stage(1)` errors with "multiplies the decision x
by the uncertainty xi (random recourse)"), so "tech" is not supported today either, although with a
deterministic `x` the robust counterpart would still be linear (inferred). A law spanning several stages
is accepted; the dual bound is certified only if its conditional expectations are linear
(`src/formulations/dual.jl`).

| # | Data | Horizon | Uncertainty enters | Objective / constraints | Integers | Decisions / uncertainties per stage | Across stages |
|---|---|---|---|---|---|---|---|
| F1 | toy | fixed (2) | RHS | linear / linear | none | 3 (buy; sell, return) / 1 | single stage of uncertainty |
| F2 | toy | fixed (2 or 3) | RHS; obj (random sell price, random `buy_cost2`) | linear, bilinear `price x sell` / linear | `buy` in stage 1 (one variant) | 2 to 6 / 1 to 2 | SWI; joint discrete law within a stage |
| F3 | toy | `T` (data repeated) | RHS | linear / linear | none | 3 (`x`, `I`, `y`) / 1 | SWI |
| F4 | stylized (seasonal formula) | `T` = 1..72 (formula) | RHS | linear (expected or worst case) / linear, one cumulative row per factory couples all stages | none | 3 / 1 | SWI |
| F5 | stylized (25 random instances) | `T` up to 30 (formula) | RHS (demand) and obj (prices) | bilinear in rule x price / linear | none | about 4P = 16 / 4 factors | SWI; factor model inside a stage |
| F6 | toy | fixed (1 recourse stage) | RHS | linear / linear | none | 1 / 2 | - |
| F7 | toy | fixed (2 or 3) | RHS, supp (box cuts, discs, ellipsoid) | linear / linear + SOC on the support | none | 2 to 4 / 2 to 3 | correlated within a stage; `law_spanning_stages` spans 2 stages |
| F8 | toy | fixed (single stage) | RHS | convex quadratic / linear | none | about 5 / 3 | - |
| F9 | toy (stylized cuts) | fixed 2, 3; extensible (inferred: the stage data are constant, only the cuts sit at the end) | RHS | linear / linear | none | about 7 (3 thermal, hydro, spill, storage, + cost-to-go) / 1 | SWI truncated normal |
| F10 | toy | fixed (2) | RHS | convex quadratic, piecewise quadratic / linear | none | as F9 / 1 | - |
| F11 | **real** (Brazilian system, 82 historical inflow records per month) | `T` (12 monthly data sets cycled; 120 stages in SDDP.jl's copy) | RHS (water balance) | linear / linear | none | 148 (95 thermal, 4 hydro, 4 spill, 4 storage, 25 exchange, 16 deficit) / 4 | SWI in the benchmark; the data are serially correlated (note below) |
| F12 | real (F11 data) | `T` (4 to 24) | RHS | linear / linear | none | as F11 / 4 | SWI |
| F13 | real (UnitCommitment.jl 30-bus instance) | fixed (36 h, two-stage) | RHS (`demand_extra[t]`, all revealed at once) | linear / linear (MILP) | binary commitment, stage 1 only (supported) | thousands (inferred from 30 buses x 36 h) / 36 | one stage |
| F14 | synthetic | fixed (1 to 3; `quadratic_inventory` T = 16) | RHS, obj, supp | linear, quadratic / linear | none | 1 to 4 / 1 to 4 | SWI, one law spanning stages |
| F15 | toy | fixed (2) | RHS | linear / linear | none | as F9 / 1 (25 atoms) | - |

Highlights: only F11, F12 and F13 have real data; F3, F4, F5, F11 (and F9 by construction) scale in
`T`; random prices (obj) appear only in F2 and F5; nothing in the package or CL has a random technology
matrix; the only integer instances are F2 (`buy`) and F13, both in stage 1; F7 is the only family with
support constraints and cones; F8, F10 and F14 are the convex QPs.

**Inflows of F11, historical scenarios against a fitted law.** The 82 vectors per month in
`test/problems/hydro_data.jl` (from MSPPy) are historical records: Dowson, Morton and Pagnoncelli (Ann. Oper.
Res. 2022, preprint in SDDP.jl `papers/entropic/preprint.pdf`) describe the MSPPy data as "historical
data", and the scenario index is a year: (run, `corr.jl` in section 8) the correlation across the 82
indices between month m and m + 1 of the same reservoir is 0.44 to 0.98 for m = 1..11 and -0.10 to 0.08
from December back to January of the same index; across reservoirs, in months 1 and 6, it ranges from
-0.34 to 0.66; the coefficient of variation is 0.22 to 0.63, with mean > median (right skew). The
benchmark uses them in two ways: `inflow_law` makes each month an `MvDiscreteNonParametric` of the 82
joint vectors (equiprobable), independent across stages; with breakpoints, `marginal_law` keeps only the
four marginals as independent `DiscreteNonParametric` laws (cross-reservoir correlation dropped) and puts
breakpoints at their quantiles. Either way the serial correlation of the record is discarded, as in the
MSPPy / SDDP.jl models ("82 stagewise-independent realizations of inflow in each stage", SDDP.jl
`papers/biobjective_sddp/preprint.pdf`). A fitted law would change, in order of importance
(inferences from the package's semantics, not runs):

1. **Serial correlation.** A periodic AR(1) per reservoir, `inflow_t = mu_m + phi_m (inflow_{t-1} -
   mu_{m-1}) + sigma_m eps_t` with independent noises `eps_t`, writes each inflow as an affine
   expression of the noises; the model declares `eps_t` as `Uncertainty` in `Stage(t)` and keeps the
   water balance linear. Conditional expectations stay linear, so `DualLDR` remains certified. A
   log-normal PAR (a common choice for inflows) is not affine in the noises and would need the inflows
   themselves as a law spanning stages, whose bound is then not certified.
2. **Support.** `PrimalLDR` must be feasible on the whole support; a fitted normal is unbounded and has
   to be truncated (`Truncate`, `ConfidenceMvNormal` boxes or ellipsoid), and a support wider than the
   convex hull of the 82 records makes the primal more conservative, while a narrower one (e.g. 95 %)
   makes it optimistic out of sample.
3. **Breakpoints and moments.** With a continuous marginal the quantile breakpoints are no longer tied to
   82 atoms, and a joint law (e.g. `MvNormal` or a Gaussian copula per month) keeps the cross-reservoir
   correlation that `marginal_law` drops; the bound and the value depend only on the support and the
   (piecewise) moments, so a fit that matches the first two moments changes little at `k = 0` but
   changes the lifted problems.
4. **Evaluation.** Out-of-sample scores should then be computed both on the fitted law and on the
   historical years (as SDDP.jl's `CyclicHistorical` sampling in `papers/entropic/paper_code.jl` does).

## 2. Variants

"=" means the same problem in CL and in the package; "differs" is described in the last column.

### F1, F2: newsvendors (all values: max profit, `Uniform(80, 120)` unless said)

| Variant | Law / data | Breakpoints | primal / dual | Where | CL vs package |
|---|---|---|---|---|---|
| (10, 8, 15), no bp | buy 10, return 8, sell 15 | 0 | 460 / 486.67 | PT, TUT, LEG, CL | = (CL prints EV 500, `Perf` 465) |
| same, bp 1 (`[100]`) | | `[100]` | 465 / 476.67 | PT `_breakpoints_1`, CL, TUT | = |
| same, bp 2 (equispaced) | | 93.33, 106.67 | 471.11 / 476.30 | CL, TUT | PT has no bp 2 entry |
| same, bp 3 (`[90, 100, 110]`) | | 3 | 471.25 / 474.17 | PT, CL, TUT | = |
| same, 7 and 15 breakpoints | | step 5 and 2.5 | 471.25 / 471.98 and 471.328 / 471.51 | PT only | |
| same, critical breakpoint | | `[760/7]` | 471.43 / 480.95 | PT | PT only; primal is the optimum |
| same, bp 4..10 | | k | 470.40 / 472.27 (4), 471.11 / 472.41 (5), ..., 471.405 / 471.79 (10); the primal is not monotone in k | TUT piecewise_linear | |
| (10, 8, 20) | sell 20 | 0..3 | 960 / 986.67, 960 / 973.33, 960 / 968.89, 965 / 970; EV 1000 (`Perf` 940); optimum 966.67 (derived) | CL only | CL's second price set, where LDR beats EV |
| integer `buy` | | 0 | 460 / 486 | PT (`mip_rel_gap = 0`) | PT only |
| random price `U(13, 17)` | price x sell | 0 | 460 / 486.67 | PT, LEG | PT only |
| two products, independent | product law | 0 | 920 / 973.33 | PT, LEG | PT only |
| three stages: demand2, `buy_cost2 ~ U(8, 14)`, demand3 | | 0 | 940 / 1000 | PT, LEG | PT only |
| discrete demand `{80, ..., 120}`, p = (.1, .2, .4, .2, .1), salvage 5 | | 0 | 400 / 470 | PT, TUT | PT test uses salvage 5; the tutorial helper uses salvage 8 (discrete 5 atoms: 460 / 488) |
| bivariate discrete (5 atoms, correlated), salvage 5 | | 0 | 700 / 760 | PT, TUT | PT only |
| legacy sampled (1000 scenarios, Xoshiro(1234)) | | 0 | sampled 464.63 | `legacy_values.toml` only | sampled: out of scope (section 7) |

### F3: inventory

| Variant | primal / dual | Where |
|---|---|---|
| T = 2, 4, 6, 8 | 200 / 173.33, 380 / 340, 560 / 506.67, 740 / 673.33 | TUT multistage; PT (T = 4, 8) |
| T = 4, bp 1, 3, 7 on `d[2]` | 380 / 343.33, 379.375 / 344.583, 378.594 / 344.896 | TUT multistage |
| T = 4..128, timing | primal 90T + 20; sizes in `benchmark/README.md` | PB `timing.jl` (T = 128: primal 57,792 variables, dual 58,812) |
| T = 64, `stock` parameter, 20 re-solves | | PB `resolve.jl` |
| `quadratic_inventory` T = 16 | primal / dual agree with the dense oracle (rtol 1e-6, Clarabel) | PT `test_size.jl` |

### F4, F5, F6: papers

See section 6 for the data and the published numbers.

### F7: constrained supports

| Variant | Values (primal / dual unless said) | Where |
|---|---|---|
| `Uniform(50, 150)` demand, constraint `80 <= d <= 120`: none / `Truncate` / `ValidInequality` | 400 / 466.67; 460 / 486.67; 460 / 416.67 | TUT advanced_distributions |
| shared stock, 5-atom law on `[0, 100]^2`: none / valid / truncate | 0 / 306.29; 188 / 297.09; 188 / 297.09 | TUT advanced_distributions |
| two products `U(40, 80)^2`, `d1 + d2 <= total` (rejection sampling, 100 to 100000 draws) | 431.11 / 511.70 (100), 427.00 / 506.94 (100000) | TUT advanced_distributions |
| three reservoirs, four `Truncate` rows on `U(10, 50) x U(10, 40) x U(10, 60)` | no values printed | TUT advanced_distributions |
| `ConfidenceMvNormal`, alpha = 0.5, 0.8, 0.9, 0.95, 0.99: boxes (HiGHS) | 621.88 / 651.91, 601.92 / 649.67, 590.53 / 649.24, 580.76 / 649.32, 561.75 / 650.25 | TUT confidence_mv_normal |
| same, ellipsoid (SOCP, Clarabel) | 623.22 / 651.86, 603.95 / 649.59, 592.96 / 649.17, 583.53 / 649.25, 565.19 / 650.18 | TUT confidence_mv_normal |
| disc `norm(d - (100, 100)) <= 30`, 20 (truncate), with a cut; three-stage disc, with breakpoints | no known value; Clarabel against the dense oracle | PT `conic_library` |

### F8: one-reservoir QP

| Variant | Differences | Where |
|---|---|---|
| fixed `vi = 0.5` | primal -0.41277778 / dual -0.41277778 | PT, LEG (`test_0`, `test_2`) |
| `vi ~ U(0.45, 0.55)` random, and `vi ~ U(0, 0.5)` | not in PT | LEG `test_recursion`, `test_1` |
| `vi` random with `BreakPoints() = 3` | not in PT | LEG `test_recursion_pwl` |
| value-function recursion with a second model and `set_parametric_objective!` | removed | LEG |
| sampled mode, N = 4, 5, 50, one `inflow ~ U(0, 0.2)` | `test_equalities_once` (PT) checks 8 + 2 equalities; the old Ipopt failure was "too few degrees of freedom" | PT, LEG |

### F9, F10: CL hydrothermal toys

Data (all CL hydro files): hydro capacity 100, reservoir 300, initial volume 100, first-stage inflow 60,
demand 70 per stage, inflow `truncated(Normal(60, 20), 0, 120)` independent across stages, 5 cuts
`theta >= c_k v + b_k` with `c = (-300, -600, -1200, -2500, -7800)` and
`b = (75000, 135000, 210000, 310000, 390000)` on the final volume, thermal `(250, 750, 1750)` with capacity
`(10, 5, 5)` (`HydroToy`), one plant of capacity 20 (`HydroQuad`), or two pieces of 10 (`HydroQuadPw`).

| Variant | Cost | Horizon | LDR values, bp 0 / 1 / 2 / 3 (primal / dual) | Source of the numbers |
|---|---|---|---|---|
| `HydroToy2` | linear (3 plants) | 2 | 119250 / 90500; 98081.92 / 91161.04; 93847.45 / 91087.71; 93412.93 / 91707.14 (EV 90500; every LDR commits to storage 105) | `results_20_small.txt`; run: identical |
| `HydroToy3` | linear | 3 (breakpoints on both inflows) | 191500 / 90750; 119344.46 / 91806.62; 107853.18 / 91673.18; 106060.80 / 92577.47 (EV 90750) | `results_20_small.txt`, `result3.txt`; run: identical |
| `HydroQuad` (current) | `121.30 (t1^2 + t2^2)` | 2 | 138924.28 / 108064.30; 116617.88 / 109809.95; 112271.28 / 109292.91; 112818.27 / 110625.51 (EV 108064.30) | second table of `results_20_small.txt`; run with Ipopt: identical to 1e-7 |
| `HydroQuad` (older) | `121.30 * 0.51^2 / 2` (removed by c51cf5d) | 2 | 106106.31 / 79065.24; 85867.43 / 80170.52; 84960.29 / 81168.97; 83019.91 / 81317.22 (EV 78620.05) | first table of `results_20_small.txt` |
| `HydroQuadPw` | piece 1: `113.1/2 g1^2` on `[0, 10]`; piece 2: `3118.6/2 g2^2 + 1131 g2` on `[0, 10]` (C1 at 10) | 2 | 134385.84 / 101308.47; 109913.23 / 101543.53; 103938.17 / 102301.11; 104754.14 / 101980.83 (EV 101308.47) | `res_ipopt.txt`, `res_highs_hipo.txt`, `res_clarabel.txt` (values), `results_20_small.txt`; run with Ipopt: identical |
| `HydroToy3` basis variants (`BP2`, `PW3`) | linear | 3 | out of scope (sampled), section 7 | `results_20_small.txt` |

Differences between the CL hydro files and this package's hydro problems:

- CL's hydro is a single reservoir with a univariate continuous law; `benchmark/hydro` is 4 reservoirs with a
  joint discrete law of 82 atoms (`MvDiscreteNonParametric`), PT `hydro_toy` a three-inflow uniform QP with
  no future cost.
- Stage convention: `HydroToy3` has `Stage(t)` on all decisions, `Stage(2)`, `Stage(3)` on the inflows and a
  known inflow in stage 1; `benchmark/hydro` follows the same convention. `HydroToy2`, `HydroQuad*` use
  `FirstStage` on the stage-1 variables and the default last stage on the others.
- `HydroQuad`'s SAA objective is `121.30 (t1^2 + (sum t2 / N)^2)`: the square of the sample mean, not the
  sample mean of the squares that the LDR objective (an expectation of the squared rule) corresponds to. The
  SAA rows of the CL tables are therefore a different problem from the LDR rows (`HydroQuadPw` has the same
  pattern). The out-of-sample `Perf` is computed correctly.
- The CL SDDP model adds a deficit variable at cost 8000 that the LDR models do not have; the SDDP bound is
  read from 400 iterations on a sample tree of `n x 100`.
- Cost scale differs between `HydroQuad` (121.30 on the whole range) and `HydroQuadPw` (56.55 up to 10, then
  1559.3 with a linear term 1131).
- EV is the deterministic problem at the mean inflow. In `HydroToy2` every LDR commits to the same first stage
  as EV (storage 105), so the out-of-sample `Perf` coincides (92068.63 +- 219.9). In `HydroToy3` bp 0 and 1
  commit storage 110 and score 102105.80 +- 207.7 against 98814.60 +- 221.4 for EV; bp 2 and 3 commit 105 and
  tie EV (1000 scenarios).

### F11: 4-reservoir hydrothermal

| Variant | primal / dual | Where |
|---|---|---|
| T = 4, 8, 12, k = 0 | 3.759985 / 0.980316; 94.874564 / 1.960632; 468.156822 / 10.753064 | PB `hydro/`, LEG `ldr_T12.csv` (same to 12 digits) |
| T = 4, k = 1, 2 breakpoints per inflow (independent marginals) | 1.888805 / 0.989138; 1.596237 / 0.991246 | PB |
| T = 8, k = 1 | 37.295662 / 1.970324 | PB |
| average-scenario model T = 4, 8, 12 | 0.980316, 1.960632, 10.386167 (lower bounds) | PB, PT (T = 3: 0.7352371632) |
| tree, 3 branches (T = 4) or 2 (T = 8, 12) | 1.036250, 1.984511, 13.695625 (one sampled problem) | PB; PT (T = 3, 3 branches: 0.741309440590) |
| PT T = 3 | 1.888358847336 / 0.7352371632 (the dual equals the average-scenario value) | PT `test_hydro.jl` |
| legacy T = 12 with `SlackAsExpr` or explicit slacks | same values; larger models | LEG, PB `compare/` |
| legacy T = 12, breakpoints lifted in the script (k = 1, 2, 3) | primal 857,808 variables and above: time limit at 600 s, no value | LEG `ldr_T12.csv` |
| SDDP, T = 12 (legacy, 2 x 614 s) | bound 15.79, simulation 15.89 +- 0.42 (2000 scenarios) | LEG `sddp_T12_checkpoints.csv` |

## 3. How CL builds its LDRs

- All LDRs go through the legacy package (0.3.1), written in JuMP with `LinearDecisionRules.LDRModel(solver)`,
  `Uncertainty(; distribution, stage)`, `FirstStage`, `Stage(t)` through `set_attribute`, and
  `BreakPoints()` (an integer = equispaced). The value is `objective_value(ldr)`, the bound
  `objective_value(ldr; dual = true)` (0.4: `objective_bound`), the first stage `get_decision`.
- `Newsvendor`, `HydroToy2`, `HydroToy3`, `HydroQuad`, `HydroQuadPw` each have `ldr_solution` (this) beside
  `expected_value_solution`, `saa_solution` (hand-written JuMP SAA, 4 / 10 / 30 / 100 samples, 3 replications)
  and, for `HydroToy3`, `sddp_solution`. `src/Compare.jl` times them and scores each first stage on 10,000
  (1,000 in `HydroToy3`) out-of-sample inflows by solving the recourse with HiGHS.
- Hand-written LDRs (no package) are in `HydroToy/hydro_scenario.jl`: `extreme_point_ldr` (the rule fitted on
  the `2^(T-1)` extreme points of the inflow box, with `linear_basis(T)`), `extreme_point_ldr_X` (the same, in
  coefficients, with Ipopt), `primal_ldr` and the dual / decomposition variants over a `Sample` with the bases
  `BP2` (6 piecewise-linear functions of the stage inflows at 0/40/80/120) and `PW3` (BP2 plus 6 functions of
  the sum of inflows). They are scenario-wise; see section 7.
- `EVPI/` does not use the package: a grid dynamic program on a 25-atom inflow, with an independent
  re-implementation in `EVPI/verification/` (EV 90500.0000, WS 92071.0682, Z* 92246.3133, EVPI 175.2451).

## 4. Solver combinations used anywhere

| Class | Solver (settings) | Where used | Notes on failures and accuracy |
|---|---|---|---|
| LP | HiGHS dual simplex (default), HiGHS.jl 1.26.1 / HiGHS 1.15.1 | PT, PB, TUT; CL default for `Newsvendor` | the reference solver; `legacy_values.toml` was generated with it, "Ipopt only if HiGHS fails" |
| LP | HiGHS IPM = HiPO (`solver = "ipm"`) | CL default of every hydro file; `PB compare/` | `DualLDR(attributes = ["solver" => "ipm", "run_crossover" => "off"])` is the benchmark-recommended dual (hydro T = 8: 8.0 s against 145 s; inventory T = 64 is 1.2 s against 0.72 s); on the 0.4 hydro primal HiPO is unreliable (SOLVE_ERROR with crossover at T = 8, objective 44.7 against 3.76 at T = 4 without crossover, status `unk`); at hydro T = 12 `OTHER_ERROR`, the bound is `NaN` since the invalid-bound fix |
| LP | HiGHS primal simplex, IPX (no crossover), PDLP, HiPDLP | `PB compare/` (`run_modes.jl`) | IPX no-crossover is 3.3x faster on the hydro T = 8 primal; PDLP and HiPDLP are slower than HiPO on every file tried; no mode is best everywhere (spread up to 75x on one file) |
| LP | Ipopt (MUMPS, `tol = 1e-8`) | `PB compare/`, `generate_legacy_values.jl` fallback | converges on all 0.4 files tried, never the fastest (1.9x to 41x slower than the best HiGHS mode), objective off by 5e-5 to 4e-4 on the hydro files (same with `tol = 1e-10`), hits the 120 s limit on the hydro T = 8 dual; not for certified bounds |
| LP | Clarabel (conic) | run (this catalogue) | `DualLDR` of `HydroToy2` with bp 1 (cost scale 1e5): `INFEASIBLE`; with all costs and cuts multiplied by 1e-3 it is `OPTIMAL`; `PrimalLDR` is fine at both scales (119250.00005, 98081.9196) |
| LP | SDDP.jl 1.15.0 on HiGHS | CL `HydroToy3` (400 iterations, cut deletion 50), LEG `monolithic_vs_sddp.jl` (T = 12, 10-minute segments) | noisy bound against the LDR bracket; 18 "numeric issues" in the legacy log |
| QP | HiGHS active set (`qpasm`; the default for a QP) | CL commit 9affc09 ("seems stalling in both LDR and SAA models"); LEG `bug_qp_highs.jl` (`qp_iteration_limit = 50000`, a min `x^2 + z` with 1 breakpoint, 2 cuts); CL `all_solvers` "HiGHS (ASM)" (needed `"solver" => "qpasm"`, commit 6e7b72f); PT `hydro_toy` (default HiGHS, atol 1e-6, passes); run | run on `HydroQuad`: bp 0 done in 3 s with a 1.6e-4 relative error (138946.92 against 138924.28, `OPTIMAL`); bp 1 and 3 reach a 20 s time limit (bound 210367.86 at bp 1); bp 2 returns no solution; `HydroQuadPw` bp 0 is `OPTIMAL` with a bound 101938.45 above the Ipopt dual optimum 101308.47 |
| QP | HiGHS IPM = HiPO (`solver = "ipm"`) | CL commit d4679dd ("faster and as reliable as Ipopt"); CL default for `HydroQuad*`; `res_highs_hipo.txt`; run | `HydroQuad`: `OPTIMAL`, 0.01 s, equal to Ipopt to 6 digits at bp 0..3; `HydroQuadPw`: values and bounds equal to Ipopt to 1e-6 but status `OTHER_ERROR` at bp 1..3 and no bound at bp 0 (CL's 2026-09 `res_highs_hipo.txt` shows the same values, status not recorded) |
| QP | Ipopt (`max_cpu_time = 60`) | CL `res_ipopt.txt`, `all_solvers`; LEG tests; run | reliable on every LDR QP; values and bounds equal across CL (`res_ipopt.txt`) and the run (1e-7); `hydro_toy` dual -0.4127777786 against -0.4127777778 (tolerance); LEG sampled mode failed with "too few degrees of freedom" when each equality was repeated per scenario (fixed by matching equalities once, PT `test_equalities_once`) |
| QP | Clarabel | CL (`moresolvers`, `res_clarabel.txt`); PT `test_size.jl` (quadratic inventory, rtol 1e-6); run | legacy `HydroQuadPw`: values correct but `Bound` garbage (2.9e12 at bp 1, 1.5e12 at bp 2, 1.1e12 at bp 3); 0.4 run: `PrimalLDR` `OPTIMAL` (138924.28, 116617.88, 112271.28, 112818.27) but `DualLDR` `PRIMAL_INFEASIBLE` at cost scale 1e5, `OPTIMAL` with bounds 108064.30, 109809.95, 110625.51 (equal to Ipopt) once costs and cuts are multiplied by 1e-3 |
| QP / NLP | Uno | CL `solvertest`, `res_uno.txt` | EV and every SAA row converge to a wrong point (EV 386623 against 101308; first stage 50 / 110 / 20); SAA 30 and 100 print "insufficient space available for profile"; 24 s for SAA 100; the LDR rows are right |
| QP / NLP | Algencan (via NLPModelsJuMP) | CL `moresolvers`, `res_algencan.txt` | LDR objective garbage (171287.55, 116136.49, 419814.82, 906823.63 against 134385 to 104754) with first stages (hydro, storage) drifting to 46 / 116 and 35 / 132; SAA rows right; EV 4.6 s |
| SOCP | Clarabel | PT (conic problems, `shared_stock_disc`, `ConfidenceMvNormal`), TUT confidence_mv_normal, getting_started | "solves to about 1e-8"; tests use rtol 1e-6; HiGHS has no second-order cone: the ellipsoid becomes its boxes (`support = :auto` chooses) |
| MILP | HiGHS (`mip_rel_gap = 0` in PT; 0.01 in the UC notebook) | PT `newsvendor(integer = true)` (dual 486), LEG `uc-ldr.ipynb` (30-bus, 36 h, binary commitment) | exact gap needed for the integer dual to be the known value |
| LP / QP (sampled) | HiGHS, Ipopt | PT `test_sampled.jl`, CL SAA | sampled value is an estimate, not a bound; see section 7 |
| LP | CPLEX 11.2 (KWG), CPLEX 12 (GWK) | papers | KWG: T = 72 bounds in 3,519 s; GWK: 4 GB limit |

Harness caveat: CL's `res_highs_hipo.txt` and `res_ipopt.txt` give the same policies but an out-of-sample `Perf`
of 103030.40 against 103004.61 for EV (+26, 0.025 %), from two versions of the harness or of the recourse
solver; the cross-solver comparison of `Perf` in those files is therefore not exact.

## 5. Differences between CL and the package, in one list

1. Same problem in both: newsvendor (10, 8, 15), bp 0, 1, 3 (CL's bp 2 = 93.33, 106.67 is not in PT); the
   4-reservoir values of `ldr_T12.csv` equal PB's to 12 digits. The CL toy hydro files have no counterpart.
2. Salvage: PT's discrete newsvendors use salvage 5, the tutorial helper and CL use 8.
3. Inventory: PT and KWG differ in the stage of the demand (stage convention above), which changes the
   information of `x_t`.
4. Quadratic costs: PT has one QP (`hydro_toy`, no future cost) and the quadratic inventory in `test_size.jl`;
   CL has two hydro QPs with future cost, where HiGHS's QP solvers misbehave.
5. API: CL uses 0.3.1 (`objective_value(m; dual = true)`, `SolveDual` attributes in LEG); 0.4 uses
   `Formulations`, `objective_bound`, `Stage` containers, `Parameter`.
6. CL's SAA is hand-written; 0.4 has `SampledLDR`, whose equalities are matched once.

## 6. Papers: data to rebuild the problems

### KWG 2011, section 5 (F4)

Inventory of Ben-Tal et al. (2004, section 5). Warehouse and `I = 3` factories, `T` semimonthly periods.
`xi_t`: demand; `x_it in [0, 567]`: production; `c_it = alpha_i varsigma_t` with
`alpha = (1, 1.5, 2)`; `varsigma_t = 1 + sin(pi (t - 1) / 12) / 2`;
`xi_t in [(1 - theta) xi* varsigma_t, (1 + theta) xi* varsigma_t]` with `xi* = 1000`, `theta = 0.3`;
cumulative capacity per factory `sum_t x_it <= 13,600 T / 24`; warehouse `500 <= x0 + sum_{s<=t} sum_i x_is - sum_{s<=t} xi_s <= 2000`
for every `t`; `x_it` observes `xi_1..xi_t`. Stochastic version (5.9): minimize `E sum c_it x_it`, `xi`
uniform on the box; robust version (5.8): minimize the worst-case cost (objective in the epigraph form,
`P` chosen freely; the uniform law gives a weak lower bound, and a point mass at `xi_t = (1 + theta) xi* varsigma_t`
makes the lower bound meet the upper one). The value of `x0` is lost in the text extraction; **`x0 = 1000`
reproduces the published table (run)**, `x0 = 500` and `2000` do not. Stochastic Table 1 (lb = dual,
ub = primal; SAA2/3/4 columns omitted):

| T | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lb | 508.3 | 1972.7 | 3825.5 | 6090.7 | 8665.4 | 11483.9 | 14433.5 | 17434.4 | 20255.9 | 22769.3 |
| ub | 558.3 | 2032.6 | 4005.3 | 6356.0 | 9064.0 | 12047.5 | 15182.7 | 18329.3 | 21279.0 | 23869.9 |

SAA (50 trees per entry, 10 / 50 / 90 % quantiles) is reported for branching 2, 3, 4 up to T = 10, 7 and 6.
Figure 1 gives the bounds for T = 1..72 (relative gap below 5 %); CPLEX 11.2 solved T = 72 in 3,519 s.

Run (this package, HiGHS): `(lb, ub)` = (508.3, 558.3), (1972.7, 2032.5), (3825.5, 4005.3), (6090.7, 6356.0),
(8665.4, 9064.0) for T = 1..5, and (11483.89, 12047.48), (17434.42, 18329.31), (22769.39, 23869.95) for
T = 6, 8, 10, in 0.01 to 0.05 s: equal to the paper to the printed digit (T = 2: 2032.5 against 2032.6).

### GWK 2015, section 7 (F5), problem (16)

`P = 4` products, `T` months, risk factors `xi_t in [-1, 1]^4` uniform and independent across `t`.
Price `R_tp = 2 + gamma_R (alpha_1p xi_t1 + alpha_2p xi_t2)`; demand
`D_tp = 2 + sin(2 pi (t - 1) / 12) + gamma_D (alpha_3p xi_t3 + alpha_4p xi_t4) / 2` for `p <= P / 2` and
`2 + cos(...) + ...` for the others; loadings `alpha` uniform on `[-1, 1]`; `gamma_R = gamma_D = 1`,
`C_B = C_I = 0.2` (backlog and holding), capacity `24`. Maximize expected profit
`sum R_tp s_tp - C_B b_tp - C_I I_t` (the text of (16) lost the indices of `I` and the `t = 1` indicators:
`I_t` appears with and without the product index, so the layout of the inventory is ambiguous). Orders `o`,
sales `s`, backlog `b`, inventory `I >= 0`; sales are served from orders of `t - 1` or earlier. Average
over 25 random instances. Table 1, relative gap between the bounds (%):

| T | 2 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LDR | 0.9 | 28.4 | 28.8 | 28.8 | 32.0 | 36.2 | 43.1 |
| AS5 (axial, 5 equispaced breakpoints) | 0.5 | 24.5 | 27.3 | 23.9 | 27.3 | 24.1 | 29.1 |
| GS1 (general segmentation, 1 breakpoint at 0) | 0.6 | 5.9 | 5.6 | 5.7 | 6.5 | 6.3 | 7.5 |
| BL (bilinear) / TL (trilinear) | 0.8 / 0.7 | 14.1 / 14.1 | 13.7 / 13.4 | 15.5 / 15.3 | 17.7 / 17.4 | 12.6 / 12.4 | 18.9 / 18.8 |
| BL-1 / TL-1 | 0.7 / 0.7 | 6.8 / 5.5 | 5.6 / 4.1 | 5.5 / 4.4 | 5.9 / 5.6 | 6.1 / 4.1 | 7.7 / 5.7 |
| SCN (scenario tree) | 1.1 | 4.1 | - | - | - | - | - |

Table 2 (CPLEX 12, primal / dual seconds) at T = 30: LDR 1.1 / 6.6, AS5 49.2 / 85.2, GS1 51.5 / 158.3. In this
package LDR and AS5 are available (`BreakPoints() = 5` on each of the four factors); GS1, BL, TL need
breakpoints on linear combinations or multilinear liftings, which are not supported (the roadmap lists
"uncertainty towers: PWL lifts on linear combinations" and "bilinear decision rules").

### GWK 2015, Examples 4.5, 5.1 and 5.11 (F6)

- 4.5: `min E x`, `x >= abs(xi2)`, `x >= abs(xi3)`, `xi2, xi3` independent `U[-1, 1]` (the program has
  `xi1 = 1`). Optimum 2/3 (`x = max(abs xi2, abs xi3)`); the best axial piecewise linear and the best linear
  rule is `x = 1`, value 1. Run (no breakpoints): primal 1.0, dual 0.3333; with `BreakPoints() = 10` per
  factor: 0.752 / 0.499. General segmentation along the diagonals is what recovers 2/3.
- 5.1: `min E(-x)`, `0 <= x <= xi2`, `x <= xi3`, independent `U[0, 1]`. Optimum -1/3 (`x = min(xi2, xi3)`);
  axial primal 0 for any number of breakpoints (also with general segmentation along `f1 = (1, 0, 0)`,
  `f2 = (0, 1, 1)`); lower bounds: -0.377 (axial, 10 equispaced segments), -0.336 (general, 10), multilinear
  lifting: upper bound -1/4, lower bound -0.416 (Example 5.11). Run: no breakpoints 0.0 / -0.41667; 9 interior
  breakpoints (10 segments) 0.0 / -0.37667, 10 interior -0.37534; the series is not monotone (11: -0.37616,
  12: -0.37525, 13: -0.37585, 20: -0.37509).

## 7. Out of scope for now: sampled-scenario variants (CL)

One line each, none is to be ported before the sampled mode is: `saa_solution` in every CL file (4 / 10 / 30 /
100 samples, 3 replications; HydroToy3 `n x 30`); `sddp_solution` (SDDP.jl, `n x 100` tree, 400
iterations); the basis-function ("Basis-LDR") models of `hydro_scenario.jl` on a `Sample` (`primal_ldr`,
`primal_ldr_benders`, `primal_ldr_scenario_generation`, bases `BP2`, `PW3`; 100 to 100,000 scenarios;
`Decomposition.md`: generation reaches 100,000 scenarios in about 1.5 s, the monolithic model needs 82 s for
10,000); the dual variants (`dual_ldr`, `dual_ldr_benders`, `dual_ldr_lagrangian`, `dual_ldr_column_generation`,
stage-measurable duals; `measurable_runner.jl`, `runner.jl`, `Dual_scenarios.md`); the sampled-LDR rows of
`legacy_values.toml` (`sampled_newsvendor_N1000_seed1234`, 464.63).
`extreme_point_ldr` (the `2^(T-1)` vertices) is scenario-wise but not sampled; it has no known value.

## 8. Reproducing the "(run)" numbers

Scratch environment: `Pkg.develop(path = <this worktree>)` plus JuMP, HiGHS, Ipopt, Clarabel, Distributions.
The models, in 0.4 syntax (`LDRModel`, `Uncertainty`, `Stage`, `BreakPoints`, `Formulations`):

```julia
# KWG 2011, section 5, stochastic inventory; returns (lb, ub) = (objective_bound, objective_value)
function kwg(T; x0 = 1000.0, theta = 0.3, xs = 1000.0)
    s = [1 + 0.5sin(pi * (t - 1) / 12) for t in 1:T]
    m = LDRModel(HiGHS.Optimizer); set_silent(m)
    @variable(m, d[t = 1:T] in Uncertainty(Uniform((1 - theta) * xs * s[t], (1 + theta) * xs * s[t])), Stage(t))
    @variable(m, 0 <= x[i = 1:3, t = 1:T] <= 567, Stage(t))
    @constraint(m, [i = 1:3], sum(x[i, t] for t in 1:T) <= 13600 * T / 24)
    @constraint(m, [t = 1:T], 500 <= x0 + sum(x[i, r] for i in 1:3, r in 1:t) - sum(d[r] for r in 1:t) <= 2000)
    @objective(m, Min, sum([1.0, 1.5, 2.0][i] * s[t] * x[i, t] for i in 1:3, t in 1:T))
    optimize!(m)
    return objective_bound(m), objective_value(m)
end

# GWK 2015, Example 5.1 (axial breakpoints on both factors)
m = LDRModel(HiGHS.Optimizer); set_silent(m)
@variable(m, a in Uncertainty(Uniform(0, 1))); @variable(m, b in Uncertainty(Uniform(0, 1)))
@variable(m, x >= 0); @constraint(m, x <= a); @constraint(m, x <= b); @objective(m, Min, -x)
set_attribute(a, BreakPoints(), 9); set_attribute(b, BreakPoints(), 9)
optimize!(m)   # objective_value 0.0, objective_bound -0.37667
```

CL models were rewritten in 0.4 from `hydro2.jl`, `hydro3.jl`, `hydroquad.jl`, `hydroquad_pw.jl` (same data,
`LDRModel`, `Stage(t)`, `BreakPoints`); HydroToy2, HydroToy3, HydroQuad and HydroQuadPw were solved
for bp 0..3 and the values compared with `results_20_small.txt` and `res_ipopt.txt`. The solver table
(section 4) rows marked "run" come from the same models with `HiGHS` (default, `qpasm`, `ipm`; 20 s limit),
`Ipopt` (20 s) and `Clarabel` (20 s), and from the cost-scale test (all costs and cuts times `1e-3`).
The CL scripts themselves were not run: their environment depends on the legacy package by relative path.

Section 1.1 (same environment):

```julia
# corr.jl: serial and cross-reservoir correlation of the F11 inflow records
include("test/problems/hydro_data.jl"); X = hydro_thermal_data().inflow_scenarios
[cor(X[i][m], X[i][m % 12 + 1]) for i in 1:4, m in 1:12]   # same index = same year
[cor(X[i][m], X[j][m]) for i in 1:4, j in 1:4]               # m = 1, 6

# tech.jl: an uncertainty times a first-stage decision in a constraint is rejected
m = LDRModel(HiGHS.Optimizer)
@variable(m, xi in Uncertainty(Uniform(2, 3)), Stage(2))
@variable(m, 0 <= x <= 10, Stage(1)); @variable(m, y >= 0, Stage(2))
@constraint(m, y <= xi * x); @objective(m, Max, 3y - 2x)
optimize!(m)   # error: "... multiplies the decision x by the uncertainty xi (random recourse) ..."
```

Section 10 (same environment): `kwg` above with three switches, `lag` (the demand of period `t` and the
decisions of period `t` go to `Stage(t + lag)`, so with `lag = 1` the decisions of `t` see only the
demands of `1..t-1`), `cumulative` (the per-factory rows) and `robust` (an epigraph variable `z` in
`Stage(1)`, the row `cost <= z`, minimize `z`):

```julia
# L1(a): kwg(T, 0.3; cumulative = false) -> dual 1972.41, 8664.40, 22764.78 (T = 2, 5, 10)
# L2 / L3: T = 24, theta in (0.025, 0.05, 0.1, 0.2), x0 = 500, lag = 1
#   robust = true  -> 35081.0, 36342.1, 38889.0, 44198.6
#   robust = false -> 33834.7, 33867.3, 34039.8, 34582.6 (dual 33824.7 to 33849.4)
```

## 9. SDDP.jl examples

Local clone `Packages/SDDP.jl` at `6b58ba9c`: every file of `docs/src/examples/` and `docs/src/tutorial/`
read, plus the test files that build concrete models. Columns as in section 1.1; x / u / xi = state
variables / controls / noise components per stage; all problems are LPs unless said; "LDR-suitable?"
means: continuous (or integer in stage 1 only), convex, fixed recourse, no random coefficient on a
decision in a constraint (section 1.1), finite or fittable law, linear policy graph. Values are the
`@test` / `@assert` lines of the file (SDDP bounds with the file's tolerance).

| File | Problem (source) | Data | Horizon / graph | Uncertainty enters (law) | Obj / cons | Integers | x / u / xi | LDR-suitable? | Same as | Known values |
|---|---|---|---|---|---|---|---|---|---|---|
| `examples/FAST_hydro_thermal.jl` | 2-stage hydro, max (FAST) | toy | 2, linear | RHS rainfall (2 atoms) | lin / lin | - | 1/2/1 | yes | - | -10 (det. eq. and bound, :42, :44) |
| `examples/FAST_production_management.jl` | 2-product production / sales (FAST) | toy | 3 (`H` parameter), linear | RHS demand {2, 10} | lin / lin | - | 2/2/1 | yes | - | -23.96 (:40) |
| `examples/FAST_quickstart.jl` | 2-stage buy / sell (FAST) | toy | 2, linear | RHS (bound of `s` in {2, 3}) | lin / lin | - | 1/1/1 | yes | - | -2 (:36, :39) |
| `examples/Hydro_thermal 2seasons.jl` | weekly hydrothermal, 2 seasons | toy | 2 nodes, cyclic | RHS (inflow, demand), 3 joint atoms | quadratic `g^2` (Ipopt) / lin | - | 1/3/2 | with changes: unroll to 52 weeks | one-reservoir toy (QP) | - |
| `examples/Hydro_thermal.jl` | one-reservoir hydrothermal (not the Brazilian one) | toy | 3 nodes, unicyclic (0.95) | RHS (inflow, demand), 3 atoms | lin / lin | - | 1/3/2 | with changes: truncate the cycle to `T` | one-reservoir toy | - |
| `examples/StochDynamicProgramming.jl_multistock.jl` | 3-stock control (StochDynamicProgramming.jl) | toy | 5, linear | RHS, 27 joint atoms | lin / lin | - | 3/3/3 | yes | - | -4.349 (:51) |
| `examples/StochDynamicProgramming.jl_stock.jl` | single stock | toy | 5 (extensible), linear | RHS, 10 atoms | lin / lin | - | 1/1/1 | yes | - | -1.471 (:30, :37) |
| `examples/StructDualDynProg.jl_prob5.2_2stages.jl` | capacity expansion, 4 technologies x 3 demand modes ("prob 5.2") | toy | 2, linear | RHS demand vector, 2 atoms (0.9 / 0.1) | lin / lin | - | 4/17/3 | yes | - | 340315.52 (:62) |
| `examples/StructDualDynProg.jl_prob5.2_3stages.jl` | same, 3 stages | toy | 3, linear | same | lin / lin | - | 4/17/3 | yes | - | 406712.49 (det. eq. and bound, :54, :57) |
| `examples/agriculture_mccardle_farm.jl` | hay farm (McCardle thesis) | thesis data | 4, Markovian with identical rows (= SWI) | **tech**: yield x `acres.in` (a state) | lin / lin | - | 4/15/1 | no in 0.4 (random coefficient on a decision); with acres fixed after stage 1 it is "tech" | - | 4074.1391 (:138) |
| `examples/air_conditioning.jl` | air-conditioner production / storage (Birge & Louveaux 1997) | textbook | 3, linear | RHS demand {100, 300} | lin / lin | **Int, all stages** | 1/3/1 | with changes: drop integrality (the LP has the same value, next row) | close to F3 (overtime instead of backlog) | 62500 (:61) |
| `examples/air_conditioning_forward.jl` | same, convex and integer copies | textbook | 3, linear | same | lin / lin | optional | 1/2/1 | yes (convex copy) | as above | 62500 both (:42, :43) |
| `examples/all_blacks.jl` | deterministic seat selling | toy | 3 | none | lin / lin | Bin, all stages | 2/1/0 | no: deterministic, binary | - | 9.0 (:37) |
| `examples/asset_management_simple.jl` | asset management (Birge & Louveaux) | textbook | 4, Markovian | **rec**: returns x holdings | lin / lin | - | 2/2/1 | no: random recourse | - | 1.514 (:56) |
| `examples/asset_management_stagewise.jl` | same + RHS, objective noise, EAVaR | textbook | 4, Markovian | rec, RHS, obj | lin + risk measure | - | 2/2/2 | no: random recourse, Markov | - | 1.278 (:71) |
| `examples/belief.jl` | partially observable inventory | toy | infinite, belief | RHS, hidden-state probabilities | lin / lin | - | 1/2/1 | no: belief state | - | bound vs simulation (:70) |
| `examples/biobjective_hydro.jl` | cost vs shortage hydro | toy | 3, linear | RHS inflow, 11 atoms | two linear objectives (weighted) | - | 1/5/1 | with changes: one weight = one LP | one-reservoir toy | 10 Pareto points (:49) |
| `examples/booking_management.jl` | hotel booking acceptance | toy | `num_requests` | RHS binary requests | lin / lin | **Bin, all stages** | - | no: binary recourse is the problem | - | 7.25 (:95), 6.13 (:103) |
| `examples/generation_expansion.jl` | 5-unit generation expansion | toy | 5, linear | RHS demand, 8 atoms | lin, penalty 5e5 | **Int investment, all stages** | 5/3/1 | with changes: LP relaxation, or invest in stage 1 only | - | 2.078860e6 (:89) |
| `examples/hydro_valley.jl` | 2-reservoir cascade, piecewise-linear turbines | toy | 3, Markovian price or linear | RHS inflow (3 atoms); **obj** price (Markov) | lin / lin (convex PWL) | - | 2/13/3 | yes without the Markov price; with changes otherwise (joint price law) | - | 835.0 det. (:229), 838.33 stagewise (:234), 851.8 Markov (:239); risk / DRO variants |
| `examples/infinite_horizon_*.jl`, `no_strong_duality.jl` | trivial cyclic models; one-reservoir self-loop | toy | infinite | none / RHS | lin | - | 1/0-3/0-2 | no (cyclic; trivial) | one-reservoir toy (hydro thermal) | 119.167, 20, 20, 2.0 |
| `examples/inner_hydro_1d.jl` | 1 reservoir, 2 thermal, deficit; inner approximation | toy (10 normal samples, seed 2) | 4 (`stages`), linear | RHS inflow, 10 atoms | lin + EAVaR | - | 1/5/1 | yes risk-neutral | one-reservoir toy | lb <= ub only (:75) |
| `examples/objective_state_newsvendor.jl` | newsvendor with AR(1) price | toy | 3, objective state | RHS (11 atoms) x price noise (4) = 44 joint; **obj** price | lin (max) / lin | - | 1/1/2 | with changes: price is affine in past noises, so a random-price model under a joint law | not F1 (other data) | 4.04 (:73, :77) |
| `examples/sldp_example_one.jl` | SLDP example 1 (Ahmed, Cabral, da Costa 2019) | toy | 8, linear | RHS, 10 atoms | lin (abs value) | Bin, all stages | 1/3/1 | with changes: relax `u` | - | <= 1.1675 (:45) |
| `examples/sldp_example_two.jl` | SLDP example 2 (same paper) | toy | 2, linear | RHS, N^2 atoms in R^2 | lin / lin | Int stage 1 (ok), **Bin stage 2** | 2/4/2 | with changes: relax `y` | - | <= -57.0, -59.33, -61.22 (:64-68) |
| `examples/stochastic_all_blacks.jl` | stochastic seat offers | toy | 3 | RHS, 3 atoms | lin / lin | Bin, all stages | 2/2/2 | no: binary recourse | - | 8.0 (:53) |
| `examples/the_farmers_problem.jl` | farmer's problem (Birge & Louveaux 2011, 1.1) | textbook | 2, linear | **tech**: yield x first-stage area (3 atoms) | lin / lin | - | 3/12/1 | not in 0.4 (tech rejected, section 1.1); yes once supported | - | 108390 (`@assert`, :259) |
| `examples/vehicle_location.jl` | ambulance dispatch | toy | 10 | obj (request location), 11 atoms | lin / lin | Bin, all stages | 18/126/1 | no: binary recourse | - | test commented out |
| `tutorial/arma.jl` | hydrothermal with AR(1) / VAR(1) inflows; Markov fit | toy | 3 | RHS noises, 3 or 9 atoms | lin / lin | - | 2-3/3/1-2 | yes for AR / VAR (affine in noises); Markov version with changes | one-reservoir toy | - |
| `tutorial/batteries.jl` | thermal unit + battery, 24 h (Philpott, Kvitfjell 2025) | toy (duck-curve load) | 24, linear | RHS load noise, 5 atoms | lin / lin (ramping) | - | 2/4/1 | yes; 24 stages | - | - |
| `tutorial/decision_hazard.jl` | here-and-now vs wait-and-see hydrothermal | toy | 4-5 | RHS inflow {2, 3} | lin / lin | - | 1-2/3/1 | yes | one-reservoir toy | text only |
| `tutorial/duality_handlers.jl`, `mdps.jl`, `warnings.jl`, `plotting.jl` | tiny MIPs, MDPs, warnings; plotting reuses `markov_uncertainty` | toy | 2-inf | none / Markov | lin; `mdps` (a) QP | Bin / Int in later stages (some) | - | no (demos) | - | 25/3 (mdps (a), printed) |
| `tutorial/example_milk_producer.jl` | milk spot sales and forward contracts | toy (log-AR(1) price simulator) | 12, Markovian (fitted) | RHS production (5 atoms); **obj** price | lin (max) + 0.5 E + 0.5 AVaR | - | 5/3/2 | with changes: price paths as a joint law, risk-neutral | - | - |
| `tutorial/example_newsvendor.jl` | 2-stage newsvendor + hand-written Kelley / L-shaped | toy: make 2, sell 5, hold 0.1, Triangular(150, 250, 200), 100 draws (no seed) | 2 | RHS | lin; CVaR / entropic variants | - | 1/1/1 | yes | **F1 family, other data** | - |
| `tutorial/example_reservoir.jl` | weekly one-reservoir hydrothermal | toy CSV inline (52 weeks) | 52 (rows; truncatable), linear or unicyclic | RHS inflow noise, 3 atoms | lin / lin | - | 1/3/1 | yes; 52 stages | one-reservoir toy (52 stages) | - |
| `tutorial/first_steps.jl` | introductory hydrothermal | toy | 3, linear | RHS inflow {0, 50, 100} | lin / lin | - | 1/3/1 | yes | one-reservoir toy; same data as `test/hydro_thermal.problem.json` | 8333.33 (`test/MSPFormat.jl`:85) |
| `tutorial/inventory.jl` | periodic-review inventory with backlog (Hillier & Lieberman; Porteus) | textbook: c 35, h 1, p 15, x0 10, U(0, 800) as 20 points | `T` + 1 (`T` = 10 parameter); also infinite | RHS demand | lin / lin | - | 3/2/1 | yes (finite version) | **F3 family, other data** | order-up-to 662 (comment, not asserted) |
| `tutorial/markov_uncertainty.jl` | wet / dry climate hydrothermal | toy | 3, Markovian (2 states) | RHS inflow, obj fuel multiplier | lin / lin | - | 1/3/2 | with changes: joint law over the climate | one-reservoir toy | - |
| `tutorial/objective_states.jl` | AR fuel price hydrothermal (Downward, Dowson, Baucke) | toy | 3, objective state | RHS inflow, **obj** price, 12 joint atoms | lin / lin | - | 1/3/2 | with changes: price affine in noises | one-reservoir toy | - |
| `tutorial/objective_uncertainty.jl` | random fuel multiplier hydrothermal | toy | 3, linear | RHS inflow and **obj** multiplier (3 joint atoms) | lin / lin | - | 1/3/2 | yes (random cost) | one-reservoir toy | - |
| `tutorial/pglib_opf.jl` | OPF on PGLib `case5_pjm` + one hydro unit | benchmark grid data (file included) | infinite, unicyclic | RHS inflow {0, 2, 5} | DC: lin; AC: **nonconvex** | - | 1/many/1 | with changes: DC version, finite horizon | - | - |
| `test/MSPFormat.jl` + `hydro_thermal.*.json` | "Simple Hydrothermal Scheduling" (MSPFormat) | toy | 3, lattice with uniform transitions (= SWI) | RHS {0, 50, 100} | lin / lin | - | 1/3/1 | yes | `first_steps` | 8333.3333 (:85, :97) |
| `test/MSPFormat.jl` + `electric.*.json` | 2-stage capacity planning, 4 technologies, demand {3, 5, 7} | toy | 2 | RHS | lin / lin | - | - | yes | - | 381.8533 (:135) |
| `test/Inner.jl`, `test/plugins/bellman_functions.jl` | one-reservoir toy: volume 5..15, x0 = 10, (inflow, demand) in {(0, 7.5), (5, 5), (10, 2.5)} | toy | `T` parameter (2, 4, 20), linear; cyclic copy | RHS, 3 joint atoms | lin / lin | - | 1/3/2 | **yes, with optima at three horizons** | one-reservoir toy | T = 2: 21.39 (Inner.jl:385), T = 4: 45.833 (:161), T = 20: 241.3833 (:353), atol 0.1 |
| `test/algorithm.jl` (`test_root_node_risk_measure`) | continuous air conditioning, demand {100, 300} each stage | textbook | 3 | RHS | lin / lin | - | 1/3/1 | yes | - | expectation 85000 (:473), worst case 107500 (:467) |
| `papers/msppy/msppy_hydro_thermal.jl` (also `papers/biobjective_sddp/brazilian_data.jl`, `papers/entropic/data.json`) | **4-reservoir Brazilian hydrothermal** (Shapiro et al. 2013 via MSPPy) | real | 120 (60 in the two papers) | RHS inflows, 82 joint atoms per month (40 in `entropic`) | lin / lin | - | 4/144/4 | yes | **F11 (same data)** | none asserted (needs Gurobi) |

Notes:

- **Same problems as the catalogue.** The Brazilian hydrothermal of F11 is in `papers/` only (not in
  `docs/`; `docs/src/examples/Hydro_thermal.jl` is a one-reservoir unicyclic toy); the newsvendor
  (`example_newsvendor.jl`) and the inventory (`tutorial/inventory.jl`) are the F1 and F3 families with
  other data; the many one-reservoir hydrothermals are cousins of F8 / F9 (linear cost, no future-cost
  cuts). No unit commitment, no KWG / Ben-Tal inventory.
- **Directly LDR-suitable** (linear graph, continuous, RHS or objective uncertainty, finite law): the three
  FAST examples, the two StochDynamicProgramming.jl stocks, StructDualDynProg prob 5.2 (2 and 3 stages),
  `air_conditioning_forward` (convex copy), `hydro_valley` without the Markov price, `inner_hydro_1d`
  (risk-neutral), `arma` (AR / VAR versions), `batteries` (24 stages), `decision_hazard`,
  `example_newsvendor`, `example_reservoir` (52 stages), `first_steps` / MSPFormat hydrothermal,
  `inventory` (finite), `objective_uncertainty`, the MSPFormat `electric` problem, the `Inner.jl`
  one-reservoir toy and the MSPPy Brazilian model. Their known values are SDDP bounds or deterministic
  equivalents of the discrete-law problem, i.e. the optimum against which `dual <= optimum <= primal`
  can be checked.
- **Not suitable as they stand:** integer recourse (`all_blacks`, `stochastic_all_blacks`,
  `booking_management`, `vehicle_location`, `sldp_example_*`, `generation_expansion`,
  `air_conditioning`), random recourse (`asset_management_*`), random technology coefficients that 0.4
  rejects (`the_farmers_problem`, `agriculture_mccardle_farm`), Markov / belief / objective-state /
  cyclic graphs (`belief`, `markov_uncertainty`, `objective_states`, `example_milk_producer`, the
  infinite-horizon examples, `Hydro_thermal*`, `pglib_opf`), nonconvex AC OPF.

## 10. LDR test problems in the literature

Only pages that were opened are cited ("opened" column; crossref = the `api.crossref.org/works/<doi>`
record, for the bibliographic data). Where only an abstract was available the row says so. KWG 2011 and
GWK 2015 are F4 to F6 (section 6). L-numbers are used in section 11.

| # | Paper | Problem | Horizon | Uncertainty (dim / stage; where) | Obj / cons; integers | Data | Reported results | Opened |
|---|---|---|---|---|---|---|---|---|
| L1 | Bodur, Luedtke, "Two-stage linear decision rules for multi-stage stochastic programming", Math. Program. 191 (2022) 347-380, doi:10.1007/s10107-018-1339-4 | (a) KWG / Ben-Tal inventory **without the cumulative capacity rows** (3 factories, `500 <= s_t <= 2000`, `x <= 567`) | T = 2..10 (formula) | 1; RHS (demand), uniform, seasonal | LP; none | in the paper (F4 data) | static LDR lb / ub: T = 2: 1972.4 / 2026.0; T = 5: 8664.4 / 9021.3; T = 10: 22764.8 / 23738.3 (< 0.02 s); two-stage LDR gap 0.93 to 1.23 % (SAA 250, eval 1e5) | [ar5iv](https://ar5iv.arxiv.org/html/1701.04102), [Opt. Online](https://optimization-online.org/?p=15100), crossref |
| L1 | (same) | (b) capacity expansion of the German system (data of de Maere d'Aertrycke, Shapiro, Smeers 2013): coal, CCGT, OCGT; 8 load periods x 5 wind regimes | T = 5, 10, 15, 20; caps C = 50, 100 | 2 (lognormal demand and wind growth, products over time); RHS through `max{d0 prod xi^g - eta K prod xi^w, 0}` | LP; none | **full data in appendix A** (Tables 6-8: costs, 8 demand levels and hours, 5 wind regimes) | normalized to 2S-LDR lb = 100: static ub 138.0 to 232.9 (T = 5, 10) against 2S 100.8 to 114.6; SDDP lb (same time) -21.4 % to +7.1 % from the 2S-LDR lb; 2S-LDR lb time 171 to 3827 s | (same) |
| L2 | Ben-Tal, Goryashko, Guslitzer, Nemirovski, "Adjustable robust solutions of uncertain linear programs", Math. Program. 99 (2004) 351-376, doi:10.1007/s10107-003-0454-y | production-inventory, 3 factories, one warehouse; the origin of F4 | 24 semimonthly periods | 1; RHS demand in a box, levels 2.5, 5, 10, 20 % | LP (affinely adjustable robust counterpart, worst case); none | pp. 369-370 of the paper (not seen) | worst-case costs (from L3): 35105, 36389, 38990, 44273 | [IAOR abstract](https://www.iaorifors.com/paper/46581), crossref |
| L3 | de Ruiter, Brekelmans, den Hertog, "The impact of the existence of multiple adjustable robust solutions", Math. Program. 160 (2016) 531-545, doi:10.1007/s10107-016-0978-6 | L2's instance re-solved (Gurobi 6.0) | 24 | as L2 | LP | as L2 | multiple robust optima: mean cost differs up to 21.9 %, 59.4 % per realization; LDR minimizing the mean with hard constraints (Table 3): mean 33919, 34031, 34311, 35066 (worst case 35108, 36412, 39040, 44298) | [preprint PDF](https://optimization-online.org/wp-content/uploads/2015/09/5105.pdf), crossref |
| L4 | Ben-Tal, Golany, Nemirovski, Vial, "Retailer-supplier flexible commitments contracts: a robust optimization approach", MSOM 7(3) (2005) 248-271 | single-item inventory with commitments fixed up front and orders affine in past demands | T = 12 (24 also reported; extensible) | 1; RHS demand, box or ellipsoid | LP (box) / one conic row (ellipsoid); none | data appendix referenced, not in the working paper read | T = 12: 1656 variables, 2231 rows; box worst case 53610 against DP 53660 on a relaxed model; > 1000 solves in 10-20 s | (literature subagent) [IAOR](https://www.iaorifors.com/paper/49685), 2003 working paper from the Geneva open archive |
| L5 | Chen, Sim, Sun, "A robust optimization perspective on stochastic programming", Oper. Res. 55(6) (2007) 1058-1071; Chen, Sim, Sun, Zhang, "A linear decision-based approximation approach to stochastic programming", Oper. Res. 56(2) (2008) 344-357 | 2007: project management with uncertain activity times, chance constraints; 2008: a two-stage problem with complete recourse (deflected LDRs) | 2007 multistage; 2008 two-stage | partial distributional information (mean, support, covariance) | SOCP; none | abstracts only seen | not seen | [Duke 2007](https://scholars.duke.edu/publication/801223), [Duke 2008](https://scholars.duke.edu/publication/801220) |
| L6 | Goh, Sim, "Robust optimization made easy with ROME", Oper. Res. 59(4) (2011) 973-985 | service-constrained inventory (fill-rate constraints), deflected rules (BDLDR) | T = 10 | 1; RHS demand, mean known, covariance `sigma L(alpha) L(alpha)'` (AR with one parameter `alpha`, `sigma = 20`), support `[0, zmax]^T` | distributionally robust; SOCP; none | parameters in the paper (c = 1, h, mu, zmax, beta varied; Tables B.1-B.4); MATLAB code said to be at robustopt.com/examples.html (not opened) | basic LDR infeasible at `zmax = 200` even with fill rate 0.5; BDLDR uniformly better | [Opt. Online](https://optimization-online.org/2009/07/2353/), [PDF](https://optimization-online.org/wp-content/uploads/2009/07/2353.pdf) |
| L7 | Bertsimas, Georghiou, "Design of near optimal decision rules in multistage adaptive mixed-integer optimization", Oper. Res. 63(3) (2015) 610-627, doi:10.1287/opre.2015.1365 | (a) inventory with advance (`c_x`) and instant (`c_z`) orders, backlog, cumulative order budget; (b) the same with N binary order blocks | (a) T up to 30; (b) up to 10 | 1; RHS demand; non-box polyhedral set | worst case (epigraph); (a) LP, (b) **binary decisions in later stages** | 25 random instances, ranges in the paper | (a) average gap LDR 23.6 to 52.5 %, two-piece rule 0.4 to 2.9 % (T = 5..30), LDR < 10 s | [Opt. Online](https://optimization-online.org/2013/09/4027/), [PDF](https://optimization-online.org/wp-content/uploads/2013/09/4027.pdf), crossref |
| L8 | Georghiou, Kuhn, Wiesemann, "The decision rule approach to optimization under uncertainty: methodology and applications", Comput. Manag. Sci. 16 (2019) 545-576, doi:10.1007/s10287-018-0338-5 | (a) production planning: 5 products, one machine (R = 168 h/week), backlog, inventory, maximize expected profit; optional sequence-dependent changeovers (binaries) | T flexible ("52 weeks and more"); backtest 52 weeks, rolling T = 4 | 5; RHS demand, independent uniform on `[(1 - theta) d_ti, (1 + theta) d_ti]`, `d_ti = (1 + sin(pi (t - 2) / 26) / 2) d_i` | LP; changeovers are **binaries in stage 1** (supported) | **full data in Tables 1-2** (prices, backlog costs, rates, mean demands, changeover costs and times; the row labels of Table 1 are scrambled in the text extraction) | primal-dual gap < 15 % (LDR), about 5 % (one breakpoint) at theta = 0.2 (figure only) | [preprint PDF (2011)](https://optimization-online.org/wp-content/uploads/2011/12/3290.pdf), crossref |
| L8 | (same) | (b) supply-chain design: London plant, warehouses in the 10 most populous of 40 European capitals (K = 5, c = 40, C = 100), 40 distribution centres | two-stage | 40; RHS demand, uniform on a polytope (sum = 100, each within `(1 +- theta) 100 mu_j`) | LP + **binary location in stage 1** | populations and geodesic distances (derivable, not tabulated; inferred) | installed capacities for theta = 0, 0.5, 1 (Table 3) | (same) |
| L9 | Gauvin, Delage, Gendreau, "Decision rule approximations for the risk averse reservoir management problem", EJOR 261(1) (2017) 317-336, doi:10.1016/j.ejor.2017.01.044 | Gatineau river (Hydro-Quebec): 5 reservoirs, 3 run-of-river plants, delays, spill curves, flow-variation limits | T = 30 days; rolling horizon with 7-day look-ahead | 1 (perfect spatial correlation); RHS inflows, box `mu_t +- 3 sigma_t` | LP, CVaR of flood / drought violations; none | **real, proprietary** historical inflows 1999-2004 (inferred) | affine-rule bounds 2.78e3, 5.57e3, 9.77e3 in 11.7, 68.6, 537 s (CPLEX 12.4); lifted rules up to 88 % better, up to 67 h | (literature subagent) thesis chapter 4, [PolyPublie](https://publications.polymtl.ca/2535/1/2017_CharlesGauvin.pdf) |
| L10 | Gauvin, Delage, Gendreau, "A stochastic program with time series and affine decision rules for the reservoir management problem", EJOR 267(2) (2018) 716-732, doi:10.1016/j.ejor.2017.12.007 | same river, freshet | T = 59 days, rolling L = 30 (first 7 uncertain) | ARMA(1,1)-GARCH(1,1) inflows, affine in independent shocks | convex quadratic flood penalty, SOCP | real, proprietary (in-sample 1999-2004, out-of-sample 2008-2013) | not extracted | (literature subagent) thesis chapter 5 (same PDF) |
| L11 | Rocha, Kuhn, "Multistage stochastic portfolio optimisation in deregulated electricity markets using linear decision rules", EJOR 216(2) (2012) 397-408, doi:10.1016/j.ejor.2011.08.001 | electricity retailer: derivatives portfolio + spot purchases, mean-variance, stage aggregation | not seen | prices and demand (not seen) | convex QP | not seen (abstract only) | not seen | (literature subagent) [IDEAS](https://ideas.repec.org/a/eee/ejores/v216y2012i2p397-408.html) |
| L12 | Lorca, Sun, Litvinov, Zheng, "Multistage adaptive robust optimization for the unit commitment problem", Oper. Res. 64(1) (2016) 32-51, doi:10.1287/opre.2015.1456 | unit commitment with simplified affine dispatch policies | T = 24 h | nodal net load (99 loads on the 118-bus; budget set, deviation 0.1 d); RHS | MILP: **commitment binaries in stage 1** (supported), affine dispatch | **public**: IEEE 118-bus and Polish 2736-bus (MATPOWER) | gap to fully adaptive <= 0.94 %; 47-178 s (118-bus), 0.4-3.6 h (2736-bus) | (literature subagent) [preprint PDF](https://optimization-online.org/wp-content/uploads/2014/09/4527.pdf) |
| L13 | Warrington, Goulart, Mariethoz, Morari, "Policy-based reserves for power systems", IEEE Trans. Power Syst. 28(4) (2013) 4427-4437 (volume and pages not verified) | reserves as affine policies of forecast errors; modified 39-bus network, 7 generators, 2 storage units, 3 wind farms | T = 8 x 15 min, receding over 288 steps | 2 correlated wind errors / step; RHS (power balance) | expected quadratic cost, robust linear constraints (convex QP); none (commitment fixed) | **parameters in Table 1**; load profile from UK national demand data | reserve cost -38.4 % (full policy) against no time coupling; 4.4 s per solve (CPLEX) | (literature subagent) [arXiv 1212.2119](https://arxiv.org/abs/1212.2119v2) |

Notes:

- **Reproductions (run, `bental.jl` in section 8).** L1(a): the package's `PrimalLDR` / `DualLDR` on F4
  without the cumulative rows give dual 1972.41, 8664.40, 22764.78 for T = 2, 5, 10: L1's static lower
  bounds to the printed digit; the primal values 2032.15, 9062.32, 23865.16 are 0.3 to 0.5 % above L1's
  static upper bounds (reason not identified; L1 may report a simulated value of a differently computed
  rule). L2 / L3: with F4's data (x0 = 1000, demand seen in its own stage) the worst-case costs at T = 24
  are 33827 to 42288, far from L2; with **x0 = 500 and the demand of period t seen only by the decisions
  of t + 1** they are 35081, 36342, 38889, 44199 against 35105, 36389, 38990, 44273 (0.07 to 0.26 %), and
  the mean-cost LDR gives 33835, 33867, 34040, 34583 against L3's 33919, 34031, 34311, 35066 (all
  `OPTIMAL`, 1.3 to 3 s): close, not exact, so L2's data (pp. 369-370) differ in a detail that was not
  recovered.
- **Public or in-paper data:** L1 (both problems), L6 (parameters), L7 (ranges, random), L8 (production
  planning fully; supply chain derivable), L12 (MATPOWER cases), L13 (Table 1), and F4 to F6. Proprietary:
  L9, L10 (Hydro-Quebec). Not seen: L2's tables, L4's appendix, L5, L11.
- **Long horizon / real world:** L1(b) (T = 20, real German data), L8(a) (T = 52), L12 (24 h, real grids),
  L9 / L10 (real river), L13 (receding horizon, real load profile).
- **Fit with 0.4:** L1(a), L2, L3, L4 (box), L8(a) and L8(b) are LPs with RHS uncertainty and at most
  first-stage binaries; L2 / L3 / L7 minimize the worst case, which a primal LDR expresses with an
  epigraph variable `z` in stage 1 and the row `cost <= z` (run above), and whose bound needs a point
  mass or is not available; L6 and L5 are distributionally robust (a fixed law gives a different
  problem); L7(b) has later-stage binaries; L1(b) has unbounded lognormal laws and a `max{., 0}` in the
  RHS (needs truncation and an approximation; the cumulative products have linear conditional
  expectations, so the dual can stay certified, inferred); L13 is a QP with robust constraints.
  Gauvin's ARMA-GARCH (L10) is the "fitted law" route of section 1.1 applied to real inflows.

## 11. Candidates to port, and solver comparisons worth running

### 11.1 Proposed benchmark set

Balanced between toys with exact values (tests), problems that scale in `T` (timing and gap as the
horizon grows) and real-world instances. "Exact" means an optimum or a published bound to check
`dual <= optimum <= primal` or equality against.

| Role | Problem | Source | Reason (one line) |
|---|---|---|---|
| toy, exact | Newsvendor (10, 8, 15) and sell = 20, bp 0..3 and the critical breakpoint | F1 (PT, CL) | closed-form optima 3300/7 and 966.67; already in PT except the second price set |
| toy, exact | GWK Examples 4.5 and 5.1 | F6 | published optimum and axial bounds (-0.377 at 10 segments), and a documented limit (no general segmentation) |
| toy, exact | One-reservoir hydro with 25-atom inflow, and `HydroToy2` / `HydroToy3` | F15, F9 (CL) | optimum 92246.31 from two independent codes; all 8 LDR (value, bound) pairs reproduced by 0.4 |
| toy, exact, several `T` | One-reservoir toy of SDDP.jl `test/Inner.jl` (3 joint (inflow, demand) atoms) | section 9 | optimum known at T = 2, 4, 20 (21.39, 45.833, 241.3833): the only toy with exact values along the horizon |
| toy, exact, multi-dimensional | StructDualDynProg prob 5.2, 2 and 3 stages | section 9 | deterministic-equivalent optima 340315.52 / 406712.49 with 3 demand components and 17 controls per stage |
| scalable | KWG / Ben-Tal inventory, T = 1..72, with the variants without cumulative rows (L1(a)) and worst case (L2 / L3) | F4, L1, L2, L3 | published bounds at T <= 10 reproduced, L1's static bounds reproduced, L3's worst case within 0.26 %; one model, three literatures |
| scalable | GKW production planning, 5 products, T up to 52; changeover variant with stage-1 binaries | L8(a) | full data in the paper; 5 uncertainties per stage at T = 52 is the largest published LDR LP with in-paper data, and the changeover version is a MILP the package accepts |
| scalable | GWK multi-product inventory with random prices | F5 | random objective coefficients at scale (the quadratic reformulation); only statistical agreement possible |
| scalable | Inventory with backlog (timing, re-solve) | F3 (PB) | already the timing benchmark; keeps continuity of the timing tables |
| real-world | Brazilian 4-reservoir hydrothermal: historical `MvDiscreteNonParametric` (current), and a fitted periodic AR(1) on the noises | F11, section 1.1 | real data, `T` up to 120 with SDDP.jl comparisons; the fitted variant keeps the serial correlation (0.44 to 0.98) and a certified dual |
| real-world | Capacity expansion of the German system | L1(b) | real data in the paper, T up to 20, and published two-stage-LDR and SDDP bounds; needs a truncated law and a treatment of `max{., 0}` |
| real-world, integer | Unit commitment: LEG 30-bus 36 h, then IEEE 118-bus 24 h | F13, L12 | stage-1 binaries with RHS load uncertainty; public MATPOWER data and a published affine-policy gap (<= 0.94 %) for the larger one |
| real-world, QP | Policy-based reserves, 39-bus | L13 | convex QP with robust linear constraints and all parameters in the paper; the only real-world QP |

Left out on purpose: SDDP.jl's farmer and McCardle farm (random technology coefficients, rejected by 0.4;
the first test problems once "tech" is supported), the integer-recourse and Markov / belief / objective-state
examples (section 9), the distributionally robust L5 / L6 and the proprietary L9 / L10 (no data). The QP
hydros of CL (F10) stay a solver benchmark (item 4 below), not a modelling one.

### 11.2 Porting notes

1. **KWG 2011 inventory (F4), T = 1..10 as tests, T = 24, 48, 72 as a benchmark.** Published lb / ub for T = 1..10
   that the package reproduces to 0.1 at 0.05 s; the benchmark adds a multi-constraint multistage LP with
   box support at a size (72 stages) that the paper solved in 3,519 s, to compare primal against dual and
   HiGHS modes (dual: `ipm` without crossover). The robust version with a point-mass law should give lb = ub.
2. **GWK Examples 5.1 and 4.5 (F6).** Published optimum, axial primal (0) and axial bound (-0.377, 10 segments);
   a two-uncertainty breakpoint test with literature values, and a documented limit (no general segmentation).
3. **CL `HydroToy2` / `HydroToy3` LDR rows (F9).** 0.4 reproduces all 8 (value, bound) pairs exactly (bp 0..3,
   2 and 3 stages); this is the "multistage + breakpoints" test that `Paper milestones.md` lists as missing,
   with a non-box (truncated normal) law and first stages that every variant agrees on.
4. **CL `HydroQuad` / `HydroQuadPw` (F10) as a QP solver benchmark** (with the Ipopt values as the regression
   reference): they separate the solvers (HiGHS `qpasm` stalls or is inexact, HiPO fine on one and
   `OTHER_ERROR` on the other, Ipopt reliable, Clarabel fails the dual at cost scale 1e5 and succeeds at
   1e-3). Worth a test that the inner solve of a dual QP is flagged (not trusted) when HiGHS reports a
   non-optimal status, and an automatic-scaling experiment for conic solvers.
5. **CL EVPI base instance (F15) with a `DiscreteNonParametric` 25-atom inflow.** An exactly known optimum
   (92246.31, EV 90500, WS 92071.07) from two independent implementations, so a test can assert
   `dual <= 92246.31 <= primal` with a non-uniform discrete law, with and without breakpoints.
6. **CL newsvendor with sell = 20 (F1).** Primal 960 / 960 / 960 / 965 against an exact optimum of 966.67, with
   dual bounds 986.67 / 973.33 / 968.89 / 970: a second price set where the primal is not monotone in the
   number of breakpoints (the first set is already in PT); five numbers, instant.
7. **GWK section 7 inventory (F5), T = 2..30, as a benchmark.** Random prices make the objective bilinear
   in the rules (random objective coefficients, not random recourse), exercise the quadratic
   reformulation at scale, and the paper gives the gap of LDR and AS5 (28 % to 43 % and 24 % to 29 %);
   only statistical agreement is possible (25 random instances of `alpha`), and the layout of (16) needs
   to be fixed first.
8. **LEG unit commitment (F13) as an integer first-stage benchmark.** The only MILP-with-LDR instance outside
   `newsvendor(integer = true)`; HiGHS with `mip_rel_gap` 0.01; no known value, so it is a timing and gap
   problem. L12's IEEE 118-bus case (24 h, 99 loads) is the next size, with public data.
9. **L1(a) as tests next to F4.** Three dual values (1972.41, 8664.40, 22764.78) equal to the paper; the
   variant is F4 with `cumulative = false`, so it costs one keyword. The worst-case version of L2 / L3
   needs the 2004 data (pp. 369-370) before its values can be asserted: x0 = 500 with one period of
   delay is within 0.26 %, not exact.
10. **L8(a) production planning.** Read Tables 1-2 from the published version (the preprint's extraction
    scrambles the row labels of Table 1); `T` and `theta` free; compare the LDR and one-breakpoint gaps
    with the figure (< 15 %, about 5 % at theta = 0.2). The changeover version adds `O(T |I|^2)` stage-1
    binaries and big-M rows: a MILP benchmark with RHS uncertainty only.
11. **F11 with a fitted law.** Per month and reservoir, fit `mu_m`, `sigma_m`, `phi_m` on the 82 records
    (indices are years); declare independent standardized noises per stage (truncated normal, or the 82
    empirical residuals as a discrete law) and write the inflows as affine expressions; compare value,
    bound and out-of-sample cost on the historical years against the current stagewise-independent model.
12. **SDDP.jl `Inner.jl` toy.** Three horizons with known optima from a 3-atom joint law: a cheap test that
    `objective_bound <= optimum <= objective_value` for `T = 2, 4, 20`, with and without breakpoints.

Solver comparisons worth running (none run here beyond the sanity runs above): (a) HiGHS `qpasm`, `ipm`,
Ipopt, Clarabel scaled and unscaled on the two QP hydros, primal and dual, bp 0..3, recording status, relative
error against Ipopt and time; (b) HiGHS dual simplex against `ipm` without crossover on the KWG inventory at
T = 24 to 72 (primal and dual); (c) Clarabel against HiGHS on the hydro LP dual with and without objective
scaling, including the 4-reservoir T = 4 benchmark; (d) a conic solver (Clarabel; MOSEK or Gurobi if licensed)
on `ConfidenceMvNormal` at larger dimension, since nothing in the three code bases compares SOCP solvers.
