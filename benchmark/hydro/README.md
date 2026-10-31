# Hydrothermal benchmark: a larger stage dimension

The 4-reservoir, 5-subsystem hydrothermal problem of Shapiro, Tekaya, da Costa and Soares (2013),
with data from the MSPPy / SDDP.jl example (`test/problems/hydro_data.jl`, copied from
`examples/recursion/recursion.jl` of the previous repository). Each stage reveals 4 joint inflows
(82 equiprobable scenarios per month, independent across stages), so the stage dimension is 4,
against 1 in the inventory benchmark. Horizons T = 4, 8, 12. The models are in
`test/problems/hydro.jl`, which the tests share, one function per model:

| Model | Function | What it is |
|---|---|---|
| LDR | `ldr_model(data, T)` | one monolithic multistage model; `PrimalLDR` (value) and `DualLDR` (bound) |
| average | `average_scenario_model(data, T)` | plain JuMP, every inflow at its mean |
| tree | `scenario_tree_model(data, T, branches)` | plain JuMP, `branches` joint inflow vectors per stage (all four reservoirs branch together), drawn from the same law with a fixed seed, equal probabilities, one copy of the stage variables per node: `branches^(T-1)` leaves |

Stage convention (that of the previous repository's `monolithic_vs_sddp.jl`): stage t is month t,
its decisions are taken after the inflows of month t are known, stage 1 has the known inflows
`inflow_initial`, so there are T - 1 uncertain vectors (`Uncertainty` with `Stage(t)`, t = 2:T).

```
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/hydro/run.jl                  # T = 4, 8, 12
julia --project=benchmark benchmark/hydro/run.jl 4 tree=3 only=ldr_primal,tree
julia --project=benchmark benchmark/hydro/run.jl 4 bp=0,1,2   # k breakpoints per inflow
# `only=ldr_dual_ipm` runs DualLDR with attributes ["solver" => "ipm", "run_crossover" => "off"]
```

`run.jl` appends one row per T and variant to `../results/hydro.csv`, with the columns of
`timing.jl` plus `branches`, `reps`, the inner solver's `status`, and `breakpoints`, `oos_mean`,
`oos_stderr` and `oos_violated` (the last three only for primal LDR variants: see
[Breakpoints](#breakpoints)). A variant is run once for warm-up, then three times with the best
kept (smallest `build_s + optimize_s`), unless its first run takes more than 10 s, in which case
that run is the result (`reps = 1`). The recordings are named by their `src_commit`, as in the
[benchmark README](../README.md#running).

## The syntax, side by side

Stage t of the LDR model (the variables and constraints come from the helpers `stage_variables`
and `stage_constraints`, shared by the three models, the stage of each variable group is set with
`LDR.set_stage.(x, t)`):

```julia
m = LDRModel(HiGHS.Optimizer)                       # m = Model(HiGHS.Optimizer)
for t in 1:T
    v = stage_variables(m, data, t)
    for x in (v.spill, v.hydro, v.thermal, v.exchange, v.deficit, v.stored)
        LDR.set_stage.(x, t)                        # (nothing)
    end
    inflow = t == 1 ? data.inflow_initial :
             @variable(m, [1:4] in Uncertainty(inflow_law(data, t)), Stage(t))
    stage_constraints(m, data, t, v, stored_in, inflow)   # water balance: stored[i] + ... == inflow[i]
    ...
end
```

The average-scenario model replaces the `@variable(... in Uncertainty(law), Stage(t))` line by
`Distributions.mean(inflow_law(data, t))` and uses `Model(HiGHS.Optimizer)`. The tree has no
uncertainty at all: each node is a call that creates its own copy of the stage variables with its
own inflow vector and its children:

```julia
function add_node!(t, stored_in, inflow, probability, name)
    v = stage_variables(m, data, t; name)
    stage_constraints(m, data, t, v, stored_in, inflow)
    add_to_expression!(cost, probability, stage_cost(data, v))
    t == T && return
    for b in 1:branches                              # one child per sampled inflow vector
        add_node!(t + 1, v.stored, next_inflow(t, b), probability / branches, "$(name)_$b")
    end
end
```

The law of the LDR model is `MvDiscreteNonParametric(atoms, p)` over the 82 joint scenarios of the
month; the tree draws `branches` of the same atoms without replacement (`tree_scenarios`).

## Current

Recorded at 7dbebf0 (rows with `commit` 901bd78): `run.jl 4 8 12
only=ldr_primal,ldr_dual,ldr_dual_ipm`, Windows 11, Julia 1.12.6, HiGHS 1.15.1 (default options
unless stated). The average and tree models do not use the package: their rows are those of 5f20d11
(see History). Times in seconds: build is the JuMP model plus everything `optimize!` does besides
the inner solve (`build_s + optimize_s - solver_s`: for the LDR the canonical form, the moments and
the reformulation, for the others the copy to HiGHS); solve is the inner solver's time. Sizes are
those of the model that is solved (constraints without variable bounds). The tree has 3 branches
at T = 4 (27 leaves) and 2 at T = 8 and 12 (128 and 2048 leaves). The costs are in the units of the
data (thousands of dollars, as in the SDDP.jl example).

| T | Model | Variables | Constraints | Nonzeros | Objective | Build | Solve | Runs |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 4 | LDR primal (value) | 11,152 | 8,136 | 33,644 | 3.759985 | 0.010 | 0.270 | 3 |
| 4 | LDR dual (bound) | 12,904 | 23,028 | 99,344 | 0.980316 | 0.016 | 3.385 | 3 |
| 4 | LDR dual, ipm without crossover | 12,904 | 23,028 | 99,344 | 0.980316 | 0.023 | 0.552 | 3 |
| 4 | average scenario | 592 | 36 | 680 | 0.980316 | 0.0030 | 0.0011 | 3 |
| 4 | tree, 3 branches | 5,920 | 360 | 6,836 | 1.036250 | 0.0181 | 0.0112 | 3 |
| 8 | LDR primal (value) | 50,464 | 35,828 | 153,264 | 94.874564 | 0.044 | 10.9 | 1 |
| 8 | LDR dual (bound) | 54,552 | 103,280 | 453,732 | 1.960632 | 0.076 | 115.7 | 1 |
| 8 | LDR dual, ipm without crossover | 54,552 | 103,280 | 453,732 | 1.960632 | 0.092 | 5.34 | 3 |
| 8 | average scenario | 1,184 | 72 | 1,364 | 1.960632 | 0.0049 | 0.0022 | 3 |
| 8 | tree, 2 branches | 37,740 | 2,295 | 43,601 | 1.984511 | 0.13 | 0.13 | 3 |
| 12 | LDR primal (value) | 117,936 | 82,784 | 358,580 | 468.156822 | 0.100 | 91.6 | 1 |
| 12 | LDR dual (bound) | 124,360 | 240,172 | 1,062,008 | 10.753064 | 0.226 | 142.1 | 1 |
| 12 | LDR dual, ipm without crossover | 124,360 | 240,172 | 1,062,008 | NaN (status OTHER_ERROR) | 0.219 | 25.6 | 1 |
| 12 | average scenario | 1,776 | 108 | 2,048 | 10.386167 | 0.0059 | 0.0032 | 3 |
| 12 | tree, 2 branches | 606,060 | 36,855 | 700,241 | 13.695625 | 3.44 | 55.7 | 1 |

The `ipm without crossover` rows use `DualLDR(; attributes = ["solver" => "ipm", "run_crossover" =>
"off"])`, the recommended dual mode. It is 6 times faster than the default at T = 4 and 22 times at
T = 8, with the same bound (to 1e-11). At T = 12 it is 5.5 times faster, but HiGHS ends with status
`OTHER_ERROR`; an inner solve ending in an error gives no bound, so the value is `NaN` (see
History).

### Observations

The primal-dual gap is large: the primal LDR values 3.76, 94.87 and 468.16 sit far above the dual
bounds 0.98, 1.96 and 10.75 at T = 4, 8 and 12 (the bound is 26%, 2.1% and 2.3% of the value), and
the relative gap goes from 74% at T = 4 to about 98% at T = 8 and 12. Both the dual LDR and the
average-scenario model are lower bounds on the optimal value: the inflows enter only the
right-hand sides, so the cost-to-go is convex in them, and with stagewise independence Jensen's
inequality gives average <= optimum (this fails with random costs or matrix coefficients, or with
integers), while the dual bound is the one of Kuhn, Wiesemann and Georghiou (2011). They are equal
to 10 digits at T = 4 and 8 (0.9803, 1.9606); at T = 12 the dual is 3.5% higher (10.753 against
10.386), so neither dominates. The tree's value (1.036, 1.985, 13.696) is the optimum of one
sampled problem: an in-sample SAA estimate, neither a lower nor an upper bound on the true optimum
(here it is above the average value and far below the primal LDR, which is an upper bound).

The primal LDR grows roughly quadratically in variables (11k, 50k, 118k), since a stage-t decision
has 1 + 4(t - 1) coefficients, and its solve time grows faster, 0.27 s, 10.9 s and 91.6 s, while
the reformulation stays at or below 0.1 s. The dual has 99k, 454k and 1.06M nonzeros, and its solve
time is 3.4 s, 115.7 s and 142.1 s (not monotone in size, as often with HiGHS). The tree's solve
time is 0.01 s, 0.13 s and 55.7 s for 27, 128 and 2048 leaves, so at T = 12 its 606k variables
cost about 0.6 times the primal LDR's solve time. The simulated primal policy at T = 12 costs
468.30 ± 2.30 on 2000 scenarios of the true law, with no violated constraint.

### Breakpoints

`run.jl 4 bp=0,1,2` and `run.jl 8 bp=0,1` add the variants `ldr_primal_bp<k>` and `ldr_dual_bp<k>`: `k`
breakpoints on each inflow, at the equiprobable quantiles of its law. `BreakPoints` apply to
univariate declarations only and the law of a month is multivariate (82 joint scenarios), so these
variants declare the four inflows of a stage as independent marginals; `k = 0` is the same law with
no breakpoints, and its primal value is that of the joint law (the cost is linear, so only the means
enter). Every primal run is also simulated on 2000 scenarios of the true joint law (one of the 82
vectors per stage, independent across stages): `oos_mean` is the sample mean of the cost, and no
constraint is violated in any of them.

The table was recorded at 27b63fa, with the previous recording, 5f20d11, in parentheses (values
unchanged). The machine was about 1.6 times slower that day: the dual with k = 1 at T = 4, the
same model at both commits, took 10.3 s against 6.0 s at 7dbebf0. The primal is smaller since then:
the closed form of box supports (7dbebf0) applies with k = 0, and with k ≥ 1 the primal leaves out
of the support the bounds of the pieces that their convex-hull rows imply (`lifted_box`, see
[Past studies](../README.md#redundant-box-rows-of-liftings-october-2026-lifted_box)).

| T | k | Primal value | Out-of-sample mean | Dual bound | Primal variables | Primal solve | Dual solve |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 0 | 3.7600 | 3.7587 ± 0.0124 | 0.9803 | 11,152 (18,160) | 0.53 s (0.58) | 2.4 s (1.3) |
| 4 | 1 | 1.8888 | 1.8875 ± 0.0150 | 0.9891 | 28,720 (42,736) | 2.7 s (2.2) | 10.3 s (6.0) |
| 4 | 2 | 1.5962 | 1.5991 ± 0.0135 | 0.9912 | 39,280 (67,312) | 6.4 s (5.4) | 53 s (33) |
| 8 | 0 | 94.875 | 95.584 ± 0.572 | 1.9606 | 50,464 (83,168) | 18.1 s (13.5) | 103 s (65) |
| 8 | 1 | 37.296 | 37.717 ± 0.548 | 1.9703 | 132,448 (197,856) | 104 s (127) | 105 s (68) |

One breakpoint per inflow halves the primal value at T = 4 (3.76 to 1.89; two give 1.60) and cuts it by
61% at T = 8, while the dual bound moves by 1% or less, so the bound goes from 26% to 52% (62% with two)
of the value at T = 4 and from 2.1% to 5.3% at T = 8. The primal LP has 2.6 times the variables with
one breakpoint and its solve takes 5.2 times as long at T = 4 and 5.8 times at T = 8 (104 s, under
the 3-minute limit of these runs). The out-of-sample means are within 1.3 standard errors of the
primal values.

## Against 0.3

T = 12, no breakpoints. Previous repository, `examples/recursion/results/ldr_T12.csv` (k = 0,
HiGHS) and `spike/dualnnz/sizes.csv` (`cur_*` columns); the times of the previous version come
from another run and are indicative only.

| T = 12 | Variables | Constraints | Nonzeros | Objective | Build | Solve |
|---|--:|--:|--:|--:|--:|--:|
| legacy primal | 356,208 | 166,044 | 758,328 | 468.156821622 | 5.2 | 103.3 |
| 0.4 `PrimalLDR` (5f20d11) | 195,024 | 82,784 | 435,668 | 468.156821622 | 0.1 | 128.2 |
| 0.4 `PrimalLDR` (7dbebf0) | 117,936 | 82,784 | 358,580 | 468.156821622 | 0.1 | 91.6 |
| legacy dual (dense cone) | 198,528 | 470,892 | 14,161,128 | 10.753064193 | n/a | 230.3 |
| 0.4 `DualLDR` (5f20d11) | 124,360 | 240,172 | 1,062,008 | 10.753064193 | 0.2 | 147.8 |

(Legacy build is `build_s + canonicalize_s` of `ldr_T12.csv`; the legacy rows' solve is the
inner solver's time.) The values agree to 12 digits, so this is the same instance and the same
reformulated problem. The 0.4 primal had 55% of the variables and 57% of the nonzeros before the
closed form of box supports, and has 33% and 47% with it; the dual has 13 times fewer nonzeros than
the dense one.

## History

**7dbebf0 against 5f20d11: the closed form of box supports.** The primal has fewer variables; the
dual's sizes and all objectives are identical. LDR rows only, 7dbebf0 then 5f20d11 in parentheses:

| T | Model | Variables | Nonzeros | Build | Solve |
|--:|---|--:|--:|--:|--:|
| 4 | LDR primal | 11,152 (18,160) | 33,644 (40,652) | 0.010 (0.0104) | 0.270 (0.57) |
| 4 | LDR dual | 12,904 | 99,344 | 0.016 (0.0182) | 3.385 (3.45) |
| 4 | LDR dual, ipm | 12,904 | 99,344 | 0.023 (0.0176) | 0.552 (0.56) |
| 8 | LDR primal | 50,464 (83,168) | 153,264 (185,968) | 0.044 (0.19) | 10.9 (13.6) |
| 8 | LDR dual | 54,552 | 453,732 | 0.076 (0.12) | 115.7 (116.9) |
| 8 | LDR dual, ipm | 54,552 | 453,732 | 0.092 (0.098) | 5.34 (5.4) |
| 12 | LDR primal | 117,936 (195,024) | 358,580 (435,668) | 0.100 (0.135) | 91.6 (128.2) |
| 12 | LDR dual | 124,360 | 1,062,008 | 0.226 (0.23) | 142.1 (147.8) |
| 12 | LDR dual, ipm | 124,360 | 1,062,008 | 0.219 (0.24) | 25.6 (25.8) |

Here the smaller primal solves faster: 2.1 times at T = 4, 1.25 at T = 8 and 1.4 at T = 12 (the last
two are single runs), with 61%, 61% and 60% of the variables and 83%, 82% and 82% of the nonzeros.
(In the inventory benchmark the primal also loses 36% of its variables at T = 128 and the solve is
16% slower, so the size does not predict the HiGHS time.) The dual solve times are within 4%.

**5f20d11 against 6307590.** `run.jl 4 8 12`, same machine, options and protocol. "Previous solve"
is the solve time of 6307590. Sizes are identical between the two recordings, as are the
objectives (to 1e-12 relative for the LPs).

| T | Model | Variables | Constraints | Nonzeros | Objective | Build | Solve | Previous solve | Runs |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | LDR primal (value) | 18,160 | 8,136 | 40,652 | 3.759985 | 0.0104 | 0.57 | 0.57 | 3 |
| 4 | LDR dual (bound) | 12,904 | 23,028 | 99,344 | 0.980316 | 0.0182 | 3.45 | 3.45 | 3 |
| 4 | LDR dual, ipm without crossover | 12,904 | 23,028 | 99,344 | 0.980316 | 0.0176 | 0.56 | | 3 |
| 4 | average scenario | 592 | 36 | 680 | 0.980316 | 0.0030 | 0.0011 | 0.0011 | 3 |
| 4 | tree, 3 branches | 5,920 | 360 | 6,836 | 1.036250 | 0.0181 | 0.0112 | 0.0115 | 3 |
| 8 | LDR primal (value) | 83,168 | 35,828 | 185,968 | 94.874564 | 0.19 | 13.6 | 13.3 | 1 |
| 8 | LDR dual (bound) | 54,552 | 103,280 | 453,732 | 1.960632 | 0.12 | 116.9 | 118.8 | 1 |
| 8 | LDR dual, ipm without crossover | 54,552 | 103,280 | 453,732 | 1.960632 | 0.098 | 5.4 | | 3 |
| 8 | average scenario | 1,184 | 72 | 1,364 | 1.960632 | 0.0049 | 0.0022 | 0.0024 | 3 |
| 8 | tree, 2 branches | 37,740 | 2,295 | 43,601 | 1.984511 | 0.13 | 0.13 | 0.14 | 3 |
| 12 | LDR primal (value) | 195,024 | 82,784 | 435,668 | 468.156822 | 0.135 | 128.2 | 130.5 | 1 |
| 12 | LDR dual (bound) | 124,360 | 240,172 | 1,062,008 | 10.753064 | 0.23 | 147.8 | 151.8 | 1 |
| 12 | LDR dual, ipm without crossover | 124,360 | 240,172 | 1,062,008 | 10.75337 (status OTHER_ERROR) | 0.24 | 25.8 | | 1 |
| 12 | average scenario | 1,776 | 108 | 2,048 | 10.386167 | 0.0059 | 0.0032 | 0.0036 | 3 |
| 12 | tree, 2 branches | 606,060 | 36,855 | 700,241 | 13.695625 | 3.44 | 55.7 | 64.6 | 1 |

The 6307590 builds: T = 4: 0.0096, 0.0184, 0.0028, 0.0194; T = 8: 0.037, 0.077, 0.0045, 0.14;
T = 12: 0.13, 0.44, 0.0064, 3.84 (primal, dual, average, tree). Solve times agree within 2% for the
LPs that were solved to optimality in about the same way (the runs with `Runs = 1` vary by a few
percent between recordings; the tree at T = 12 was 14% faster, a single run). The T = 8 primal build
of 0.19 s (0.037 s before) is one run with a garbage-collection pause, not repeated.

At 5f20d11 the T = 12 ipm run ended with status `OTHER_ERROR` and a bound 3e-5 (relative) above the
simplex one (10.753370 against 10.753064; the dual LP has one optimal value, so the difference is
the inaccuracy of the unfinished ipm run). That bound was invalid (above the optimum of a
minimization): the package was then fixed so that an inner solve ending in an error gives no bound
(`NaN`); only a time or iteration limit keeps the dual objective as a bound.
