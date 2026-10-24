# Hydrothermal benchmark: a larger stage dimension

The 4-reservoir, 5-subsystem hydrothermal problem of Shapiro, Tekaya, da Costa and Soares (2013),
with data from the MSPPy / SDDP.jl example (`data.jl`, copied from `examples/recursion/recursion.jl`
of the previous repository). Each stage reveals 4 joint inflows (82 equiprobable scenarios per
month, independent across stages), so the stage dimension is 4, against 1 in the inventory benchmark.
Horizons T = 4, 8, 12; `model.jl` has one function per model:

| Model | Function | What it is |
|---|---|---|
| LDR | `ldr_model(data, T)` | one monolithic multistage model; `PrimalLDR` (value) and `DualLDR` (bound) |
| average | `average_scenario_model(data, T)` | plain JuMP, every inflow at its mean |
| tree | `scenario_tree_model(data, T, branches)` | plain JuMP, `branches` joint inflow vectors per stage (all four reservoirs branch together), drawn from the same law with a fixed seed, equal probabilities, one copy of the stage variables per node: `branches^(T-1)` leaves |

Stage convention (that of the previous repository's `monolithic_vs_sddp.jl`): stage t is month t,
its decisions are taken after the inflows of month t are known, stage 1 has the known inflows
`inflow_initial`, so there are T - 1 uncertain vectors (`Uncertainty` with `Stage(t)`, t = 2:T).

```
julia --project=benchmark/hydro -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
julia --project=benchmark/hydro benchmark/hydro/run.jl                  # T = 4, 8, 12
julia --project=benchmark/hydro benchmark/hydro/run.jl 4 tree=3 only=ldr_primal,tree
julia --project=benchmark/hydro benchmark/hydro/run.jl 4 bp=0,1,2   # k breakpoints per inflow
```

`run.jl` appends one row per T and variant to `../results/hydro.csv`, with the columns of
`timing.jl` plus `branches`, `reps`, the inner solver's `status`, and `breakpoints`, `oos_mean`,
`oos_stderr` and `oos_violated` (the last three only for primal LDR variants: see below). A variant is run once for
warm-up, then three times with the best kept (smallest `build_s + optimize_s`), unless its first
run takes more than 10 s, in which case that run is the result (`reps = 1`).

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

## Results

Windows 11, Julia 1.12.6, HiGHS 1.26.1 (default options), one run of `run.jl 4 8 12`
(the `commit` column of `results/hydro.csv` is the commit of `src/`). Times in seconds: build is
the JuMP model plus everything `optimize!` does besides the inner solve (`build_s + optimize_s - solver_s`: for the LDR the canonical form, the moments and the reformulation, for the others the copy to HiGHS); solve is
the inner solver's time. Sizes are those of the model that is solved (constraints without variable
bounds). The tree has 3 branches at T = 4 (27 leaves) and 2 at T = 8 and 12 (128 and 2048 leaves).

| T | Model | Variables | Constraints | Nonzeros | Objective | Build | Solve | Runs |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 4 | LDR primal (value) | 18,160 | 8,136 | 40,652 | 3.759985 | 0.0096 | 0.57 | 3 |
| 4 | LDR dual (bound) | 12,904 | 23,028 | 99,344 | 0.980316 | 0.0184 | 3.45 | 3 |
| 4 | average scenario | 592 | 36 | 680 | 0.980316 | 0.0028 | 0.0011 | 3 |
| 4 | tree, 3 branches | 5,920 | 360 | 6,836 | 1.036250 | 0.0194 | 0.0115 | 3 |
| 8 | LDR primal (value) | 83,168 | 35,828 | 185,968 | 94.874564 | 0.0372 | 13.3 | 1 |
| 8 | LDR dual (bound) | 54,552 | 103,280 | 453,732 | 1.960632 | 0.0766 | 118.8 | 1 |
| 8 | average scenario | 1,184 | 72 | 1,364 | 1.960632 | 0.0045 | 0.0024 | 3 |
| 8 | tree, 2 branches | 37,740 | 2,295 | 43,601 | 1.984511 | 0.14 | 0.14 | 3 |
| 12 | LDR primal (value) | 195,024 | 82,784 | 435,668 | 468.156822 | 0.13 | 130.5 | 1 |
| 12 | LDR dual (bound) | 124,360 | 240,172 | 1,062,008 | 10.753064 | 0.44 | 151.8 | 1 |
| 12 | average scenario | 1,776 | 108 | 2,048 | 10.386167 | 0.0064 | 0.0036 | 3 |
| 12 | tree, 2 branches | 606,060 | 36,855 | 700,241 | 13.695625 | 3.84 | 64.6 | 1 |

The costs are in the units of the data (thousands of dollars, as in the SDDP.jl example).

### Against the previous version (T = 12, no breakpoints)

Previous repository, `examples/recursion/results/ldr_T12.csv` (k = 0, HiGHS) and
`spike/dualnnz/sizes.csv` (`cur_*` columns); the times of the previous version come from another
run and are indicative only.

| T = 12 | Variables | Constraints | Nonzeros | Objective | Build | Solve |
|---|--:|--:|--:|--:|--:|--:|
| legacy primal | 356,208 | 166,044 | 758,328 | 468.156821622 | 5.2 | 103.3 |
| 0.4 `PrimalLDR` | 195,024 | 82,784 | 435,668 | 468.156821622 | 0.1 | 130.5 |
| legacy dual (dense cone) | 198,528 | 470,892 | 14,161,128 | 10.753064193 | n/a | 230.3 |
| 0.4 `DualLDR` | 124,360 | 240,172 | 1,062,008 | 10.753064193 | 0.4 | 151.8 |

(Legacy build is `build_s + canonicalize_s` of `ldr_T12.csv`; the legacy rows' solve is the
inner solver's time.) The values agree to 12 digits, so this is the same instance and the same
reformulated problem; the 0.4 primal has 55% of the variables and 57% of the nonzeros, and the dual
has 13 times fewer nonzeros than the dense one.

### Breakpoints

`run.jl 4 bp=0,1,2` and `run.jl 8 bp=0,1` add the variants `ldr_primal_bp<k>` and `ldr_dual_bp<k>`: `k`
breakpoints on each inflow, at the equiprobable quantiles of its law. `BreakPoints` apply to
univariate declarations only and the law of a month is multivariate (82 joint scenarios), so these
variants declare the four inflows of a stage as independent marginals; `k = 0` is the same law with
no breakpoints, and its primal value is that of the joint law (the cost is linear, so only the means
enter). Every primal run is also simulated on 2000 scenarios of the true joint law (one of the 82
vectors per stage, independent across stages): `oos_mean` is the sample mean of the cost, and no
constraint is violated in any of them.

| T | k | Primal value | Out-of-sample mean | Dual bound | Primal variables | Primal solve | Dual solve |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 0 | 3.7600 | 3.7587 ± 0.0124 | 0.9803 | 18,160 | 0.60 s | 1.3 s |
| 4 | 1 | 1.8888 | 1.8875 ± 0.0150 | 0.9891 | 42,736 | 2.2 s | 6.0 s |
| 4 | 2 | 1.5962 | 1.5991 ± 0.0135 | 0.9912 | 67,312 | 5.4 s | 33 s |
| 8 | 0 | 94.875 | 95.584 ± 0.572 | 1.9606 | 83,168 | 13.3 s | 64 s |
| 8 | 1 | 37.296 | 37.717 ± 0.548 | 1.9703 | 197,856 | 124 s | 66 s |

One breakpoint per inflow halves the primal value at T = 4 (3.76 to 1.89; two give 1.60) and cuts it by
61% at T = 8, while the dual bound moves by 1% or less, so the bound goes from 26% to 52% (62% with two)
of the value at T = 4 and from 2.1% to 5.3% at T = 8. The primal LP has 2.4 times the variables with
one breakpoint and its solve takes 3.7 times as long at T = 4 and 9.3 times at T = 8 (124 s, under the
3-minute limit of these runs). The out-of-sample means are within 1.3 standard errors of the primal
values.

## Observations

The primal-dual gap is large: the primal LDR values 3.76, 94.87 and 468.16 sit far above the dual
bounds 0.98, 1.96 and 10.75 at T = 4, 8 and 12 (the bound is 26%, 2.1% and 2.3% of the value), and the
relative gap goes from 74% at T = 4 to about 98% at T = 8 and 12. Both the dual LDR and the average-scenario
model are lower bounds on the optimal value: the inflows enter only the right-hand sides, so the
cost-to-go is convex in them, and with stagewise independence Jensen's inequality gives
average <= optimum (this fails with random costs or matrix coefficients, or with integers), while
the dual bound is the one of Kuhn, Wiesemann and Georghiou (2011). They are equal to 10 digits at
T = 4 and 8 (0.9803, 1.9606); at T = 12 the dual is 3.5% higher (10.753 against 10.386), so neither
dominates. The tree's value (1.036, 1.985, 13.696) is the optimum of one sampled problem: an in-sample
SAA estimate, neither a lower nor an upper bound on the true optimum (here it is above the average
value and far below the primal LDR, which is an upper bound). The primal LDR grows roughly
quadratically in variables (18k, 83k, 195k), since a stage-t decision has 1 + 4(t - 1) coefficients,
and its solve time grows faster, 0.57 s, 13.3 s, 130.5 s, while the reformulation stays below 0.2 s;
the dual has 100k, 454k and 1.06M nonzeros, its solve time is 3.4 s, 118.8 s and 151.8 s (not
monotone in size, as often with HiGHS), and the tree's solve time is 0.01 s, 0.14 s and 64.6 s for
27, 128 and 2048 leaves, so at T = 12 its 606k variables cost about half the primal LDR's solve time.
