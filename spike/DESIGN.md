# Design of LinearDecisionRules.jl 0.4

This document freezes the plan agreed by the authors after the Phase 0 spike (October 2026).
It replaces the 0.1–0.3 implementation, which is kept in the `legacy` branch and used only as a
source of reference values. Changes to this design should be discussed before they are coded.

## Goals

The package keeps the mathematics of the previous versions — primal and dual linear decision
rules (Kuhn, Wiesemann and Georghiou, 2011), axis-aligned piecewise liftings (a special case of
Georghiou, Wiesemann and Kuhn, 2015), a sampled formulation, stages — and rebuilds the pipeline
around one typed canonical form, so that each planned feature is a small addition.

The lessons that drive the design, measured on the previous version:

- **Building the model costs more than solving it.** Inventory benchmark, T = 128: the primal LP
  had 223k variables against 58k in a hand-written model.
- **The dual exploded in multistage models**, because `S M W'ᵀ` was dense: 300M nonzeros on the
  12-stage hydro instance with 3 breakpoints, 2.6B with 24 stages. The structured dual below has
  13–40× fewer nonzeros on the same instances, with identical objectives (spike, `dualnnz`).
- **Implicit syntax was error-prone:** a constraint without decisions silently conditioned the law.
- **Multistage was untested in CI.**

## Pipeline

```
user JuMP.Model ──► LinearDecisionRules.Optimizer ──► CanonicalForm ──► transforms ──► moments
                                                          (information structure, lifting, support blocks)
                    ──► one reformulated model per formulation (PrimalLDR, DualLDR, SampledLDR) ──► results
```

Every stage is a typed struct with documented fields; nothing is stored in `ext` dictionaries.

## User layer: an MOI optimizer (decision 1)

The user writes a plain `JuMP.Model`; `LDRModel(optimizer) = Model(() -> Optimizer(optimizer);
add_bridges = false)`. `Optimizer <: MOI.AbstractOptimizer` implements `MOI.optimize!(dest, src)`
and reads JuMP's cache directly. It was chosen over a wrapper `AbstractModel` (forwarding of every
JuMP function, `MethodError` for the rest, third-party builders typed `::JuMP.Model` do not compose)
and over an optimize hook (results through a test-utility mock optimizer and internal fields).

- **No bridges.** The optimizer supports natively exactly what the extraction reads: affine and
  quadratic rows in `LessThan`, `GreaterThan`, `EqualTo`, `Interval` (a product in a row must
  involve a JuMP `Parameter`; any other product is an error of `optimize!` that names it);
  vector-affine rows in `Nonnegatives`, `Nonpositives`, `Zeros`, read one output at a time;
  variable bounds, `Integer`, `ZeroOne`; a quadratic objective. Bridges would add auxiliary
  variables with no stage. Vector-affine rows in `SecondOrderCone` are read only over the
  uncertainties and tagged with a Ξ role (October 2026): they describe Ξ, as conic rows.
- **Metadata are part of the model:** `Uncertainty(dist)` is an MOI set (scalar or vector); stages
  are `VariableIndex`-in-`Stage(t)` constraints, so they appear in `print` and are deleted with their
  variable; `BreakPoints` and the Ξ roles are MOI attributes. The `@variable`/`@constraint` tags use
  `build_variable`/`build_constraint` on our own types only (no type piracy).
- **Checks at declaration** whenever they are local and JuMP gives a hook on our own types:
  unbounded distributions (or laws without `minimum` and `maximum`), and bounds or integrality given
  in the `@variable` of an uncertainty. A bound, `fix` or integrality set later, and an untagged
  uncertainty-only row, reach the model through JuMP's own `set_lower_bound`, `fix` and
  `build_constraint` methods; catching them there would be type piracy, so `canonicalize` reports
  them at `optimize!`. Whole-model checks run at `optimize!`, with messages that name variables and
  constraints (an unnamed constraint is printed, such as `d[1] + d[3] <= 190`). The assumptions of
  the dual bound are warnings (October 2026), since the primal and sampled rules do not need them:
  a quadratic objective with `P` not positive semidefinite (`-P` for a maximization), by a
  Cholesky factorization of the block of `P`, naming its decisions; and a singular second-moment
  matrix `M`, block by block on the correlation matrices (a discrete law with at most as many
  atoms as dimensions, a constant uncertainty, a lifted law with at most as many atoms as pieces),
  naming the uncertainties and rank, which also marks the bound as not certified.
- **Solver settings:** `InnerOptimizer()`; `Silent` and raw attributes are forwarded. `TimeLimitSec`
  limits the whole `optimize!`, rejection sampling included (see Moments): each formulation, built
  and solved in turn, gets the time that remains.

## Results (decision 2)

- Result 1 is the primal-LDR rule (`FEASIBLE_POINT`: robustly feasible on Ξ). Result 2 is reserved
  for the sampled rule (`UNKNOWN_RESULT_STATUS`: feasible on the drawn scenarios only). The slots
  are fixed: a slot whose formulation was not solved or has no solution is `NO_SOLUTION`, and
  `ResultCount` is 2 when `SampledLDR` was solved, else 1 when `PrimalLDR` was. `Formulations()`
  takes at most one formulation of each type.
- The dual LDR takes no result slot, since its rule is not a policy: its value is `objective_bound`,
  so `relative_gap` is the primal–dual certificate. The bound needs no primal point (an inner MIP
  proves it without an incumbent), and it is `±Inf` when the dual LDR is unbounded.
- `TerminationStatus` reports `INFEASIBLE` only on the dual's evidence; an infeasible primal LDR
  alone gives `LOCALLY_INFEASIBLE`.
- Per formulation: `termination_status(model, DualLDR())`, `objective_value(model, DualLDR())`;
  our own functions take `formulation = DualLDR()`.
- `value(x) = E[x(ξ)]`, `value(d) = E[d]` (decision 6). JuMP's `value(expr)` plugs in these means,
  which is exact for affine expressions only; `expected_value(expr)` is exact for quadratic ones.
- `decision_rule(x)` returns a callable rule on a scenario of the original uncertainties: a
  constant, the coefficients of the uncertainties that are not lifted, and for each lifted one a
  `LiftedTerm` (its lifting and the coefficients on its pieces, the slopes for `AxisPiecewise`); it
  applies the lifting internally.
- `evaluate(model, scenario)` applies the rule of a formulation to one scenario of the original
  uncertainties (lifting it first): the decisions, the objective at the scenario and the violated
  constraints by name. `simulate(model, n)` draws `n` scenarios from the laws (conditioned by
  `Truncate()` rows, by rejection unless exact), in the original coordinates, and `simulate(model, scenarios)` takes scenarios from elsewhere (for
  example SDDP simulations); it returns the objective of each, its mean and standard error, and
  the violations. The primal rule has none on the support, so the sample mean estimates
  `objective_value`; the sampled rule is feasible on its drawn scenarios only, and `simulate`
  measures it out of sample.
- `FormulationResult.certified` (October 2026) says whether `objective_value` and
  `objective_bound` are exact for the law: `false` when they use moments estimated by rejection
  sampling, for a bound when the second-moment matrix is singular or a law spanning several stages
  is not known to have linear conditional expectations (see Moments), and always for
  `SampledLDR`; `raw_status` then ends with `[not certified: <reason>]` (not for `SampledLDR`,
  whose values are estimates by construction). Decided (question 11): it is exposed at the model
  level as the attribute `Certified()`, read for one formulation,
  `get_attribute(model, FormulationAttribute(DualLDR(), Certified()))`, with the helper
  `is_certified(model, f = DualLDR())`; `Certified()` without a formulation is an error that
  says how to read it.
- Decided (October 2026): low-level access to the rules is public, since users post-process rules
  themselves and `_rule_coefficient` would have kept them on `decision_rule(x).coefficients`, which
  drops zeros and hides the pieces of a lifted uncertainty. `rule_coefficient(x, η; formulation)`
  is the coefficient of `η` in the rule of the decision `x` (`rule_coefficient(x)` the constant),
  and `rule_matrix(model; formulation)` returns a `RuleMatrix`: the matrix `X`, the decisions of
  its rows and the labels of its columns (`:constant`, an uncertainty, or `(η, k)` for the `k`-th
  piece of a lifted `η`). The matrix is in lifted coordinates, `x(ξ) = X lift(map, ξ)`. For a
  lifted `η`, `rule_coefficient` returns the vector of the coefficients on its pieces, in order.
  Still open: the representation for lifted linear combinations of several uncertainties (the
  coefficients on the lifted coordinates and the map `R`), until such liftings exist.

## Modelling conventions

- One `Stage(t)` tag for decisions and uncertainties; `FirstStage` is `Stage(1)`. An unmarked
  variable is in the last stage T = max(2, largest declared stage), with a warning when T > 2
  (decision 3).
- Ξ roles (decision 4): `@constraint(m, expr, ValidInequality())` tightens Ξ only;
  `@constraint(m, expr, Truncate())` conditions the law on the set C where such rows hold, and
  tightens Ξ (role `:truncate`). An untagged constraint over uncertainties only is an error naming
  both tags. Decided in October 2026:
  - a univariate law whose `Truncate()` rows involve it alone (a box) is truncated exactly, and the
    rows are dropped since its box replaces them: `Distributions.truncated` for a continuous law
    (moments by QuadGK when Distributions has none), the atoms in the interval as a
    `DiscreteNonParametric` for a discrete law with a finite support, rejection otherwise;
  - the declarations coupled by `Truncate()` rows (union–find on declarations) merge into one law,
    a `ConditionedDistribution` (the product of their laws conditioned on `W η ≥ h`, also
    declarable directly), with one covariance block; its draws are by rejection;
  - decided (question 12): when all the merged laws have finitely many atoms
    (`MvDiscreteNonParametric`, univariate laws with a finite support, or a declared
    `ConditionedDistribution` of such laws) and their product at most 10⁶, the conditioned law is
    exact: the atoms of the product that satisfy the rows, renormalized, as an
    `MvDiscreteNonParametric` (`_discrete_conditioned_law`); the rows stay in Ξ, the moments are
    exact and the values certified, and no atom left is an error naming the rows;
  - a tagged equality, a tagged row with decisions, and a truncation of probability zero are
    errors; valid rows are checked at the mean of the conditioned law (not for estimated laws);
  - an uncertainty whose law is estimated by rejection cannot be lifted (breakpoints) yet.
- Constraint stages are dropped (decision 5): support blocks make slack masking automatic. A
  `Stage` tag on a constraint is reserved for a possible extension: an expectation constraint
  "at time t".
- Formulations are chosen with `set_attribute(model, Formulations(), [PrimalLDR(), DualLDR()])`,
  using types so that `SampledLDR(N; seed)` can carry parameters.

## Canonical form and transforms

- `CanonicalForm`: decisions (stage, integrality, bounds, names); uncertainty columns; rows
  `B_l ξ ≤ A x ≤ B_u ξ` (equalities and intervals native); objective `P, C, Q, d, f`; Ξ as a list of
  `SupportBlock(W, h, cones, roles)`: `W η_b - h ∈ K_b`, with `K_b` the orthant of the linear rows
  times one second-order cone per conic row (`cones` lists the pieces, linear rows first).
- `InformationStructure` answers `observes(i, j)`. With nested stages, columns are sorted by stage
  and each history is a prefix; after lifting, columns are sorted again.
- Liftings provide `L` (applied to scenarios) and a sparse linear retraction `R`; the form is lifted
  by `B ← BR`, `C ← CR`, `W ← WR` plus hull rows. A lifted coordinate's stage is the largest stage
  of the coordinates it depends on. `AxisPiecewise` (breakpoints) ships first.
- `lift(form, liftings)` returns a `CanonicalForm` over the lifted columns (`canonicalize` calls it
  for `BreakPoints`), with sparse products, `d ← Rᵀd`, `Q ← RᵀQR` and `f` unchanged. Its field
  `lifting` is a `LiftingMap` (kept columns, one block per lifting with its input and output
  columns, the assembled `R`); `lift(map, ξ)` applies L to outcomes. The law of a lifted
  uncertainty is a `LiftedDistribution`, whose box replaces the original one; the hull rows join
  the valid rows with no constraint (role `:hull`), so the dense oracles read a lifted form as any
  other. For now, a lifted uncertainty must be declared alone, with a univariate law.
- Ξ is a product of support blocks: connected components of coordinates coupled by support, valid,
  conditioning or hull rows.
- Conic Ξ (decided October 2026). A `VectorAffineFunction`-in-`SecondOrderCone` constraint over
  uncertainties, tagged `ValidInequality()` or `Truncate()`, is a `ConicRow` of the form (`W η - h`
  in the cone), lifted as `W ← WR`; untagged or with decisions, it is an error at `optimize!`. A
  conic `Truncate()` row conditions its laws by rejection (`ConditionedDistribution` takes a list
  of cones), never by exact truncation, except for laws with finitely many atoms. The support of a law beyond its box comes from the public
  hook `support_constraints(d) -> (W, h, cones)` (role `:law` in the blocks), empty by default;
  `simulate` checks given scenarios against it.
- `ConfidenceMvNormal(μ, Σ, α; support = :auto | :ellipsoid | :boxes)` (decided October 2026):
  `:ellipsoid` is one second-order cone row `(ρ, L⁻¹(η - μ))` (SOCP inner problems); `:boxes` is
  the box along the principal axes of the previous version, beside the bounding box of the law
  (LP inner problems; a valid outer approximation, so the primal stays feasible and the dual a
  valid, looser bound). `:auto` is `:ellipsoid` when the inner optimizer supports
  `VectorAffineFunction`-in-`SecondOrderCone` (asked at `optimize!` through a bridged instance,
  only when some law has `:auto`) and no decision is integer (MISOCP is much harder than MILP),
  and `:boxes` otherwise; `canonicalize(src, info; conic_support)` takes the optimizer's answer.
  The dimension is not part of the heuristic: the ellipsoid is one cone of dimension `d + 1`,
  while the boxes add `2d` rows. Draws are by rejection in the standardized space (`1/α` attempts
  per draw), which is cheaper than the exact radial method (a `χ²` quantile per draw) for the
  usual levels. Measured on a shared stock with 30 correlated demands (primal and dual, build
  and solve; main #97, 87f5979): the boxes' LP took 8.8 s with HiGHS and 1.3 s with Clarabel, the
  ellipsoid's SOCP 0.9 s with Clarabel, with a primal value 30% better and a slightly tighter
  bound. Decided (question 16): `:auto` stays as it is; with the same solver (Clarabel), the
  SOCP (0.9 s) is faster than the LP of the boxes (1.3 s), so the conic representation costs
  nothing when the solver supports it, and the 8.8 s of the LP with HiGHS is a solver difference.
  Decided (question 17): the draws stay by rejection. A candidate `z ~ N(0, I)` is accepted iff
  `‖z‖² ≤ ρ² = quantile(Chisq(d), α)`, and `‖z‖² ~ χ²(d)`, so the acceptance rate is exactly `α`
  in any dimension: only a small `α` is costly (documented in the docstring).

## Moments

- `UncertaintyBlock(dist, columns, stages)`; separate declarations are independent.
- `M = μ̄μ̄ᵀ + blockdiag(Σ_b)` is stored with its structure. Moments are analytic (exact sums for
  discrete laws, conditioned ones included), by quadrature for lifted blocks, or by rejection
  sampling for the other conditioned blocks, and cached by their inputs.
  A `LiftedDistribution` computes its moments once, on construction: exactly for discrete laws
  and, for `AxisPiecewise`, in closed form for `Uniform` and by QuadGK (already a dependency of
  Distributions) of the density on each piece otherwise.
- Decided (question 15): the dual bound needs the conditional expectations of later stages given
  earlier ones to be linear (condition (i) of `docs/math`, Kuhn, Wiesemann and Georghiou, 2011),
  which, since declarations are independent, is a property of each law spanning several stages.
  It replaces the covariance test, which flagged correlated laws that have it and missed
  uncorrelated but dependent ones. The internal trait
  `_linear_conditional_expectations(dist, stages)::Union{Bool,Nothing}` is

  | Law | Value | Reason |
  |---|---|---|
  | in one stage | `true` | nothing to condition on |
  | univariate | `true` | one stage |
  | `product_distribution` of univariate laws | `true` | independent components |
  | `MvNormal` | `true` | the conditional law is normal with an affine mean |
  | `ConfidenceMvNormal` | `true` | the conditional law is symmetric about the affine mean of the normal law (appendix C) |
  | `MvDiscreteNonParametric` | `true` or `false` | checked exactly: at each stage, the conditional means of the later components per distinct history must fit an affine function of the history with no residual |
  | other multivariate laws (`ConditionedDistribution`, …) | `nothing` | unknown |

  The dual warns, naming the laws, and the bound is not certified, when a law spanning several
  stages returns `false` or `nothing`; blocks estimated by Monte Carlo and a singular `M` also
  mark the bound as uncertified (`is_certified`).
- Rejection sampling (October 2026). A `ConditionedDistribution` keeps up to
  `RejectionSamplingMaxSamples()` draws (10 000; the old `MaxIterations`, 1000, had this meaning)
  within its time limit; its moments are those of the empirical law of the draws (divided by n),
  so the mean is in C and the moments are consistent. Decided (questions 13 and 14): the
  sampling counts against `TimeLimitSec` (one limit for the whole `optimize!`, as for the
  formulations). `RejectionSamplingTimeLimit()` is `nothing` by default: then, with
  `TimeLimitSec`, the sampling gets half of the time that remains of it, shared by the laws to
  sample (each gets an equal share of what the previous ones left; cached laws take no time), and
  the formulations the rest; without `TimeLimitSec`, 10 s per law. An explicit
  `RejectionSamplingTimeLimit()` (per law) wins. The default sample size stays 10 000: the standard
  error of the moments is of order `1/√N` (1% of the standard deviation; 1 000 draws gave about
  3%), and the time is proportional to `N` divided by the acceptance rate. The k-th conditioned law draws from `Xoshiro(s[k])`, with
  `s = rand(Xoshiro(RejectionSamplingSeed()), UInt64, K)` (seed 1234): laws never share a stream
  (the old code reused one seed for every group). A draw is never replaced by a default value (the
  old code counted a zero vector after failed attempts, so `WarnAttempts` is gone): fewer than
  `min(MaxSamples, 100)` draws is an error naming the constraints and the acceptance rate, raised
  early when even the rule-of-three upper bound of the rate cannot reach it in time; fewer than
  `MaxSamples` is one warning per `optimize!` listing the laws. `CovarianceBlock.num_samples`
  records the sample size (0 when exact). The optimizer caches complete estimates by (law with its
  rows, derived seed, sample size), keeping those of the last solve, so a re-solve that changes
  other data does not sample again. `SampledLDR` and `simulate` draw their scenarios from the
  conditioned laws with their own seeds.

## Formulations

Each formulation is a type with `capabilities()` and a method of `_build!(model, f, form, mom)`,
which builds the inner problem and returns the rule variables. `_build!` is internal: its contract
may change while the liftings and conic Ξ arrive, so it becomes public with the `Experimental`
submodule (Phase 5), when formulations from outside the package need it.

- **PrimalLDR:** robust counterparts per (row, support block) with structural nonzeros, one
  coupling inequality per row; a box block is treated like any other block (the closed form for
  boxes is the Phase 4 fast path). On a conic block the multipliers are
  in the dual cone of each piece (`λ ≥ 0` on the linear rows, `λ ∈ SOC` on a conic row, which is
  self-dual): Kuhn, Wiesemann and Georghiou (2011) state Proposition 1 for a polyhedral Ξ; the
  conic version needs a point of Ξ strictly inside the cones for strong duality, and the rule is
  feasible on Ξ without it.
- **DualLDR:** with `u = S μ̄`, the cone constraint `S M W'ᵀ ≥ 0` becomes
  `u (W' μ̄)ᵀ + S Σ̄ W'ᵀ ≥ 0`; blocks outside a row's support collapse into `u_r ≥ 0`; S is masked
  by history. On a conic block, `S M W'ᵀ ∈ K` row by row, one second-order cone constraint per
  conic row (the inclusion `K_P ⊂ K` of their Proposition 3 holds for any closed convex cone, so
  the bound stays valid); a piece with no masked column is implied and skipped.
- Solve times (`benchmark/compare/`, October 2026): smaller is not always faster. After presolve,
  the new primal is the same LP as the previous version with `SlackAsExpr`, with the same times.
  The structured dual is 17–150× faster than the previous dense dual on the inventory problem, but
  4–6× slower on hydro, where presolve cannot remove the auxiliaries `u = S μ̄`: substituting `u`
  out recovers the previous iteration counts on hydro and is 20× slower on inventory (denser rows).
  Two options for Phase 4: a density rule choosing whether to keep `u`, and solver hints per
  formulation (the dual needs only its bound: HiGHS's IPM without crossover took 8 s instead of
  145 s on hydro T = 8, at 2e-10 relative accuracy).
  Done (October 2026): `PrimalLDR(; attributes)` and `DualLDR(; attributes)` carry raw attributes
  of the inner optimizer, set over those of the optimizer; the benchmark's
  recommendation is in the docstrings, not the default (it fails on small models, where HiPO
  stagnates and HiGHS reports `unknown`). When an inner LP is solved to optimality, the bound is
  its dual objective value (weak duality, valid without crossover), not the primal objective.
- **SampledLDR:** equality rows without random recourse are matched once, not per scenario.
- Inner models are JuMP models built from sparse blocks (decision 7); objectives use `dot`, never
  `tr` of matrix products; the mean `E[x_i] = X_i μ̄` of a decision in `E[x'Px]` is a variable with
  one equality row, so that the term has `nnz(P)` entries and not one per pair of columns (a
  coupled quadratic inventory, T = 48: 749 700 terms become 20 776, build 0.17 s to 0.015 s, and
  Clarabel solves 10× faster; HiGHS's QP solver is erratic on both forms). The dense formulations of the previous version are kept in
  `test/oracles` and must agree with the structured ones.

## Testing

- A problem library with known values, run under every formulation: newsvendor (primal 460,
  optimum 3300/7, dual 1460/3; integer first stage: dual 486), two products (920), three stages
  (940 / 1000), the T-period inventory (90T + 20), the 1-D hydro toy, value-function chaining.
- Golden values from the previous version in `test/reference/legacy_values.toml`; for
  `ConfidenceMvNormal` (whose support was the boxes), in `test/jump/test_conic_support.jl`.
- Second-order cone programs are solved by Clarabel in the tests (a conic problem library against
  the dense oracles, extended to conic Ξ); linear ones by HiGHS.
- Invariants: primal ≤ optimum ≤ dual (max), monotone bounds over nested breakpoints, primal rules
  feasible on samples of the support, the stage mask.
- Every file `test/**/test_*.jl` runs in CI; Aqua checks piracy and ambiguities.

## Open questions 11–17 (settled October 2026)

| # | Question | Decision | Where |
|---|---|---|---|
| 11 | Model-level certification | `Certified()` through `FormulationAttribute`, and `is_certified(model, f = DualLDR())` | Results |
| 12 | Exact conditioning of discrete laws | atoms of the product that satisfy the `Truncate()` rows, renormalized; certified | Modelling conventions |
| 13 | Rejection sampling and `TimeLimitSec` | the sampling counts against the one limit of `optimize!` | Moments |
| 14 | Default sampling time and size | half of what remains of `TimeLimitSec`, shared by the laws (else 10 s per law); an explicit `RejectionSamplingTimeLimit()` wins; 10 000 draws | Moments |
| 15 | Stagewise warning of the dual | linear conditional expectations, law by law, instead of correlation | Moments |
| 16 | `:auto` for `ConfidenceMvNormal` | kept; same-solver comparison recorded | Canonical form |
| 17 | Draws of `ConfidenceMvNormal` | rejection kept; acceptance rate exactly `α` in any dimension | Canonical form |

## Roadmap

| Phase | Content | Exit |
|---|---|---|
| 0 | Spike (done); repository, CI, golden values, this document | Golden values committed |
| 1 | Canonical form, information structure, support blocks, structured moments, block primal, structured dual, sampled, optimizer, results | Golden values and dense oracles match on non-lifted problems |
| 2 | `AxisPiecewise` lifting, piecewise moments, `evaluate`, `simulate` | Multistage + breakpoint tests |
| 3 | Ξ roles and cached rejection sampling (`Truncate()`, done), conic Ξ and `SecondOrderCone` rows (with the boxes alternative for `ConfidenceMvNormal`) and the `support_constraints` hook (done), low-level rule access (done) | Tutorials ported; release 0.4.0 (decision 8) |
| 4 | Box fast path, MOI hot paths if needed, JuMP Parameters for re-solves; keep-or-substitute `u` in the dual by a density rule; solver hints per formulation (done as `attributes`; IPM without crossover for the dual's bound); measured against `benchmark/compare/` | Benchmark targets |
| 5 | `Experimental` submodule (decision 9): LFDR value functions, new liftings, CVaR | Deck experiments E1–E4 |

Research backlog: auto-tuned breakpoints, constant-depth decision rules, piecewise-affine rules of
Thomä–Schiffer–Wiesemann, random recourse via SDP, infinite horizon, expectation constraints at
time t.

## Repository (decision 10)

This history becomes the new `main` of `github.com/bfpc/LinearDecisionRules.jl`; the previous
`main` is renamed `legacy`, which keeps the registered versions' tree hashes reachable.
