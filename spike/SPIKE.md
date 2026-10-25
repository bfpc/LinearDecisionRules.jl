# Phase 0 spike: choosing the user layer

This directory is a throwaway prototype. It decides decisions 1 (user layer) and 2 (results API) of the plan
"LinearDecisionRules.jl — new package plan". Nothing here is the final package. Do not modify anything outside
`spike/`, do not commit, and do not touch the old package's environments (`test/`, `docs/`, `examples/`, root
`Project.toml`/`Manifest.toml`). Run Julia with `julia --project=spike` from the repository root
(`c:\Users\bernardo.paulo\Packages\LinearDecisionRules.jl`); use the Bash tool for Julia commands (PowerShell
mangles quotes). The environment already has JuMP 1.32.1, MathOptInterface 1.54, Distributions, HiGHS, Aqua,
Test, LinearAlgebra, SparseArrays, Random. Do not add packages without saying why in your report.

## Layout

```
spike/
  SPIKE.md                  this brief (the contract)
  core/SpikeCore.jl         shared: canonical form + dense primal/dual LDR (one implementation for all designs)
  core/test_core.jl         core tests, plain JuMP models + hand-built ModelInfo
  wrapper/LDRWrapper.jl     design A: LDRModel <: JuMP.AbstractModel wrapping an inner JuMP.Model
  wrapper/checklist.jl
  optimizer/LDROptimizer.jl design B: LDR.Optimizer <: MOI.AbstractOptimizer under a plain JuMP.Model (add_bridges = false)
  optimizer/checklist.jl
  hook/LDRHook.jl           design C: plain JuMP.Model + typed struct in model.ext + JuMP.set_optimize_hook
  hook/checklist.jl
  dualnnz/                  separate experiment: structured-dual size on the hydro instance
  REPORT.md                 final comparison (written last)
```

Each design is one module in one file (split only if it exceeds ~600 lines) and must not edit `core/`.
If a design needs something from the core, write it in your report as a request; do not fork the core.

## Mathematics (from the old package; it is the specification)

Notation: ξ = (1, η) with η the uncertainties; rules x(ξ) = X ξ; M = E[ξξᵀ]. Constraints
`A x(ξ) (≤, =, ≥) B ξ`, objective `E[x(ξ)ᵀ P x(ξ) + c(ξ)ᵀ x(ξ)] + r` written as `tr(XᵀPXM) + tr(CᵀXM) + r`
(compute with `dot`, not `tr` of matrix products). Ξ = {ξ : Wξ ≥ h, ξ₁ = 1}, a box from the distributions'
`minimum`/`maximum` plus rows tagged `ValidInequality()`.
- Primal LDR (Kuhn–Wiesemann–Georghiou 2011): `src/solve_primal.jl` (Λ ≥ 0, ΛW = S, Λh ≥ 0 per inequality row).
- Dual LDR: `src/solve_dual.jl` (`S M W'ᵀ ≥ 0`, W' = W − h e₁ᵀ without its two zero rows).
- Canonical split and moments: `src/canonical.jl` (`_canonical`, `_second_moment_matrix`, `_objective_constant`).
- Stage mask: X[i, j] exists iff j == 1 or stage(η_{j−1}) ≤ stage(x_i). Integer/binary decisions only in stage 1,
  and then only their constant column (they must not see stage-1 uncertainty: X[i, j>1] = 0 for them).
- Stages: T = max(2, largest declared stage). An unmarked decision or uncertainty is Stage(T); warn once
  (naming the variables) when T > 2 and something is unmarked.
- Independence: separate declarations are independent; within one multivariate declaration use its `cov`.

## Core contract (`core/SpikeCore.jl`, module `SpikeCore`)

```julia
struct UncertaintyDecl
    dist::Distributions.Distribution          # Univariate (1 var) or Multivariate (length(vars) vars)
    vars::Vector{MOI.VariableIndex}
    stages::Vector{Union{Int,Nothing}}        # per component; nothing = unmarked
end
Base.@kwdef struct ModelInfo
    uncertainties::Vector{UncertaintyDecl} = UncertaintyDecl[]
    decision_stage::Dict{MOI.VariableIndex,Int} = Dict()   # absent = unmarked
    valid_rows::Set{MOI.ConstraintIndex} = Set()            # uncertainty-only rows tagged ValidInequality()
    truncate_rows::Set{MOI.ConstraintIndex} = Set()         # tagged Truncate(): error "not in the spike"
end
canonicalize(src::MOI.ModelLike, info::ModelInfo) -> Canonical
struct FormulationResult
    termination_status::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    objective_value::Float64                  # in the user's sense (Max stays Max)
    X::Matrix{Float64}                        # n_decisions × (1 + n_uncertainties); column 1 = constant
end
solve_primal(c::Canonical, optimizer_factory; silent = true) -> FormulationResult
solve_dual(c::Canonical, optimizer_factory; silent = true) -> FormulationResult
decisions(c), uncertainties(c)                -> Vector{MOI.VariableIndex} in canonical order
mean_xi(c)                                    -> E[ξ] (length 1 + n)
second_moment(c)                              -> M
expected_decision(c, res, vi)                 -> X[i, :] ⋅ E[ξ]   (vi a decision); E[η] for an uncertainty
rule(c, res, vi)                              -> (constant, Dict{MOI.VariableIndex,Float64} of nonzero coefficients)
expected_value(c, res, f::MOI.ScalarQuadraticFunction or ScalarAffineFunction) -> exact E[f(x(ξ), η)] using M
```
`canonicalize` reads only: `ScalarAffineFunction` in LessThan/GreaterThan/EqualTo/Interval; `VariableIndex` in
LessThan/GreaterThan/EqualTo/Interval/Integer/ZeroOne; objective `VariableIndex`/`ScalarAffineFunction`/
`ScalarQuadraticFunction`, with its sense. Any other constraint type present is an error naming the type,
unless `SpikeCore.is_metadata_constraint(F, S)` returns true (default false; a design may add methods for its
own set types). Errors with clear messages: an uncertainty-only row that is untagged (the message names both
`Truncate()` and `ValidInequality()`); bounds on an uncertainty variable; integer decisions after stage 1;
uncertainty × decision products in constraints (random recourse); a quadratic constraint.

## The checklist (every design implements it in `<design>/checklist.jl` with `@testset`s)

Preferred API (from the plan) in parentheses; a design that cannot provide it uses the closest form and
records the deviation. Known values are mandatory `@test`s; do not weaken them.

1. **Newsvendor.** `buy ≥ 0, Stage(1)`; `d in Uncertainty(Uniform(80, 120))` unmarked; `sell, ret ≥ 0` unmarked;
   `sell + ret ≤ buy`, `sell ≤ d`; `Max −10buy + 8ret + 15sell`. Primal objective 460 (`objective_value`), dual
   1460/3 (`objective_bound`), primal status FEASIBLE_POINT, rule buy = 120, sell = d (`decision_rule`),
   `value(buy) == 120`, `value(sell) == 100`, `value(d) == 100`.
2. **Integer first stage.** Same with `@variable(m, buy ≥ 0, Int, Stage(1))`: 460.
3. **Vector uncertainty.** Two products, `@variable(m, d[1:2] in Uncertainty(product_distribution([Uniform(80,120), Uniform(80,120)])))`:
   920, and the rule for sell[1] has no coefficient on d[2].
4. **Containers and stages.** The 3-stage newsvendor of `test/test_stages.jl::test_3_stage_newsvendor` written with
   `Stage(t)` tags, including at least one indexed container `@variable(m, x[t = 1:2] ≥ 0, Stage(t))` and
   uncertainties declared with `Stage(t)`: primal 940, dual 1000; stage-2 rules have no coefficient on stage-3
   uncertainty. Also `@variable(m, d in Uncertainty(dist), Stage(3))` and `stage(x)` / `set_stage(x, t)` work.
5. **print.** `print(model)` runs and shows the uncertainties and stages in a readable way (record the output).
6. **delete.** Add a variable `extra` and a constraint using it, solve, `delete` both, re-solve: back to 460, and
   `decision_rule(sell)` is still correct afterwards.
7. **Modification.** Add `cap: buy ≤ 200`, solve, `set_normalized_rhs(cap, 100)`, re-solve: the objective equals
   that of a model built from scratch with `buy ≤ 100`.
8. **Results API.** Per-formulation status and objective; `objective_bound`/`relative_gap`; a disabled formulation
   gives NO_SOLUTION / a clear error; `expected_value(p * sell)` with a random price p differs from
   `value(p) * value(sell)` and is exact (check against a sampled estimate within 1%).
9. **Ξ roles.** `@constraint(m, d1 + d2 ≤ 230, ValidInequality())` is accepted and changes the primal; an untagged
   uncertainty-only constraint errors with a message naming `Truncate()` and `ValidInequality()`.
10. **Attributes.** `set_attribute(d, BreakPoints(), [100.0])` and `get_attribute(d, BreakPoints())` round-trip,
    and survive deleting another variable (no lifting in the spike).
11. **Unsupported input.** `@constraint(m, sell^2 ≤ 1)` gives a clear error, never a silent reformulation; no
    auxiliary (bridge) variables ever reach the core.
12. **Extension probes** (record what happens and how much code it would take; tests may be `@test_broken`):
    a. JuMP parameters: `@variable(m, p in Parameter(5.0))` used in a right-hand side, then `set_parameter_value`.
    b. A third-party builder `build!(m::JuMP.Model)` (typed!) adding variables/constraints to the LDR model.
    c. A tag on a constraint, `@constraint(m, expr, Stage(2))`, stored and readable (future meaning: an
       expectation constraint "at time t").
    d. `copy_model(model)` keeps the LDR metadata.

## Decision criteria (the report scores each design on all four)

1. **Checklist**: pass/fail per item, and every deviation from the preferred API.
2. **Code-base complexity**: non-blank non-comment lines of the design file; number of methods it adds to
   JuMP/MOI/Base functions; number of forwarding methods; new types; Aqua `test_piracies` and
   `test_ambiguities` on the design module; what a contributor must understand (JuMP extension API vs MOI).
3. **Extension possibilities**: how the design accommodates item 12 and the roadmap (new formulations, liftings
   with variable attributes, conic Ξ, Parameters/MPC, constraint tags, third-party models, other JuMP extensions).
4. **User-facing semantics**: where errors appear (declaration vs `optimize!`), `value`, `print`, error messages.

## Reports

Each agent ends with a short report (≤ 400 words): what was built, how to run it, test results (copy the final
test summary), deviations, problems, and its own measurements for criteria 2–3. Be factual: if a test fails or
was skipped, say so.
