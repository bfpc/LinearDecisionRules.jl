# Phase 0 spike: report

**Recommendation.** Decision 1: design B, an `LDR.Optimizer` under a plain `JuMP.Model`, with two borrowings:

- from C: encode stages and uncertainty as constraints, so they print and are deleted with their variable;
- from A: check at declaration whatever can be checked locally.

Decision 2: B's results semantics. Result 1 is the primal policy, the dual LDR is `objective_bound`, and result 2 is
kept for the sampled rule. The package's own functions take a `formulation =` keyword.

The hydro experiment confirms the planned structured dual: it has 13–40× fewer nonzeros, with identical objectives.

All numbers below come from `audit/AUDIT.md`, which was produced by an independent agent, and from the design
agents' reports. Every test file can be re-run with `julia --project=spike <file>`.

## What was built

| Part | Files | Tests (auditor's re-run) |
|---|---|---|
| Shared core: canonical form, dense primal and dual LDR | `core/SpikeCore.jl` | 67 / 67 pass |
| A: wrapper `LDRModel <: AbstractModel`, own `LDRVariable` | `wrapper/` | 141 pass, 1 broken |
| B: `LDR.Optimizer` under `JuMP.Model` (`add_bridges = false`) | `optimizer/` | 123 pass, 3 broken |
| C: plain `JuMP.Model` + `set_optimize_hook` | `hook/` | 132 pass, 1 broken |
| Audit: metrics, 43 adversarial probes per design | `audit/` | — |
| Hydro sizes: block primal and structured dual | `dualnnz/` | 12 / 12 validations agree to 1.2e-12 |

All mandatory known values hold in every design: 460, 1460/3, 920, 940, 1000, and the rules. Every design gives the same objective on every model solved.

## The four criteria

| Criterion | A: wrapper | B: optimizer | C: hook |
|---|---|---|---|
| **Checklist** | Everything except 12b. A third-party `build!(::JuMP.Model)` cannot take the model; this is structural | Everything except item 5, where stages are missing from `print` (fixable; see below). Parameters need a 2-line core change | Everything except Parameters (same 2-line core change) |
| **Complexity**: code lines; methods on other modules' functions; forwarding methods | 575; 114 (99 on JuMP); ≈42 | 310; 59 (42 on MOI); 0 | 261; 20; 0 |
| What a contributor must know | JuMP's `AbstractModel` API. Every JuMP function a user calls needs a method of ours; missing ones give a `MethodError` (`solution_summary`, `set_objective_coefficient`, `relax_integrality`, `write_to_file`, `set_time_limit_sec`, …) | The MOI optimizer API: `optimize!(dest, src)`, attributes, result indices, `map_indices` | The JuMP tag API, plus `MockOptimizer` and an internal field (`backend(model).model_to_optimizer_map`) |
| **Extension**: third-party builders, `copy_model`, other JuMP extensions | Builders no; `copy_model` with about 30 lines of our own; extensions that assume `VariableRef` or `JuMP.Model` do not compose | All work unchanged | All work unchanged |
| Extension: new formulations, liftings, conic Ξ, constraint tags (decision-5 note) | Easy, since we own everything | Easy: attribute values, `supports_constraint`, constraint attributes | Easy |
| **Semantics**: where errors appear | Mostly at declaration, the best of the three | Declaration for unsupported types, with JuMP's generic message; otherwise `optimize!` | Always at `optimize!` |
| Semantics: results | Own API, `formulation =` keyword. `objective_value(m; result = 2)` silently returns the primal | MOI-native: `objective_bound`, `relative_gap`, `solution_summary` all work. LOCALLY_INFEASIBLE when only the primal LDR fails. Errors instead of NaN | JuMP queries work through a test-utility mock. Solver attributes and time limits are accepted but silently ignored. NaN on infeasible models |

**Results after `set_silent`.** B and C discard results after `set_silent` or a time limit, as plain JuMP does with HiGHS (checked). A keeps them, which is non-standard rather than an advantage.

## Why B

- **No forwarding, no MethodError surface.** A plain `JuMP.Model` gets every JuMP function, and any other package's, for free. A's 42 forwarders and its list of missing functions would grow with JuMP.
- **Results through public MOI attributes.** C reaches the same user experience through `MockOptimizer` and an internal field, and ignores solver settings. B's results path is the standard one, as in Dualization.jl and MultiObjectiveAlgorithms.jl.
- **B's extra size is in a stable API.** It is 49 more lines than C and 39 more foreign methods, almost all `MOI.get`/`MOI.set`. Joaquim's MOI expertise makes that the cheapest kind of code for this team.
- **A's real advantage is early errors.** B can adopt most of it in `build_variable`: unbounded distributions, bounds or `fix` on an uncertainty, untagged uncertainty-only rows.

## Changes to carry into Phase 1

Front-end (design B):

- **Stages and uncertainty as constraints.** Encode stages as `VariableIndex`-in-`Stage(t)` constraints and uncertainty as a set, as C does. Stages then appear in `print` and disappear on `delete`.
- **Declaration-time checks**, as listed above. Improve the message when a user calls `set_optimizer(m, HiGHS.Optimizer)` by mistake.
- **Raw solver attributes.** Forward `MOI.RawOptimizerAttribute` to the inner solver; today B rejects them.
- **Helpers.** Add `formulation =` helpers on our own functions (`decision_rule`, `expected_value`, a per-formulation objective and status), so users don't write `FormulationAttribute` by hand.

Core:

1. **Deterministic models.** With no bounded uncertainty, the dual LDR keeps no inequality row, so a deterministic `Max 2x, x ≤ 3` reports `objective_bound = 0` instead of 6.
2. **Failed solves.** Values from an infeasibility certificate are reported as `objective_value`/`value`. Check the primal status before filling results.
3. **Integer first stage.** The dual LDR bound uses the inner MIP's incumbent; use its `objective_bound`. The resulting 486.0 for the integer newsvendor is a valid bound, tighter than 1460/3: the integer optimum is 471.41.
4. **Messages.** Name constraints in error messages (use `MOI.ConstraintName`), and warn only once about unmarked variables.
5. **Parameters.** Treat JuMP `Parameter`s as fixed stage-1 values in `canonicalize` (2 lines).
6. **Tests.** Make `delete` (item 6) bind before deletion, and test `expected_value(p * sell)` where sell depends on p, as B does.

## Hydro: block primal and structured dual (`dualnnz/README.md`)

Same objectives as today's formulations on 12 validation cases (max relative difference 1.2e-12). Nonzeros:

| T, k | Current dual | Structured dual | Ratio | Current primal | Block primal |
|---|---|---|---|---|---|
| 12, 0 | 14.2M | 1.06M | 13× | 758k | 450k |
| 12, 3 | 301M | 15.5M | 19× | 3.59M | 2.05M |
| 24, 0 | 121M | 4.4M | 27× | 3.13M | 1.84M |
| 24, 3 | 2.62B | 64.9M | 40× | 15.0M | 8.5M |

Most of the gain comes from collapsing untouched blocks into one `u_r ≥ 0` per row; masking S alone gives about 2×.
When one law couples all stages with a dense covariance, the dual gains only 2×. That is the case the planned
innovations lifting addresses.
