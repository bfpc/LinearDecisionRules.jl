# Phase 0 spike — independent audit

Auditor's scope: `spike/core/SpikeCore.jl` and designs A (`wrapper/`), B (`optimizer/`), C (`hook/`), measured
against `spike/SPIKE.md`. No design or core file was modified. Everything here is reproducible:

```
julia --project=spike spike/core/test_core.jl            # and the three <design>/checklist.jl
julia --project=spike spike/audit/metrics.jl             # criteria 2 (one fresh process per design)
julia --project=spike spike/audit/adversarial.jl         # 43 probes x 3 designs + math check of item 2
```
Raw outputs: `spike/audit/logs/*.log` (Julia 1.12.6, JuMP 1.32.1, MOI 1.54, HiGHS).

## 1. Test runs (re-run by the auditor, unchanged files)

| File | Test summary | Wall time |
|---|---|---|
| core/test_core.jl | `SpikeCore | Pass 67, Total 67` | 12 s |
| wrapper/checklist.jl (A) | `Design A: LDRModel wrapper | Pass 141, Broken 1, Total 142` | 28 s |
| optimizer/checklist.jl (B) | `Design B: LDROptimizer | Pass 123, Broken 3, Total 126` | 27 s |
| hook/checklist.jl (C) | `Design C: hook | Pass 132, Broken 1, Total 133` | 36 s |

No failures or errors. Broken: A 12b (typed `build!(::JuMP.Model)` cannot take an `LDRModel`); B item 5 (print does
not show stages) and 12a ×2 (JuMP `Parameter`); C 12a (JuMP `Parameter`).

## 2. Honesty check of the checklists (items 1–12)

✓ = implemented and the mandatory values are asserted exactly as specified.

| Item | A | B | C |
|---|---|---|---|
| 1 newsvendor (460, 1460/3, FEASIBLE_POINT, rules, values 120/100/100) | ✓ | ✓ | ✓ |
| 2 integer first stage (460) | ✓ (+ prints bound 486.0, asserts only bound ≥ 460) | ✓ (bound not shown) | ✓ (bound not shown) |
| 3 vector uncertainty (920, no d[2] coef) | ✓ | ✓ | ✓ |
| 4 3-stage (940, 1000, no stage-3 coefs, container `Stage(t)`, `d in Uncertainty, Stage(3)`, `stage`/`set_stage`) | ✓; unmarking a variable for the warning test writes the internal field `m.stage` (no public `unset_stage`) | ✓; scalar form tested with `Stage(2)` (`buy_cost2`), not `Stage(3)` — equivalent code path | ✓ |
| 5 print shows uncertainties and stages | ✓ (own `print`: tags, distributions, "unmarked") | **partial**: uncertainties shown (`d in Uncertainty(...)`), stages not shown; honestly `@test_broken` | ✓ (`buy in Stage(1)`, `d in Uncertainty(...)`; unmarked variables simply not listed) |
| 6 delete → back to 460, rule correct | ✓ but **weak**: the extra constraint `extra + sell <= 1000` never binds, so the first solve is already 460 | ✓ (binding `sell + extra <= 90`: obj < 460 before delete) | ✓ but **weak** (same non-binding `<= 1000`) |
| 7 `set_normalized_rhs` = model from scratch | ✓ | ✓ | ✓ |
| 8 results API | ✓ — but `expected_value(p*sell)` equals `value(p)*value(sell)` (=1500) in its model; the "differs" and 1 % sampled checks are done on `p*q` with an added `q == 2p` | ✓ — literal: `sell` depends on `p` (`Max p*sell − 15 sell`), 626.67 vs 600, sampled within 1 % | ✓ — "differs" shown on `p*z` with an added `z == p`; `p*sell` = 1500 is sampled within 1 % but does not differ. Disabled primal: `objective_value` returns `NaN` and `termination_status` OPTIMAL (only noted, not asserted) |
| 9 Ξ roles (ValidInequality changes primal; untagged error names both tags) | ✓ (920 → 940; error at declaration) | ✓ (error at `optimize!`) | ✓ (error at `optimize!`) |
| 10 BreakPoints round-trip, survives deleting another variable | ✓ | ✓ | ✓ |
| 11 quadratic constraint → clear error, no bridge variables | ✓ (own message) | ✓ asserted, but the message is JuMP's generic "not supported by the solver … consider using a different solver" | ✓ (core message, at `optimize!`; also checked with bridges on) |
| 12a Parameter | works (parameters temporarily turned into fixed decisions during `canonicalize`); 460 then 430 asserted | `@test_broken` ×2, justified (core rejects `VariableIndex-in-Parameter`) | `@test_broken`, justified |
| 12b typed `build!(::JuMP.Model)` | `@test_broken`, justified (MethodError); escape hatch `build!(inner_model(m))` works, but writes to the inner model leave stale results visible (asserted by the test itself) | works (`m isa JuMP.Model`) | works; `Stage` tags are also accepted on any plain `JuMP.Model` |
| 12c `Stage(2)` tag on a constraint | ✓ | ✓ (one tag per constraint: JuMP macro limit) | ✓ (also containers and broadcast) |
| 12d `copy_model` keeps metadata | ✓ (own `copy_model`, keeps solver) | ✓ metadata kept; the copy has no optimizer, the test re-attaches `Optimizer(HiGHS.Optimizer)` by hand | ✓ (hook and results holder restored via `copy_extension_data`; the inner factory is kept) |

No mandatory value was weakened (all use `≈` on the specified numbers). All `@test_broken` are on items where SPIKE.md
allows them (12) except B's item 5, which is a real partial failure and is reported as such.

**Item 2 dual bound 486.0.** All three designs report it (probe k8: A, B and C all give `objective_bound = 486.0`);
it is what the shared core computes. It is mathematically valid: buy's rule is masked to its constant column, so
integrality is imposed on a deterministic first-stage decision; for any fixed `buy = b` the dual LDR is a relaxation
of the true problem with `buy = b` (the projection of any feasible policy is feasible and keeps the constant), hence
`max_{b ∈ ℤ} dual(b)` bounds the integer optimum. Numbers (spike/audit `math` run): continuous dual optimum
486.667 at buy = 106.67; dual(106) = 485.667, dual(107) = 486.0; true optimum 471.43 (buy = 108.57), integer true
optimum 471.41 (buy = 109). So 471.41 ≤ 486.0 ≤ 486.667: a valid and tighter bound. Caveat (core): the bound is the
inner MIP's `objective_value` (incumbent); with a nonzero MIP gap the incumbent of a Max dual-LDR is *not* a valid
upper bound — the inner `objective_bound` should be used when integers are present.

## 3. Code-base complexity (uniform script `spike/audit/metrics.jl`)

| Measure | A (wrapper) | B (optimizer) | C (hook) |
|---|---|---|---|
| Code lines (non-blank, non-comment, non-docstring) | **575** (723 total; 65 doc) | **310** (444; 65 doc) | **261** (375; 44 doc) |
| Methods added to functions of other modules | **114** on 78 functions | **59** on 23 functions | **20** on 9 functions |
| … by owner | JuMP 99, Base 11, Base.Broadcast 2, MOI 2 | MOI 42, JuMP 9, Base 5, SpikeCore 2, MOI.Utilities 1 | JuMP 12, Base 4, SpikeCore 3, MOI 1 |
| Core.kwcall (keyword sorters, not counted above) | 13 | 3 | 6 |
| Methods of its own functions | 55 | 21 | 28 |
| New types (concrete / abstract) | 17 / 0 (+9 aliases/unions) | 20 / 2 (+5) | 13 / 1 (+3) |
| Forwarding methods (by inspection) | ≈42 pure forwards to the inner model/backend (22 generated by three `@eval` loops: bounds, fix, integrality, names, start values) + ≈10 AbstractModel plumbing (`jump_function*`, `variable_ref_type`, `==`/`hash`/`isequal`/`copy`/`iszero`, `broadcastable`) | 0 (8 `MOI.get/set` methods store optimizer attributes) | 0 |
| Aqua `test_piracies` | PASS | PASS | PASS |
| Aqua `test_ambiguities` | PASS | PASS | PASS |
| `Test.detect_ambiguities` pairs involving the design | 0 | 0 | 0 |
| Contributor must know | JuMP AbstractModel extension API (large surface, every JuMP function a user calls needs a method) | MOI optimizer API (`copy_to`/`optimize!(dest, src)`, attributes, `is_set_by_optimize`, result indices) | JuMP `build_variable`/`add_constraint` extension + MOI sets/attributes + `MockOptimizer`/`CachingOptimizer` internals (`model_to_optimizer_map`, `*_allowed` flags) |

Method counts come from reflection over every loaded module (methods whose `m.module` is the design module, on
functions/types owned by another module, including the design's embedded `SpikeCore` copy for B and C). A method
with a default positional argument counts twice (Julia creates two methods). Aqua ran on a throw-away package whose
`src` file is `Base.include(Base.__toplevel__, "<unmodified design file>")`, identical for the three.

## 4. Adversarial probes (`spike/audit/adversarial.jl`, same code for A, B, C)

Same objective values everywhere a solve succeeded. 43 probes per design: A 20 pass / 7 fail / 16 info, B 20/7/16,
C 20/7/16 (different failures).

| Probe | A | B | C |
|---|---|---|---|
| a1 delete an unused uncertainty, re-solve | ✓ 460 | ✓ 460 | ✓ 460 |
| a2 delete a used uncertainty `d` | silently turns `sell <= d` into `sell <= 0`; obj 0 (JuMP semantics) | same | same |
| a3 delete one component of a vector uncertainty | clear own error at `delete` | MOI `DeleteNotAllowed` (VectorOfVariables) | same as B |
| a4 delete all components, deterministic remainder | obj 1 ✓, **bound 0.0** (core, see k4) | same | same |
| b optimize! twice | same results; canonical + both inner models rebuilt (no caching) | same | same |
| c no Stage marks | buy becomes Stage(2) = d; 500/500; no warning | same | same |
| d `@expression` mixing uncertainties/decisions in constraint + objective | ✓ 460 | ✓ | ✓ |
| e1 `fix(buy, 100)` | 430 / 476.667 | same | same |
| e2 `fix(d, 100)` on an uncertainty | error **at `fix`** | error at `optimize!` (names `d`) | error at `optimize!` (names `d`) |
| f1 `set_objective_coefficient` after solve | **MethodError** | ✓ invalidates, re-solve = reference | ✓ |
| f2 new `@objective` after solve | ✓ | ✓ | ✓ |
| f3 `set_objective_sense(MIN)` (unbounded) | DUAL_INFEASIBLE, `objective_value` = −10.0 (ray value) | DUAL_INFEASIBLE, `objective_value` errors | as A |
| g `set_silent` / `unset_silent` after solve | results survive | **results invalidated** (OPTIMIZE_NOT_CALLED) | **results invalidated** |
| g `set_time_limit_sec` after solve | MethodError (not implemented) | results invalidated | results invalidated |
| g raw `set_attribute("presolve", …)` after solve | survives | `UnsupportedAttribute` error | results invalidated |
| k16 attributes reach the inner HiGHS | raw attribute yes; time limit not available | raw attribute **rejected**; time limit yes | both **accepted and silently ignored** (only `Silent` is read) |
| h1 untagged uncertainty-only row: message names it? | at declaration, no name (stack trace gives the line) | at `optimize!`, `ConstraintIndex{…}(3)`, no name | same as B |
| h2 bound on uncertainty | at `set_lower_bound`, generic | at `optimize!`, names `d` | at `optimize!`, names `d` |
| h3 integer decision in stage 2 | at `optimize!`, names `sell` | same | same |
| h4 random recourse `d*sell` | at declaration, generic "only supports affine…" | at declaration, JuMP's "not supported by the solver … choose a different solver" | at `optimize!`, "multiplies an uncertainty by a decision (random recourse)", `ConstraintIndex` not name |
| i1 unbounded primal LDR | DUAL_INFEASIBLE / INFEASIBILITY_CERTIFICATE but `objective_value` = 1.0, `value(x)` = 1.0, `objective_bound` = 1.5 (ray values) | `objective_value`/`value` error (ResultCount 0) but `has_values` true and `objective_bound` = 1.5 | as A |
| i2 infeasible model | INFEASIBLE; clear errors on `objective_value`/`value`/bound | INFEASIBLE; `ResultIndexBoundsError`; `objective_bound` = NaN | INFEASIBLE; `objective_value`, `value`, bound all return **NaN** |
| i3 primal LDR infeasible, truth feasible | INFEASIBLE, bound 0.583 | **LOCALLY_INFEASIBLE** (only dual proves INFEASIBLE), bound 0.583 | INFEASIBLE, bound 0.583 |
| j1 `set_optimizer(m, HiGHS.Optimizer)` on the LDR model | stored as the inner solver, works | accepted; `optimize!` errors "ScalarUncertainty not supported by the solver … choose a different solver" | hook reinstalls its results holder, works |
| j2 plain `Model(HiGHS.Optimizer)` + design tags | MethodError at declaration | declarations accepted; error at `optimize!` (as j1) | error at declaration (`VariableIndex-in-Stage` not supported by the solver) |
| k1 `value(p*sell)` vs `expected_value` (sell depends on p) | 600 vs 626.67; `value(objective_function)` = 0 vs `objective_value` = 26.67 | same | same |
| k2 Min sense | −460 / −486.667 ✓ | ✓ | ✓ |
| k3 `solution_summary` | MethodError | ✓ | ✓ |
| k4 deterministic LP (no uncertainty) | obj 6 ✓, **bound 0.0** | same | same |
| k5 empty model | INVALID_MODEL | same | same |
| k6 `Uncertainty(Normal(0,1))` (unbounded) | error at declaration | **no error**: LOCALLY_INFEASIBLE, bound 0.0 | **no error**: INFEASIBLE, bound 0.0 |
| k7 anonymous `set = Uncertainty(...)` | ✓ | ✓ | ✓ |
| k8 integer first stage, dual | 486.0 | 486.0 | 486.0 |
| k9 `relax_integrality` | MethodError | ✓ 486.667 | ✓ |
| k10 `dual(c)` / `has_duals` | `has_duals` MethodError; `dual` GetAttributeNotAllowed | `has_duals` false; GetAttributeNotAllowed | `has_duals` false; "No mock dual is set" |
| k11 `write_to_file` (MOF) | MethodError | MOF fails on `ScalarUncertainty` (no `head_name`) | MOF fails on `UnivariateUncertainty` |
| k12 decision in Stage(5), no stage-5 uncertainty | T = 5, warning lists unmarked `d, sell` | same | same |
| k13 `value(d)`, `num_variables` includes uncertainties | ✓ 100, 4 | ✓ | ✓ |
| k14 `@variable(m, w >= 0, set = Uncertainty(...))` | error at declaration | error at `optimize!` (names `w`) | same as B |
| k15 `copy_model` after solve, solve copy | ✓ | ✓ (optimizer re-attached by hand) | ✓ |
| k19 `is_solved_and_feasible` | ✓ | ✓ | ✓ |
| k20 `result = 2` | `objective_value(m; result = 2)` silently returns the primal 460; `value(x; result=2)` ArgumentError | `result_count` 1; result 2 NO_SOLUTION (reserved for a sampled rule) | `result_count` 2; result 2 = dual LDR (486.667, `value` 100) |

Core-level findings that every design inherits (probes a4, k4, i1, k6, math check): (1) with no uncertainty (or
none with a finite box) the dual LDR keeps no inequality row, so HiGHS returns DUAL_INFEASIBLE with
FEASIBLE_POINT/0.0 and every design reports `objective_bound = 0.0` for `Max 2x, x ≤ 3` (true 6); (2)
`FormulationResult.objective_value`/`X` are filled from an infeasibility certificate whenever `has_values` is true,
so unbounded models produce ray "values" (A and C pass them through as `objective_value`/`value`; all three as
`objective_bound`); (3) integer dual bound uses the incumbent (see §2).

## 5. Cross-design consistency

* Objective values: identical on every model solved in the three checklists and the 43 probes (460, 486.667, 486.0,
  920, 940, 1000, 430/476.667, 960, 500, 26.67, 0.583, …).
* Results API:

| | A | B | C |
|---|---|---|---|
| primal LDR | `objective_value(m)`, `value(x)` (= E[x(ξ)]) | same | same |
| dual LDR objective | `objective_bound(m)` and `objective_value(m; formulation = :dual)` | `objective_bound(m)`; `get_attribute(m, FormulationAttribute(DualLDR(), ObjectiveValue()))` | `objective_bound(m)` and `objective_value(m; result = 2)` |
| dual LDR values / rules | `value(x; formulation = :dual)`, `decision_rule(x; formulation = :dual)` | `decision_rule(x; formulation = DualLDR())` only (result 2 reserved for a sampled rule) | `value(x; result = 2)`, `decision_rule(x; dual = true)` |
| per-formulation status | `termination_status(m; formulation = …)` | `FormulationAttribute(f, TerminationStatus())` | `primal_status(m; result = r)`; one aggregated `termination_status`; `raw_status` string |
| choose formulations | `set_attribute(m, SolveDual(), false)` | `set_attribute(m, Formulations(), [PrimalLDR()])` | `set_attribute(m, Formulations(), :primal)` |
| disabled formulation queried | clear error naming `SolveDual` | `GetAttributeNotAllowed` "DualLDR() was not solved" | bound = Inf, gap = Inf; primal disabled → `objective_value` NaN, `termination_status` OPTIMAL |
| aggregated termination | primal's | dual INFEASIBLE ⇒ INFEASIBLE; primal-only infeasibility ⇒ LOCALLY_INFEASIBLE | primal's (or dual's if primal disabled) |
| JuMP generic queries (`solution_summary`, `solve_time`, `raw_status`, `result_count`, `relax_integrality`, `set_objective_coefficient`, `set_time_limit_sec`) | not available unless re-implemented (MethodError) | available | available |

* Where errors appear: A rejects at declaration everything it can see locally (untagged Ξ row, quadratic/recourse/
  conic constraints, bounds/fix/integrality on uncertainties, unbounded distributions); only whole-model errors
  (integer after stage 1, Truncate) wait for `optimize!`. B rejects at declaration what `supports_constraint` excludes
  (quadratic, conic) with JuMP's generic solver message, everything else at `optimize!`. C rejects everything at
  `optimize!` (core messages, MOI indices instead of names for constraints), except unsupported sets on a plain
  `Model(HiGHS.Optimizer)`.

## 6. Conclusions (facts only)

1. All three checklists pass with the mandatory known values asserted unweakened; B has one genuine partial item
   (5: stages absent from `print`), and A and C test items 6 (non-binding extra constraint) and 8 (gap shown on an
   auxiliary `q == 2p`/`z == p` rather than `p*sell`) less stringently than B.
2. The integer-first-stage dual bound of 486.0 is reproduced by all three designs and is a valid, tighter bound
   than 1460/3 (true integer optimum 471.41), with the caveat that the core reports the inner MIP incumbent.
3. On uniform measurements, A is 575 code lines with 114 foreign methods (≈42 pure forwards, 99 on JuMP), B is
   310 lines with 59 (42 on MOI, no forwards), C is 261 lines with 20 (no forwards); Aqua piracy and ambiguity
   tests pass for all three.
4. A is the only design where solver settings (`set_silent`, raw attributes) leave results intact and the only one
   that supports JuMP `Parameter`s, but every JuMP function not re-implemented fails with a MethodError
   (`set_objective_coefficient`, `solution_summary`, `relax_integrality`, `set_time_limit_sec`, `write_to_file`).
5. B and C invalidate results on `set_silent`/`set_time_limit_sec`; C also accepts solver attributes and time
   limits without forwarding them to the inner solver, and B rejects raw solver attributes.
6. Error timing differs: A reports most input errors at declaration (including unbounded distributions, which B
   and C accept and turn into an INFEASIBLE/LOCALLY_INFEASIBLE solve with bound 0.0); C reports all of them at
   `optimize!`, and B and C messages identify constraints by MOI index, not by name.
7. Failed inner solves are reported inconsistently: A and C expose infeasibility-certificate values as
   `objective_value`/`value` for unbounded models and C returns NaN for infeasible ones, while B raises
   result-index errors; the core's deterministic-model dual bound (0.0) is wrong in all three.
