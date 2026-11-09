# Phase 0 spike (archived)

The throwaway prototype that chose the user layer of version 0.4 (decisions 1 and 2 of `DESIGN.md`),
copied here from the untracked `spike/` folder of the legacy repository so that it stays available.
Paths inside these files refer to the legacy repository (`LinearDecisionRules.jl`, branch `Multistage`, now `legacy`)
and to the environments there; the code is not maintained and is not part of the package.

- `SPIKE.md`: the brief, the checklist and the decision criteria given to every design.
- `core/`: the shared dense core (`SpikeCore`).
- `wrapper/` (design A, `LDRModel <: AbstractModel`), `optimizer/` (design B, chosen), `hook/` (design C):
  each design with its checklist.
- `audit/`: the independent audit (`AUDIT.md`, `metrics.jl`, `adversarial.jl`).
- `dualnnz/`: the hydro experiment that validated the block primal and the structured dual.
- `REPORT.md`: the comparison and the recommendation.
- `IMPLEMENTATION_BRIEF.md`: the conventions given to the agents that wrote Phases 1–3.
- `legacy_env/`: the environment used to run the legacy package (for `test/reference/generate_legacy_values.jl`).
