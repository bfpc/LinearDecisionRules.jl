# Implementation brief: LinearDecisionRules.jl 0.4 (design B)

New repository: `C:\Users\bernardo.paulo\Packages\LinearDecisionRules.jl.new` (git, branch `main`).
Read its `DESIGN.md` first: it is the agreed design. Prototypes to port from (old repository, untracked):
`C:\Users\bernardo.paulo\Packages\LinearDecisionRules.jl\spike\` — `core/SpikeCore.jl` (canonical form, dense
LDRs), `optimizer/LDROptimizer.jl` (design B front-end), `hook/LDRHook.jl` (stages encoded as constraints),
`dualnnz/measure.jl` (block primal and structured dual, validated), `REPORT.md`, `audit/AUDIT.md` (bugs to fix).
The previous implementation (`src/` of the old repository) is the mathematical specification.

## Rules for every agent

- Work only in the files you own (below). Do not commit: the lead commits. End your report with a proposed
  list of atomic commits (message + files), e.g. a feature with its tests, or "Add tests for X" separately.
- Use the Bash tool for Julia. Run tests with `julia --project=. -e 'using Pkg; Pkg.test()'` from the new
  repository, or one file with `LDR_TEST_FILES=test_x.jl`. Add test-only packages to `[extras]`/`[targets]`
  in Project.toml only if needed, and say so.
- Style: JuMP's conventions, except that lines may go up to 100 columns and should not be broken
  needlessly — a simple idea stays on one line.
  - Every file starts with the copyright header of `src/LinearDecisionRules.jl`.
  - `import` packages and qualify names (`MOI.get`, `JuMP.value`), as JuMP does internally; no `using` in `src/`.
  - Internal names start with `_`; public names are exported from `src/LinearDecisionRules.jl` and documented
    with a docstring whose first line is the signature (JuMP style), then one sentence, then details/example.
  - Functions end with an explicit `return`; keyword arguments after `;`; no type piracy.
  - Comments explain why, not what; match the density of JuMP's source.
  - Types: concrete, documented fields; no `Dict{Symbol,Any}` bags.
- Tests: one module per file, `module TestX ... end; TestX.runtests()`, with `test_*` functions (see
  `test/test_aqua.jl`). Use HiGHS. Known values are exact `@test ... ≈`; never weaken one to pass.

## File ownership and interfaces

```
src/LinearDecisionRules.jl          lead (includes, exports)
src/canonical/info.jl               canonical agent: ModelInfo, UncertaintyDeclaration
src/canonical/form.jl               canonical agent: CanonicalForm, canonicalize
src/canonical/information.jl        canonical agent: InformationStructure, observes
src/canonical/support.jl            canonical agent: SupportBlock, support_blocks
src/probability/moments.jl          canonical agent: Moments (structured), moments
src/probability/distributions/*.jl  canonical agent: MvDiscreteNonParametric (port), ConfidenceMvNormal later
src/formulations/*.jl               formulations agent
src/moi/*.jl, src/jump/*.jl         front-end agent
test/...                            each agent its own files, named after the source file
```

Canonical layer (input = MOI model + metadata; no JuMP model needed):

```julia
struct UncertaintyDeclaration          # one @variable call; separate declarations are independent
    distribution::Distributions.Distribution
    variables::Vector{MOI.VariableIndex}
    stages::Vector{Union{Nothing,Int}} # per component; nothing = unmarked
end
Base.@kwdef struct ModelInfo
    uncertainties::Vector{UncertaintyDeclaration} = UncertaintyDeclaration[]
    decision_stages::Dict{MOI.VariableIndex,Int} = Dict{MOI.VariableIndex,Int}()   # absent = unmarked
    valid_rows::Set{MOI.ConstraintIndex} = Set{MOI.ConstraintIndex}()
    truncate_rows::Set{MOI.ConstraintIndex} = Set{MOI.ConstraintIndex}()
    breakpoints::Dict{MOI.VariableIndex,Vector{Float64}} = Dict{MOI.VariableIndex,Vector{Float64}}() # Phase 2
end
canonicalize(src::MOI.ModelLike, info::ModelInfo) -> CanonicalForm
_is_metadata_constraint(::Type{F}, ::Type{S}) = false   # front-end adds methods for its own sets (Uncertainty, Stage)
moments(form::CanonicalForm) -> Moments                  # structured: mean over (1, η) and covariance blocks
```

Formulation layer:

```julia
abstract type AbstractFormulation end
struct PrimalLDR <: AbstractFormulation end
struct DualLDR <: AbstractFormulation end
struct SampledLDR <: AbstractFormulation; num_scenarios::Int; seed::Int; end   # SampledLDR(N; seed = 1234)
struct FormulationResult
    formulation::AbstractFormulation
    termination_status::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    raw_status::String
    objective_value::Float64             # user's sense; NaN unless primal_status == FEASIBLE_POINT
    objective_bound::Float64             # inner solver's bound (for MIPs), user's sense
    solve_time::Float64
    X::Matrix{Float64}                   # decisions × (1 + uncertainty columns); NaN if no solution
    model::JuMP.Model                    # the reformulated model, for inspection and write_to_file
end
solve_formulation(form, f::AbstractFormulation, mom::Moments, optimizer_factory; silent, time_limit, raw_attributes) -> FormulationResult
```

## Bugs from the spike audit to fix (and test)

1. With no bounded uncertainty, the dual dropped every inequality row (deterministic `Max 2x, x ≤ 3` gave
   bound 0 instead of 6).
2. Values from an infeasibility certificate were reported as results: fill X/objective only when the inner
   primal status is FEASIBLE_POINT.
3. With integers, the dual bound must use the inner MIP's objective bound, not its incumbent.
4. Messages must name variables and constraints (`MOI.VariableName`, `MOI.ConstraintName`), not indices.
5. JuMP Parameters (`VariableIndex`-in-`MOI.Parameter`) are fixed stage-1 values in `canonicalize`.
6. The unmarked-variable warning is emitted once per `optimize!`, not repeatedly.
