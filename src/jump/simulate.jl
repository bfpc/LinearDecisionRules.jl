# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# `evaluate` and `simulate` apply the rule `X` of a formulation to outcomes of the original
# uncertainties, in batches (the columns of a matrix): `ξ = (1, η)` is lifted with
# `lift(form.lifting, ξ)` when there are breakpoints, the decisions are `x = X ξ̃`, and the
# objective and the rows of the canonical form are evaluated at `(x, ξ̃)`.

# One law of the original uncertainties, on the columns of the original ξ.
struct _OriginalLaw
    distribution::Distributions.Distribution
    columns::Vector{Int}
end

# A group of rows that violations are reported by: a constraint, or the lower or upper bound of
# the decision `i`, as `(:lower, i)` or `(:upper, i)`.
const _RowGroup = Union{MOI.ConstraintIndex,Tuple{Symbol,Int}}

# The data of the last solve of one formulation. The rows of the canonical form (an equality as a
# `≤` and a `≥`) and the bounds of the decisions are stacked as `G x ≤ H ξ̃`, with `ξ̃₁ = 1`.
struct _Simulator
    form::CanonicalForm
    X::Matrix{Float64}
    decision_index::Dict{MOI.VariableIndex,Int}
    # The original uncertainties; `uncertainties[j]` is the column `1 + j` of the original ξ.
    uncertainties::Vector{MOI.VariableIndex}
    uncertainty_names::Vector{String}
    column::Dict{MOI.VariableIndex,Int}
    laws::Vector{_OriginalLaw}
    # The box of the support of each column of the original ξ (the first entries are unused).
    lower::Vector{Float64}
    upper::Vector{Float64}
    G::_SparseMatrix
    H::_SparseMatrix
    group_of_row::Vector{Int}
    groups::Vector{_RowGroup}
end

# The laws of the original uncertainties, from the laws of the form, which are lifted if there are
# breakpoints.
function _original_laws(form::CanonicalForm)
    lifting = form.lifting
    if lifting === nothing
        return [_OriginalLaw(law.distribution, law.columns) for law in form.laws]
    end
    original = Dict(k => j for (j, k) in enumerate(lifting.kept) if j > 1 && k > 0)
    pieces = Dict(block.columns => block for block in lifting.blocks)
    return map(form.laws) do law
        if law.distribution isa LiftedDistribution
            return _OriginalLaw(law.distribution.original, pieces[law.columns].inputs)
        end
        return _OriginalLaw(law.distribution, [original[c] for c in law.columns])
    end
end

function _stack_rows(form::CanonicalForm)
    groups = _RowGroup[]
    group_index = Dict{_RowGroup,Int}()
    function group(key)
        return get!(group_index, key) do
            push!(groups, key)
            return length(groups)
        end
    end
    constraints = vcat(form.e_constraints, form.e_constraints, form.u_constraints)
    append!(constraints, form.l_constraints)
    group_of_row = Int[group(ci) for ci in constraints]
    G = vcat(form.A_e, -form.A_e, form.A_u, -form.A_l)
    H = vcat(form.B_e, -form.B_e, form.B_u, -form.B_l)
    # `x_i ≤ u_i` and `-x_i ≤ -l_i`, with the bound as `ξ̃₁` times a constant.
    I, J, V = Int[], Int[], Float64[]
    rows, rhs = Int[], Float64[]
    function bound!(i, sign, value, key)
        push!(I, length(rhs) + 1)
        push!(J, i)
        push!(V, sign)
        push!(rhs, sign * value)
        push!(group_of_row, group(key))
        return
    end
    for i in eachindex(form.decisions)
        isfinite(form.lower[i]) && bound!(i, -1.0, form.lower[i], (:lower, i))
        isfinite(form.upper[i]) && bound!(i, 1.0, form.upper[i], (:upper, i))
    end
    nb = length(rhs)
    Gb = SparseArrays.sparse(I, J, V, nb, length(form.decisions))
    Hb = SparseArrays.sparse(1:nb, fill(1, nb), rhs, nb, size(H, 2))
    return vcat(G, Gb), vcat(H, Hb), group_of_row, groups
end

function MOI.get(o::Optimizer, attr::SimulationData)
    r = _require_result(o, FormulationAttribute(attr.formulation, MOI.PrimalStatus()))
    if !_has_solution(r)
        throw(MOI.GetAttributeNotAllowed(attr, "$(attr.formulation) has no solution."))
    end
    form = o.form
    lifting = form.lifting
    uncertainties = copy(lifting === nothing ? form.uncertainties : lifting.uncertainties)
    names = copy(lifting === nothing ? form.uncertainty_names : lifting.uncertainty_names)
    laws = _original_laws(form)
    lower, upper = fill(-Inf, 1 + length(uncertainties)), fill(Inf, 1 + length(uncertainties))
    for law in laws
        lower[law.columns], upper[law.columns] = _support_box(law.distribution)
    end
    G, H, group_of_row, groups = _stack_rows(form)
    return _Simulator(
        form,
        r.X,
        copy(o.decision_rows),
        uncertainties,
        names,
        copy(o.uncertainty_columns),
        laws,
        lower,
        upper,
        G,
        H,
        group_of_row,
        groups,
    )
end

function _simulator(model::JuMP.GenericModel, f::AbstractFormulation)
    if JuMP.termination_status(model) == MOI.OPTIMIZE_NOT_CALLED
        error("The model has no solution: call `optimize!(model)` first (it changed since).")
    elseif capabilities(f).rule_status == MOI.NO_SOLUTION
        throw(
            ArgumentError(
                "$f has no decision rule to apply, since its value is a bound. Use " *
                "`PrimalLDR()` or `SampledLDR(...)`.",
            ),
        )
    end
    return MOI.get(model, SimulationData(f))
end

const _Given = Vector{Pair{JuMP.VariableRef,Float64}}

# A scenario maps each uncertainty to a value, and a container of uncertainties to a container of
# values of the same size.
function _add_scenario!(out::_Given, scenario)
    for (key, value) in scenario
        _add_pair!(out, key, value)
    end
    return out
end

_add_scenario!(out::_Given, scenario::Pair) = _add_pair!(out, scenario.first, scenario.second)

_add_pair!(out, x::JuMP.VariableRef, value::Real) = push!(out, x => Float64(value))

function _add_pair!(out, xs::AbstractArray{JuMP.VariableRef}, values::AbstractArray{<:Real})
    if size(xs) != size(values)
        throw(ArgumentError("A scenario has $(length(values)) values for $(length(xs)) variables."))
    end
    for (x, value) in zip(xs, values)
        push!(out, x => Float64(value))
    end
    return out
end

function _add_pair!(out, key, value)
    return throw(
        ArgumentError(
            "A scenario maps an uncertainty (or an array of them) to a value (or an array of " *
            "values of the same size), but it has $key => $value.",
        ),
    )
end

# The outcome ξ = (1, η) of a scenario, in the columns of the original ξ.
function _outcome(sim::_Simulator, model::JuMP.GenericModel, scenario)
    given = _add_scenario!(_Given(), scenario)
    ξ = fill(NaN, 1 + length(sim.uncertainties))
    ξ[1] = 1.0
    for (x, value) in given
        j = JuMP.owner_model(x) === model ? get(sim.column, JuMP.index(x), 0) : 0
        if j == 0
            error(
                "The variable `$x` is not an uncertainty of the model: a scenario gives the " *
                "values of the uncertainties, not of the decisions.",
            )
        elseif !isfinite(value)
            error("The value of the uncertainty `$x` in the scenario is $value.")
        end
        ξ[j] = value
    end
    missing_names = [sim.uncertainty_names[j-1] for j in eachindex(ξ) if isnan(ξ[j])]
    if !isempty(missing_names)
        error(
            "The scenario has no value for the uncertainties $(join(missing_names, ", ")): " *
            "give a value to every uncertainty of the model.",
        )
    end
    return ξ
end

function _check_policy(outside_support::Symbol)
    if !(outside_support in (:error, :warn, :ignore))
        msg = "`outside_support` is `:error`, `:warn` or `:ignore`; got `:$outside_support`."
        throw(ArgumentError(msg))
    end
    return
end

# The results of a batch of outcomes: the decisions and the objective of each, and the violation of
# each group of rows (the largest of its rows, 0 if satisfied), which is `flagged` beyond tolerance.
struct _Batch
    ξ̃::Matrix{Float64}
    decisions::Matrix{Float64}
    objective::Vector{Float64}
    excess::Matrix{Float64}
    flagged::BitMatrix
end

function _assess(sim::_Simulator, ξ::Matrix{Float64}, tolerance::Float64)
    form = sim.form
    ξ̃ = form.lifting === nothing ? ξ : lift(form.lifting, ξ)
    x = sim.X * ξ̃
    η = ξ̃[2:end, :]
    objective = vec(sum(x .* (form.P * x); dims = 1)) .+ vec(sum(x .* (form.C * ξ̃); dims = 1))
    objective .+= vec(sum(η .* (form.Q * η); dims = 1)) .+ vec(η' * form.d) .+ form.f
    lhs, rhs = sim.G * x, sim.H * ξ̃
    excess = zeros(length(sim.groups), size(ξ, 2))
    flagged = falses(length(sim.groups), size(ξ, 2))
    for s in axes(ξ, 2), r in axes(lhs, 1)
        violation = lhs[r, s] - rhs[r, s]
        violation > 0 || continue
        g = sim.group_of_row[r]
        excess[g, s] = max(excess[g, s], violation)
        scale = 1 + max(abs(lhs[r, s]), abs(rhs[r, s]))
        flagged[g, s] |= violation > tolerance * scale
    end
    return _Batch(ξ̃, x, objective, excess, flagged)
end

function _group_name(model::JuMP.GenericModel, sim::_Simulator, g::Int)
    group = sim.groups[g]
    if group isa MOI.ConstraintIndex
        return _constraint_name(JuMP.backend(model), group)
    end
    side, i = group
    name = sim.form.decision_names[i]
    bound = side == :lower ? sim.form.lower[i] : sim.form.upper[i]
    return "$name $(side == :lower ? ">=" : "<=") $(_number_string(bound))"
end

# For each outcome, why it is not in the support Ξ, or `nothing`: outside the box of a law, or a
# violated `ValidInequality()` (the convex-hull rows of the liftings follow from the lifting).
function _support_reasons(
    model::JuMP.GenericModel,
    sim::_Simulator,
    ξ::Matrix{Float64},
    batch::_Batch,
    tolerance::Float64,
)
    form = sim.form
    valid = findall(!isnothing, form.valid_constraints)
    slack = form.W_valid[valid, :] * batch.ξ̃[2:end, :] .- form.h_valid[valid]
    reasons = Vector{Union{Nothing,String}}(nothing, size(ξ, 2))
    for s in axes(ξ, 2)
        for j in 2:size(ξ, 1)
            lo, hi = sim.lower[j], sim.upper[j]
            slack_lo, slack_hi = ξ[j, s] - lo, hi - ξ[j, s]
            if min(slack_lo, slack_hi) < -tolerance * (1 + abs(lo) + abs(hi))
                name = sim.uncertainty_names[j-1]
                reasons[s] = "$name = $(ξ[j, s]) is outside its support [$lo, $hi]"
                break
            end
        end
        reasons[s] === nothing || continue
        for (k, r) in enumerate(valid)
            if slack[k, s] < -tolerance * (1 + abs(form.h_valid[r]))
                name = _constraint_name(JuMP.backend(model), form.valid_constraints[r])
                reasons[s] = "the valid inequality $name is violated by $(-slack[k, s])"
                break
            end
        end
    end
    return reasons
end

"""
    Evaluation

The result of [`evaluate`](@ref): the decision rule applied to one scenario.

## Fields

 * `formulation::AbstractFormulation`
 * `scenario::Dict{JuMP.VariableRef,Float64}`: the value of each uncertainty.
 * `decisions::Dict{JuMP.VariableRef,Float64}`: the value of each decision.
 * `objective_value::Float64`: the objective at the scenario, in the sense of the model.
 * `max_violation::Float64`: the largest violation of a constraint or of a bound of a decision,
   0 if there is none.
 * `violations::Vector{Pair{String,Float64}}`: the constraints (or bounds, as `x >= 0`) violated by
   more than `tolerance`, with their violation, the largest first. A violation `v` of a row with
   sides `a` and `b` counts if `v > tolerance * (1 + max(|a|, |b|))`.
 * `in_support::Bool`: whether the scenario is in the support Ξ.
 * `tolerance::Float64`

Read a value with `value(result, x)` or `result[x]`, for a decision, an uncertainty or a
parameter, and `objective_value(result)`.
"""
struct Evaluation
    formulation::AbstractFormulation
    scenario::Dict{JuMP.VariableRef,Float64}
    decisions::Dict{JuMP.VariableRef,Float64}
    objective_value::Float64
    max_violation::Float64
    violations::Vector{Pair{String,Float64}}
    in_support::Bool
    tolerance::Float64
end

function _lookup(r::Evaluation, x::JuMP.VariableRef)
    haskey(r.decisions, x) && return r.decisions[x]
    haskey(r.scenario, x) && return r.scenario[x]
    JuMP.is_parameter(x) && return JuMP.parameter_value(x)
    return throw(KeyError(x))
end

"""
    JuMP.value(result::Evaluation, x::JuMP.VariableRef) -> Float64
    JuMP.value(result::Evaluation, expr::JuMP.AbstractJuMPScalar) -> Float64

The value of a decision, an uncertainty or a parameter, or of an expression of them, in the scenario
of `result`. `result[x]` is the same as `value(result, x)` for a variable.
"""
JuMP.value(r::Evaluation, x::JuMP.VariableRef) = _lookup(r, x)

JuMP.value(r::Evaluation, expr::JuMP.AbstractJuMPScalar) = JuMP.value(x -> _lookup(r, x), expr)

Base.getindex(r::Evaluation, x::JuMP.VariableRef) = _lookup(r, x)

"""
    JuMP.objective_value(result::Evaluation) -> Float64

The objective value at the scenario of `result`.
"""
JuMP.objective_value(r::Evaluation) = r.objective_value

function Base.show(io::IO, r::Evaluation)
    print(io, "Evaluation of ", r.formulation, ": objective ", r.objective_value)
    if isempty(r.violations)
        print(io, ", feasible (max violation ", r.max_violation, ")")
    else
        print(io, ", ", length(r.violations), " violated: ", join(first.(r.violations), ", "))
    end
    return
end

# What to do with an outcome outside the support: `:error`, `:warn` or `:ignore`.
function _outside_support(policy::Symbol, reason::String, where_::String)
    message = "The scenario$where_ is outside the support: $reason."
    if policy == :error
        error(message * " Use `outside_support = :warn` or `:ignore` to evaluate it anyway.")
    elseif policy == :warn
        @warn message
    end
    return
end

"""
    evaluate(
        model::JuMP.Model,
        scenario::AbstractDict;
        formulation::AbstractFormulation = PrimalLDR(),
        tolerance::Real = 1e-6,
        outside_support::Symbol = :error,
    ) -> Evaluation
    evaluate(model, scenario::Pair...; kwargs...)

Apply the decision rule of `formulation` to a scenario of the uncertainties, and return the
[`Evaluation`](@ref): the value of every decision, the objective at the scenario, and the
constraints that it violates.

`scenario` maps each uncertainty of the model to a value, as a `Dict` or as pairs; an array of
uncertainties maps to an array of values. Every uncertainty needs a value, in the original
coordinates (not lifted by [`BreakPoints`](@ref)), or `evaluate` is an error that names the missing
ones. The scenario must be in the support of the uncertainties (their bounds, and the constraints
tagged `ValidInequality()`): `outside_support` is `:error`, or `:warn` or `:ignore` to evaluate it
anyway.

The rule of [`PrimalLDR`](@ref) violates no constraint on the support, up to the tolerance of the
inner solver; that of [`SampledLDR`](@ref) may, outside the scenarios that it was solved on.
A violation larger than `tolerance` (relative to the size of the row, see [`Evaluation`](@ref)) is
listed in `violations`.

## Example

```julia
optimize!(model)
r = evaluate(model, d => 90.0)
value(r, sell)       # the decision `sell` at d = 90
r.objective_value
r.violations         # constraints violated by more than the tolerance
```
"""
function evaluate(
    model::JuMP.GenericModel,
    scenario::AbstractDict;
    formulation::AbstractFormulation = PrimalLDR(),
    tolerance::Real = 1e-6,
    outside_support::Symbol = :error,
)
    _check_policy(outside_support)
    sim = _simulator(model, formulation)
    ξ = reshape(_outcome(sim, model, scenario), :, 1)
    batch = _assess(sim, ξ, Float64(tolerance))
    reason = only(_support_reasons(model, sim, ξ, batch, Float64(tolerance)))
    reason === nothing || _outside_support(outside_support, reason, "")
    decisions = Dict(
        JuMP.VariableRef(model, v) => batch.decisions[i, 1] for (v, i) in sim.decision_index
    )
    given = Dict(JuMP.VariableRef(model, v) => ξ[j] for (v, j) in sim.column)
    violated = findall(batch.flagged[:, 1])
    violations = [_group_name(model, sim, g) => batch.excess[g, 1] for g in violated]
    sort!(violations; by = last, rev = true)
    return Evaluation(
        formulation,
        given,
        decisions,
        batch.objective[1],
        maximum(batch.excess; init = 0.0),
        violations,
        reason === nothing,
        Float64(tolerance),
    )
end

function evaluate(model::JuMP.GenericModel, scenario::Pair...; kwargs...)
    return evaluate(model, Dict{Any,Any}(scenario); kwargs...)
end

function evaluate(model::JuMP.GenericModel, scenario::AbstractVector{<:Pair}; kwargs...)
    return evaluate(model, Dict{Any,Any}(scenario); kwargs...)
end

"""
    Simulation

The result of [`simulate`](@ref): the decision rule applied to many scenarios.

## Fields

 * `formulation::AbstractFormulation`
 * `objectives::Vector{Float64}`: the objective at each scenario.
 * `objective_mean::Float64`, `objective_stderr::Float64`: the sample mean of the objective and its
   standard error (`NaN` with one scenario).
 * `violated::BitVector`: whether each scenario violates a constraint by more than `tolerance`.
 * `max_violations::Vector{Float64}`: the largest violation at each scenario.
 * `violation_fraction::Float64`: the fraction of scenarios with a violation.
 * `max_violation::Float64`: the largest violation of all the scenarios.
 * `violation_counts::Vector{Pair{String,Int}}`: for each constraint (or bound, as `x >= 0`) that
   some scenario violates, the number of scenarios that do, the largest first.
 * `decisions::Union{Nothing,Dict{JuMP.VariableRef,Vector{Float64}}}`: the value of each decision
   at each scenario, if `keep_decisions = true`.
 * `num_outside_support::Int`: the number of scenarios outside the support (given by the user).
 * `tolerance::Float64`: as in [`Evaluation`](@ref).
"""
struct Simulation
    formulation::AbstractFormulation
    objectives::Vector{Float64}
    objective_mean::Float64
    objective_stderr::Float64
    violated::BitVector
    max_violations::Vector{Float64}
    violation_fraction::Float64
    max_violation::Float64
    violation_counts::Vector{Pair{String,Int}}
    decisions::Union{Nothing,Dict{JuMP.VariableRef,Vector{Float64}}}
    num_outside_support::Int
    tolerance::Float64
end

function Base.show(io::IO, s::Simulation)
    n = length(s.objectives)
    print(io, "Simulation of ", s.formulation, " on ", n, " scenarios\n")
    mean, stderr = round(s.objective_mean; sigdigits = 8), round(s.objective_stderr; sigdigits = 3)
    print(io, "  objective: mean ", mean, " ± ", stderr, " (standard error)\n")
    percent = round(100 * s.violation_fraction; sigdigits = 3)
    print(io, "  violations: ", sum(s.violated), " scenarios (", percent, "%)")
    print(io, ", max violation ", round(s.max_violation; sigdigits = 3))
    print(io, " (tolerance ", s.tolerance, ")")
    if s.num_outside_support > 0
        print(io, "\n  ", s.num_outside_support, " scenarios outside the support")
    end
    return
end

# The scenarios `first:last`, as the columns of a matrix of outcomes: drawn from the original laws.
function _draw_chunk(rng::Random.AbstractRNG, sim::_Simulator, num::Int)
    ξ = ones(1 + length(sim.uncertainties), num)
    for law in sim.laws
        draws = rand(rng, law.distribution, num)
        univariate = law.distribution isa Distributions.UnivariateDistribution
        ξ[law.columns, :] = univariate ? reshape(draws, 1, :) : draws
    end
    return ξ
end

function _simulate(
    model::JuMP.GenericModel,
    sim::_Simulator,
    formulation::AbstractFormulation,
    n::Int,
    outcomes::Function,
    outside_support::Union{Nothing,Symbol},
    keep_decisions::Bool,
    tolerance::Float64,
)
    objectives, max_violations = zeros(n), zeros(n)
    violated = falses(n)
    counts = zeros(Int, length(sim.groups))
    kept = keep_decisions ? zeros(size(sim.X, 1), n) : nothing
    outside, first_reason = 0, ""
    # Batches bound the memory: one value per row and scenario.
    chunk = clamp(500_000 ÷ (size(sim.G, 1) + length(sim.groups) + size(sim.X, 1) + 1), 1, 4096)
    for start in 1:chunk:n
        scenarios = start:min(start + chunk - 1, n)
        ξ = outcomes(start, length(scenarios))
        batch = _assess(sim, ξ, tolerance)
        objectives[scenarios] = batch.objective
        max_violations[scenarios] = maximum(batch.excess; dims = 1, init = 0.0)
        violated[scenarios] = vec(any(batch.flagged; dims = 1))
        counts .+= vec(sum(batch.flagged; dims = 2))
        kept === nothing || (kept[:, scenarios] = batch.decisions)
        outside_support === nothing && continue
        reasons = _support_reasons(model, sim, ξ, batch, tolerance)
        for (k, reason) in enumerate(reasons)
            reason === nothing && continue
            outside += 1
            if outside == 1
                first_reason = reason
                outside_support == :error && _outside_support(:error, reason, " $(start + k - 1)")
            end
        end
    end
    if outside > 0 && outside_support == :warn
        @warn "$outside scenarios are outside the support; the first has $first_reason."
    end
    mean = sum(objectives) / n
    stderr = n > 1 ? sqrt(sum(abs2, objectives .- mean) / (n - 1) / n) : NaN
    violation_counts = [_group_name(model, sim, g) => c for (g, c) in enumerate(counts) if c > 0]
    sort!(violation_counts; by = last, rev = true)
    decisions = nothing
    if kept !== nothing
        decisions = Dict(JuMP.VariableRef(model, v) => kept[i, :] for (v, i) in sim.decision_index)
    end
    return Simulation(
        formulation,
        objectives,
        mean,
        stderr,
        violated,
        max_violations,
        count(violated) / n,
        maximum(max_violations),
        violation_counts,
        decisions,
        outside,
        tolerance,
    )
end

"""
    simulate(
        model::JuMP.Model,
        n::Integer;
        formulation::AbstractFormulation = PrimalLDR(),
        rng::Random.AbstractRNG = Random.default_rng(),
        keep_decisions::Bool = false,
        tolerance::Real = 1e-6,
    ) -> Simulation
    simulate(
        model::JuMP.Model,
        scenarios::AbstractVector;
        formulation::AbstractFormulation = PrimalLDR(),
        keep_decisions::Bool = false,
        tolerance::Real = 1e-6,
        outside_support::Symbol = :error,
    ) -> Simulation

Apply the decision rule of `formulation` to `n` scenarios drawn from the laws of the uncertainties
(in the original, not lifted, coordinates, one independent draw of each declaration), or to the
`scenarios` given, each as in [`evaluate`](@ref), for example the simulations of another method.

The [`Simulation`](@ref) has the objective at each scenario, its sample mean and standard error, and
the violations of the constraints: the fraction of scenarios with a violation, the largest one, and
how often each constraint is violated. The decisions of each scenario are kept only with
`keep_decisions = true`, since they take `n` values per decision.

For the rule of [`PrimalLDR`](@ref), which is feasible on the support, there is no violation (up to
the tolerance), and the sample mean of the objective estimates `objective_value(model)`. The rule of
[`SampledLDR`](@ref) is feasible on the scenarios that it was solved on only, so `simulate` reports
its out-of-sample violations. Given `scenarios` must be in the support, with `outside_support` as in
`evaluate`.

## Example

```julia
optimize!(model)
s = simulate(model, 10_000; rng = Random.Xoshiro(1))
s.objective_mean, s.objective_stderr   # compare with objective_value(model)
s.violation_fraction                   # 0.0 for the primal rule
simulate(model, [Dict(d => 90.0), Dict(d => 110.0)])
```
"""
function simulate(
    model::JuMP.GenericModel,
    n::Integer;
    formulation::AbstractFormulation = PrimalLDR(),
    rng::Random.AbstractRNG = Random.default_rng(),
    keep_decisions::Bool = false,
    tolerance::Real = 1e-6,
)
    if n < 1
        throw(ArgumentError("The number of scenarios must be positive; got $n."))
    end
    sim = _simulator(model, formulation)
    draw(_, num) = _draw_chunk(rng, sim, num)
    tol = Float64(tolerance)
    return _simulate(model, sim, formulation, Int(n), draw, nothing, keep_decisions, tol)
end

function simulate(
    model::JuMP.GenericModel,
    scenarios::AbstractVector;
    formulation::AbstractFormulation = PrimalLDR(),
    keep_decisions::Bool = false,
    tolerance::Real = 1e-6,
    outside_support::Symbol = :error,
)
    _check_policy(outside_support)
    if isempty(scenarios)
        throw(ArgumentError("There are no scenarios to simulate."))
    end
    sim = _simulator(model, formulation)
    function outcomes(start, num)
        ξ = zeros(1 + length(sim.uncertainties), num)
        for k in 1:num
            ξ[:, k] = _outcome(sim, model, scenarios[start+k-1])
        end
        return ξ
    end
    return _simulate(
        model,
        sim,
        formulation,
        length(scenarios),
        outcomes,
        outside_support,
        keep_decisions,
        Float64(tolerance),
    )
end
