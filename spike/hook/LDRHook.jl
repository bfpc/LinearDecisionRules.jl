"""
    LDRHook

Design C of the Phase 0 spike: a plain `JuMP.Model` whose LDR data lives in MOI constraints/attributes of its
cache (plus a small typed struct in `model.ext[:LDR]`), solved by `JuMP.set_optimize_hook`.

* `Stage(t)` and `Uncertainty(dist)` are MOI *sets* (`x in Stage(2)`); `@variable(..., Stage(t))` composes via
  `build_variable`/`add_variable` on our own types.
* Constraint tags (`ValidInequality()`, `Truncate()`, `Stage(t)`) and `BreakPoints()` are MOI attributes.
* The backend's optimizer is a `MOI.Utilities.MockOptimizer` that only *holds results*: after the LDR solve the
  hook copies the model into it and writes the results, so JuMP's own `objective_value`, `objective_bound`,
  `termination_status`, `value`, `primal_status`, `relative_gap` (and `result = 2` for the dual LDR) work.
"""
module LDRHook

import Distributions
import JuMP
import MathOptInterface as MOI

include(joinpath(@__DIR__, "..", "core", "SpikeCore.jl"))

export LDRModel, Stage, Uncertainty, ValidInequality, Truncate, BreakPoints, Formulations,
    stage, set_stage, decision_rule, expected_value, constraint_tag, uncertainty_distribution

const U = MOI.Utilities
const VI = MOI.VariableIndex

# ------------------------------------------------------------------------------------------ tags and sets

"`Stage(t)`: MOI set (`x in Stage(2)`), variable tag in `@variable`, and constraint tag in `@constraint`."
struct Stage <: MOI.AbstractScalarSet
    t::Int
end

"Set of an uncertainty variable with a univariate distribution (build it with [`Uncertainty`](@ref))."
struct UnivariateUncertainty <: MOI.AbstractScalarSet
    dist::Distributions.UnivariateDistribution
end

"Set of a vector of uncertainty variables with a joint multivariate distribution."
struct MultivariateUncertainty <: MOI.AbstractVectorSet
    dist::Distributions.MultivariateDistribution
end

"`Uncertainty(dist)`: use as `d in Uncertainty(Uniform(80, 120))` (scalar) or `d[1:2] in Uncertainty(mvdist)`."
Uncertainty(dist::Distributions.UnivariateDistribution) = UnivariateUncertainty(dist)
Uncertainty(dist::Distributions.MultivariateDistribution) = MultivariateUncertainty(dist)

MOI.dimension(s::MultivariateUncertainty) = length(s.dist)
Base.copy(s::MultivariateUncertainty) = s

"Common supertype of constraint tags; an `MOI.AbstractSet` so that MOI can copy attribute values without a `map_indices` method."
abstract type AbstractTag <: MOI.AbstractSet end
"Constraint tag: an uncertainty-only row that describes the support Ξ."
struct ValidInequality <: AbstractTag end
"Constraint tag: an uncertainty-only row that truncates the distribution (not in the spike)."
struct Truncate <: AbstractTag end

"Variable attribute (`set_attribute(d, BreakPoints(), [100.0])`): stored in the model cache."
struct BreakPoints <: MOI.AbstractVariableAttribute end
"Constraint attribute holding the tag of a constraint."
struct ConstraintTag <: MOI.AbstractConstraintAttribute end
"Model attribute with value `:both` (default), `:primal` or `:dual`: which formulations `optimize!` runs."
struct Formulations <: MOI.AbstractModelAttribute end

SpikeCore.is_metadata_constraint(::Type{VI}, ::Type{Stage}) = true
SpikeCore.is_metadata_constraint(::Type{VI}, ::Type{UnivariateUncertainty}) = true
SpikeCore.is_metadata_constraint(::Type{MOI.VectorOfVariables}, ::Type{MultivariateUncertainty}) = true

# printing: JuMP prints `x in <set>` using `show` of the set
Base.show(io::IO, s::Stage) = print(io, "Stage(", s.t, ")")
Base.show(io::IO, s::UnivariateUncertainty) = print(io, "Uncertainty(", s.dist, ")")
Base.show(io::IO, s::MultivariateUncertainty) = print(io, "Uncertainty(", typeof(s.dist).name.name, " of dim ", length(s.dist), ")")

# ------------------------------------------------------------------------------------------ typed model data

mutable struct LDRData
    factory::Any
    bridges::Bool
    canonical::Union{Nothing,SpikeCore.Canonical}
    primal::Union{Nothing,SpikeCore.FormulationResult}
    dual::Union{Nothing,SpikeCore.FormulationResult}
    seconds::Float64
end

_data(model::JuMP.Model) = get(model.ext, :LDR, nothing)::Union{Nothing,LDRData}

# the optimizer under the model: a mock over a universal fallback, so it accepts every constraint/attribute
_results_optimizer() = U.MockOptimizer(U.UniversalFallback(U.Model{Float64}());
    eval_objective_value = false, eval_dual_objective_value = false, eval_variable_constraint_dual = false)
_allow!(mk, b::Bool) = (mk.add_var_allowed = mk.add_con_allowed = mk.modify_allowed = mk.delete_allowed = b)

"""
    LDRModel(optimizer_factory)

A plain `JuMP.Model` with LDR data in `model.ext[:LDR]` and an optimize hook. `optimizer_factory` (e.g.
`HiGHS.Optimizer`) is used for the LDR reformulations.
"""
function LDRModel(factory; bridges::Bool = false)
    model = JuMP.Model(_results_optimizer; add_bridges = bridges)
    _attach!(model, LDRData(factory, bridges, nothing, nothing, nothing, 0.0))
    return model
end

function _attach!(model::JuMP.Model, data::LDRData)
    model.ext[:LDR] = data
    JuMP.set_optimize_hook(model, _optimize_hook)
    return model
end

# `copy_model` copies the cache (with all sets/attributes) and the hook, but not the optimizer
function JuMP.copy_extension_data(data::LDRData, new_model::JuMP.Model, ::JuMP.Model)
    JuMP.set_optimizer(new_model, _results_optimizer; add_bridges = data.bridges)
    return LDRData(data.factory, data.bridges, nothing, nothing, nothing, 0.0)
end

# ------------------------------------------------------------------------------------------ @variable tags

"A variable (or variable-in-set) builder carrying a `Stage` tag."
struct StagedVariable{V<:JuMP.AbstractVariable} <: JuMP.AbstractVariable
    variable::V
    stage::Int
end

JuMP.build_variable(error_fn::Function, info::JuMP.VariableInfo, s::Stage) =
    StagedVariable(JuMP.build_variable(error_fn, info), s.t)
# `d in Uncertainty(dist), Stage(t)`
JuMP.build_variable(::Function, v::StagedVariable, set::UnivariateUncertainty) =
    StagedVariable(JuMP.VariableConstrainedOnCreation(v.variable, set), v.stage)
JuMP.build_variable(::Function, vs::AbstractArray{<:StagedVariable}, set::UnivariateUncertainty) =
    map(v -> StagedVariable(JuMP.VariableConstrainedOnCreation(v.variable, set), v.stage), vs)
JuMP.add_variable(model::JuMP.Model, vs::AbstractArray{<:StagedVariable}, names::AbstractArray{<:String}) =
    JuMP.add_variable.(model, vs, names)
JuMP.build_variable(::Function, vs::Vector{<:StagedVariable}, set::MultivariateUncertainty) =
    StagedVariables(JuMP.VariablesConstrainedOnCreation([v.variable for v in vs], set), [v.stage for v in vs])

"A vector of variables constrained on creation, with one stage per component."
struct StagedVariables{V<:JuMP.AbstractVariable} <: JuMP.AbstractVariable
    variables::V
    stages::Vector{Int}
end

function JuMP.add_variable(model::JuMP.Model, v::StagedVariables, names)
    xs = JuMP.add_variable(model, v.variables, names)
    foreach((x, t) -> JuMP.add_constraint(model, JuMP.ScalarConstraint(x, Stage(t))), xs, v.stages)
    return xs
end

function JuMP.add_variable(model::JuMP.Model, v::StagedVariable, name::String = "")
    x = JuMP.add_variable(model, v.variable, name)
    JuMP.add_constraint(model, JuMP.ScalarConstraint(x, Stage(v.stage)))
    return x
end

# ------------------------------------------------------------------------------------------ @constraint tags

"A constraint builder carrying a tag."
struct TaggedConstraint{C<:JuMP.AbstractConstraint,T} <: JuMP.AbstractConstraint
    constraint::C
    tag::T
end

const _Tag = Union{ValidInequality,Truncate,Stage}
const _ScalarSet = Union{MOI.LessThan,MOI.GreaterThan,MOI.EqualTo,MOI.Interval}

JuMP.build_constraint(error_fn::Function, f, set::_ScalarSet, tag::_Tag; kwargs...) =
    TaggedConstraint(JuMP.build_constraint(error_fn, f, set; kwargs...), tag)

function JuMP.add_constraint(model::JuMP.Model, c::TaggedConstraint, name::String = "")
    ref = JuMP.add_constraint(model, c.constraint, name)
    MOI.set(model, ConstraintTag(), ref, c.tag)
    return ref
end

"`constraint_tag(cref)`: the tag given in `@constraint(m, expr, tag)`, or `nothing`."
constraint_tag(cref::JuMP.ConstraintRef) = MOI.get(JuMP.owner_model(cref), ConstraintTag(), cref)

# ------------------------------------------------------------------------------------------ queries on variables

# `x in Stage(t)` constraints: variable index => (constraint index, stage)
function _stage_constraints(src)
    return Dict{VI,Tuple{MOI.ConstraintIndex{VI,Stage},Int}}(
        MOI.get(src, MOI.ConstraintFunction(), ci) => (ci, MOI.get(src, MOI.ConstraintSet(), ci).t)
        for ci in MOI.get(src, MOI.ListOfConstraintIndices{VI,Stage}()))
end

"`stage(x)`: the declared stage of the variable, or `nothing` if unmarked."
function stage(x::JuMP.VariableRef)
    d = _stage_constraints(JuMP.backend(JuMP.owner_model(x)))
    return haskey(d, JuMP.index(x)) ? d[JuMP.index(x)][2] : nothing
end

"`set_stage(x, t)`: declare or change the stage of a variable."
function set_stage(x::JuMP.VariableRef, t::Integer)
    model = JuMP.owner_model(x)
    d = _stage_constraints(JuMP.backend(model))
    haskey(d, JuMP.index(x)) && JuMP.delete(model, JuMP.constraint_ref_with_index(model, d[JuMP.index(x)][1]))
    JuMP.add_constraint(model, JuMP.ScalarConstraint(x, Stage(t)))
    return x
end

"`uncertainty_distribution(x)`: the distribution `x` was declared with, or `nothing` for a decision."
function uncertainty_distribution(x::JuMP.VariableRef)
    src = JuMP.backend(JuMP.owner_model(x))
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{VI,UnivariateUncertainty}())
        MOI.get(src, MOI.ConstraintFunction(), ci) == JuMP.index(x) && return MOI.get(src, MOI.ConstraintSet(), ci).dist
    end
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorOfVariables,MultivariateUncertainty}())
        JuMP.index(x) in MOI.get(src, MOI.ConstraintFunction(), ci).variables &&
            return MOI.get(src, MOI.ConstraintSet(), ci).dist
    end
    return nothing
end

# ------------------------------------------------------------------------------------------ ModelInfo from the cache

function _model_info(src)
    stages = _stage_constraints(src)
    decls = SpikeCore.UncertaintyDecl[]
    unc = Set{VI}()
    stage_or_nothing(v) = haskey(stages, v) ? stages[v][2] : nothing
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{VI,UnivariateUncertainty}())
        v = MOI.get(src, MOI.ConstraintFunction(), ci)
        push!(unc, v)
        push!(decls, SpikeCore.UncertaintyDecl(MOI.get(src, MOI.ConstraintSet(), ci).dist, [v],
            Union{Int,Nothing}[stage_or_nothing(v)]))
    end
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorOfVariables,MultivariateUncertainty}())
        vs = MOI.get(src, MOI.ConstraintFunction(), ci).variables
        union!(unc, vs)
        push!(decls, SpikeCore.UncertaintyDecl(MOI.get(src, MOI.ConstraintSet(), ci).dist, vs,
            Union{Int,Nothing}[stage_or_nothing(v) for v in vs]))
    end
    dstage = Dict{VI,Int}(v => s for (v, (_, s)) in stages if !(v in unc))
    valid, trunc = Set{MOI.ConstraintIndex}(), Set{MOI.ConstraintIndex}()
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        ConstraintTag() in MOI.get(src, MOI.ListOfConstraintAttributesSet{F,S}()) || continue
        for ci in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
            tag = MOI.get(src, ConstraintTag(), ci)
            tag isa ValidInequality && push!(valid, ci)
            tag isa Truncate && push!(trunc, ci)
        end
    end
    return SpikeCore.ModelInfo(uncertainties = decls, decision_stage = dstage, valid_rows = valid,
        truncate_rows = trunc)
end

function _formulations(src)
    f = MOI.get(src, Formulations())
    return (primal = f !== :dual, dual = f !== :primal)
end

function _silent(src)
    s = try MOI.get(src, MOI.Silent()) catch; nothing end
    return s === nothing ? true : s
end

# ------------------------------------------------------------------------------------------ the hook

function _optimize_hook(model::JuMP.Model; kwargs...)
    isempty(kwargs) || error("LDR `optimize!` takes no keyword arguments; got $(keys(kwargs)).")
    data = _data(model)
    # a user-called `set_optimizer` would have replaced the results holder
    JuMP.unsafe_backend(model) isa U.MockOptimizer ||
        JuMP.set_optimizer(model, _results_optimizer; add_bridges = data.bridges)
    src = JuMP.backend(model)
    # forget old results first, so that a failure below never leaves stale ones
    data.canonical = data.primal = data.dual = nothing
    U.reset_optimizer(src)
    t0 = time_ns()
    forms, silent = _formulations(src), _silent(src)
    c = SpikeCore.canonicalize(src, _model_info(src))
    primal = forms.primal ? SpikeCore.solve_primal(c, data.factory; silent) : nothing
    dual = forms.dual ? SpikeCore.solve_dual(c, data.factory; silent) : nothing
    data.canonical, data.primal, data.dual, data.seconds = c, primal, dual, (time_ns() - t0) / 1e9
    _publish!(model, data)
    return
end

# Copy the model into the results optimizer (through JuMP's own `optimize!`), then write the results into it.
function _publish!(model::JuMP.Model, data::LDRData)
    src = JuMP.backend(model)
    mk = JuMP.unsafe_backend(model)
    _allow!(mk, true)
    JuMP.optimize!(model; ignore_optimize_hook = true)
    map = src.model_to_optimizer_map
    c = data.canonical
    MOI.set(mk, MOI.ResultCount(), 2)
    MOI.set(mk, MOI.RawStatusString(), "LDR primal: $(data.primal === nothing ? "disabled" : data.primal.termination_status); " *
                                     "dual: $(data.dual === nothing ? "disabled" : data.dual.termination_status)")
    MOI.set(mk, MOI.SolveTimeSec(), data.seconds)
    for (r, res) in enumerate((data.primal, data.dual))
        if res === nothing
            MOI.set(mk, MOI.PrimalStatus(r), MOI.NO_SOLUTION)
            continue
        end
        MOI.set(mk, MOI.PrimalStatus(r), res.primal_status)
        MOI.set(mk, MOI.ObjectiveValue(r), res.objective_value)
        for vi in MOI.get(src, MOI.ListOfVariableIndices())
            MOI.set(mk, MOI.VariablePrimal(r), map[vi], SpikeCore.expected_decision(c, res, vi))
        end
    end
    first_res = something(data.primal, data.dual, Some(nothing))
    MOI.set(mk, MOI.TerminationStatus(),
        first_res === nothing ? MOI.OPTIMIZE_NOT_CALLED : first_res.termination_status)
    # the dual LDR objective is the bound; with the dual disabled the bound is trivial (and the gap infinite)
    bound = data.dual !== nothing ? data.dual.objective_value : (c.sense == MOI.MAX_SENSE ? Inf : -Inf)
    MOI.set(mk, MOI.ObjectiveBound(), bound)
    p = data.primal === nothing ? NaN : data.primal.objective_value
    MOI.set(mk, MOI.RelativeGap(), isfinite(bound) && isfinite(p) ? abs(bound - p) / max(abs(p), eps()) : Inf)
    _allow!(mk, false)    # from now on any model modification resets the optimizer, i.e. invalidates the results
    return
end

# ------------------------------------------------------------------------------------------ result helpers

function _result(x::JuMP.VariableRef, dual::Bool)
    data = _data(JuMP.owner_model(x))
    data === nothing && error("Not an LDRModel.")
    JuMP.termination_status(JuMP.owner_model(x)) == MOI.OPTIMIZE_NOT_CALLED &&
        error("No LDR solution: call `optimize!` (the model was modified or never solved).")
    res = dual ? data.dual : data.primal
    res === nothing && error("The $(dual ? "dual" : "primal") formulation is disabled (see `Formulations`).")
    return data.canonical, res
end

"""
    DecisionRule

Affine rule `x(η) = constant + Σ coefficients[u] * u`; callable as `rule(d => 90.0)`, `rule(Dict(d => 90.0))`
or `rule(90.0)` when there is a single uncertainty.
"""
struct DecisionRule
    constant::Float64
    coefficients::Dict{JuMP.VariableRef,Float64}
end

function (r::DecisionRule)(values::AbstractDict)
    return r.constant + sum((c * values[u] for (u, c) in r.coefficients); init = 0.0)
end
(r::DecisionRule)(pairs::Pair...) = r(Dict(pairs))
function (r::DecisionRule)(v::Real)
    length(r.coefficients) <= 1 || error("The rule depends on $(length(r.coefficients)) uncertainties; pass `u => value` pairs.")
    return r.constant + sum((c * v for (_, c) in r.coefficients); init = 0.0)
end

"`decision_rule(x; dual = false)`: the affine rule of `x` (a [`DecisionRule`](@ref)) from the primal or dual LDR."
function decision_rule(x::JuMP.VariableRef; dual::Bool = false)
    c, res = _result(x, dual)
    k, coefs = SpikeCore.rule(c, res, JuMP.index(x))
    model = JuMP.owner_model(x)
    return DecisionRule(k, Dict(JuMP.VariableRef(model, vi) => a for (vi, a) in coefs))
end

function _any_variable(expr)
    expr isa JuMP.AbstractVariableRef && return expr
    for (_, v) in JuMP.linear_terms(expr)
        return v
    end
    expr isa JuMP.GenericQuadExpr && for (_, v, _) in JuMP.quad_terms(expr)
        return v
    end
    return nothing
end

"`expected_value(expr; dual = false)`: exact `E[expr(x(ξ), η)]` for an affine or quadratic JuMP expression."
function expected_value(expr::JuMP.AbstractJuMPScalar; dual::Bool = false)
    x = _any_variable(expr)
    x === nothing && return JuMP.value(_ -> 0.0, expr)
    c, res = _result(x, dual)
    return SpikeCore.expected_value(c, res, JuMP.moi_function(expr))
end
expected_value(x::Real; dual::Bool = false) = Float64(x)

end # module
