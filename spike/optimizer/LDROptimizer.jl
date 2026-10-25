"""
    LDROptimizer

Design B of the Phase 0 spike: the user writes a plain `JuMP.Model` whose optimizer is
`LDROptimizer.Optimizer <: MOI.AbstractOptimizer` (no bridges). At `optimize!`, the optimizer reads JuMP's
cache directly (`MOI.optimize!(dest, src)` returning `copied = false`, so no third copy is kept and the
CachingOptimizer never attaches), builds a `SpikeCore.ModelInfo` from the metadata stored in the cache
(uncertainty sets, variable/constraint attributes) and solves the primal/dual LDR with the inner optimizer.
"""
module LDROptimizer

import Distributions
import JuMP
import LinearAlgebra
import MathOptInterface as MOI

include(joinpath(@__DIR__, "..", "core", "SpikeCore.jl"))
import .SpikeCore

export LDRModel, Uncertainty, Stage, ValidInequality, Truncate, BreakPoints, StageAttribute, ConstraintTags,
    Formulations, PrimalLDR, DualLDR, FormulationAttribute, InnerOptimizer, stage, set_stage, decision_rule,
    expected_value, constraint_tags

const VI = MOI.VariableIndex
const SAF = MOI.ScalarAffineFunction{Float64}
const SQF = MOI.ScalarQuadraticFunction{Float64}
const BOUNDS = (MOI.LessThan{Float64}, MOI.GreaterThan{Float64}, MOI.EqualTo{Float64}, MOI.Interval{Float64})

# ============================================================== uncertainty sets

"Univariate uncertainty set: `@variable(m, d in Uncertainty(Uniform(80, 120)))`."
struct ScalarUncertainty{D<:Distributions.UnivariateDistribution} <: MOI.AbstractScalarSet
    dist::D
end

"Multivariate uncertainty set: `@variable(m, d[1:2] in Uncertainty(product_distribution(...)))`."
struct VectorUncertainty{D<:Distributions.MultivariateDistribution} <: MOI.AbstractVectorSet
    dist::D
end

"""
    Uncertainty(dist)

The set of an uncertainty variable distributed as `dist`: a `ScalarUncertainty` for a univariate
distribution, a `VectorUncertainty` for a multivariate one.
"""
Uncertainty(d::Distributions.UnivariateDistribution) = ScalarUncertainty(d)
Uncertainty(d::Distributions.MultivariateDistribution) = VectorUncertainty(d)

const AnyUncertainty = Union{ScalarUncertainty,VectorUncertainty}
MOI.dimension(s::VectorUncertainty) = length(s.dist)
Base.copy(s::S) where {S<:AnyUncertainty} = S(deepcopy(s.dist))
Base.:(==)(a::S, b::S) where {S<:AnyUncertainty} = a.dist == b.dist
Base.hash(s::AnyUncertainty, h::UInt) = hash(s.dist, hash(typeof(s), h))
Base.show(io::IO, s::AnyUncertainty) = print(io, "Uncertainty(", s.dist, ")")

SpikeCore.is_metadata_constraint(::Type{VI}, ::Type{<:ScalarUncertainty}) = true
SpikeCore.is_metadata_constraint(::Type{MOI.VectorOfVariables}, ::Type{<:VectorUncertainty}) = true

# ============================================================== tags and attributes

"Tags accepted as extra positional arguments of `@variable` (`Stage`) and `@constraint` (all)."
abstract type LDRTag end
"`Stage(t)`: stage of a variable (`@variable(m, x, Stage(1))`) or tag of a constraint."
struct Stage <: LDRTag
    t::Int
end
"Constraint tag: an uncertainty-only row describing the support Ξ."
struct ValidInequality <: LDRTag end
"Constraint tag: an uncertainty-only row truncating the distribution (not in the spike)."
struct Truncate <: LDRTag end

"Variable attribute: the stage (`Int`) of a decision or uncertainty; `nothing` when unmarked."
struct StageAttribute <: MOI.AbstractVariableAttribute end
"Constraint attribute: the `Vector{LDRTag}` given in `@constraint(m, expr, tags...)`."
struct ConstraintTags <: MOI.AbstractConstraintAttribute end
"Variable attribute: breakpoints of an uncertainty (stored only; no lifting in the spike)."
struct BreakPoints <: MOI.AbstractVariableAttribute end

abstract type AbstractFormulation end
"The primal (conservative) LDR: result 1 and `ObjectiveValue`."
struct PrimalLDR <: AbstractFormulation end
"The dual (progressive) LDR: `ObjectiveBound`."
struct DualLDR <: AbstractFormulation end

"Optimizer attribute: the formulations solved by `optimize!` (default `[PrimalLDR(), DualLDR()]`)."
struct Formulations <: MOI.AbstractOptimizerAttribute end
"Optimizer attribute: the factory of the optimizer used for each formulation."
struct InnerOptimizer <: MOI.AbstractOptimizerAttribute end

"""
    FormulationAttribute(formulation, attr)

Per-formulation result: `attr` is `MOI.TerminationStatus()`, `MOI.PrimalStatus()` or `MOI.ObjectiveValue()`.
"""
struct FormulationAttribute{A<:MOI.AbstractModelAttribute} <: MOI.AbstractModelAttribute
    formulation::AbstractFormulation
    attr::A
end

# Internal results used by the JuMP-side helpers (indices are mapped back by the CachingOptimizer).
struct RuleCoefficients <: MOI.AbstractVariableAttribute   # Vector{Float64}: rule over ξ = (1, η)
    formulation::AbstractFormulation
end
struct XiVariables <: MOI.AbstractModelAttribute end       # Vector{VI}: η in canonical order
struct XiMoments <: MOI.AbstractModelAttribute end         # (E[ξ], E[ξξᵀ])

MOI.is_set_by_optimize(::Union{FormulationAttribute,RuleCoefficients,XiVariables,XiMoments}) = true

const OwnAttr = Union{StageAttribute,ConstraintTags,BreakPoints,Formulations,InnerOptimizer,FormulationAttribute,
    RuleCoefficients,XiMoments}
MOI.Utilities.map_indices(::Any, ::OwnAttr, x) = x      # values hold no indices (needed by copy_to/get/set)

# ============================================================== JuMP declaration syntax

"A variable (or array/vector of variables) under construction carrying a `Stage`."
struct StagedVariable{V} <: JuMP.AbstractVariable
    inner::V
    stage::Any        # Int, or Vector{Int} for a vector-constrained declaration
end

function JuMP.build_variable(error_fn::Function, info::JuMP.VariableInfo, s::Stage; kwargs...)
    return StagedVariable(JuMP.build_variable(error_fn, info; kwargs...), s.t)
end
function JuMP.build_variable(error_fn::Function, v::StagedVariable, set::MOI.AbstractScalarSet)
    return StagedVariable(JuMP.build_variable(error_fn, v.inner, set), v.stage)
end
function JuMP.build_variable(error_fn::Function, vs::AbstractArray{<:StagedVariable}, set::MOI.AbstractScalarSet)
    return map(v -> JuMP.build_variable(error_fn, v, set), vs)
end
function JuMP.build_variable(error_fn::Function, vs::Vector{<:StagedVariable}, set::MOI.AbstractVectorSet)
    inner = JuMP.build_variable(error_fn, [v.inner for v in vs], set)
    return StagedVariable(inner, [v.stage for v in vs])
end

function JuMP.add_variable(model::JuMP.GenericModel, v::StagedVariable, name)
    x = JuMP.add_variable(model, v.inner, name)
    x isa AbstractArray ? foreach(set_stage, x, v.stage) : set_stage(x, v.stage)
    return x
end
function JuMP.add_variable(model::JuMP.GenericModel, vs::AbstractArray{<:StagedVariable}, names::AbstractArray)
    return JuMP.add_variable.(model, vs, names)
end

"""
    stage(x::VariableRef) -> Union{Int,Nothing}
    stage(c::ConstraintRef) -> Union{Int,Nothing}

The declared stage (`nothing` when unmarked).
"""
stage(x::JuMP.GenericVariableRef) = MOI.get(JuMP.owner_model(x), StageAttribute(), x)
function stage(c::JuMP.ConstraintRef)
    k = findfirst(t -> t isa Stage, constraint_tags(c))
    return k === nothing ? nothing : constraint_tags(c)[k].t
end
"`set_stage(x, t)`: set the stage of a decision or uncertainty."
set_stage(x::JuMP.GenericVariableRef, t::Integer) = MOI.set(JuMP.owner_model(x), StageAttribute(), x, Int(t))

"A constraint under construction carrying tags (`ValidInequality()`, `Truncate()`, `Stage(t)`)."
struct TaggedConstraint{C<:JuMP.AbstractConstraint} <: JuMP.AbstractConstraint
    inner::C
    tags::Vector{LDRTag}
end

# JuMP's @constraint accepts a single extra positional argument, so one tag per constraint.
function JuMP.build_constraint(error_fn::Function, f::JuMP.AbstractJuMPScalar, set::MOI.AbstractScalarSet, tag::LDRTag)
    return TaggedConstraint(JuMP.build_constraint(error_fn, f, set), LDRTag[tag])
end

function JuMP.add_constraint(model::JuMP.GenericModel, c::TaggedConstraint, name::String = "")
    cref = JuMP.add_constraint(model, c.inner, name)
    MOI.set(model, ConstraintTags(), cref, c.tags)
    return cref
end

"`constraint_tags(c)`: the tags of a constraint (empty if none)."
constraint_tags(c::JuMP.ConstraintRef) = something(MOI.get(JuMP.owner_model(c), ConstraintTags(), c), LDRTag[])

# ============================================================== the optimizer

"""
    Optimizer(inner_factory = nothing)

LDR optimizer for a plain JuMP model: `Model(() -> LDROptimizer.Optimizer(HiGHS.Optimizer); add_bridges = false)`
(or `LDRModel(HiGHS.Optimizer)`).
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    inner::Any
    silent::Bool
    time_limit::Union{Nothing,Float64}
    formulations::Vector{AbstractFormulation}
    canonical::Union{Nothing,SpikeCore.Canonical}
    results::Dict{AbstractFormulation,SpikeCore.FormulationResult}
    solve_time::Float64
end
Optimizer(inner = nothing) = Optimizer(inner, false, nothing, [PrimalLDR(), DualLDR()], nothing,
    Dict{AbstractFormulation,SpikeCore.FormulationResult}(), NaN)

"`LDRModel(factory)`: a plain `JuMP.Model` with an LDR `Optimizer` over `factory`, without bridges."
LDRModel(factory) = JuMP.Model(() -> Optimizer(factory); add_bridges = false)

MOI.is_empty(o::Optimizer) = o.canonical === nothing && isempty(o.results)
function MOI.empty!(o::Optimizer)   # keeps the optimizer attributes
    o.canonical = nothing
    empty!(o.results)
    o.solve_time = NaN
    return
end

MOI.get(::Optimizer, ::MOI.SolverName) = "LDROptimizer"
MOI.supports_incremental_interface(::Optimizer) = false

# --- what the core can read (anything else errors at declaration, since there are no bridges)
MOI.supports_constraint(::Optimizer, ::Type{VI}, ::Type{<:Union{BOUNDS...,MOI.Integer,MOI.ZeroOne}}) = true
MOI.supports_constraint(::Optimizer, ::Type{SAF}, ::Type{<:Union{BOUNDS...}}) = true
MOI.supports_constraint(::Optimizer, ::Type{VI}, ::Type{<:ScalarUncertainty}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.VectorOfVariables}, ::Type{<:VectorUncertainty}) = true
MOI.supports_add_constrained_variable(::Optimizer, ::Type{<:ScalarUncertainty}) = true
MOI.supports_add_constrained_variables(::Optimizer, ::Type{<:VectorUncertainty}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{<:Union{VI,SAF,SQF}}) = true
MOI.supports(::Optimizer, ::Union{MOI.VariableName,StageAttribute,BreakPoints}, ::Type{VI}) = true
MOI.supports(::Optimizer, ::Union{MOI.ConstraintName,ConstraintTags}, ::Type{<:MOI.ConstraintIndex}) = true

# --- optimizer attributes
MOI.supports(::Optimizer, ::Union{MOI.Silent,MOI.TimeLimitSec,Formulations,InnerOptimizer}) = true
MOI.get(o::Optimizer, ::MOI.Silent) = o.silent
MOI.set(o::Optimizer, ::MOI.Silent, v::Bool) = (o.silent = v; nothing)
MOI.get(o::Optimizer, ::MOI.TimeLimitSec) = o.time_limit
MOI.set(o::Optimizer, ::MOI.TimeLimitSec, v::Union{Nothing,Real}) = (o.time_limit = v; nothing)
MOI.get(o::Optimizer, ::InnerOptimizer) = o.inner
MOI.set(o::Optimizer, ::InnerOptimizer, v) = (o.inner = v; nothing)
MOI.get(o::Optimizer, ::Formulations) = copy(o.formulations)
function MOI.set(o::Optimizer, ::Formulations, v)
    isempty(v) && throw(ArgumentError("Formulations() needs at least one of PrimalLDR(), DualLDR()."))
    o.formulations = collect(AbstractFormulation, v)
    return
end

# --- copy and solve: read the src (JuMP's cache) in place; identity index map, no copy kept
function _get_if_set(src, attr, idx, present::Bool)
    present || return nothing
    return MOI.get(src, attr, idx)
end

function _model_info(src::MOI.ModelLike)
    has_stage = StageAttribute() in MOI.get(src, MOI.ListOfVariableAttributesSet())
    vstage(v) = _get_if_set(src, StageAttribute(), v, has_stage)
    uncs = SpikeCore.UncertaintyDecl[]
    valid, trunc = Set{MOI.ConstraintIndex}(), Set{MOI.ConstraintIndex}()
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        cis = MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
        if S <: AnyUncertainty
            for ci in cis
                f, s = MOI.get(src, MOI.ConstraintFunction(), ci), MOI.get(src, MOI.ConstraintSet(), ci)
                vars = f isa VI ? [f] : f.variables
                push!(uncs, SpikeCore.UncertaintyDecl(s.dist, vars, Union{Int,Nothing}[vstage(v) for v in vars]))
            end
        end
        ConstraintTags() in MOI.get(src, MOI.ListOfConstraintAttributesSet{F,S}()) || continue
        for ci in cis
            tags = MOI.get(src, ConstraintTags(), ci)
            tags === nothing && continue
            any(t -> t isa ValidInequality, tags) && push!(valid, ci)
            any(t -> t isa Truncate, tags) && push!(trunc, ci)
        end
    end
    unc_vars = Set(v for u in uncs for v in u.vars)
    dstage = Dict{VI,Int}()
    for v in MOI.get(src, MOI.ListOfVariableIndices())
        s = v in unc_vars ? nothing : vstage(v)
        s === nothing || (dstage[v] = s)
    end
    return SpikeCore.ModelInfo(uncertainties = uncs, decision_stage = dstage, valid_rows = valid, truncate_rows = trunc)
end

function _identity_map(src::MOI.ModelLike)
    map = MOI.Utilities.IndexMap()
    for v in MOI.get(src, MOI.ListOfVariableIndices())
        map[v] = v
    end
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent()), ci in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
        map[ci] = ci
    end
    return map
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    MOI.empty!(dest)
    dest.canonical = SpikeCore.canonicalize(src, _model_info(src))
    return _identity_map(src)
end

function MOI.optimize!(dest::Optimizer, src::MOI.ModelLike)
    index_map = MOI.copy_to(dest, src)
    MOI.optimize!(dest)
    return index_map, false     # `false`: nothing to modify incrementally; the next optimize! re-reads src
end

function _inner_factory(o::Optimizer)
    o.inner === nothing && error("LDROptimizer.Optimizer has no inner optimizer: use " *
                                 "`LDRModel(HiGHS.Optimizer)` or `set_attribute(model, InnerOptimizer(), HiGHS.Optimizer)`.")
    o.time_limit === nothing && return o.inner
    return () -> (inner = MOI.instantiate(o.inner); MOI.set(inner, MOI.TimeLimitSec(), o.time_limit); inner)
end

function MOI.optimize!(o::Optimizer)
    o.canonical === nothing && error("LDROptimizer.Optimizer: call `optimize!(dest, src)` (through JuMP).")
    factory, start = _inner_factory(o), time()
    for f in o.formulations
        solve = f isa PrimalLDR ? SpikeCore.solve_primal : SpikeCore.solve_dual
        o.results[f] = solve(o.canonical, factory; silent = o.silent)
    end
    o.solve_time = time() - start
    return
end

# --- results: result 1 = primal rule; result 2 = sampled rule (not in the spike); dual = ObjectiveBound
_result(o::Optimizer, f::AbstractFormulation) = get(o.results, f, nothing)

function _require(o::Optimizer, f::AbstractFormulation, attr)
    r = _result(o, f)
    r === nothing && throw(MOI.GetAttributeNotAllowed(attr, "$f was not solved; add it to `Formulations()` and re-solve."))
    return r
end

function MOI.get(o::Optimizer, ::MOI.TerminationStatus)
    isempty(o.results) && return MOI.OPTIMIZE_NOT_CALLED
    p, d = _result(o, PrimalLDR()), _result(o, DualLDR())
    d !== nothing && d.termination_status == MOI.INFEASIBLE && return MOI.INFEASIBLE   # only the dual proves it
    p === nothing && return d.termination_status
    return p.termination_status == MOI.INFEASIBLE ? MOI.LOCALLY_INFEASIBLE : p.termination_status
end

function MOI.get(o::Optimizer, ::MOI.RawStatusString)
    return join(("$f: $(r.termination_status)" for (f, r) in o.results), "; ")
end

function MOI.get(o::Optimizer, ::MOI.ResultCount)
    p = _result(o, PrimalLDR())
    return p !== nothing && p.primal_status == MOI.FEASIBLE_POINT ? 1 : 0
end

function MOI.get(o::Optimizer, attr::MOI.PrimalStatus)
    attr.result_index == 1 || return MOI.NO_SOLUTION     # result 2 (sampled rule) is not in the spike
    p = _result(o, PrimalLDR())
    return p === nothing ? MOI.NO_SOLUTION : p.primal_status
end
MOI.get(::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION

function MOI.get(o::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(o, attr)
    return o.results[PrimalLDR()].objective_value
end
MOI.get(o::Optimizer, attr::MOI.ObjectiveBound) = _require(o, DualLDR(), attr).objective_value
function MOI.get(o::Optimizer, attr::MOI.RelativeGap)
    f, b = MOI.get(o, MOI.ObjectiveValue()), MOI.get(o, MOI.ObjectiveBound())
    return abs(b - f) / abs(f)
end
MOI.get(o::Optimizer, ::MOI.SolveTimeSec) = o.solve_time

function MOI.get(o::Optimizer, attr::MOI.VariablePrimal, vi::VI)
    MOI.check_result_index_bounds(o, attr)
    return SpikeCore.expected_decision(o.canonical, o.results[PrimalLDR()], vi)
end

function MOI.get(o::Optimizer, attr::FormulationAttribute)
    r = _result(o, attr.formulation)
    if attr.attr isa MOI.TerminationStatus
        return r === nothing ? MOI.OPTIMIZE_NOT_CALLED : r.termination_status
    elseif attr.attr isa MOI.PrimalStatus
        return r === nothing ? MOI.NO_SOLUTION : r.primal_status
    elseif attr.attr isa MOI.ObjectiveValue
        return _require(o, attr.formulation, attr).objective_value
    end
    throw(MOI.UnsupportedAttribute(attr, "FormulationAttribute supports TerminationStatus, PrimalStatus, ObjectiveValue."))
end

function MOI.get(o::Optimizer, attr::RuleCoefficients, vi::VI)
    r = _require(o, attr.formulation, attr)
    k, coefs = SpikeCore.rule(o.canonical, r, vi)
    unc = SpikeCore.uncertainties(o.canonical)
    return [k; [get(coefs, u, 0.0) for u in unc]]
end
MOI.get(o::Optimizer, ::XiVariables) = copy(SpikeCore.uncertainties(o.canonical))
MOI.get(o::Optimizer, ::XiMoments) = (SpikeCore.mean_xi(o.canonical), SpikeCore.second_moment(o.canonical))

# ============================================================== JuMP-side results

"""
    DecisionRule

Affine rule `x(η) = constant + Σ coefficients[u] * u`. Call it with a `Dict(u => value)` or pairs `u => value`.
"""
struct DecisionRule
    constant::Float64
    coefficients::Dict{JuMP.VariableRef,Float64}
end
(r::DecisionRule)(vals::AbstractDict) = r.constant + sum((c * vals[u] for (u, c) in r.coefficients); init = 0.0)
(r::DecisionRule)(vals::Pair...) = r(Dict(vals))
function Base.show(io::IO, r::DecisionRule)
    print(io, "DecisionRule: ", r.constant)
    for (u, c) in sort!(collect(r.coefficients); by = p -> JuMP.index(p[1]).value)
        print(io, c < 0 ? " - " : " + ", abs(c), " ", u)
    end
end

"""
    decision_rule(x::VariableRef; formulation = PrimalLDR()) -> DecisionRule

The affine rule of `x` from the last solve (indices refreshed at each `optimize!`).
"""
function decision_rule(x::JuMP.VariableRef; formulation::AbstractFormulation = PrimalLDR())
    model = JuMP.owner_model(x)
    a = MOI.get(model, RuleCoefficients(formulation), x)
    unc = MOI.get(model, XiVariables())
    return DecisionRule(a[1], Dict(JuMP.VariableRef(model, u) => a[1+j] for (j, u) in enumerate(unc) if !iszero(a[1+j])))
end

_some_variable(x::JuMP.VariableRef) = x
_some_variable(x::JuMP.AffExpr) = isempty(x.terms) ? nothing : first(keys(x.terms))
_some_variable(x::JuMP.QuadExpr) = isempty(x.terms) ? _some_variable(x.aff) : first(keys(x.terms)).a

"""
    expected_value(expr; formulation = PrimalLDR()) -> Float64

Exact `E[expr(x(ξ), η)]` of an affine or quadratic JuMP expression under the solved rules.
"""
function expected_value(expr::JuMP.AbstractJuMPScalar; formulation::AbstractFormulation = PrimalLDR())
    v = _some_variable(expr)
    v === nothing && return Float64(JuMP.constant(expr))
    model = JuMP.owner_model(v)
    mu, M = MOI.get(model, XiMoments())
    g = convert(SQF, JuMP.moi_function(expr))
    a(vi) = MOI.get(model, RuleCoefficients(formulation), JuMP.VariableRef(model, vi))
    val = g.constant + sum((t.coefficient * LinearAlgebra.dot(a(t.variable), mu) for t in g.affine_terms); init = 0.0)
    for t in g.quadratic_terms
        coef = t.variable_1 == t.variable_2 ? t.coefficient / 2 : t.coefficient
        val += coef * LinearAlgebra.dot(a(t.variable_1), M, a(t.variable_2))
    end
    return val
end

end # module
