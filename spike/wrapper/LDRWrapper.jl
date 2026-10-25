"""
    LDRWrapper

Design A of the Phase 0 spike: `LDRModel <: JuMP.AbstractModel`, a wrapper that owns

  * an inner `JuMP.Model` used only as an MOI store (variables, rows, objective, names, deletion), and
  * the LDR metadata (stages, uncertainty declarations, row tags, attributes) in side tables keyed by MOI indices.

User variables are `LDRVariable <: JuMP.AbstractVariableRef` (InfiniteOpt style), never the inner `VariableRef`s, so
that `value(x)` is the expected decision and `stage(x)` needs no back-pointer. See `spike/SPIKE.md`.
"""
module LDRWrapper

isdefined(Main, :SpikeCore) || include(joinpath(@__DIR__, "..", "core", "SpikeCore.jl"))

import Distributions
import JuMP
import MathOptInterface as MOI
import ..SpikeCore

export LDRModel, LDRVariable, Stage, Uncertainty, ValidInequality, Truncate, BreakPoints, SolvePrimal, SolveDual,
    DecisionRule, stage, set_stage, decision_rule, expected_value, inner_model

const VI = MOI.VariableIndex
const CI = MOI.ConstraintIndex
const BoundSet = Union{MOI.LessThan{Float64},MOI.GreaterThan{Float64},MOI.EqualTo{Float64},MOI.Interval{Float64}}

# ------------------------------------------------------------------------------------------------ user-facing tags

"""
    Stage(t)

Stage tag. In `@variable(m, x >= 0, Stage(2))` it marks a decision (or, with `in Uncertainty(...)`, an uncertainty);
in `@constraint(m, expr <= 1, Stage(2))` it is stored on the constraint (future: expectation constraint at time `t`).
"""
struct Stage
    t::Int
    function Stage(t::Integer)
        t >= 1 || throw(ArgumentError("Stage index must be at least 1, got $t."))
        return new(Int(t))
    end
end

"""
    Uncertainty(dist)

Used as `@variable(m, d in Uncertainty(dist))` with a univariate (scalar or container of independent scalars) or
multivariate (vector; correlations from `cov(dist)`) `Distributions.jl` distribution with bounded support.
"""
struct Uncertainty{D<:Distributions.Distribution}
    dist::D
end

"Constraint tag: an uncertainty-only row describing the support Ξ."
struct ValidInequality end
"Constraint tag: an uncertainty-only row truncating the distribution (not supported by the spike core)."
struct Truncate end
const _RowTag = Union{ValidInequality,Truncate,Stage}

"Variable attribute: break points of a piecewise-linear lifting of a scalar uncertainty (stored only)."
struct BreakPoints end
"Model option: solve the primal (conservative) formulation. Default `true`."
struct SolvePrimal end
"Model option: solve the dual (bound) formulation. Default `true`."
struct SolveDual end
const _Option = Union{SolvePrimal,SolveDual}
_formulation(::SolvePrimal) = :primal
_formulation(::SolveDual) = :dual

# ------------------------------------------------------------------------------------------------ model and variable

mutable struct _Declaration
    dist::Distributions.Distribution
    vars::Vector{VI}
end

struct _Solution
    canonical::SpikeCore.Canonical
    results::Dict{Symbol,SpikeCore.FormulationResult}
end

"""
    LDRModel([optimizer_factory])

JuMP model for linear decision rules. The optimizer is the *inner* solver used for the LDR reformulations.
"""
mutable struct LDRModel <: JuMP.AbstractModel
    inner::JuMP.Model
    solver::Any
    solver_attributes::Vector{Pair{String,Any}}
    silent::Bool
    enabled::Dict{Symbol,Bool}
    stage::Dict{VI,Int}
    declarations::Vector{_Declaration}
    uncertainty::Dict{VI,_Declaration}
    attributes::Dict{Tuple{VI,DataType},Any}
    valid_rows::Set{CI}
    truncate_rows::Set{CI}
    constraint_stage::Dict{CI,Int}
    solution::Union{Nothing,_Solution}
    is_model_dirty::Bool                 # JuMP's generic code (e.g. set_normalized_rhs) writes this field
    obj_dict::Dict{Symbol,Any}
    ext::Dict{Symbol,Any}
    function LDRModel(solver = nothing)
        return new(JuMP.Model(), solver, Pair{String,Any}[], false, Dict(:primal => true, :dual => true),
            Dict{VI,Int}(), _Declaration[], Dict{VI,_Declaration}(), Dict{Tuple{VI,DataType},Any}(), Set{CI}(),
            Set{CI}(), Dict{CI,Int}(), nothing, false, Dict{Symbol,Any}(), Dict{Symbol,Any}())
    end
end

"`LDRVariable(model, index)`: user-facing variable (decision or uncertainty) of an `LDRModel`."
struct LDRVariable <: JuMP.AbstractVariableRef
    model::LDRModel
    index::VI
end

const LDRAffExpr = JuMP.GenericAffExpr{Float64,LDRVariable}
const LDRQuadExpr = JuMP.GenericQuadExpr{Float64,LDRVariable}
const _Scalar = Union{LDRVariable,LDRAffExpr,LDRQuadExpr}

_dirty!(m::LDRModel) = (m.solution = nothing; m.is_model_dirty = true; nothing)
# JuMP's generic code (set_normalized_rhs, set_normalized_coefficient, ...) only sets `is_model_dirty`: that is the truth.
_current(m::LDRModel) = m.is_model_dirty ? nothing : m.solution
_backend(m::LDRModel) = JuMP.backend(m.inner)
_inner(x::LDRVariable) = JuMP.VariableRef(x.model.inner, x.index)
_wrap(m::LDRModel, x::JuMP.VariableRef) = LDRVariable(m, JuMP.index(x))
_wrap(m::LDRModel, x::AbstractArray) = _wrap.(Ref(m), x)
_is_uncertainty(m::LDRModel, x::LDRVariable) = haskey(m.uncertainty, x.index)

"The inner `JuMP.Model`. Variables created there (e.g. by a third-party `build!(::JuMP.Model)`) become unmarked decisions."
inner_model(m::LDRModel) = m.inner
function LDRVariable(m::LDRModel, x::JuMP.VariableRef)
    JuMP.owner_model(x) === m.inner || error("The variable does not belong to the inner model of this LDRModel.")
    return LDRVariable(m, JuMP.index(x))
end

Base.broadcastable(m::LDRModel) = Ref(m)
Base.broadcastable(x::LDRVariable) = Ref(x)
Base.:(==)(a::LDRVariable, b::LDRVariable) = a.model === b.model && a.index == b.index
Base.isequal(a::LDRVariable, b::LDRVariable) = a == b
Base.hash(x::LDRVariable, h::UInt) = hash(x.index, h)
Base.copy(x::LDRVariable) = x
Base.iszero(::LDRVariable) = false
JuMP.index(x::LDRVariable) = x.index
JuMP.variable_ref_type(::Type{LDRModel}) = LDRVariable
JuMP.object_dictionary(m::LDRModel) = m.obj_dict
JuMP.backend(m::LDRModel) = _backend(m)
Base.show(io::IO, m::LDRModel) = print(io, "LDRModel with ", JuMP.num_variables(m), " variables, ",
    length(m.declarations), " uncertainty declarations")

# ------------------------------------------------------------------------------------------------ declarations

struct _Staged <: JuMP.AbstractVariable
    variable::JuMP.ScalarVariable
    stage::Int
end
struct _UncertainVariable <: JuMP.AbstractVariable
    variable::JuMP.ScalarVariable
    stage::Union{Int,Nothing}
    dist::Distributions.UnivariateDistribution
end
struct _UncertainVector <: JuMP.AbstractVariable
    variables::Vector{JuMP.ScalarVariable}
    stages::Vector{Union{Int,Nothing}}
    dist::Distributions.Distribution            # univariate (one variable) or multivariate
end
const _Plain = Union{JuMP.ScalarVariable,_Staged}
_scalar(v::JuMP.ScalarVariable) = v
_scalar(v::_Staged) = v.variable
_stage_of(::JuMP.ScalarVariable) = nothing
_stage_of(v::_Staged) = v.stage

JuMP.build_variable(::Function, info::JuMP.VariableInfo, s::Stage) = _Staged(JuMP.ScalarVariable(info), s.t)

function JuMP.build_variable(err::Function, v::_Plain, u::Uncertainty{<:Distributions.UnivariateDistribution})
    _check_support(err, u.dist)
    return _UncertainVariable(_scalar(v), _stage_of(v), u.dist)
end
function JuMP.build_variable(err::Function, vs::AbstractArray{<:_Plain},
    u::Uncertainty{<:Distributions.UnivariateDistribution})
    return JuMP.build_variable.(err, vs, Ref(u))
end
function JuMP.build_variable(err::Function, vs::AbstractVector{<:_Plain},
    u::Uncertainty{<:Distributions.MultivariateDistribution})
    length(vs) == length(u.dist) || err("A $(length(u.dist))-dimensional distribution needs a vector of that length, got $(length(vs)).")
    _check_support(err, u.dist)
    return _UncertainVector(JuMP.ScalarVariable[_scalar(v) for v in vs],
        Union{Int,Nothing}[_stage_of(v) for v in vs], u.dist)
end

function _check_support(err, dist)
    lo, hi = Distributions.minimum(dist), Distributions.maximum(dist)
    all(isfinite, lo) && all(isfinite, hi) || err("The distribution $dist has unbounded support; Ξ must be bounded.")
    return nothing
end

function _check_plain_info(v::JuMP.ScalarVariable)
    i = v.info
    (i.has_lb || i.has_ub || i.has_fix || i.binary || i.integer) &&
        error("An uncertainty cannot have bounds, be fixed or be integer: its support comes from its distribution.")
    return nothing
end

function JuMP.add_variable(m::LDRModel, v::JuMP.ScalarVariable, name::String = "")
    _dirty!(m)
    return _wrap(m, JuMP.add_variable(m.inner, v, name))
end
function JuMP.add_variable(m::LDRModel, v::_Staged, name::String = "")
    x = JuMP.add_variable(m, v.variable, name)
    m.stage[x.index] = v.stage
    return x
end
function JuMP.add_variable(m::LDRModel,
    v::Union{JuMP.VariableConstrainedOnCreation,JuMP.VariablesConstrainedOnCreation}, name)
    _dirty!(m)
    return _wrap(m, JuMP.add_variable(m.inner, v, name))
end
function JuMP.add_variable(m::LDRModel, vs::AbstractArray{<:Union{_Staged,_UncertainVariable,JuMP.VariableConstrainedOnCreation}},
    names::AbstractArray)
    return JuMP.add_variable.(m, vs, names)
end

function JuMP.add_variable(m::LDRModel, v::_UncertainVariable, name::String = "")
    return JuMP.add_variable(m, _UncertainVector([v.variable], [v.stage], v.dist), [name])[1]
end
function JuMP.add_variable(m::LDRModel, v::_UncertainVector, names)
    foreach(_check_plain_info, v.variables)
    xs = LDRVariable[JuMP.add_variable(m, s, n) for (s, n) in zip(v.variables, names)]
    decl = _Declaration(v.dist, VI[x.index for x in xs])
    push!(m.declarations, decl)
    for (x, s) in zip(xs, v.stages)
        m.uncertainty[x.index] = decl
        s === nothing || (m.stage[x.index] = s)
    end
    return xs
end

# ------------------------------------------------------------------------------------------------ stages

"""
    stage(x)

Declared stage of a variable or constraint (`nothing` if unmarked).
"""
stage(x::LDRVariable) = get(x.model.stage, x.index, nothing)
stage(c::JuMP.ConstraintRef{LDRModel}) = get(c.model.constraint_stage, JuMP.index(c), nothing)

"""
    set_stage(x, t)

Mark a variable (decision or uncertainty) or a constraint as belonging to stage `t`.
"""
function set_stage(x::LDRVariable, t::Integer)
    JuMP.is_valid(x.model, x) || error("The variable does not belong to the model (or was deleted).")
    x.model.stage[x.index] = Stage(t).t
    _dirty!(x.model)
    return
end
function set_stage(c::JuMP.ConstraintRef{LDRModel}, t::Integer)
    c.model.constraint_stage[JuMP.index(c)] = Stage(t).t
    _dirty!(c.model)
    return
end

# ------------------------------------------------------------------------------------------------ variable forwarding

for f in (:has_lower_bound, :lower_bound, :has_upper_bound, :upper_bound, :is_fixed, :fix_value, :is_integer,
    :is_binary, :start_value, :name)
    @eval JuMP.$f(x::LDRVariable) = JuMP.$f(_inner(x))
end
for f in (:set_lower_bound, :set_upper_bound, :set_start_value, :set_name, :set_parameter_value)
    @eval function JuMP.$f(x::LDRVariable, v)
        $(f in (:set_lower_bound, :set_upper_bound) ? :(_forbid_for_uncertainty(x, $(QuoteNode(f)))) : nothing)
        _dirty!(x.model)
        return JuMP.$f(_inner(x), v)
    end
end
for f in (:delete_lower_bound, :delete_upper_bound, :unfix, :set_integer, :unset_integer, :set_binary, :unset_binary)
    @eval function JuMP.$f(x::LDRVariable)
        $(f in (:unfix, :delete_lower_bound, :delete_upper_bound) ? nothing : :(_forbid_for_uncertainty(x, $(QuoteNode(f)))))
        _dirty!(x.model)
        return JuMP.$f(_inner(x))
    end
end
function JuMP.fix(x::LDRVariable, v; force::Bool = false)
    _forbid_for_uncertainty(x, :fix)
    _dirty!(x.model)
    return JuMP.fix(_inner(x), v; force)
end
JuMP.parameter_value(x::LDRVariable) = JuMP.parameter_value(_inner(x))
JuMP.is_parameter(x::LDRVariable) = JuMP.is_parameter(_inner(x))

function _forbid_for_uncertainty(x::LDRVariable, op)
    _is_uncertainty(x.model, x) && error("`$op` is not allowed on an uncertainty: its support comes from its distribution.")
    return nothing
end

JuMP.is_valid(m::LDRModel, x::LDRVariable) = x.model === m && MOI.is_valid(_backend(m), x.index)
JuMP.num_variables(m::LDRModel) = JuMP.num_variables(m.inner)
JuMP.all_variables(m::LDRModel) = _wrap(m, JuMP.all_variables(m.inner))
function JuMP.variable_by_name(m::LDRModel, name::String)
    x = JuMP.variable_by_name(m.inner, name)
    return x === nothing ? nothing : _wrap(m, x)
end

function JuMP.delete(m::LDRModel, x::LDRVariable)
    return JuMP.delete(m, [x])
end
function JuMP.delete(m::LDRModel, xs::AbstractVector{LDRVariable})
    all(x -> JuMP.is_valid(m, x), xs) || error("Cannot delete a variable that does not belong to this model.")
    gone = Set(x.index for x in xs)
    for x in xs
        d = get(m.uncertainty, x.index, nothing)
        if d !== nothing
            all(in(gone), d.vars) || error("$(JuMP.name(x)) belongs to a vector uncertainty; delete all its components at once.")
            filter!(e -> e !== d, m.declarations)
            foreach(v -> delete!(m.uncertainty, v), d.vars)
        end
        delete!(m.stage, x.index)
        delete!(m.attributes, (x.index, BreakPoints))
        JuMP.delete(m.inner, _inner(x))
    end
    # row metadata of constraints removed together with the variable (none can exist for variable-in-set rows)
    _dirty!(m)
    return
end

# ------------------------------------------------------------------------------------------------ constraints

struct _TaggedConstraint{C<:JuMP.AbstractConstraint,T} <: JuMP.AbstractConstraint
    constraint::C
    tag::T
end
function JuMP.build_constraint(err::Function, f, set::MOI.AbstractScalarSet, tag::_RowTag; kwargs...)
    return _TaggedConstraint(JuMP.build_constraint(err, f, set; kwargs...), tag)
end

_unsupported(f) = error("LDRModel only supports affine constraints in LessThan/GreaterThan/EqualTo/Interval; got a " *
                        "function of type $(typeof(f)) (quadratic and nonlinear constraints, and uncertainty × " *
                        "decision products, are not supported).")

_variables(f::LDRVariable) = [f]
_variables(f::LDRAffExpr) = collect(keys(f.terms))

function JuMP.add_constraint(m::LDRModel, c::JuMP.ScalarConstraint, name::String = "")
    return _add_constraint(m, c, name, nothing)
end
function JuMP.add_constraint(m::LDRModel, c::_TaggedConstraint, name::String = "")
    return _add_constraint(m, c.constraint, name, c.tag)
end
function JuMP.add_constraint(m::LDRModel, c::JuMP.AbstractConstraint, name::String = "")
    return _unsupported(c)
end

function _add_constraint(m::LDRModel, c::JuMP.ScalarConstraint, name::String, tag)
    f, set = c.func, c.set
    f isa Union{LDRVariable,LDRAffExpr} || _unsupported(f)
    set isa BoundSet || (f isa LDRVariable && set isa Union{MOI.Integer,MOI.ZeroOne}) ||
        error("LDRModel does not support constraints in the set $(typeof(set)).")
    JuMP.check_belongs_to_model(f, m)
    vars = _variables(f)
    uncertain = [_is_uncertainty(m, v) for v in vars]
    if f isa LDRVariable
        uncertain[1] && error("`$(JuMP.name(f))` is an uncertainty: bounds and integrality come from its distribution.")
    elseif isempty(vars) || all(uncertain)
        tag isa Union{ValidInequality,Truncate} ||
            error("This constraint involves only uncertainties. Tag it with `ValidInequality()` (it describes the " *
                  "support Ξ) or `Truncate()` (it truncates the distribution).")
    elseif tag isa Union{ValidInequality,Truncate}
        error("A `$(typeof(tag))()` constraint may only involve uncertainties.")
    end
    ci = MOI.add_constraint(_backend(m), _moi(f), set)
    isempty(name) || f isa LDRVariable || MOI.set(_backend(m), MOI.ConstraintName(), ci, name)
    tag isa ValidInequality && push!(m.valid_rows, ci)
    tag isa Truncate && push!(m.truncate_rows, ci)
    tag isa Stage && (m.constraint_stage[ci] = tag.t)
    _dirty!(m)
    return _cref(m, ci)
end

_cref(m::LDRModel, ci::CI{<:MOI.AbstractScalarFunction}) = JuMP.ConstraintRef(m, ci, JuMP.ScalarShape())
_cref(m::LDRModel, ci::CI) = JuMP.ConstraintRef(m, ci, JuMP.VectorShape())

JuMP.is_valid(m::LDRModel, c::JuMP.ConstraintRef) = c.model === m && MOI.is_valid(_backend(m), c.index)
function JuMP.delete(m::LDRModel, c::JuMP.ConstraintRef)
    JuMP.is_valid(m, c) || error("Cannot delete a constraint that does not belong to this model.")
    MOI.delete(_backend(m), c.index)
    delete!(m.valid_rows, c.index)
    delete!(m.truncate_rows, c.index)
    delete!(m.constraint_stage, c.index)
    _dirty!(m)
    return
end
JuMP.delete(m::LDRModel, cs::AbstractVector{<:JuMP.ConstraintRef}) = (foreach(c -> JuMP.delete(m, c), cs); nothing)

MOI.get(m::LDRModel, a::MOI.AbstractConstraintAttribute, c::JuMP.ConstraintRef) = MOI.get(_backend(m), a, c.index)
function MOI.set(m::LDRModel, a::MOI.AbstractConstraintAttribute, c::JuMP.ConstraintRef, v)
    _dirty!(m)
    return MOI.set(_backend(m), a, c.index, v)
end

function JuMP.constraint_by_name(m::LDRModel, name::String)
    ci = MOI.get(_backend(m), MOI.ConstraintIndex, name)
    return ci === nothing ? nothing : _cref(m, ci)
end
JuMP.list_of_constraint_types(m::LDRModel) =
    Tuple{Type,Type}[(JuMP.jump_function_type(m, F), S) for (F, S) in MOI.get(_backend(m), MOI.ListOfConstraintTypesPresent())]
JuMP.num_constraints(m::LDRModel, ::Type{F}, ::Type{S}) where {F<:_Scalar,S<:MOI.AbstractSet} =
    MOI.get(_backend(m), MOI.NumberOfConstraints{JuMP.moi_function_type(F),S}())
function JuMP.num_constraints(m::LDRModel; count_variable_in_set_constraints::Bool)
    return sum(JuMP.num_constraints(m, F, S) for (F, S) in JuMP.list_of_constraint_types(m)
               if F != LDRVariable || count_variable_in_set_constraints; init = 0)
end
function JuMP.all_constraints(m::LDRModel, ::Type{F}, ::Type{S}) where {F<:_Scalar,S<:MOI.AbstractSet}
    return [_cref(m, ci) for ci in MOI.get(_backend(m), MOI.ListOfConstraintIndices{JuMP.moi_function_type(F),S}())]
end

function JuMP.all_constraints(m::LDRModel; include_variable_in_set_constraints::Bool)
    return JuMP.ConstraintRef[c for (F, S) in JuMP.list_of_constraint_types(m)
                              if F != LDRVariable || include_variable_in_set_constraints
                              for c in JuMP.all_constraints(m, F, S)]
end

# ------------------------------------------------------------------------------------------------ MOI <-> JuMP functions

_moi(x::LDRVariable) = x.index
function _moi(f::LDRAffExpr)
    return MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{Float64}[MOI.ScalarAffineTerm(c, v.index)
                                                                  for (c, v) in JuMP.linear_terms(f)], f.constant)
end
function _moi(q::LDRQuadExpr)
    qt = MOI.ScalarQuadraticTerm{Float64}[MOI.ScalarQuadraticTerm(a == b ? 2c : c, a.index, b.index)
                                          for (c, a, b) in JuMP.quad_terms(q)]
    aff = _moi(q.aff)
    return MOI.ScalarQuadraticFunction(qt, aff.terms, aff.constant)
end
_moi(x::Real) = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{Float64}[], Float64(x))

JuMP.jump_function_type(::LDRModel, ::Type{MOI.VariableIndex}) = LDRVariable
JuMP.jump_function_type(::LDRModel, ::Type{MOI.ScalarAffineFunction{Float64}}) = LDRAffExpr
JuMP.jump_function_type(::LDRModel, ::Type{MOI.ScalarQuadraticFunction{Float64}}) = LDRQuadExpr
JuMP.jump_function(m::LDRModel, f::MOI.VariableIndex) = LDRVariable(m, f)
function JuMP.jump_function(m::LDRModel, f::MOI.ScalarAffineFunction{Float64})
    a = LDRAffExpr(f.constant)
    foreach(t -> JuMP.add_to_expression!(a, t.coefficient, LDRVariable(m, t.variable)), f.terms)
    return a
end
function JuMP.jump_function(m::LDRModel, f::MOI.ScalarQuadraticFunction{Float64})
    q = LDRQuadExpr(JuMP.jump_function(m, MOI.ScalarAffineFunction(f.affine_terms, f.constant)))
    for t in f.quadratic_terms
        coef = t.variable_1 == t.variable_2 ? t.coefficient / 2 : t.coefficient
        JuMP.add_to_expression!(q, coef, LDRVariable(m, t.variable_1), LDRVariable(m, t.variable_2))
    end
    return q
end

# ------------------------------------------------------------------------------------------------ objective

function JuMP.set_objective_function(m::LDRModel, f::Union{_Scalar,Real})
    f isa Real || JuMP.check_belongs_to_model(f, m)
    mf = _moi(f)
    MOI.set(_backend(m), MOI.ObjectiveFunction{typeof(mf)}(), mf)
    _dirty!(m)
    return
end
JuMP.set_objective_function(::LDRModel, f::JuMP.AbstractJuMPScalar) = _unsupported(f)
JuMP.objective_sense(m::LDRModel) = MOI.get(_backend(m), MOI.ObjectiveSense())
function JuMP.set_objective_sense(m::LDRModel, s::MOI.OptimizationSense)
    MOI.set(_backend(m), MOI.ObjectiveSense(), s)
    _dirty!(m)
    return
end
JuMP.objective_function_type(m::LDRModel) =
    JuMP.jump_function_type(m, MOI.get(_backend(m), MOI.ObjectiveFunctionType()))
JuMP.objective_function(m::LDRModel) = JuMP.objective_function(m, JuMP.objective_function_type(m))
function JuMP.objective_function(m::LDRModel, ::Type{F}) where {F<:_Scalar}
    return JuMP.jump_function(m, MOI.get(_backend(m), MOI.ObjectiveFunction{JuMP.moi_function_type(F)}()))
end

# ------------------------------------------------------------------------------------------------ attributes

function _check_breakpoints(x::LDRVariable)
    d = get(x.model.uncertainty, x.index, nothing)
    (d === nothing || length(d.vars) != 1 || !(d.dist isa Distributions.UnivariateDistribution)) &&
        error("`BreakPoints` only applies to scalar uncertainties.")
    return nothing
end
function JuMP.set_attribute(x::LDRVariable, ::BreakPoints, v::Union{Nothing,AbstractVector{<:Real}})
    _check_breakpoints(x)
    isempty(something(v, [1])) && error("Provide at least one break point.")
    v === nothing ? delete!(x.model.attributes, (x.index, BreakPoints)) :
    (x.model.attributes[(x.index, BreakPoints)] = Float64.(v))
    _dirty!(x.model)
    return
end
function JuMP.get_attribute(x::LDRVariable, ::BreakPoints)
    _check_breakpoints(x)
    return get(x.model.attributes, (x.index, BreakPoints), nothing)
end
JuMP.set_attribute(m::LDRModel, o::_Option, v::Bool) = (m.enabled[_formulation(o)] = v; _dirty!(m); nothing)
JuMP.get_attribute(m::LDRModel, o::_Option) = m.enabled[_formulation(o)]
function JuMP.set_attribute(m::LDRModel, name::String, v)
    filter!(p -> p.first != name, m.solver_attributes)
    push!(m.solver_attributes, name => v)
    return
end
JuMP.get_attribute(m::LDRModel, name::String) = last(m.solver_attributes[findlast(p -> p.first == name, m.solver_attributes)])

# ------------------------------------------------------------------------------------------------ solve and results

JuMP.set_optimizer(m::LDRModel, factory) = (m.solver = factory; nothing)
JuMP.set_silent(m::LDRModel) = (m.silent = true; nothing)
JuMP.unset_silent(m::LDRModel) = (m.silent = false; nothing)

function _info(m::LDRModel)
    decls = [SpikeCore.UncertaintyDecl(d.dist, copy(d.vars), Union{Int,Nothing}[get(m.stage, v, nothing) for v in d.vars])
             for d in m.declarations]
    dstage = Dict{VI,Int}(v => s for (v, s) in m.stage if !haskey(m.uncertainty, v))
    return SpikeCore.ModelInfo(uncertainties = decls, decision_stage = dstage, valid_rows = copy(m.valid_rows),
        truncate_rows = copy(m.truncate_rows))
end

# JuMP Parameters: the core has no notion of them, so for the duration of `canonicalize` every parameter p is turned
# into a decision fixed by `p == value` (a bound the core reads as a constant rule), then restored.
function _with_parameters_fixed(f, m::LDRModel)
    b = _backend(m)
    cis = MOI.get(b, MOI.ListOfConstraintIndices{VI,MOI.Parameter{Float64}}())
    fixed = [(MOI.get(b, MOI.ConstraintFunction(), ci), MOI.get(b, MOI.ConstraintSet(), ci)) for ci in cis]
    isempty(fixed) && return f()
    foreach(ci -> MOI.delete(b, ci), cis)
    eqs = [MOI.add_constraint(b, v, MOI.EqualTo(s.value)) for (v, s) in fixed]
    try
        return f()
    finally
        foreach(ci -> MOI.delete(b, ci), eqs)
        foreach(((v, s),) -> MOI.add_constraint(b, v, s), fixed)
    end
end

function JuMP.optimize!(m::LDRModel)
    m.solver === nothing && error("No optimizer set: use `LDRModel(HiGHS.Optimizer)` or `set_optimizer(model, HiGHS.Optimizer)`.")
    any(values(m.enabled)) || error("All formulations are disabled (SolvePrimal and SolveDual are both false).")
    factory = isempty(m.solver_attributes) ? m.solver : JuMP.optimizer_with_attributes(m.solver, m.solver_attributes...)
    c = _with_parameters_fixed(() -> SpikeCore.canonicalize(_backend(m), _info(m)), m)
    results = Dict{Symbol,SpikeCore.FormulationResult}()
    m.enabled[:primal] && (results[:primal] = SpikeCore.solve_primal(c, factory; silent = m.silent))
    m.enabled[:dual] && (results[:dual] = SpikeCore.solve_dual(c, factory; silent = m.silent))
    m.solution = _Solution(c, results)
    m.is_model_dirty = false
    return
end

function _check_formulation(f::Symbol)
    f in (:primal, :dual) || throw(ArgumentError("`formulation` must be :primal or :dual, got :$f."))
    return f
end
function _result(m::LDRModel, f::Symbol)
    _check_formulation(f)
    m.enabled[f] || error("The :$f formulation is disabled; enable it with `set_attribute(model, " *
                          "$(f == :primal ? "SolvePrimal" : "SolveDual")(), true)`.")
    _current(m) === nothing && error("No results: call `optimize!(model)` (the model was never solved or was modified since).")
    return m.solution.canonical, m.solution.results[f]
end
function _solved(m::LDRModel, f::Symbol)
    c, r = _result(m, f)
    r.primal_status == MOI.NO_SOLUTION && error("No solution in the :$f formulation (termination status $(r.termination_status)).")
    return c, r
end
_maybe(m::LDRModel, f::Symbol) =
    (_check_formulation(f); _current(m) === nothing || !m.enabled[f] ? nothing : m.solution.results[f])

function JuMP.termination_status(m::LDRModel; formulation::Symbol = :primal)
    r = _maybe(m, formulation)
    return r === nothing ? MOI.OPTIMIZE_NOT_CALLED : r.termination_status
end
function JuMP.primal_status(m::LDRModel; result::Int = 1, formulation::Symbol = :primal)
    r = _maybe(m, formulation)
    return r === nothing || result != 1 ? MOI.NO_SOLUTION : r.primal_status
end
JuMP.dual_status(::LDRModel; kwargs...) = MOI.NO_SOLUTION
JuMP.has_values(m::LDRModel; result::Int = 1, formulation::Symbol = :primal) =
    JuMP.primal_status(m; result, formulation) != MOI.NO_SOLUTION
JuMP.is_solved_and_feasible(m::LDRModel; formulation::Symbol = :primal, kwargs...) =
    JuMP.termination_status(m; formulation) == MOI.OPTIMAL && JuMP.primal_status(m; formulation) == MOI.FEASIBLE_POINT
JuMP.objective_value(m::LDRModel; result::Int = 1, formulation::Symbol = :primal) = _solved(m, formulation)[2].objective_value
"Objective of the dual formulation: a bound on the other side of the primal (policy) objective."
JuMP.objective_bound(m::LDRModel) = JuMP.objective_value(m; formulation = :dual)
JuMP.relative_gap(m::LDRModel) = abs(JuMP.objective_bound(m) - JuMP.objective_value(m)) / abs(JuMP.objective_value(m))

function JuMP.value(x::LDRVariable; result::Int = 1, formulation::Symbol = :primal)
    result == 1 || throw(ArgumentError("LDRModel has a single result."))
    c, r = _solved(x.model, formulation)
    return SpikeCore.expected_decision(c, r, x.index)
end

"""
    expected_value(expr; formulation = :primal)

Exact `E[expr(x(ξ), η)]` for an affine or quadratic expression, using the decision rules and the second moments.
"""
function expected_value(f::_Scalar; formulation::Symbol = :primal)
    m = JuMP.owner_model(f)
    c, r = _solved(m, formulation)
    return SpikeCore.expected_value(c, r, _moi(f))
end

"""
    DecisionRule

Affine rule `x(η) = constant + Σ coefficients[u] * u`. Callable with `Pair`s (`rule(d => 90.0)`), a `Dict`, a vector in
the model's uncertainty order, or a number when the model has a single uncertainty.
"""
struct DecisionRule
    constant::Float64
    coefficients::Dict{LDRVariable,Float64}
    uncertainties::Vector{LDRVariable}
end
function (r::DecisionRule)(values::AbstractDict)
    return r.constant + sum((c * values[u] for (u, c) in r.coefficients); init = 0.0)
end
(r::DecisionRule)(pairs::Pair...) = r(Dict(pairs))
(r::DecisionRule)(x::Real) = (length(r.uncertainties) == 1 || isempty(r.coefficients)) ? r(Dict(r.uncertainties .=> x)) :
    error("The model has $(length(r.uncertainties)) uncertainties; pass `u => value` pairs or a vector.")
(r::DecisionRule)(v::AbstractVector{<:Real}) = r(Dict(zip(r.uncertainties, v)))
function Base.show(io::IO, r::DecisionRule)
    print(io, r.constant)
    for (u, c) in r.coefficients
        print(io, c < 0 ? " - " : " + ", abs(c), " ", JuMP.name(u))
    end
end

"""
    decision_rule(x; formulation = :primal) -> DecisionRule
"""
function decision_rule(x::LDRVariable; formulation::Symbol = :primal)
    c, r = _solved(x.model, formulation)
    k, coefs = SpikeCore.rule(c, r, x.index)
    return DecisionRule(k, Dict{LDRVariable,Float64}(LDRVariable(x.model, v) => a for (v, a) in coefs),
        LDRVariable[LDRVariable(x.model, v) for v in SpikeCore.uncertainties(c)])
end

# ------------------------------------------------------------------------------------------------ print, copy

function JuMP.objective_function_string(mode, m::LDRModel)
    return JuMP.function_string(mode, JuMP.objective_function(m))
end
function JuMP.constraints_string(mode, m::LDRModel)
    out = String[]
    for (F, S) in JuMP.list_of_constraint_types(m), c in JuMP.all_constraints(m, F, S)
        tag = JuMP.index(c) in m.valid_rows ? "   [ValidInequality]" : JuMP.index(c) in m.truncate_rows ? "   [Truncate]" :
              haskey(m.constraint_stage, JuMP.index(c)) ? "   [Stage($(m.constraint_stage[JuMP.index(c)]))]" : ""
        push!(out, JuMP.constraint_string(mode, c; in_math_mode = true) * tag)
    end
    return out
end
function Base.print(io::IO, m::LDRModel)
    mode = MIME("text/plain")
    s = JuMP.objective_sense(m)
    println(io, s == MOI.MAX_SENSE ? "Max " : s == MOI.MIN_SENSE ? "Min " : "Feasibility", s == MOI.FEASIBILITY_SENSE ? "" : JuMP.objective_function_string(mode, m))
    println(io, "Subject to")
    foreach(c -> println(io, " ", c), JuMP.constraints_string(mode, m))
    println(io, "Uncertainties")
    for d in m.declarations
        names = [JuMP.name(LDRVariable(m, v)) for v in d.vars]
        println(io, " ", length(names) == 1 ? names[1] : "[" * join(names, ", ") * "]", " ~ ", d.dist, "   [",
            join((string(get(m.stage, v, "unmarked")) for v in d.vars), ", "), "]")
    end
    println(io, "Decision stages")
    for v in JuMP.all_variables(m)
        _is_uncertainty(m, v) || println(io, " ", JuMP.name(v), ": ", get(m.stage, v.index, "unmarked"))
    end
    return
end
Base.print(m::LDRModel) = print(stdout, m)

struct LDRReferenceMap
    model::LDRModel
    index_map::MOI.IndexMap
end
Base.getindex(r::LDRReferenceMap, x::LDRVariable) = LDRVariable(r.model, r.index_map[x.index])
Base.getindex(r::LDRReferenceMap, c::JuMP.ConstraintRef) = _cref(r.model, r.index_map[c.index])

"""
    copy_model(model) -> (new_model, reference_map)

Copy the model with its LDR metadata (stages, uncertainties, tags, attributes, options, solver).
"""
function JuMP.copy_model(m::LDRModel)
    new = LDRModel(m.solver)
    inner, ref = JuMP.copy_model(m.inner)
    new.inner = inner
    imap = ref.index_map
    new.solver_attributes = copy(m.solver_attributes)
    new.silent = m.silent
    new.enabled = copy(m.enabled)
    for (v, s) in m.stage
        new.stage[imap[v]] = s
    end
    for d in m.declarations
        nd = _Declaration(d.dist, [imap[v] for v in d.vars])
        push!(new.declarations, nd)
        foreach(v -> new.uncertainty[v] = nd, nd.vars)
    end
    for ((v, T), a) in m.attributes
        new.attributes[(imap[v], T)] = copy(a)
    end
    new.valid_rows = Set{CI}(imap[c] for c in m.valid_rows)
    new.truncate_rows = Set{CI}(imap[c] for c in m.truncate_rows)
    new.constraint_stage = Dict{CI,Int}(imap[c] => s for (c, s) in m.constraint_stage)
    rmap = LDRReferenceMap(new, imap)
    for (k, v) in m.obj_dict       # registered names; objects deleted since registration are skipped
        try
            new.obj_dict[k] = _remap(rmap, v)
        catch err
            err isa Union{KeyError,MethodError} || rethrow()
        end
    end
    return new, rmap
end
_remap(r::LDRReferenceMap, v::Union{LDRVariable,JuMP.ConstraintRef}) = r[v]
_remap(r::LDRReferenceMap, v::AbstractArray) = _remap.(Ref(r), v)

end # module
