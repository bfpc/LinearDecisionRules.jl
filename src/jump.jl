Base.@kwdef mutable struct LDRModel <: JuMP.AbstractModel
    cache_model::JuMP.Model = JuMP.Model()
    primal_model::JuMP.Model = JuMP.Model()
    dual_model::JuMP.Model = JuMP.Model()

    # maps
    cache_first_stage::Set{JuMP.VariableRef} = Set{JuMP.VariableRef}()
    uncertainty_to_distribution::Dict{JuMP.VariableRef,Tuple{Int,Int}} = Dict{JuMP.VariableRef,Tuple{Int,Int}}()
    cache_scalar_distributions::Vector{Distributions.Distribution} = Vector{Distributions.Distribution}()
    cache_vector_distributions::Vector{Distributions.Distribution} = Vector{Distributions.Distribution}()

    # options
    solver::Any = nothing
    solve_primal::Bool = true
    solve_dual::Bool = true
    silent::Bool = false

    ext::Dict{Symbol,Any} = Dict{Symbol,Any}()

    # API part
    obj_dict::Dict{Symbol,Any} = Dict{Symbol,Any}()
end

function LDRModel(
    optimizer_factory
)
    model = LDRModel()
    model.solver = optimizer_factory
    return model
end

struct SolvePrimal end
struct SolveDual end
function JuMP.set_attribute(model::LDRModel, ::SolvePrimal, value::Bool)
    model.solve_primal = value
end
function JuMP.get_attribute(model::LDRModel, ::SolvePrimal)
    return model.solve_primal
end
function JuMP.set_attribute(model::LDRModel, ::SolveDual, value::Bool)
    model.solve_dual = value
end
function JuMP.get_attribute(model::LDRModel, ::SolveDual)
    return model.solve_dual
end

_primal_disabled() = error("Primal solution mode is disabled.")
_dual_disabled() = error("Dual solution mode is disabled.")

function JuMP.termination_status(model::LDRModel; dual = false)
    if dual
        if !model.solve_dual
            return MOI.OPTIMIZE_NOT_CALLED
        else
            return JuMP.termination_status(model.dual_model)
        end
    else
        if !model.solve_primal
            return MOI.OPTIMIZE_NOT_CALLED
        else
            return JuMP.termination_status(model.primal_model)
        end
    end
end

function JuMP.primal_status(model::LDRModel; dual = false)
    if dual
        if !model.solve_dual
            return MOI.NO_SOLUTION
        else
            return JuMP.primal_status(model.dual_model)
        end
    else
        if !model.solve_primal
            return MOI.NO_SOLUTION
        else
            return JuMP.primal_status(model.primal_model)
        end
    end
end

function JuMP.dual_status(model::LDRModel; dual = false)
    return MOI.NO_SOLUTION
end

function JuMP.objective_value(model::LDRModel; dual = false)
    if dual
        if !model.solve_dual
            _dual_disabled()
        else
            return JuMP.objective_value(model.dual_model)
        end
    else
        if !model.solve_primal
            _primal_disabled()
        else
            return JuMP.objective_value(model.primal_model)
        end
    end
end

function JuMP.set_silent(model::LDRModel)
    model.silent = true
    return nothing
end

function JuMP.unset_silent(model::LDRModel)
    model.silent = false
    return nothing
end


struct _LDR_SolutionSummary
    primal::Any
    dual::Any
end

function Base.show(io::IO, summary::_LDR_SolutionSummary)
    println(io, "Primal LDR model:")
    if summary.primal !== nothing
        show(io, summary.primal)
    else
        println(io, "* Primal solution mode is disabled.")
    end
    println(io)
    println(io, "Dual LDR model:")
    if summary.dual !== nothing
        show(io, summary.dual)
    else
        println(io, "* Dual solution mode is disabled.")
    end
    return
end

function JuMP.solution_summary(
    model::LDRModel;
    result::Int = 1,
    verbose::Bool = false,
)
    _LDR_SolutionSummary(
        if model.solve_primal
            JuMP.solution_summary(model.primal_model, result = result, verbose = verbose)
        else
            nothing
        end,
        if model.solve_dual
            JuMP.solution_summary(model.dual_model, result = result, verbose = verbose)
        else
            nothing
        end
    )
end


Base.broadcastable(model::LDRModel) = Ref(model)

JuMP.object_dictionary(model::LDRModel) = model.obj_dict

JuMP.variable_ref_type(::Union{LDRModel,Type{LDRModel}}) = JuMP.VariableRef

function JuMP.optimize!(model::LDRModel)
    if model.solver === nothing
        error("solver not set")
    end
    if !model.solve_primal && !model.solve_dual
        error("Both primal and dual solution modes are disabled.")
    else
        _prepare_data(model)
    end
    if model.solve_primal
        _solve_primal_ldr(model)
    end
    if model.solve_dual
        _solve_dual_ldr(model)
    end
    return
end

struct VectorUncertainty
    distribution::Distributions.Distribution
    function VectorUncertainty(distribution::Distributions.Distribution)
        if !(distribution isa Distributions.MultivariateDistribution)
            error("Only MultivariateDistribution distributions are supported, got a $(typeof(distribution)).")
        end
        return new(distribution)
    end
end

struct _VectorUncertainty <: JuMP.AbstractVariable
    info::Vector{JuMP.VariableInfo}
    distribution::Distributions.Distribution
end

function JuMP.build_variable(
    _err::Function,
    info::Vector{<:JuMP.ScalarVariable},
    set::VectorUncertainty;
    kwargs...
)
    infos = [i.info for i in info]
    n1 = length(infos)
    n2 = length(set.distribution)
    @assert n1 == n2
    return _VectorUncertainty(infos, set.distribution)
end

function JuMP.add_variable(
    model::LDRModel,
    uncertainty::_VectorUncertainty,
    names::Vector{String},
)
    _info = uncertainty.info
    dist = uncertainty.distribution
    ret = Vector{JuMP.VariableRef}(undef, length(_info))
    push!(model.cache_vector_distributions, dist)
    dist_index = length(model.cache_vector_distributions)
    upper = maximum(dist)
    lower = minimum(dist)
    for i in 1:length(_info)
        _has_lb = _info[i].has_lb
        _lower_bound = _info[i].lower_bound
        _has_ub = _info[i].has_ub
        _upper_bound = _info[i].upper_bound
        _has_fix = _info[i].has_fix
        _fixed_value = _info[i].fixed_value
        _has_start = _info[i].has_start
        _start = _info[i].start
        _binary = _info[i].binary
        _integer = _info[i].integer

        if lower[i] == -Inf
            error("Lower bound of the distribution ($dist) is -Inf.")
        end
        if upper[i] == Inf
            error("Upper bound of the distribution ($dist) is +Inf.")
        end
        if _has_lb
            error("Enforce bounds on the distribution only")
        else
            _lower_bound = lower[i]
            _has_lb = true
        end
        if _has_ub
            error("Enforce bounds on the distribution only")
        else
            _upper_bound = upper[i]
            _has_ub = true
        end    

        info = VariableInfo(
            _has_lb,
            _lower_bound,
            _has_ub,
            _upper_bound,
            _has_fix,
            _fixed_value,
            _has_start,
            _start,
            _binary,
            _integer,
        )

        var = JuMP.add_variable(
            model.cache_model,
            JuMP.ScalarVariable(info),
            names[i],
        )
        ret[i] = var
        model.uncertainty_to_distribution[var] = (dist_index, i)
    end
    return ret
end

function Uncertainty(; distribution::Distributions.Distribution = nothing)
    if distribution === nothing
        error("distribution is required.")
    elseif distribution isa Distributions.UnivariateDistribution
        return ScalarUncertainty(distribution)
    elseif distribution isa Distributions.MultivariateDistribution
        return VectorUncertainty(distribution)
    end
    error("Only univariate distributions are supported, got a $(typeof(distribution)).")
end

struct ScalarUncertainty
    distribution::Distributions.Distribution
    function ScalarUncertainty(distribution::Distributions.Distribution)
        if !(distribution isa Distributions.UnivariateDistribution)
            error("Only univariate distributions are supported, got a $(typeof(distribution)).")
        end
        return new(distribution)
    end
end

struct _ScalarUncertainty <: JuMP.AbstractVariable
    info::JuMP.VariableInfo
    distribution::Distributions.Distribution
end

function JuMP.build_variable(
    _err::Function,
    info::JuMP.ScalarVariable,
    set::ScalarUncertainty;
    kwargs...
)
    return _ScalarUncertainty(info.info, set.distribution)
end

function JuMP.add_variable(
    model::LDRModel,
    uncertainty::_ScalarUncertainty,
    name::String,
)
    _has_lb = uncertainty.info.has_lb
    _lower_bound = uncertainty.info.lower_bound
    _has_ub = uncertainty.info.has_ub
    _upper_bound = uncertainty.info.upper_bound
    _has_fix = uncertainty.info.has_fix
    _fixed_value = uncertainty.info.fixed_value
    _has_start = uncertainty.info.has_start
    _start = uncertainty.info.start
    _binary = uncertainty.info.binary
    _integer = uncertainty.info.integer

    dist = uncertainty.distribution
    if !(dist isa Distributions.UnivariateDistribution)
        error("Only univariate distributions are supported, got a $(typeof(dist)).")
    end
    upper = maximum(dist)
    lower = minimum(dist)
    if lower == -Inf
        error("Lower bound of the distribution ($dist) is -Inf.")
    end
    if upper == Inf
        error("Upper bound of the distribution ($dist) is +Inf.")
    end
    if _has_lb
        error("Enforce bounds on the distribution only")
        # if _lower_bound != lower
        #     error("Lower bound of the distribution ($dist) is different from the lower bound of the variable.")
        # end
    else
        _lower_bound = lower
        _has_lb = true
    end
    if _has_ub
        error("Enforce bounds on the distribution only")
        # if _upper_bound != upper
        #     error("Upper bound of the distribution ($dist) is different from the upper bound of the variable.")
        # end
    else
        _upper_bound = upper
        _has_ub = true
    end

    info = VariableInfo(
        _has_lb,
        _lower_bound,
        _has_ub,
        _upper_bound,
        _has_fix,
        _fixed_value,
        _has_start,
        _start,
        _binary,
        _integer,
    )

    var = JuMP.add_variable(
        model.cache_model,
        JuMP.ScalarVariable(info),
        name,
    )

    push!(model.cache_scalar_distributions, dist)
    dist_index = length(model.cache_scalar_distributions)
    model.uncertainty_to_distribution[var] = (dist_index, 0)

    return var
end


struct FirstStage <: JuMP.AbstractVariableRef
    info::JuMP.VariableInfo
end

function JuMP.build_variable(
    _err::Function,
    info::JuMP.VariableInfo,
    ::Type{FirstStage};
    kwargs...
)
    return FirstStage(info)
end

function JuMP.add_variable(
    model::LDRModel,
    first_stage::FirstStage,
    name::String,
)
    var = JuMP.add_variable(
        model.cache_model,
        JuMP.ScalarVariable(first_stage.info),
        name,
    )
    push!(model.cache_first_stage, var)
    return var
end

##

function JuMP.add_variable(
    model::LDRModel,
    v::JuMP.AbstractVariable,
    name::String = "",
)
    return JuMP.add_variable(model.cache_model, v, name)
end

function JuMP.add_variable(
    model::LDRModel,
    variable::JuMP.VariableConstrainedOnCreation,
    name::String,
)
    return JuMP.add_variable(model.cache_model, variable, name)
end

function JuMP.add_variable(
    model::LDRModel,
    variable::JuMP.VariablesConstrainedOnCreation,
    names,
)
    return JuMP.add_variable(model.cache_model, variable, names)
end

function JuMP.delete(model::LDRModel, vref::JuMP.AbstractVariableRef)
    error("not implemented")
    JuMP.delete(model.cache_model, vref)
    # delete!(model.cache_uncertainty, vref)
    return
end

function JuMP.delete(model::LDRModel, vref::Vector{V}) where {V<:JuMP.AbstractVariableRef}
    JuMP.delete.(model, vref)
    return
end

function JuMP.is_valid(model::LDRModel, vref::JuMP.AbstractVariableRef)
    return JuMP.is_valid(model.cache_model, vref)
end

function JuMP.all_variables(model::LDRModel)
    return JuMP.all_variables(model.cache_model)
end

JuMP.num_variables(m::LDRModel) = JuMP.num_variables(m.cache_model)

function JuMP.add_constraint(
    model::LDRModel,
    c::JuMP.AbstractConstraint,
    name::String = "",
)
    return JuMP.add_constraint(model.cache_model, c, name)
end

function JuMP.delete(model::LDRModel, constraint_ref::JuMP.ConstraintRef)
    JuMP.delete(model.cache_model, constraint_ref)
    # TODO: fix maps
    error("fix maps")
    return
end

function JuMP.delete(model::LDRModel, constraint_ref::Vector{<:JuMP.ConstraintRef})
    JuMP.delete.(model, constraint_ref)
    return
end

function JuMP.is_valid(model::LDRModel, constraint_ref::JuMP.ConstraintRef)
    return JuMP.is_valid(model.cache_model, constraint_ref)
end

function JuMP.all_constraints(
    model::LDRModel,
    ::Type{F},
    ::Type{S},
) where {F,S<:JuMP.MOI.AbstractSet}
    return JuMP.all_constraints(model.cache_model, F, S)
end

function JuMP.constraint_object(cref::JuMP.ConstraintRef)
    return JuMP.constraint_object(cref.model.cache_model, cref)
end

function JuMP.num_constraints(
    model::LDRModel,
    ::Type{F},
    ::Type{S},
) where {F,S<:JuMP.MOI.AbstractSet}
    return JuMP.num_constraints(model.cache_model, F, S)
end

function JuMP.num_constraints(
    model::LDRModel;
    count_variable_in_set_constraints::Bool,
)
    return JuMP.num_constraints(
        model.cache_model;
        count_variable_in_set_constraints = count_variable_in_set_constraints,
    )
end

function JuMP.set_objective_function(
    m::LDRModel,
    f::Union{JuMP.AbstractJuMPScalar,Vector{<:JuMP.AbstractJuMPScalar}, Real},
)
    JuMP.set_objective_function(m.cache_model, f)
    return
end

JuMP.objective_sense(model::LDRModel) = JuMP.objective_sense(model.cache_model)

function JuMP.set_objective_sense(model::LDRModel, sense)
    JuMP.set_objective_sense(model.cache_model, sense)
    return
end

JuMP.objective_function_type(model::LDRModel) = JuMP.objective_function_type(model.cache_model)

JuMP.objective_function(model::LDRModel) = JuMP.objective_function(model.cache_model)

function JuMP.objective_function(model::LDRModel, FT::Type)
    return JuMP.objective_function(model.cache_model, FT)
end

function JuMP.variable_by_name(model::LDRModel, name::String)
    return JuMP.variable_by_name(model.cache_model, name)
end

function JuMP.constraint_by_name(model::LDRModel, name::String)
    return JuMP.constraint_by_name(model.cache_model, name)
end

JuMP.show_backend_summary(io::IO, model::LDRModel) = JuMP.show_backend_summary(io, model.cache_model)

function JuMP.show_objective_function_summary(io::IO, model::LDRModel)
    JuMP.show_objective_function_summary(io, model.cache_model)
    return
end

function JuMP.objective_function_string(print_mode, model::LDRModel)
    return JuMP.objective_function_string(print_mode, model.cache_model)
end

function JuMP.show_constraints_summary(io::IO, model::LDRModel)
    JuMP.show_constraints_summary(io, model.cache_model)
    return
end

function JuMP.constraints_string(print_mode, model::LDRModel)
    return JuMP.constraints_string(print_mode, model.cache_model)
end

function JuMP.set_optimizer(model::LDRModel, optimizer_factory)
    model.solver = optimizer_factory
    return nothing
end