# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The stage of each variable, from its `VariableIndex`-in-`Stage` constraint. Such a constraint has
# the index of its variable, so a variable has at most one.
function _variable_stages(src::MOI.ModelLike)
    stages = Dict{MOI.VariableIndex,Int}()
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex,Stage}())
        stages[MOI.get(src, MOI.ConstraintFunction(), ci)] = MOI.get(src, MOI.ConstraintSet(), ci).t
    end
    return stages
end

function _uncertainty_declarations(src::MOI.ModelLike, stages::Dict{MOI.VariableIndex,Int})
    declarations = UncertaintyDeclaration[]
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        if !(S <: _AnyUncertainty)
            continue
        end
        for ci in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
            f = MOI.get(src, MOI.ConstraintFunction(), ci)
            variables = f isa MOI.VariableIndex ? [f] : copy(f.variables)
            set = MOI.get(src, MOI.ConstraintSet(), ci)
            variable_stages = Union{Nothing,Int}[get(stages, v, nothing) for v in variables]
            declaration = UncertaintyDeclaration(set.distribution, variables, variable_stages)
            push!(declarations, declaration)
        end
    end
    return declarations
end

# The constraints tagged with a role, by role.
function _xi_roles(src::MOI.ModelLike)
    valid, truncate = Set{MOI.ConstraintIndex}(), Set{MOI.ConstraintIndex}()
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        if !(XiRole() in MOI.get(src, MOI.ListOfConstraintAttributesSet{F,S}()))
            continue
        end
        for ci in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
            role = MOI.get(src, XiRole(), ci)
            if role isa ValidInequality
                push!(valid, ci)
            elseif role isa Truncate
                push!(truncate, ci)
            end
        end
    end
    return valid, truncate
end

# The breakpoints of each variable: a number of breakpoints or a vector of breakpoints. `nothing`
# (the value of a variable without breakpoints, or of cleared ones) means no lifting.
function _breakpoints(src::MOI.ModelLike)
    breakpoints = Dict{MOI.VariableIndex,Union{Int,Vector{Float64}}}()
    if !(BreakPoints() in MOI.get(src, MOI.ListOfVariableAttributesSet()))
        return breakpoints
    end
    for v in MOI.get(src, MOI.ListOfVariableIndices())
        points = MOI.get(src, BreakPoints(), v)
        if points isa Integer
            breakpoints[v] = Int(points)
        elseif points isa AbstractVector{<:Real}
            breakpoints[v] = Vector{Float64}(points)
        elseif points !== nothing
            error(
                "The breakpoints of $(_variable_name(src, v)) must be an integer (their " *
                "number) or a vector of numbers; got $(repr(points)).",
            )
        end
    end
    return breakpoints
end

# The combinations with breakpoints of `Experimental.set_breakpoints`.
function _combinations(src::MOI.ModelLike)
    out = Pair{MOI.ScalarAffineFunction{Float64},Union{Int,Vector{Float64}}}[]
    if !(CombinationBreakPoints() in MOI.get(src, MOI.ListOfModelAttributesSet()))
        return out
    end
    for (f, points) in something(MOI.get(src, CombinationBreakPoints()), [])
        push!(out, f => (points isa Integer ? Int(points) : Vector{Float64}(points)))
    end
    return out
end

"""
    model_info(src::MOI.ModelLike) -> ModelInfo

Read the linear-decision-rule metadata of `src`: the uncertainty declarations, the stages, the
roles of the uncertainty-only rows, the breakpoints, and the combinations of uncertainties with
breakpoints of `Experimental.set_breakpoints`. The result is meant for [`canonicalize`](@ref).

A stage on an uncertainty goes to its declaration, and a stage on any other variable to
`decision_stages`. It only reads: the checks of the model itself are those of `canonicalize`.

## Example

```julia
import LinearDecisionRules as LDR
info = LDR.model_info(backend(model))
form = LDR.canonicalize(backend(model), info)
```
"""
function model_info(src::MOI.ModelLike)
    stages = _variable_stages(src)
    declarations = _uncertainty_declarations(src, stages)
    in_uncertainty = Set(v for d in declarations for v in d.variables)
    decision_stages = Dict(v => t for (v, t) in stages if !(v in in_uncertainty))
    valid_rows, truncate_rows = _xi_roles(src)
    return ModelInfo(;
        uncertainties = declarations,
        decision_stages,
        valid_rows,
        truncate_rows,
        breakpoints = _breakpoints(src),
        combinations = _combinations(src),
    )
end
