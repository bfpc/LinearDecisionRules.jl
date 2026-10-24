# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    UncertaintyDeclaration(
        distribution::Distributions.Distribution,
        variables::Vector{MOI.VariableIndex},
        stages::Vector{Union{Nothing,Int}} = fill(nothing, length(variables)),
    )

One uncertainty declaration: the variables of one `@variable` call and their joint law.

A `Univariate` distribution has one variable, a `Multivariate` one has `length(distribution)`
variables, in the order of its components. Separate declarations are independent.

## Fields

 * `distribution::Distributions.Distribution`: the law of the variables.
 * `variables::Vector{MOI.VariableIndex}`: the variables, one per component.
 * `stages::Vector{Union{Nothing,Int}}`: the stage of each component; `nothing` means unmarked,
   that is, in the last stage.
"""
struct UncertaintyDeclaration
    distribution::Distributions.Distribution
    variables::Vector{MOI.VariableIndex}
    stages::Vector{Union{Nothing,Int}}
end

function UncertaintyDeclaration(
    distribution::Distributions.Distribution,
    variables::Vector{MOI.VariableIndex},
)
    return UncertaintyDeclaration(distribution, variables, fill(nothing, length(variables)))
end

"""
    ModelInfo(;
        uncertainties = UncertaintyDeclaration[],
        decision_stages = Dict{MOI.VariableIndex,Int}(),
        valid_rows = Set{MOI.ConstraintIndex}(),
        truncate_rows = Set{MOI.ConstraintIndex}(),
        breakpoints = Dict{MOI.VariableIndex,Vector{Float64}}(),
    )

The linear-decision-rule metadata of an MOI model, read by [`canonicalize`](@ref).

Every variable that is neither in an [`UncertaintyDeclaration`](@ref) nor a JuMP `Parameter` is a
decision.

## Fields

 * `uncertainties::Vector{UncertaintyDeclaration}`: the uncertainty declarations.
 * `decision_stages::Dict{MOI.VariableIndex,Int}`: the stage of each decision; an absent decision
   is unmarked, that is, in the last stage.
 * `valid_rows::Set{MOI.ConstraintIndex}`: uncertainty-only rows tagged `ValidInequality()`; they
   describe the support Ξ without changing the law.
 * `truncate_rows::Set{MOI.ConstraintIndex}`: uncertainty-only rows tagged `Truncate()`; they
   condition the law (not supported yet).
 * `breakpoints::Dict{MOI.VariableIndex,Vector{Float64}}`: breakpoints of the piecewise lifting
   of each uncertainty (not supported yet).
"""
Base.@kwdef struct ModelInfo
    uncertainties::Vector{UncertaintyDeclaration} = UncertaintyDeclaration[]
    decision_stages::Dict{MOI.VariableIndex,Int} = Dict{MOI.VariableIndex,Int}()
    valid_rows::Set{MOI.ConstraintIndex} = Set{MOI.ConstraintIndex}()
    truncate_rows::Set{MOI.ConstraintIndex} = Set{MOI.ConstraintIndex}()
    breakpoints::Dict{MOI.VariableIndex,Vector{Float64}} = Dict{MOI.VariableIndex,Vector{Float64}}()
end
