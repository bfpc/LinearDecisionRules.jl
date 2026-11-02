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
        breakpoints = Dict{MOI.VariableIndex,Union{Int,Vector{Float64}}}(),
        combinations = Pair{MOI.ScalarAffineFunction{Float64},Union{Int,Vector{Float64}}}[],
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
   condition the law on the set where they hold, and bound Ξ (see [`canonicalize`](@ref)).
 * `breakpoints::Dict{MOI.VariableIndex,Union{Int,Vector{Float64}}}`: the breakpoints of the
   [`AxisPiecewise`](@ref) lifting of an uncertainty declared alone (with a univariate law): the
   interior breakpoints, or their number `n`, for `n` equally spaced breakpoints inside the
   support. An empty vector or `0` means no lifting.
 * `combinations::Vector{Pair{MOI.ScalarAffineFunction{Float64},Union{Int,Vector{Float64}}}}`: the
   combinations of uncertainties with breakpoints, each with its breakpoints or their number, for
   the [`CombinationPiecewise`](@ref) liftings of `Experimental` (see
   [`Experimental.set_breakpoints`](@ref)).
"""
Base.@kwdef struct ModelInfo
    uncertainties::Vector{UncertaintyDeclaration} = UncertaintyDeclaration[]
    decision_stages::Dict{MOI.VariableIndex,Int} = Dict{MOI.VariableIndex,Int}()
    valid_rows::Set{MOI.ConstraintIndex} = Set{MOI.ConstraintIndex}()
    truncate_rows::Set{MOI.ConstraintIndex} = Set{MOI.ConstraintIndex}()
    breakpoints::Dict{MOI.VariableIndex,Union{Int,Vector{Float64}}} =
        Dict{MOI.VariableIndex,Union{Int,Vector{Float64}}}()
    combinations::Vector{Pair{MOI.ScalarAffineFunction{Float64},Union{Int,Vector{Float64}}}} =
        Pair{MOI.ScalarAffineFunction{Float64},Union{Int,Vector{Float64}}}[]
end
