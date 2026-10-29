# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The interface of a formulation: its type, its capabilities, and the build of its inner problem.

"""
    AbstractFormulation

A reformulation of a [`CanonicalForm`](@ref) as a deterministic problem over the coefficients of
linear decision rules, solved by [`solve_formulation`](@ref).

The decision rule of decision `i` is `x_i(ξ) = X[i, :] ⋅ ξ` with `ξ = (1, η)`, where `X[i, j]` is
zero for the columns that decision `i` does not observe (see [`InformationStructure`](@ref)).
"""
abstract type AbstractFormulation end

_attribute_pairs(attributes) = Pair{String,Any}[String(k) => v for (k, v) in attributes]

_formulation_attributes(::AbstractFormulation) = Pair{String,Any}[]

"""
    FormulationCapabilities

What a formulation provides, returned by [`capabilities`](@ref).

## Fields

 * `rule_status::MOI.ResultStatusCode`: the status of its decision rule as a solution of the
   stochastic program when the inner problem is solved: `FEASIBLE_POINT` (feasible for every outcome
   in Ξ), `UNKNOWN_RESULT_STATUS` (feasible on the drawn outcomes only) or `NO_SOLUTION` (the rule is
   not a policy).
 * `bound::Bool`: whether its objective value bounds the optimal value of the stochastic program
   from the side opposite to the sense (an upper bound of a maximization).
"""
struct FormulationCapabilities
    rule_status::MOI.ResultStatusCode
    bound::Bool
end

"""
    capabilities(f::AbstractFormulation) -> FormulationCapabilities

Return the [`FormulationCapabilities`](@ref) of the formulation `f`.
"""
function capabilities end

# `_build!(model, f, form, mom)` builds the inner problem of the formulation `f` in `model` and
# returns its rule variables `X` (see `_rule_variables!`).
function _build! end
