# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    InformationStructure(
        decision_stages::Vector{Int},
        column_stages::Vector{Int},
        integer::BitVector,
    )
    InformationStructure(form::CanonicalForm)

Which columns of ξ each decision observes (see [`observes`](@ref)).

Decision `i` observes column `j` if `j == 1` (the constant) or if the stage of column `j` is at most
the stage of decision `i`; integer decisions observe only the constant. Since columns are sorted by
stage, the observed columns of every decision are a prefix `1:num_visible[i]`
(see [`visible_columns`](@ref)).

## Fields

 * `decision_stages::Vector{Int}`: the stage of each decision.
 * `column_stages::Vector{Int}`: the stage of each column of ξ, nondecreasing; the constant column
   is in stage 1.
 * `num_visible::Vector{Int}`: the number of columns that each decision observes.
"""
struct InformationStructure
    decision_stages::Vector{Int}
    column_stages::Vector{Int}
    num_visible::Vector{Int}

    function InformationStructure(
        decision_stages::Vector{Int},
        column_stages::Vector{Int},
        integer::BitVector,
    )
        if !issorted(column_stages)
            throw(ArgumentError("The columns of ξ must be sorted by stage."))
        end
        num_visible = map(eachindex(decision_stages)) do i
            return integer[i] ? 1 : max(1, searchsortedlast(column_stages, decision_stages[i]))
        end
        return new(decision_stages, column_stages, num_visible)
    end
end

function InformationStructure(form::CanonicalForm)
    return InformationStructure(form.decision_stages, [1; form.uncertainty_stages], form.integer)
end

"""
    observes(observed::InformationStructure, i::Int, j::Int) -> Bool

Return `true` if decision `i` observes column `j` of ξ, that is, if its decision rule may have a
nonzero coefficient on `ξ[j]`.
"""
observes(observed::InformationStructure, i::Int, j::Int) = j <= observed.num_visible[i]

"""
    visible_columns(observed::InformationStructure, i::Int) -> UnitRange{Int}

Return the columns of ξ that decision `i` observes, a prefix `1:k`.
"""
visible_columns(observed::InformationStructure, i::Int) = 1:observed.num_visible[i]
