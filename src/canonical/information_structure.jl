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

The rule of a decision may also use some later columns, `extra[i]`: [`DualLDR`](@ref) relaxes its
information structure that way for the laws whose conditional expectations are not linear (see
its docstring). The primal and sampled rules never do, so their `extra` are empty.

## Fields

 * `decision_stages::Vector{Int}`: the stage of each decision.
 * `column_stages::Vector{Int}`: the stage of each column of ξ, nondecreasing; the constant column
   is in stage 1.
 * `num_visible::Vector{Int}`: the number of columns that each decision observes.
 * `extra::Vector{Vector{Int}}`: the later columns that the rule of each decision may also use,
   sorted, all after `num_visible[i]`.
"""
struct InformationStructure
    decision_stages::Vector{Int}
    column_stages::Vector{Int}
    num_visible::Vector{Int}
    extra::Vector{Vector{Int}}

    function InformationStructure(
        decision_stages::Vector{Int},
        column_stages::Vector{Int},
        integer::BitVector,
        extra::Vector{Vector{Int}} = [Int[] for _ in decision_stages],
    )
        if !issorted(column_stages)
            throw(ArgumentError("The columns of ξ must be sorted by stage."))
        end
        num_visible = map(eachindex(decision_stages)) do i
            return integer[i] ? 1 : max(1, searchsortedlast(column_stages, decision_stages[i]))
        end
        for i in eachindex(extra)
            if !(issorted(extra[i]) && allunique(extra[i]) && all(>(num_visible[i]), extra[i]))
                throw(ArgumentError("The extra columns of a decision must be later and sorted."))
            end
        end
        return new(decision_stages, column_stages, num_visible, extra)
    end
end

function InformationStructure(form::CanonicalForm)
    return InformationStructure(form.decision_stages, [1; form.uncertainty_stages], form.integer)
end

"""
    observes(observed::InformationStructure, i::Int, j::Int) -> Bool

Return `true` if decision `i` observes column `j` of ξ, that is, if its decision rule may have a
nonzero coefficient on `ξ[j]`: a column of its prefix, or one of its extra columns.
"""
function observes(observed::InformationStructure, i::Int, j::Int)
    return j <= observed.num_visible[i] || insorted(j, observed.extra[i])
end

"""
    visible_columns(observed::InformationStructure, i::Int) -> Vector{Int}

Return the columns of ξ that the rule of decision `i` may use, sorted: the prefix
`1:num_visible[i]`, then its extra columns.
"""
function visible_columns(observed::InformationStructure, i::Int)
    return vcat(1:observed.num_visible[i], observed.extra[i])
end
