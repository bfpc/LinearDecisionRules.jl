# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The support block of each column of ξ (0 for the constant column).
function _block_of_columns(blocks::Vector{SupportBlock}, k::Int)
    block_of = zeros(Int, k)
    for (b, block) in enumerate(blocks)
        block_of[block.columns] .= b
    end
    return block_of
end

# The support blocks with a column where the row is structurally nonzero. Blocks are sorted by
# their first column, so those with a column in `2:num_visible` are a prefix.
function _touched_blocks(row::_LinearRow, first_columns::Vector{Int}, block_of::Vector{Int})
    touched = collect(1:searchsortedlast(first_columns, row.num_visible))
    for j in SparseArrays.nonzeroinds(row.b)
        if j > row.num_visible
            push!(touched, block_of[j])
        end
    end
    return unique!(sort!(touched))
end

# The multipliers of the rows of `block`, in the dual cone of each piece of its cone: the
# nonnegative orthant and the second-order cone are self-dual. The linear piece is `λ[r,b]` and
# the `p`-th piece, a second-order cone, `λ[r,b,p]`.
function _multipliers!(model::JuMP.Model, block::SupportBlock, r::Int, b::Int)
    named = JuMP.set_string_names_on_creation(model)
    λ = JuMP.VariableRef[]
    for (p, (cone, rows)) in enumerate(zip(block.cones, _cone_rows(block)))
        isempty(rows) && continue
        if cone isa MOI.Nonnegatives
            name = named ? "λ[$r,$b]" : ""
            append!(λ, JuMP.@variable(model, [1:length(rows)], lower_bound = 0.0, base_name = name))
        else
            name = named ? "λ[$r,$b,$p]" : ""
            set = JuMP.SecondOrderCone()
            append!(λ, JuMP.@variable(model, [1:length(rows)] in set, base_name = name))
        end
    end
    return λ
end

# With Ξ = Π_b {η_b : W_b η_b - h_b ∈ K_b} and the slack `s(ξ) = s₁ + Σ_b s_b ⋅ η_b`, conic duality
# gives
#   min_{η_b ∈ Ξ_b} s_b ⋅ η_b = max {h_b ⋅ λ_b : W_b' λ_b = s_b, λ_b ∈ K_b*},
# so `s(ξ) ≥ 0` on Ξ iff there are `λ_b ∈ K_b*` with `W_b' λ_b = s_b` and
# `s₁ + Σ_b h_b ⋅ λ_b ≥ 0` (Kuhn, Wiesemann and Georghiou, 2011, Proposition 1, for a polyhedral
# Ξ_b; with second-order cones, strong duality needs a point of Ξ_b strictly inside them, and the
# "if" direction holds without it, so the rule stays feasible on Ξ).
# A block where `s_b` is structurally zero needs no multiplier, since Ξ_b is nonempty.
# The slack is an expression of X, as `SlackAsExpr` in the previous version: explicit slack
# variables would add one variable and one equality per (row, column) and save nothing.
function _add_robust_counterpart!(
    model::JuMP.Model,
    X,
    r::Int,
    ineq::_Inequality,
    blocks::Vector{SupportBlock},
    touched::Vector{Int},
)
    row, sign = ineq.row, ineq.sign
    # The coupling `s₁ + Σ_b h_b ⋅ λ_b ≥ 0`, with `s₁ = sign * (b₁ - a ⋅ X₁)`.
    coupling = _add_row_column!(JuMP.AffExpr(sign * row.b[1]), X, row, 1, -sign)
    for b in touched
        block = blocks[b]
        λ = _multipliers!(model, block, r, b)
        for (k, h) in enumerate(block.h)
            iszero(h) || JuMP.add_to_expression!(coupling, h, λ[k])
        end
        W = block.W
        rows, values = SparseArrays.rowvals(W), SparseArrays.nonzeros(W)
        for (c, j) in enumerate(block.columns)
            expr = _add_row_column!(JuMP.AffExpr(0.0), X, row, j, sign)
            for p in SparseArrays.nzrange(W, c)
                JuMP.add_to_expression!(expr, values[p], λ[rows[p]])
            end
            JuMP.@constraint(model, expr == sign * row.b[j])
        end
    end
    JuMP.@constraint(model, coupling >= 0)
    return
end

function _build!(model::JuMP.Model, ::PrimalLDR, form::CanonicalForm, mom::Moments)
    info = InformationStructure(form)
    X = _rule_variables!(model, form, info)
    _add_equalities!(model, form, info, X)
    blocks = support_blocks(form)
    block_of = _block_of_columns(blocks, 1 + length(form.uncertainties))
    first_columns = [first(block.columns) for block in blocks]
    for (r, ineq) in enumerate(_inequalities(form, info))
        if _is_deterministic(ineq.row)
            _add_deterministic!(model, X, ineq)
        else
            touched = _touched_blocks(ineq.row, first_columns, block_of)
            _add_robust_counterpart!(model, X, r, ineq, blocks, touched)
        end
    end
    _set_objective!(model, form, mom, X)
    return X
end
