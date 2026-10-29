# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    PrimalLDR(; attributes = [], box_closed_form::Bool = true)

The primal linear decision rule: every constraint holds for every outcome in the support Ξ.

Its rule is a policy, so its objective value is attained (a lower bound of a maximization, an upper
bound of a minimization). Each inequality is replaced by its robust counterpart over the
[`SupportBlock`](@ref)s on which it depends (Kuhn, Wiesemann and Georghiou, 2011), with
multipliers in the dual cone of each block: nonnegative for its linear rows, in a second-order
cone for each of its second-order cone rows, so the inner problem is then a second-order cone
program. On a block that is a box (finite bounds of its coordinates, no other row), the
multipliers are replaced by the closed form `min(lo s_j, hi s_j)` of each coefficient `s_j` of the
slack, with one auxiliary variable per coefficient; `box_closed_form = false` keeps the
multipliers, to compare both forms.

`attributes` are `name => value` raw attributes of the inner optimizer, set after (so over) those
of the optimizer, for this formulation only: `PrimalLDR(; attributes = ["solver" => "ipx"])`.
Formulations are compared by type: the attributes are settings, not part of the identity.

With HiGHS, keep the default dual simplex for this formulation: on inventory and hydro problems
(HiGHS 1.15), it is within 2 times of the best mode almost everywhere.
The exception is a long hydro problem (T = 8), where the interior point method IPX without
crossover, `["solver" => "ipx", "run_crossover" => "off"]`, is 3 times faster; the other modes
cost 3 to 16 times more on the larger inventory problem, or fail on the hydro one.
"""
struct PrimalLDR <: AbstractFormulation
    attributes::Vector{Pair{String,Any}}
    box_closed_form::Bool
end

function PrimalLDR(; attributes = Pair{String,Any}[], box_closed_form::Bool = true)
    return PrimalLDR(_attribute_pairs(attributes), box_closed_form)
end

# The formulations of a type are the same formulation whatever their attributes.
Base.:(==)(::PrimalLDR, ::PrimalLDR) = true

Base.hash(::PrimalLDR, h::UInt) = hash(:PrimalLDR, h)

function capabilities(::PrimalLDR)
    return FormulationCapabilities(MOI.FEASIBLE_POINT, false)
end

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

# The bounds `(lower, upper)` of the coordinates of a box block, or `nothing` for any other block.
# A box block has rows of role `:box` only, each the bound of one coordinate, and both bounds of
# every coordinate are finite (an infinite bound has no row).
function _box_bounds(block::SupportBlock)
    if !all(==(:box), block.roles) || !all(cone -> cone isa MOI.Nonnegatives, block.cones)
        return nothing
    end
    lower, upper = fill(-Inf, length(block.columns)), fill(Inf, length(block.columns))
    W = block.W
    rows, values = SparseArrays.rowvals(W), SparseArrays.nonzeros(W)
    count = zeros(Int, size(W, 1))
    for c in axes(W, 2), p in SparseArrays.nzrange(W, c)
        w = values[p]
        iszero(w) && continue
        count[rows[p]] += 1
        bound = block.h[rows[p]] / w
        if w > 0
            lower[c] = max(lower[c], bound)
        else
            upper[c] = min(upper[c], bound)
        end
    end
    if !all(==(1), count) || !all(isfinite, lower) || !all(isfinite, upper)
        return nothing
    end
    return lower, upper
end

# The closed form of the robust counterpart on a box block: with `s_j` the coefficient of the
# slack on column `j`, whose coordinate ranges over `[lo, hi]`,
#   min_{η_j ∈ [lo, hi]} s_j η_j = min(lo s_j, hi s_j) = hi s_j - (hi - lo) max(0, s_j)
#                                                  = lo s_j - (hi - lo) max(0, -s_j),
# so one auxiliary `w ≥ 0` with `w ≥ s_j` (or `w ≥ -s_j`) adds `hi s_j - (hi - lo) w` (or
# `lo s_j - (hi - lo) w`) to the coupling: the coupling holds for some `w` iff it holds at the
# smallest one, `max(0, ±s_j)`.
# The endpoint of smaller magnitude multiplies `s_j`, so that a zero endpoint leaves it out of the
# coupling. Against the two multipliers and one equality of the general form, this has one variable
# fewer and as many rows; a constant `s_j` or a point `lo = hi` needs no auxiliary. The form
# `c s_j - ρ t` with `t ≥ ±s_j` would need two rows and three copies of `s_j`, and
# `z ≤ lo s_j`, `z ≤ hi s_j` two rows.
function _add_box_counterpart!(
    model::JuMP.Model,
    coupling::JuMP.AffExpr,
    X,
    r::Int,
    ineq::_Inequality,
    block::SupportBlock,
    lower::Vector{Float64},
    upper::Vector{Float64},
)
    row, sign = ineq.row, ineq.sign
    named = JuMP.set_string_names_on_creation(model)
    for (c, j) in enumerate(block.columns)
        lo, hi = lower[c], upper[c]
        s = _add_row_column!(JuMP.AffExpr(sign * row.b[j]), X, row, j, -sign)
        if isempty(s.terms)
            JuMP.add_to_expression!(coupling, min(lo * s.constant, hi * s.constant))
        elseif lo == hi
            iszero(lo) || JuMP.add_to_expression!(coupling, lo, s)
        else
            w = JuMP.@variable(model, lower_bound = 0.0, base_name = named ? "w[$r,$j]" : "")
            upper_endpoint = abs(hi) <= abs(lo)
            endpoint = upper_endpoint ? hi : lo
            if upper_endpoint
                JuMP.@constraint(model, w - s >= 0)
            else
                JuMP.@constraint(model, w + s >= 0)
            end
            iszero(endpoint) || JuMP.add_to_expression!(coupling, endpoint, s)
            JuMP.add_to_expression!(coupling, lo - hi, w)
        end
    end
    return
end

# With Ξ = Π_b {η_b : W_b η_b - h_b ∈ K_b} and the slack `s(ξ) = s₁ + Σ_b s_b ⋅ η_b`, conic duality
# gives
#   min_{η_b ∈ Ξ_b} s_b ⋅ η_b = max {h_b ⋅ λ_b : W_b' λ_b = s_b, λ_b ∈ K_b*},
# so `s(ξ) ≥ 0` on Ξ iff there are `λ_b ∈ K_b*` with `W_b' λ_b = s_b` and
# `s₁ + Σ_b h_b ⋅ λ_b ≥ 0` (Kuhn, Wiesemann and Georghiou, 2011, Proposition 1, for a polyhedral
# Ξ_b; with second-order cones, strong duality needs a point of Ξ_b strictly inside them, and the
# "if" direction holds without it, so the rule stays feasible on Ξ).
# A block where `s_b` is structurally zero needs no multiplier, since Ξ_b is nonempty.
# The slack is an expression of X: explicit slack variables would add one variable and one
# equality per (row, column) and save nothing.
function _add_robust_counterpart!(
    model::JuMP.Model,
    X,
    r::Int,
    ineq::_Inequality,
    blocks::Vector{SupportBlock},
    boxes::Vector,
    touched::Vector{Int},
)
    row, sign = ineq.row, ineq.sign
    # The coupling `s₁ + Σ_b h_b ⋅ λ_b ≥ 0`, with `s₁ = sign * (b₁ - a ⋅ X₁)`.
    coupling = _add_row_column!(JuMP.AffExpr(sign * row.b[1]), X, row, 1, -sign)
    for b in touched
        block = blocks[b]
        if boxes[b] !== nothing
            _add_box_counterpart!(model, coupling, X, r, ineq, block, boxes[b]...)
            continue
        end
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

function _build!(model::JuMP.Model, f::PrimalLDR, form::CanonicalForm, mom::Moments)
    info = InformationStructure(form)
    X = _rule_variables!(model, form, info)
    _add_equalities!(model, form, info, X)
    blocks = support_blocks(form)
    block_of = _block_of_columns(blocks, 1 + length(form.uncertainties))
    first_columns = [first(block.columns) for block in blocks]
    # The bounds of the box blocks, for their closed form; `nothing` for the general form.
    boxes = Any[f.box_closed_form ? _box_bounds(block) : nothing for block in blocks]
    for (r, ineq) in enumerate(_inequalities(form, info))
        if _is_deterministic(ineq.row)
            _add_deterministic!(model, X, ineq)
        else
            touched = _touched_blocks(ineq.row, first_columns, block_of)
            _add_robust_counterpart!(model, X, r, ineq, blocks, boxes, touched)
        end
    end
    _set_objective!(model, form, mom, X)
    return X
end
