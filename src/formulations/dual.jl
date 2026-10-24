# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The rows `W'ξ = W_b η_b - h_b ξ₁ ∈ K_b` of one support block, seen through the moments:
# `E[s(ξ) W'ξ] = (S μ) (W'μ) + S Σ W'ᵀ` for the slack `s(ξ) = S ξ`.
struct _DualCone
    V::_SparseMatrix         # Σ W'ᵀ: one column per row of the block, one row per column of ξ
    Wμ::Vector{Float64}      # W'μ ∈ K_b, since the mean is in Ξ
    cones::Vector{_SupportCone}
    rows::Vector{UnitRange{Int}} # the rows of each piece of the cone
end

function _covariance_matrix(mom::Moments)
    k = length(mom.mean)
    I, J, V = Int[], Int[], Float64[]
    for block in mom.blocks, (b, j) in enumerate(block.columns), (a, i) in enumerate(block.columns)
        if !iszero(block.covariance[a, b])
            push!(I, i)
            push!(J, j)
            push!(V, block.covariance[a, b])
        end
    end
    return SparseArrays.sparse(I, J, V, k, k)
end

function _dual_cones(blocks::Vector{SupportBlock}, mom::Moments)
    Σ = _covariance_matrix(mom)
    return map(blocks) do block
        V = SparseArrays.dropzeros!(Σ[:, block.columns] * SparseArrays.sparse(transpose(block.W)))
        Wμ = block.W * mom.mean[block.columns] - block.h
        return _DualCone(V, Wμ, block.cones, _cone_rows(block))
    end
end

# With the slack rule `s(ξ) = S ξ`, `u = S μ`, the relaxation of `s(ξ) ≥ 0` on Ξ is
# `E[s(ξ) ξ₁] = u ≥ 0` and `E[s(ξ) W'ξ] = u (W'μ) + S Σ W'ᵀ ∈ K` for every support block, since
# `s(ξ) W'ξ ∈ K` on Ξ and K is a closed convex cone (Kuhn, Wiesemann and Georghiou, 2011,
# Proposition 3, for the nonnegative orthant). S is masked by the columns where the row is
# structurally nonzero, and Σ is block diagonal, so a piece of the cone with no masked column in
# its covariance block reduces to `u (W'μ)_piece ∈ K_piece`, implied by `u ≥ 0` and `W'μ ∈ K`.
function _add_dual_cone!(
    model::JuMP.Model,
    X,
    r::Int,
    ineq::_Inequality,
    μ::Vector{Float64},
    cones::Vector{_DualCone},
    cones_of::Vector{Vector{Int}},
)
    row, sign = ineq.row, ineq.sign
    S = Dict{Int,JuMP.VariableRef}()
    named = JuMP.set_string_names_on_creation(model)
    u = JuMP.@variable(model, lower_bound = 0.0, base_name = named ? "u[$r]" : "")
    mean = JuMP.AffExpr(0.0, u => 1.0)
    touched = Int[]
    for j in _structural_columns(row)
        S[j] = JuMP.@variable(model, base_name = named ? "S[$r,$j]" : "")
        # a ⋅ X_j + sign * S_j = b_j, that is, S_j = sign * (b_j - a ⋅ X_j).
        expr = _add_row_column!(JuMP.AffExpr(0.0), X, row, j, 1.0)
        JuMP.add_to_expression!(expr, sign, S[j])
        JuMP.@constraint(model, expr == row.b[j])
        iszero(μ[j]) || JuMP.add_to_expression!(mean, -μ[j], S[j])
        append!(touched, cones_of[j])
    end
    JuMP.@constraint(model, mean == 0)
    for b in unique!(sort!(touched))
        cone = cones[b]
        V = cone.V
        # The masked part `(S Σ W'ᵀ)_k` of the row `k` of the block.
        function masked(k)
            expr = JuMP.AffExpr(0.0)
            for p in SparseArrays.nzrange(V, k)
                s = get(S, SparseArrays.rowvals(V)[p], nothing)
                s === nothing || JuMP.add_to_expression!(expr, SparseArrays.nonzeros(V)[p], s)
            end
            return expr
        end
        for (piece, rows) in zip(cone.cones, cone.rows)
            if piece isa MOI.Nonnegatives
                for k in rows
                    expr = masked(k)
                    isempty(expr.terms) && continue
                    iszero(cone.Wμ[k]) || JuMP.add_to_expression!(expr, cone.Wμ[k], u)
                    JuMP.@constraint(model, expr >= 0)
                end
            else
                exprs = [masked(k) for k in rows]
                all(expr -> isempty(expr.terms), exprs) && continue
                for (expr, k) in zip(exprs, rows)
                    iszero(cone.Wμ[k]) || JuMP.add_to_expression!(expr, cone.Wμ[k], u)
                end
                JuMP.@constraint(model, exprs in JuMP.SecondOrderCone())
            end
        end
    end
    return
end

# The dual bound relies on stagewise independence: the laws whose covariance block correlates
# uncertainties of different stages, as `"a, b (stages 2, 3)"`.
function _correlated_stage_laws(form::CanonicalForm, mom::Moments)
    laws = String[]
    for block in mom.blocks
        stages = form.uncertainty_stages[block.columns.-1]
        correlated = any(
            stages[a] != stages[b] && !iszero(block.covariance[a, b]) for
            a in eachindex(stages), b in eachindex(stages)
        )
        if correlated
            names = join((form.uncertainty_names[j-1] for j in block.columns), ", ")
            push!(laws, "$names (stages $(join(unique(stages), ", ")))")
        end
    end
    return laws
end

function _warn_correlated_stages(form::CanonicalForm, mom::Moments)
    laws = _correlated_stage_laws(form, mom)
    isempty(laws) && return
    @warn(
        "The uncertainties of different stages are correlated in the law of " *
        "$(join(laws, "; ")). Stagewise independence fails, so the bound of DualLDR() is not " *
        "certified.",
    )
    return
end

function _build!(model::JuMP.Model, ::DualLDR, form::CanonicalForm, mom::Moments)
    _warn_correlated_stages(form, mom)
    info = InformationStructure(form)
    X = _rule_variables!(model, form, info)
    _add_equalities!(model, form, info, X)
    cones = _dual_cones(support_blocks(form), mom)
    # The cones whose rows involve each column of ξ.
    cones_of = [Int[] for _ in mom.mean]
    for (b, cone) in enumerate(cones), j in unique(SparseArrays.rowvals(cone.V))
        push!(cones_of[j], b)
    end
    for (r, ineq) in enumerate(_inequalities(form, info))
        if _is_deterministic(ineq.row)
            _add_deterministic!(model, X, ineq)
        else
            _add_dual_cone!(model, X, r, ineq, mom.mean, cones, cones_of)
        end
    end
    _set_objective!(model, form, mom, X)
    return X
end
