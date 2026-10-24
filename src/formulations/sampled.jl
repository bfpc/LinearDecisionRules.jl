# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The outcomes ξ_s = (1, η_s) as the columns of a matrix: one draw of each law per scenario, in
# the order of the declarations.
function _sample_outcomes(form::CanonicalForm, f::SampledLDR)
    rng = Random.Xoshiro(f.seed)
    Ξ = zeros(1 + length(form.uncertainties), f.num_scenarios)
    Ξ[1, :] .= 1.0
    for s in axes(Ξ, 2), law in form.laws
        Ξ[law.columns, s] .= rand(rng, law.distribution)
    end
    return Ξ
end

# The empirical moments `μ̂ = Σ_s ξ_s / N` and `M̂ = Σ_s ξ_s ξ_sᵀ / N`, with one covariance block:
# the empirical covariance of separate laws is not zero.
function _empirical_moments(Ξ::Matrix{Float64})
    N = size(Ξ, 2)
    μ = vec(sum(Ξ; dims = 2)) ./ N
    μ[1] = 1.0
    if size(Ξ, 1) == 1
        return Moments(μ, CovarianceBlock[])
    end
    D = Ξ[2:end, :] .- μ[2:end]
    return Moments(μ, [CovarianceBlock(collect(2:size(Ξ, 1)), D * D' ./ N)])
end

function _build!(model::JuMP.Model, f::SampledLDR, form::CanonicalForm, ::Moments)
    info = InformationStructure(form)
    X = _rule_variables!(model, form, info)
    # Without random recourse, coefficient matching is exact and needs no scenario; one equality
    # per scenario would leave too few degrees of freedom to interior-point solvers.
    _add_equalities!(model, form, info, X)
    Ξ = _sample_outcomes(form, f)
    for ineq in _inequalities(form, info)
        if _is_deterministic(ineq.row)
            _add_deterministic!(model, X, ineq)
            continue
        end
        row, sign = ineq.row, ineq.sign
        for s in axes(Ξ, 2)
            # sign * a ⋅ X ξ_s ≤ sign * b ⋅ ξ_s
            expr = JuMP.AffExpr(0.0)
            for (i, a) in zip(row.decisions, row.coefficients), j in eachindex(X[i])
                iszero(Ξ[j, s]) || JuMP.add_to_expression!(expr, sign * a * Ξ[j, s], X[i][j])
            end
            rhs = 0.0
            for (j, b) in zip(SparseArrays.nonzeroinds(row.b), SparseArrays.nonzeros(row.b))
                rhs += b * Ξ[j, s]
            end
            JuMP.@constraint(model, expr <= sign * rhs)
        end
    end
    _set_objective!(model, form, _empirical_moments(Ξ), X)
    return X
end
