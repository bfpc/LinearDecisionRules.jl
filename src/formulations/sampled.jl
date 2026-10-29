# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    SampledLDR(num_scenarios::Integer; seed::Integer = 1234)

The sampled linear decision rule: every inequality holds on `num_scenarios` outcomes drawn from the
laws, and the objective uses the empirical moments of the outcomes.

Outcomes are drawn with `Random.Xoshiro(seed)`, one draw per declaration per scenario. Equality rows
hold for every outcome, as in [`PrimalLDR`](@ref). The rule is feasible on the drawn outcomes only.

Results are read with the same number of scenarios and seed: after solving `SampledLDR(100)`,
`objective_value(model, SampledLDR(10))` is an error.
"""
struct SampledLDR <: AbstractFormulation
    num_scenarios::Int
    seed::Int

    function SampledLDR(num_scenarios::Integer, seed::Integer)
        if num_scenarios < 1
            throw(ArgumentError("The number of scenarios must be positive; got $num_scenarios."))
        end
        return new(num_scenarios, seed)
    end
end

SampledLDR(num_scenarios::Integer; seed::Integer = 1234) = SampledLDR(num_scenarios, seed)

function Base.show(io::IO, f::SampledLDR)
    print(io, "SampledLDR(", f.num_scenarios, "; seed = ", f.seed, ")")
    return
end

function capabilities(::SampledLDR)
    return FormulationCapabilities(MOI.UNKNOWN_RESULT_STATUS, false)
end

# The outcomes ξ_s = (1, η_s) as the columns of a matrix: one draw of each law per scenario, in
# the order of the declarations.
function _sample_outcomes(form::CanonicalForm, f::SampledLDR)
    rng = Random.Xoshiro(f.seed)
    outcomes = zeros(1 + length(form.uncertainties), f.num_scenarios)
    outcomes[1, :] .= 1.0
    for s in axes(outcomes, 2), law in form.laws
        outcomes[law.columns, s] .= rand(rng, law.distribution)
    end
    return outcomes
end

# The empirical moments `μ̂ = Σ_s ξ_s / N` and `M̂ = Σ_s ξ_s ξ_sᵀ / N`, with one covariance block:
# the empirical covariance of separate laws is not zero.
function _empirical_moments(outcomes::Matrix{Float64})
    N = size(outcomes, 2)
    μ = vec(sum(outcomes; dims = 2)) ./ N
    μ[1] = 1.0
    if size(outcomes, 1) == 1
        return Moments(μ, CovarianceBlock[])
    end
    D = outcomes[2:end, :] .- μ[2:end]
    return Moments(μ, [CovarianceBlock(collect(2:size(outcomes, 1)), D * D' ./ N)])
end

function _build!(model::JuMP.Model, f::SampledLDR, form::CanonicalForm, ::Moments)
    observed = InformationStructure(form)
    X = _rule_variables!(model, form, observed)
    # Without random recourse, coefficient matching is exact and needs no scenario; one equality
    # per scenario would leave too few degrees of freedom to interior-point solvers.
    _add_equalities!(model, form, observed, X)
    outcomes = _sample_outcomes(form, f)
    for ineq in _inequalities(form, observed)
        if _is_deterministic(ineq.row)
            _add_deterministic!(model, X, ineq)
            continue
        end
        row, sign = ineq.row, ineq.sign
        for s in axes(outcomes, 2)
            # sign * a ⋅ X ξ_s ≤ sign * b ⋅ ξ_s
            expr = JuMP.AffExpr(0.0)
            for (i, a) in zip(row.decisions, row.coefficients), j in eachindex(X[i])
                ξj = outcomes[j, s]
                iszero(ξj) || JuMP.add_to_expression!(expr, sign * a * ξj, X[i][j])
            end
            rhs = 0.0
            for (j, b) in zip(SparseArrays.nonzeroinds(row.b), SparseArrays.nonzeros(row.b))
                rhs += b * outcomes[j, s]
            end
            JuMP.@constraint(model, expr <= sign * rhs)
        end
    end
    _set_objective!(model, form, _empirical_moments(outcomes), X)
    return X
end
