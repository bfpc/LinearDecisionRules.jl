# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The dense primal and dual linear decision rules of the previous version (`solve_primal.jl`,
# `solve_dual.jl`, and `spike/core/SpikeCore.jl`), written against a `CanonicalForm` and the dense
# second moment. Used only by the tests, as an oracle for the structured formulations.
#
#   opt  tr(X'PXM) + tr(C'XM) + r
#   s.t. E X = F,  S = H - G X,  where G X ξ ≤ H ξ holds all inequalities and finite bounds,
#        primal: Λ ∈ K*, Λ W = S, Λ h ≥ 0, with Ξ = {ξ : W ξ - h ∈ K} and rows ±e₁ for ξ₁ = 1;
#        dual:   S M W'ᵀ ∈ K, with W' = [e₁'; W - h e₁'] without the rows ±e₁,
# row by row of Λ and S, where K is the nonnegative orthant of the linear rows times one
# second-order cone (self-dual) per conic row.
#
# The row e₁' of W' (that is, E[s(ξ)] ≥ 0) is the fix of the spike audit's bug 1: the previous
# version had no inequality left in the dual when no uncertainty was bounded.

module DenseOracle

using JuMP

import LinearAlgebra
import LinearDecisionRules
import LinearDecisionRules: CanonicalForm, InformationStructure, Moments, second_moment

# The box of a law is the definition of Ξ, not part of the formulations, so it is shared.
const _box = LinearDecisionRules._support_box

struct DenseData
    mask::BitMatrix
    E::Matrix{Float64}
    F::Matrix{Float64}
    G::Matrix{Float64}
    H::Matrix{Float64}
    W::Matrix{Float64}
    h::Vector{Float64}
    # The rows of W in each second-order cone; the others are linear.
    cones::Vector{UnitRange{Int}}
    M::Matrix{Float64}
    r::Float64
end

function DenseData(form::CanonicalForm, mom::Moments)
    n, m = length(form.decisions), length(form.uncertainties)
    k = 1 + m
    info = InformationStructure(form)
    mask = BitMatrix([j <= info.num_visible[i] for i in 1:n, j in 1:k])
    e(i) = [l == i ? 1.0 : 0.0 for l in 1:n]'
    e1 = [1.0 zeros(1, m)]
    G = [Matrix(form.A_u); -Matrix(form.A_l)]
    H = [Matrix(form.B_u); -Matrix(form.B_l)]
    for i in 1:n
        isfinite(form.upper[i]) && (G = [G; e(i)]; H = [H; form.upper[i] * e1])
        isfinite(form.lower[i]) && (G = [G; -e(i)]; H = [H; -form.lower[i] * e1])
    end
    lb, ub = fill(-Inf, m), fill(Inf, m)
    for law in form.laws
        lo, hi = _box(law.distribution)
        lb[law.columns.-1], ub[law.columns.-1] = lo, hi
    end
    I = Matrix{Float64}(LinearAlgebra.I, m, m)
    Wη = [I[isfinite.(lb), :]; -I[isfinite.(ub), :]; Matrix(form.W_valid)]
    hη = [lb[isfinite.(lb)]; -ub[isfinite.(ub)]; form.h_valid]
    W = [1 zeros(1, m); -1 zeros(1, m); zeros(length(hη)) Wη]
    h = [1; -1; hη]
    cones = UnitRange{Int}[]
    # The rows of the support of each law (`support_constraints`), over all the columns of ξ.
    for law in form.laws
        rows = LinearDecisionRules.support_constraints(law.distribution)
        offset = 0
        for cone in rows.cones
            k = offset .+ (1:MOI.dimension(cone))
            offset += MOI.dimension(cone)
            Wk = zeros(length(k), 1 + m)
            Wk[:, law.columns] = rows.W[k, :]
            cone isa MOI.SecondOrderCone && push!(cones, length(h) .+ (1:length(k)))
            W = [W; Wk]
            h = [h; rows.h[k]]
        end
    end
    for row in form.conic_rows
        push!(cones, length(h) .+ (1:length(row.h)))
        W = [W; zeros(length(row.h)) Matrix(row.W)]
        h = [h; row.h]
    end
    M = second_moment(mom)
    r = form.f + form.d' * M[2:end, 1] + sum(Matrix(form.Q) .* M[2:end, 2:end]; init = 0.0)
    return DenseData(mask, Matrix(form.A_e), Matrix(form.B_e), G, H, W, h, cones, M, r)
end

function _skeleton(form::CanonicalForm, data::DenseData)
    model = Model()
    n, k = size(data.mask)
    X = [
        data.mask[i, j] ? AffExpr(0.0, @variable(model) => 1.0) : AffExpr(0.0) for i in 1:n,
        j in 1:k
    ]
    for i in 1:n
        form.binary[i] && set_binary(first(keys(X[i, 1].terms)))
        form.integer[i] && !form.binary[i] && set_integer(first(keys(X[i, 1].terms)))
    end
    @constraint(model, data.E * X .== data.F)
    S = data.H - data.G * X
    if form.sense != MOI.FEASIBILITY_SENSE
        P, C, M = Matrix(form.P), Matrix(form.C), data.M
        obj = QuadExpr(AffExpr(data.r))
        CM = C * M
        for i in 1:n, j in 1:k
            add_to_expression!(obj, CM[i, j], X[i, j])
        end
        for I in findall(!iszero, P), J in findall(!iszero, M)
            add_to_expression!(obj, P[I] * M[J] * X[I[1], J[1]], X[I[2], J[2]])
        end
        set_objective(model, form.sense, obj)
    end
    return model, X, S
end

"""
    build_primal(form, mom) -> (model, X)

The dense primal LDR of the previous version, without an optimizer.
"""
function build_primal(form::CanonicalForm, mom::Moments)
    data = DenseData(form, mom)
    model, X, S = _skeleton(form, data)
    @variable(model, Λ[1:size(S, 1), 1:length(data.h)])
    linear = setdiff(1:length(data.h), data.cones...)
    set_lower_bound.(Λ[:, linear], 0.0)
    for rows in data.cones, i in axes(Λ, 1)
        @constraint(model, Λ[i, rows] in SecondOrderCone())
    end
    @constraint(model, Λ * data.W .== S)
    @constraint(model, Λ * data.h .>= 0)
    return model, X
end

"""
    build_dual(form, mom) -> (model, X)

The dense dual LDR of the previous version, with the row `E[s(ξ)] ≥ 0`, without an optimizer.
"""
function build_dual(form::CanonicalForm, mom::Moments)
    data = DenseData(form, mom)
    model, X, S = _skeleton(form, data)
    m = size(data.W, 2) - 1
    W2 = data.W .- data.h * [1 zeros(1, m)]
    W′ = [1 zeros(1, m); W2[3:end, :]]
    # Column 1 of `S M W'ᵀ` is the row e₁'; the rows of W are shifted by one after dropping ±e₁.
    SMW = S * (data.M * W′')
    cones = [rows .- 1 for rows in data.cones]
    linear = setdiff(1:size(W′, 1), cones...)
    @constraint(model, SMW[:, linear] .>= 0)
    for rows in cones, i in axes(SMW, 1)
        @constraint(model, SMW[i, rows] in SecondOrderCone())
    end
    return model, X
end

"""
    solve(builder, form, mom, optimizer) -> NamedTuple

Build with `builder` (`build_primal` or `build_dual`), solve, and return the statuses, the
objective value (`NaN` without a feasible point) and the rule `X`.
"""
function solve(builder, form::CanonicalForm, mom::Moments, optimizer)
    model, X = builder(form, mom)
    set_optimizer(model, optimizer)
    set_silent(model)
    optimize!(model)
    feasible = primal_status(model) == MOI.FEASIBLE_POINT
    objective = !feasible ? NaN : form.sense == MOI.FEASIBILITY_SENSE ? 0.0 : objective_value(model)
    return (
        termination_status = termination_status(model),
        primal_status = primal_status(model),
        objective_value = objective,
        X = feasible ? value.(X) : fill(NaN, size(X)),
        model = model,
    )
end

end # module DenseOracle
