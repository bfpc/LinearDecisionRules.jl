# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The rescaling of the uncertainties that `optimize!` applies before the formulations, with
# `RescaleUncertainties()`. Each column of ξ = (1, η) is written `η_j = c_j + s_j ζ_j`: `c_j` is
# the center and `s_j` the half-width of the box of the law of `η_j`, so that `ζ_j ∈ [-1, 1]`, or,
# for a law without a bounded box, its mean and standard deviation. With `ξ̃ = (1, ζ)`,
#
#     ξ = T ξ̃,   T = [1 0; c S],   S = Diagonal(s),
#
# and the form in `ξ̃` has `B T` and `C T` for `B_*` and `C`, `S Q S`, `S (d + 2 Q c)` and
# `f + cᵀ Q c + dᵀ c` for the quadratic, linear and constant terms in η, `W S` and `h - W c` for
# the rows of Ξ (then divided by their largest coefficient, see `_rescale_rows`), the constants of
# the parameters shifted like the first column of `B T`, and the laws of ζ. The decisions, the
# values and the bounds are the same (`prop:affine-invariance` of `docs/math`, where `T` is `G`);
# the rules `X̃` in `ξ̃` are `X = X̃ T⁻¹` in ξ. The moments of a uniform law on a box of width 10⁴
# have entries of order 10⁸ against 1 for the constant column, which is what this avoids.

struct _Rescaling
    center::Vector{Float64}
    scale::Vector{Float64}
end

# The law of `ζ = (η - center) ./ scale` for a law of η: what the formulations read from a law (its
# box, the rows of its support, the trait of the dual bound and its draws), moved to ζ.
struct _RescaledLaw{F<:Distributions.VariateForm,S<:Distributions.ValueSupport,D} <:
       Distributions.Distribution{F,S}
    law::D
    center::Vector{Float64}
    scale::Vector{Float64}
end

function _RescaledLaw(
    law::Distributions.Distribution{F,S},
    center::Vector{Float64},
    scale::Vector{Float64},
) where {F,S}
    return _RescaledLaw{F,S,typeof(law)}(law, center, scale)
end

Base.length(d::_RescaledLaw{Distributions.Multivariate}) = length(d.center)

function _rescale_bounds(d::_RescaledLaw, (lo, hi))
    return (lo .- d.center) ./ d.scale, (hi .- d.center) ./ d.scale
end

_support_box(d::_RescaledLaw) = _rescale_bounds(d, _support_box(d.law))

_box_rows(d::_RescaledLaw, lifted_box::Bool) = _rescale_bounds(d, _box_rows(d.law, lifted_box))

function support_constraints(d::_RescaledLaw)
    rows = support_constraints(d.law)
    W, h = _rescale_rows(rows.W, rows.h, d.center, d.scale, rows.cones)
    return (W = Matrix(W), h = h, cones = rows.cones)
end

function _linear_conditional_expectations(d::_RescaledLaw, stages::AbstractVector)
    return _linear_conditional_expectations(d.law, stages)
end

function Base.rand(rng::Random.AbstractRNG, d::_RescaledLaw{Distributions.Univariate})
    return (rand(rng, d.law) - d.center[1]) / d.scale[1]
end

function Base.rand(rng::Random.AbstractRNG, d::_RescaledLaw{Distributions.Multivariate})
    return (rand(rng, d.law) .- d.center) ./ d.scale
end

# How the pieces of an `AxisPiecewise` lifting are rescaled, for the study of
# `benchmark/scaling/`, which reads it through `benchmark/internals.jl`; the choice it makes stays,
# and the others go:
#
#  * `:per_piece`, the default: each piece on its own box, `[η̲, b₁]` and `[0, Δᵢ]`, so that a piece
#    is -1 below its breakpoint;
#  * `:before_lifting`: as if the uncertainty were rescaled to `[-1, 1]` and then lifted, so that
#    every piece has the half-width `s` of the uncertainty as its scale, the first piece its center
#    and the others 0: the pieces after the first are 0 below their breakpoint;
#  * `:hybrid`: the first piece on its own box, and the others in `[0, 1]` (center 0, scale Δᵢ).
#
# Every choice is an affine map of each column alone, so the values do not depend on it
# (`prop:affine-invariance`).
const _LIFTED_RESCALING = Ref(:per_piece)

# The center and the scale of each uncertainty of `form`: those of the box of its law if it is
# bounded and not a point, else its mean and standard deviation, else no change. The blocks of
# `mom` are those of `form.laws`, in their order (see `moments`). The pieces of an `AxisPiecewise`
# lifting follow `_LIFTED_RESCALING`.
function _rescaling(form::CanonicalForm, mom::Moments)
    m = length(form.uncertainties)
    center, scale = zeros(m), ones(m)
    for (law, block) in zip(form.laws, mom.blocks)
        lo, hi = _support_box(law.distribution)
        for (k, j) in enumerate(law.columns)
            if isfinite(lo[k]) && isfinite(hi[k]) && hi[k] > lo[k]
                center[j-1], scale[j-1] = (lo[k] + hi[k]) / 2, (hi[k] - lo[k]) / 2
            elseif block.covariance[k, k] > 0 && isfinite(block.covariance[k, k])
                center[j-1], scale[j-1] = mom.mean[j], sqrt(block.covariance[k, k])
            end
        end
        mode = _LIFTED_RESCALING[]
        if law.distribution isa LiftedDistribution{<:Any,AxisPiecewise} && mode != :per_piece
            _rescale_pieces!(center, scale, law.columns .- 1, lo, hi, mode)
        end
    end
    return _Rescaling(center, scale)
end

# The pieces of one uncertainty, the columns `η` of η, with the boxes `lo` and `hi`.
function _rescale_pieces!(center, scale, η, lo, hi, mode::Symbol)
    all(isfinite, lo) && all(isfinite, hi) || return
    if mode == :before_lifting
        # The box of the uncertainty is [lo₁, hi₁ + Σᵢ₌₂ hiᵢ], the pieces after the first in [0, Δᵢ].
        s = (hi[1] - lo[1] + sum(hi[2:end])) / 2
        center[η[1]] = lo[1] + s
        center[η[2:end]] .= 0.0
        scale[η] .= s
    elseif mode == :hybrid
        center[η[2:end]] .= 0.0
        scale[η[2:end]] .= hi[2:end]
    else
        throw(ArgumentError("unknown rescaling of the pieces: $mode"))
    end
    return
end

# `T = [1 0; c S]`, with `1 + m` rows and columns.
function _rescaling_matrix(r::_Rescaling)
    m = length(r.center)
    I = [1; 2:m+1; 2:m+1]
    J = [1; ones(Int, m); 2:m+1]
    V = [1.0; r.center; r.scale]
    return SparseArrays.sparse(I, J, V, m + 1, m + 1)
end

# The rows `W S ζ - (h - W c) ∈ K` of the rows `W η - h ∈ K`, divided by their largest
# coefficient: each row of `MOI.Nonnegatives` by its own, the rows of another cone by one factor
# (`cones` lists the cones of consecutive rows; `nothing` means rows of `MOI.Nonnegatives`). A
# positive factor leaves the set unchanged, and the rows of the lifted supports, scaled by products
# of lengths of pieces (see `_hull_rows`), would keep coefficients of order 10⁶ for uncertainties of
# order 10⁴. A right-hand side within rounding of zero, such as that of the first row of the hull
# of an `AxisPiecewise` lifting, is zero.
function _rescale_rows(
    W::AbstractMatrix,
    h::Vector{Float64},
    center::Vector{Float64},
    scale::Vector{Float64},
    cones = nothing,
)
    Ws = SparseArrays.sparse(W) * LinearAlgebra.Diagonal(scale)
    hs = h - W * center
    size = abs.(h) + abs.(W) * abs.(center)
    hs[abs.(hs).<=16*eps()*size] .= 0.0
    factor = [maximum(abs, view(Ws, i, :); init = 0.0) for i in axes(Ws, 1)]
    offset = 0
    for cone in something(cones, ())
        rows = offset .+ (1:MOI.dimension(cone))
        offset += MOI.dimension(cone)
        if !(cone isa MOI.Nonnegatives)
            factor[rows] .= maximum(factor[rows]; init = 0.0)
        end
    end
    factor[iszero.(factor)] .= 1.0
    return LinearAlgebra.Diagonal(1 ./ factor) * Ws, hs ./ factor
end

function _rescale(form::CanonicalForm, r::_Rescaling)
    c, s = r.center, r.scale
    T = _rescaling_matrix(r)
    S = LinearAlgebra.Diagonal(s)
    shift(B) = B[:, 2:end] * c
    Qc = form.Q * c
    Δf = LinearAlgebra.dot(c, Qc) + LinearAlgebra.dot(form.d, c)
    p = form.parameters
    parameters = ParameterTerms(
        p.parameters,
        p.values,
        p.b_e + shift(form.B_e),
        p.Bp_e,
        p.b_u + shift(form.B_u),
        p.Bp_u,
        p.b_l + shift(form.B_l),
        p.Bp_l,
        p.f + Δf,
        p.fp,
    )
    laws = map(form.laws) do law
        η = law.columns .- 1
        return UncertaintyBlock(_RescaledLaw(law.distribution, c[η], s[η]), law.columns, law.stages)
    end
    conic_rows = map(form.conic_rows) do row
        W, h = _rescale_rows(row.W, row.h, c, s, (row.cone,))
        return ConicRow(_SparseMatrix(W), h, row.cone, row.constraint, row.conditioning)
    end
    W_valid, h_valid = _rescale_rows(form.W_valid, form.h_valid, c, s)
    return _replace(
        form;
        laws,
        B_e = _SparseMatrix(form.B_e * T),
        B_u = _SparseMatrix(form.B_u * T),
        B_l = _SparseMatrix(form.B_l * T),
        W_valid = _SparseMatrix(W_valid),
        h_valid,
        conic_rows,
        C = _SparseMatrix(form.C * T),
        Q = _SparseMatrix(S * form.Q * S),
        d = s .* (form.d + 2 * Qc),
        f = form.f + Δf,
        parameters,
    )
end

function _rescale(mom::Moments, r::_Rescaling)
    mean = copy(mom.mean)
    mean[2:end] = (mean[2:end] - r.center) ./ r.scale
    blocks = map(mom.blocks) do block
        s = r.scale[block.columns.-1]
        return CovarianceBlock(block.columns, block.covariance ./ (s * s'), block.num_samples)
    end
    return Moments(mean, blocks)
end

# The result with its rules in ξ: `X = X̃ T⁻¹`, that is, the columns of η divided by the scales,
# and their centers taken out of the constant column.
function _unscale(result::FormulationResult, r::_Rescaling)
    X̃ = result.X
    size(X̃, 2) == 1 + length(r.center) || return result
    X = similar(X̃)
    X[:, 2:end] = X̃[:, 2:end] ./ r.scale'
    X[:, 1] = X̃[:, 1] - X[:, 2:end] * r.center
    return FormulationResult(
        result.formulation,
        result.termination_status,
        result.primal_status,
        result.raw_status,
        result.objective_value,
        result.objective_bound,
        result.certified,
        result.solve_time,
        result.build_time,
        result.results_time,
        X,
        result.model,
    )
end
