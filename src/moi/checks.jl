# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Whole-model checks of the assumptions of the dual bound, run by `optimize!`: they warn, since the
# primal rule and the sampled rule do not need them, and the inner solver has the last word.

# "a, b, c and 2 more", for at most `n` names.
function _name_list(names::Vector{String}, n::Int = 5)
    length(names) <= n && return join(names, ", ")
    return "$(join(names[1:n], ", ")) and $(length(names) - n) more"
end

# The decisions with a quadratic term, if `sense * P` is not positive semidefinite (up to a
# tolerance relative to its largest entry, so that a singular semidefinite `P` passes), else nothing.
function _nonconvex_decisions(form::CanonicalForm)
    sign = form.sense == MOI.MAX_SENSE ? -1.0 : 1.0
    if form.sense == MOI.FEASIBILITY_SENSE || iszero(SparseArrays.nnz(form.P))
        return nothing
    end
    P = SparseArrays.dropzeros(sign * (form.P + form.P') / 2)
    involved = unique!(sort(SparseArrays.rowvals(P)))
    isempty(involved) && return nothing
    tolerance = 1e-9 * maximum(abs, SparseArrays.nonzeros(P))
    shifted = P[involved, involved] + tolerance * LinearAlgebra.I
    factor = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(shifted); check = false)
    LinearAlgebra.issuccess(factor) && return nothing
    return involved
end

# The dual bound needs a convex objective (the dual bound theorem of `docs/math`); the inner
# problems of the other formulations are nonconvex too.
function _check_convex_objective(form::CanonicalForm)
    involved = _nonconvex_decisions(form)
    involved === nothing && return
    what, sign = form.sense == MOI.MAX_SENSE ? ("maximization", "-P") : ("minimization", "P")
    names = _name_list(form.decision_names[involved])
    @warn(
        "The quadratic objective of this $what is not convex: the matrix $sign of its terms in " *
        "the decisions $names is not positive semidefinite. The bound of DualLDR() is no " *
        "longer proven, and the inner problems are nonconvex.",
    )
    return
end

# The covariance blocks of `mom` that are singular, as `(block, rank)`: the coordinates of a block
# lie almost surely in an affine subspace, so that `M = E[ξ ξᵀ]` is not positive definite (a
# Schur complement on the constant column: `M ≻ 0` if and only if every block is). The test is on
# the correlation matrix, to not depend on the units of the uncertainties.
function _singular_blocks(mom::Moments; tolerance::Float64 = 1e-10)
    out = Tuple{CovarianceBlock,Int}[]
    for block in mom.blocks
        C = block.covariance
        d = LinearAlgebra.diag(C)
        if all(>(0), d)
            R = C ./ sqrt.(d) ./ sqrt.(d)'
            shifted = LinearAlgebra.Symmetric(R - tolerance * LinearAlgebra.I)
            LinearAlgebra.issuccess(LinearAlgebra.cholesky(shifted; check = false)) && continue
        else
            R = Matrix(C)
        end
        λ = LinearAlgebra.eigvals(LinearAlgebra.Symmetric(R))
        rank = count(>(tolerance * max(maximum(λ), 0.0)), λ)
        push!(out, (block, rank))
    end
    return out
end

# The singular blocks as `"a, b (rank 1 of 2)"`.
function _singular_laws(form::CanonicalForm, mom::Moments)
    return map(_singular_blocks(mom)) do (block, rank)
        names = _name_list([form.uncertainty_names[j-1] for j in block.columns])
        return "$names (rank $rank of $(length(block.columns)))"
    end
end

# The second-moment matrix of the uncertainties must be positive definite for the dual bound
# (the support assumption of `docs/math`). Without it, the primal and sampled rules are still
# feasible and their objectives are the expected values of the rule, so they are unaffected.
function _warn_singular_moments(form::CanonicalForm, mom::Moments)
    laws = _singular_laws(form, mom)
    isempty(laws) && return
    @warn(
        "The second-moment matrix of the uncertainties is singular: the covariance of " *
        "$(join(laws, "; ")) is not positive definite, so they lie almost surely in an affine " *
        "subspace. A discrete law needs more atoms than its dimension (than its number of " *
        "pieces, for a lifted uncertainty). The values of PrimalLDR() and SampledLDR() are " *
        "unaffected, but the bound of DualLDR() is not certified.",
    )
    return
end
