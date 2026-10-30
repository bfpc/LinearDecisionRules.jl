# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Whole-model checks of the assumptions of the dual bound, run by `optimize!`: they warn, since the
# primal rule and the sampled rule do not need them, and the inner solver has the last word.

# The dual bound needs a convex objective (`thm:dual-bound` of `docs/math`); the inner
# problems of the other formulations are nonconvex too.
function _check_convex_objective(form::CanonicalForm)
    involved = _nonconvex_decisions(form)
    involved === nothing && return
    what, sign = form.sense == MOI.MAX_SENSE ? ("maximization", "-P") : ("minimization", "P")
    names = _name_list(form.decision_names[involved])
    @warn(
        "The quadratic objective of this $what is not convex: the matrix $sign of its terms in " *
        "the decisions $names is not positive semidefinite. The bound of DualLDR() is not " *
        "certified, and the inner problems are nonconvex.",
    )
    return
end

# The second-moment matrix of the uncertainties must be positive definite for the dual bound
# (`ass:support` of `docs/math`). Without it, the primal and sampled rules are still
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
