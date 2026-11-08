# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Whole-model checks of the assumptions of the dual bound, run by `optimize!`: they warn, since the
# primal rule and the sampled rule do not need them, and the inner solver has the last word. The
# last one warns of a choice of the information structure that a user may not expect.

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
# feasible and their objectives are the expected values of the rule, so they are unaffected:
# `optimize!` warns only when it solves a formulation with a bound. The law then lies in an affine
# subspace of Ξ, and the primal rule, robust over all of Ξ, is conservative but valid.
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

# Integer decisions observe only the constant (`sec:information` of `docs/math`), even when some
# uncertainties are revealed in stage 1, with them. The model is well defined, so it is a warning:
# under a continuous law the primal loses nothing (an integer-valued affine function of a
# continuous η is constant), but the bound of the dual is that of the problem in which the integer
# decisions are taken before any uncertainty is seen.
function _warn_integer_before_stage_one(form::CanonicalForm)
    any(form.integer) || return
    seen = _original_stages(form) .== 1
    any(seen) || return
    decisions = _name_list(form.decision_names[form.integer])
    uncertainties = _name_list(_original_names(form)[seen])
    @warn(
        "The integer decisions $decisions observe no uncertainty, not even those of stage 1 " *
        "($uncertainties): their rules are constants, as if they were taken before these " *
        "uncertainties are revealed, and the bound of DualLDR() is that of this problem. " *
        "Under a continuous law, the rules of PrimalLDR() lose nothing, since an integer " *
        "affine function of a continuous uncertainty is constant.",
    )
    return
end
