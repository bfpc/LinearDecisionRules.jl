# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The checks of a canonical form once read: valid rows at the mean, observable equalities.

# A valid inequality holds on the support, so it holds at the mean: a cheap check of validity,
# which also shows that every support block contains a point (the mean). The mean is that of the
# conditioned law, which is not known before it is estimated for a `ConditionedDistribution`: the
# rows of such a law are not checked, nor the rows tagged `Truncate()`.
function _check_valid_rows_at_mean(src, form::CanonicalForm)
    μ = fill(NaN, length(form.uncertainties))
    for law in form.laws
        law.distribution isa ConditionedDistribution && continue
        μ[law.columns.-1] .= _law_mean(law.distribution)
    end
    Wt = SparseArrays.sparse(transpose(form.W_valid))
    for r in eachindex(form.h_valid)
        form.conditioning[r] && continue
        h = form.h_valid[r]
        value, scale = 0.0, abs(h)
        for k in SparseArrays.nzrange(Wt, r)
            term = SparseArrays.nonzeros(Wt)[k] * μ[SparseArrays.rowvals(Wt)[k]]
            value += term
            scale = max(scale, abs(term))
        end
        if isfinite(value) && value < h - 1e-9 * max(1.0, scale)
            error(
                "The constraint $(_constraint_name(src, form.valid_constraints[r])) is tagged " *
                "`ValidInequality()`, but the mean of the uncertainty violates it, so it does " *
                "not hold on the support of the distribution. Use `Truncate()` to condition " *
                "the law.",
            )
        end
    end
    for row in form.conic_rows
        row.conditioning && continue
        # The sparse product reads only the columns of the row, so the NaN of others do not show.
        Wμ = row.W * μ
        all(isfinite, Wμ) || continue
        scale = max(1.0, maximum(abs, Wμ; init = 0.0), maximum(abs, row.h; init = 0.0))
        if _cone_violation(row.cone, Wμ - row.h) > 1e-9 * scale
            error(
                "The constraint $(_constraint_name(src, row.constraint)) is tagged " *
                "`ValidInequality()`, but the mean of the uncertainty violates it, so it does " *
                "not hold on the support of the distribution. Use `Truncate()` to condition " *
                "the law.",
            )
        end
    end
    return
end

# An equality row holds for every ξ only if, for each uncertainty in it, some decision of the row
# observes that uncertainty.
function _check_observable_equalities(src::MOI.ModelLike, form::CanonicalForm)
    isempty(form.e_constraints) && return
    info = InformationStructure(form)
    At = SparseArrays.sparse(transpose(form.A_e))
    Bt = SparseArrays.sparse(transpose(form.B_e))
    for r in eachindex(form.e_constraints)
        decisions = view(SparseArrays.rowvals(At), SparseArrays.nzrange(At, r))
        visible = maximum((info.num_visible[i] for i in decisions); init = 1)
        for k in SparseArrays.nzrange(Bt, r)
            j = SparseArrays.rowvals(Bt)[k]
            j <= visible && continue
            name = _constraint_name(src, form.e_constraints[r])
            integer = any(i -> form.integer[i], decisions) ? " Integer decisions observe none." : ""
            error(
                "The equality constraint $name has the uncertainty " *
                "$(form.uncertainty_names[j-1]) (Stage($(form.uncertainty_stages[j-1]))), which " *
                "none of its decisions observes; no decision rule satisfies it for every " *
                "outcome.$integer",
            )
        end
    end
    return
end
