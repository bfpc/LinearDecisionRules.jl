# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The laws conditioned by the rows tagged `Truncate()` (see `canonicalize`): a univariate law whose
# rows involve it alone is truncated exactly, and its rows are dropped; the other laws coupled by
# such rows are merged, by union–find, into one `ConditionedDistribution`, in the place of the
# first one.
function _condition(src::MOI.ModelLike, form::CanonicalForm)
    any(form.conditioning) || return form
    law_of = zeros(Int, length(form.uncertainties))
    for (l, law) in enumerate(form.laws), j in law.columns
        law_of[j-1] = l
    end
    Wt = SparseArrays.sparse(transpose(form.W_valid))
    rows = findall(form.conditioning)
    # The η columns of each conditioning row, and the laws that they belong to.
    columns_of = Dict(r => SparseArrays.rowvals(Wt)[SparseArrays.nzrange(Wt, r)] for r in rows)
    laws_of = Dict(r => unique(law_of[columns_of[r]]) for r in rows)
    laws = copy(form.laws)
    dropped = falses(length(form.h_valid))
    for (l, law) in enumerate(form.laws)
        mine = filter(r -> l in laws_of[r], rows)
        if isempty(mine) || !all(r -> length(laws_of[r]) == 1, mine)
            continue
        end
        d = law.distribution
        d isa Distributions.UnivariateDistribution || continue
        j = only(law.columns) - 1
        (lo,), (hi,) = _support_box(d)
        for r in mine
            # `a η ≥ h`
            a, h = Wt[j, r], form.h_valid[r]
            if a > 0
                lo = max(lo, h / a)
            else
                hi = min(hi, h / a)
            end
        end
        names = _row_names(src, form, mine)
        truncated = _truncated_law(d, lo, hi, form.uncertainty_names[j], names)
        truncated === nothing && continue
        laws[l] = UncertaintyBlock(truncated, law.columns, law.stages)
        dropped[mine] .= true
    end
    parent = collect(eachindex(laws))
    for r in rows
        dropped[r] && continue
        a = _find_root!(parent, first(laws_of[r]))
        for l in laws_of[r]
            parent[_find_root!(parent, l)] = a
        end
    end
    group_rows = Dict{Int,Vector{Int}}()
    for r in rows
        dropped[r] || push!(get!(group_rows, _find_root!(parent, first(laws_of[r])), Int[]), r)
    end
    merged = UncertaintyBlock[]
    for l in eachindex(laws)
        root = _find_root!(parent, l)
        if !haskey(group_rows, root)
            push!(merged, laws[l])
        elseif l == minimum(k for k in eachindex(laws) if _find_root!(parent, k) == root)
            group = filter(k -> _find_root!(parent, k) == root, eachindex(laws))
            push!(merged, _conditioned_block(src, form, laws[group], group_rows[root], Wt))
        end
    end
    keep = .!dropped
    return _replace(
        form;
        laws = merged,
        W_valid = form.W_valid[keep, :],
        h_valid = form.h_valid[keep],
        valid_constraints = form.valid_constraints[keep],
        conditioning = form.conditioning[keep],
    )
end

# The constraints of the rows `rows` of `W_valid`, without repeats (an interval gives two rows).
function _row_names(src, form::CanonicalForm, rows)
    return unique!([_constraint_name(src, form.valid_constraints[r]) for r in rows])
end

# The law of the declarations `laws` conditioned on the rows `rows` of `W_valid`. A law that is
# already a `ConditionedDistribution` (declared so) gives its laws and its rows.
function _conditioned_block(src, form::CanonicalForm, laws, rows, Wt)
    components = Distributions.Distribution[]
    columns, stages = Int[], Int[]
    W, h, names = Matrix{Float64}[], Float64[], String[]
    for law in laws
        d = law.distribution
        k = length(columns)
        append!(columns, law.columns)
        append!(stages, law.stages)
        if d isa ConditionedDistribution
            append!(components, d.laws)
            # Its rows, over its own columns.
            block = zeros(length(d.h), length(form.uncertainties))
            block[:, law.columns.-1] = d.W
            push!(W, block)
            append!(h, d.h)
            append!(names, d.names)
        else
            push!(components, d)
        end
    end
    for r in rows
        row = zeros(1, length(form.uncertainties))
        for p in SparseArrays.nzrange(Wt, r)
            row[SparseArrays.rowvals(Wt)[p]] = SparseArrays.nonzeros(Wt)[p]
        end
        push!(W, row)
        push!(h, form.h_valid[r])
        push!(names, _constraint_name(src, form.valid_constraints[r]))
    end
    Wη = reduce(vcat, W)[:, columns.-1]
    d = ConditionedDistribution(components, Wη, h; names)
    return UncertaintyBlock(d, columns, stages)
end

# The law `d` conditioned on `lo ≤ η ≤ hi`, exactly: `Distributions.truncated` for a continuous law,
# or a `DiscreteNonParametric` law on the atoms in `[lo, hi]` for a discrete law with a finite
# support. It is `nothing` for other laws, whose truncation is then estimated by rejection
# sampling. A truncation of probability zero is an error that names the constraints.
function _truncated_law(d::Distributions.UnivariateDistribution, lo, hi, name, constraints)
    function empty_error()
        return error(
            "The constraints $(join(constraints, ", ")) tagged `Truncate()` condition the law of " *
            "the uncertainty $name on [$lo, $hi], whose probability is zero.",
        )
    end
    if d isa Distributions.ContinuousUnivariateDistribution
        if !(lo < hi) || !(Distributions.cdf(d, hi) - Distributions.cdf(d, lo) > 0)
            empty_error()
        end
        return Distributions.truncated(d, lo, hi)
    elseif d isa Distributions.DiscreteUnivariateDistribution
        # `support` throws for the laws with an infinite support, such as `Poisson`.
        (dlo,), (dhi,) = _support_box(d)
        isfinite(dlo) && isfinite(dhi) || return nothing
        atoms = try
            Distributions.support(d)
        catch err
            err isa MethodError || rethrow()
            nothing
        end
        atoms isa AbstractVector || return nothing
        # The tolerance of the rejection, so that both treat the atoms on the boundary alike.
        inside(x) = lo - 1e-9 * (1 + abs(lo)) <= x <= hi + 1e-9 * (1 + abs(hi))
        x = Float64[a for a in atoms if inside(a) && Distributions.pdf(d, a) > 0]
        isempty(x) && empty_error()
        p = [Distributions.pdf(d, a) for a in x]
        return Distributions.DiscreteNonParametric(x, p ./ sum(p))
    end
    return nothing
end
