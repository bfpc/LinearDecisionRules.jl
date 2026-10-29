# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The laws conditioned by the rows tagged `Truncate()` (see `canonicalize`): a univariate law whose
# linear rows involve it alone is truncated exactly, and its rows are dropped; the other laws
# coupled by such rows, or by conic rows tagged `Truncate()`, are merged, by union–find, into one
# `ConditionedDistribution`, in the place of the first one, or into the exact law of the atoms of
# their product in the conditioning set when they all have finitely many atoms.
function _condition(src::MOI.ModelLike, form::CanonicalForm)
    conic = findall(row -> row.conditioning, form.conic_rows)
    declared = any(law -> law.distribution isa ConditionedDistribution, form.laws)
    any(form.conditioning) || !isempty(conic) || declared || return form
    law_of = zeros(Int, length(form.uncertainties))
    for (l, law) in enumerate(form.laws), j in law.columns
        law_of[j-1] = l
    end
    Wt = SparseArrays.sparse(transpose(form.W_valid))
    rows = findall(form.conditioning)
    # The η columns of each conditioning row, and the laws that they belong to.
    columns_of = Dict(r => SparseArrays.rowvals(Wt)[SparseArrays.nzrange(Wt, r)] for r in rows)
    laws_of = Dict(r => unique(law_of[columns_of[r]]) for r in rows)
    conic_laws_of = Dict(c => unique(law_of[_nonzero_columns(form.conic_rows[c].W)]) for c in conic)
    # A conic row is not an interval, so the laws that it conditions are estimated by rejection.
    in_conic = Set(l for c in conic for l in conic_laws_of[c])
    laws = copy(form.laws)
    dropped = falses(length(form.h_valid))
    for (l, law) in enumerate(form.laws)
        mine = filter(r -> l in laws_of[r], rows)
        if isempty(mine) || l in in_conic || !all(r -> length(laws_of[r]) == 1, mine)
            continue
        end
        d = law.distribution
        d isa Distributions.UnivariateDistribution || continue
        j = only(law.columns) - 1
        (lo,), (hi,) = _support_box(d)
        # The rows `a η ≥ h`, as pairs `(a, h)`.
        pairs = [(Wt[j, r], form.h_valid[r]) for r in mine]
        for (a, h) in pairs
            if a > 0
                lo = max(lo, h / a)
            else
                hi = min(hi, h / a)
            end
        end
        names = _row_names(src, form, mine)
        truncated = _truncated_law(d, lo, hi, pairs, form.uncertainty_names[j], names)
        truncated === nothing && continue
        laws[l] = UncertaintyBlock(truncated, law.columns, law.stages)
        dropped[mine] .= true
    end
    coupled = _UnionFind(length(laws))
    for r in rows
        dropped[r] || _join!(coupled, laws_of[r])
    end
    for c in conic
        _join!(coupled, conic_laws_of[c])
    end
    groups, group = _groups(coupled)
    # The conditioning rows of each group of laws.
    group_rows, group_conic = [Int[] for _ in groups], [Int[] for _ in groups]
    for r in rows
        dropped[r] || push!(group_rows[group[first(laws_of[r])]], r)
    end
    for c in conic
        push!(group_conic[group[first(conic_laws_of[c])]], c)
    end
    merged = UncertaintyBlock[]
    for (g, members) in enumerate(groups)
        linear, cones = group_rows[g], group_conic[g]
        # A law without conditioning rows is alone in its group. A declared
        # `ConditionedDistribution` may be exact (see `_conditioned_block`).
        alone = isempty(linear) && isempty(cones)
        if alone && !(laws[only(members)].distribution isa ConditionedDistribution)
            push!(merged, laws[only(members)])
        else
            push!(merged, _conditioned_block(src, form, laws[members], linear, cones, Wt))
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

# The law of the declarations `laws` conditioned on the rows `rows` of `W_valid` and the conic rows
# `conic` of `form`. A law that is already a `ConditionedDistribution` (declared so) gives its laws
# and its rows. When every law has finitely many atoms, the conditioned law is exact (see
# `_discrete_conditioned_law`); otherwise it is a `ConditionedDistribution`.
function _conditioned_block(src, form::CanonicalForm, laws, rows, conic, Wt)
    components = Distributions.Distribution[]
    columns, stages = Int[], Int[]
    W, h, names = Matrix{Float64}[], Float64[], String[]
    cones = _SupportCone[]
    m = length(form.uncertainties)
    for law in laws
        d = law.distribution
        append!(columns, law.columns)
        append!(stages, law.stages)
        if d isa ConditionedDistribution
            append!(components, d.laws)
            # Its rows, over its own columns.
            block = zeros(length(d.h), m)
            block[:, law.columns.-1] = d.W
            push!(W, block)
            append!(h, d.h)
            append!(cones, d.cones)
            append!(names, d.names)
        else
            push!(components, d)
        end
    end
    for r in rows
        row = zeros(1, m)
        for p in SparseArrays.nzrange(Wt, r)
            row[SparseArrays.rowvals(Wt)[p]] = SparseArrays.nonzeros(Wt)[p]
        end
        push!(W, row)
        push!(h, form.h_valid[r])
        push!(names, _constraint_name(src, form.valid_constraints[r]))
    end
    isempty(rows) || push!(cones, MOI.Nonnegatives(length(rows)))
    for c in conic
        row = form.conic_rows[c]
        push!(W, Matrix(row.W))
        append!(h, row.h)
        push!(cones, row.cone)
        append!(names, fill(_constraint_name(src, row.constraint), length(row.h)))
    end
    Wη = reduce(vcat, W)[:, columns.-1]
    d = ConditionedDistribution(components, Wη, h; cones, names)
    discrete = _discrete_conditioned_law(d, join(form.uncertainty_names[columns.-1], ", "))
    return UncertaintyBlock(something(discrete, d), columns, stages)
end

# Above this number of atoms of the product of its laws, a conditioned law is estimated by
# rejection sampling rather than enumerated.
const _MAX_PRODUCT_ATOMS = 10^6

# The law `d` when its laws have finitely many atoms: the atoms of their product in the
# conditioning set, with their probabilities renormalized, as a `MvDiscreteNonParametric`. It is
# exact, and the law of the draws by rejection (with the same tolerance). `nothing` if a law has
# infinitely many atoms, or if the product has more than `_MAX_PRODUCT_ATOMS`. No atom in the set
# is an error that names the constraints and the uncertainties `names`.
function _discrete_conditioned_law(d::ConditionedDistribution, names::String)
    pieces = map(_finite_atoms, d.laws)
    any(isnothing, pieces) && return nothing
    prod(Float64(length(p)) for (_, p) in pieces) <= _MAX_PRODUCT_ATOMS || return nothing
    grid = CartesianIndices(Tuple(length(p) for (_, p) in pieces))
    atoms, probabilities = Vector{Float64}[], Float64[]
    for chunk in Iterators.partition(grid, 2^14)
        C = Matrix{Float64}(undef, length(d), length(chunk))
        p = ones(length(chunk))
        for (s, index) in enumerate(chunk), (l, (x, q)) in enumerate(pieces)
            C[d.offsets[l]:d.offsets[l+1]-1, s] .= view(x, :, index[l])
            p[s] *= q[index[l]]
        end
        for s in findall(_in_set(d, C))
            push!(atoms, C[:, s])
            push!(probabilities, p[s])
        end
    end
    if isempty(atoms)
        error(
            "The constraints $(_constraint_names(d)) tagged `Truncate()` condition the law of " *
            "the uncertainties $names on a set that contains none of its atoms, so its " *
            "probability is zero.",
        )
    end
    return MvDiscreteNonParametric(atoms, probabilities ./ sum(probabilities))
end
# The law `d` conditioned on the rows `a η ≥ h` of `pairs`, whose intersection with the support of
# `d` is `[lo, hi]`, exactly: `Distributions.truncated` for a continuous law, or a
# `DiscreteNonParametric` law on the atoms that satisfy the rows for a discrete law with a finite
# support. It is `nothing` for other laws, whose truncation is then estimated by rejection
# sampling. A truncation of probability zero is an error that names the constraints.
function _truncated_law(d::Distributions.UnivariateDistribution, lo, hi, pairs, name, constraints)
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
        atoms = _finite_atoms(d)
        atoms === nothing && return nothing
        x, p = vec(atoms[1]), atoms[2]
        # The rule of the rejection, on the rows themselves, so that both treat the atoms on the
        # boundary alike.
        a, h = first.(pairs), last.(pairs)
        cone = MOI.Nonnegatives(length(h))
        inside = findall(η -> !_outside_cone(cone, a .* η .- h, h, _CONDITIONING_TOLERANCE), x)
        isempty(inside) && empty_error()
        return Distributions.DiscreteNonParametric(x[inside], p[inside] ./ sum(p[inside]))
    end
    return nothing
end
