# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    lift(
        form::CanonicalForm,
        liftings::AbstractDict{MOI.VariableIndex,<:AbstractLifting},
    ) -> CanonicalForm

Return the lifted form of `form`, whose columns are the lifted coordinates `ξ̃ = (1, η̃)`, with
`ξ = R ξ̃` for the retraction `R` of the liftings (see [`AbstractLifting`](@ref)).

Each uncertainty in `liftings` must be declared alone, with a univariate law. Its column is
replaced by its lifted coordinates, which are in its stage; the columns are sorted by stage again.
The data are lifted with sparse products: `B ← BR` for the rows, `C ← CR`, `d ← Rᵀd` and
`Q ← RᵀQR` for the objective (whose constant `f` does not change), and `W ← WR` for the valid
and conic rows. The law of the lifted coordinates is a [`LiftedDistribution`](@ref), whose box replaces the
box of the original law, and the convex-hull rows of each lifting are added to the valid rows,
with constraint `nothing`, so that they form a support block with its box.

The field `lifting` of the result is the [`LiftingMap`](@ref) that maps the outcomes of the
original uncertainties to the columns of the lifted form. [`canonicalize`](@ref) lifts the
uncertainties with breakpoints.

## Example

```julia
import LinearDecisionRules as LDR
form = LDR.canonicalize(backend(model), LDR.model_info(backend(model)))
lifted = lift(form, Dict(index(d) => AxisPiecewise([90, 100, 110])))
```
"""
function lift(form::CanonicalForm, liftings::AbstractDict{MOI.VariableIndex,<:AbstractLifting})
    if form.lifting !== nothing
        error("The canonical form is already lifted.")
    end
    m = length(form.uncertainties)
    lifting_of = _liftings_by_column(form, liftings)
    # The lifted coordinates `(j, k)`: column `j` of ξ itself if `k == 0`, else its `k`-th lifted
    # coordinate. A lifted coordinate is in the largest stage of its inputs, here the stage of `j`.
    coordinates = Tuple{Int,Int}[]
    for j in 2:1+m
        l = lifting_of[j]
        if l === nothing
            push!(coordinates, (j, 0))
        else
            append!(coordinates, ((j, k) for k in 1:_num_lifted(l)))
        end
    end
    stages = [form.uncertainty_stages[j-1] for (j, _) in coordinates]
    order = sortperm(stages; alg = Base.Sort.DEFAULT_STABLE)
    coordinates, stages = coordinates[order], stages[order]
    m̃ = length(coordinates)
    kept = zeros(Int, 1 + m)
    kept[1] = 1
    pieces = Dict{Int,Vector{Int}}()
    for (c, (j, k)) in enumerate(coordinates)
        if k == 0
            kept[j] = 1 + c
        else
            push!(get!(pieces, j, Int[]), 1 + c)
        end
    end
    R = _Triplets()
    _push!(R, 1, 1, 1.0)
    for j in 2:1+m
        kept[j] > 0 && _push!(R, j, kept[j], 1.0)
    end
    blocks = LiftingBlock[]
    for j in sort!(collect(keys(pieces)))
        l = lifting_of[j]
        push!(blocks, LiftingBlock(l, [j], pieces[j]))
        Rl = retraction(l)
        for (_, k, p) in zip(SparseArrays.findnz(Rl)...)
            _push!(R, j, pieces[j][k], p)
        end
    end
    R = _sparse(R, 1 + m, 1 + m̃)
    Rη = R[2:end, 2:end]
    laws, W_hull, h_hull = _lifted_laws(form, lifting_of, kept, pieces, m̃)
    names = map(coordinates) do (j, k)
        name = form.uncertainty_names[j-1]
        return k == 0 ? name : "$name[piece $k]"
    end
    lifting = LiftingMap(
        copy(form.uncertainties),
        copy(form.uncertainty_names),
        copy(form.uncertainty_stages),
        kept,
        blocks,
        R,
    )
    Rηt = SparseArrays.sparse(transpose(Rη))
    # The decisions, the matrices `A`, `P`, the constant `f` and the parameters do not change: the
    # first column of `R` is the first unit vector, so `B[:, 1]` is unchanged.
    return _replace(
        form;
        uncertainties = [form.uncertainties[j-1] for (j, _) in coordinates],
        uncertainty_names = names,
        uncertainty_stages = stages,
        laws,
        B_e = form.B_e * R,
        B_u = form.B_u * R,
        B_l = form.B_l * R,
        W_valid = vcat(form.W_valid * Rη, W_hull),
        h_valid = vcat(form.h_valid, h_hull),
        valid_constraints = vcat(form.valid_constraints, fill(nothing, length(h_hull))),
        conditioning = vcat(form.conditioning, falses(length(h_hull))),
        conic_rows = [
            ConicRow(r.W * Rη, r.h, r.cone, r.constraint, r.conditioning) for r in form.conic_rows
        ],
        C = form.C * R,
        Q = Rηt * form.Q * Rη,
        d = Vector(Rηt * form.d),
        lifting,
    )
end

# The lifting of each column of ξ, or `nothing`.
function _liftings_by_column(form::CanonicalForm, liftings)
    column = Dict(v => 1 + j for (j, v) in enumerate(form.uncertainties))
    lifting_of = Vector{Union{Nothing,AbstractLifting}}(nothing, 1 + length(form.uncertainties))
    for (v, l) in liftings
        if !haskey(column, v)
            error("The variable with index $(v.value) has a lifting, but it is not an uncertainty.")
        end
        j = column[v]
        name = form.uncertainty_names[j-1]
        law = form.laws[findfirst(law -> j in law.columns, form.laws)]
        if law.distribution isa ConditionedDistribution
            error(
                "The uncertainty $name has a lifting, but its law is conditioned by the " *
                "constraints $(_constraint_names(law.distribution)) and estimated by rejection " *
                "sampling: liftings of such laws are not supported. A univariate law conditioned " *
                "on an interval by constraints on it alone is truncated exactly, and can be lifted.",
            )
        elseif !(law.distribution isa Distributions.UnivariateDistribution)
            names = join((form.uncertainty_names[c-1] for c in law.columns), ", ")
            error(
                "The uncertainty $name has a lifting, but it is a component of the joint law of " *
                "$names (declared together, or merged by `Truncate()` constraints that couple " *
                "them). Liftings apply to an uncertainty whose law is univariate and not coupled " *
                "to others by `Truncate()` constraints.",
            )
        elseif _num_inputs(l) != 1
            error("The lifting $l of the uncertainty $name must have one input.")
        end
        lifting_of[j] = l
    end
    return lifting_of
end

# The laws of the lifted form, and the convex-hull rows `W η̃ ≥ h` of the liftings.
function _lifted_laws(form::CanonicalForm, lifting_of, kept, pieces, m̃::Int)
    hull = _Triplets()
    h = Float64[]
    laws = map(form.laws) do law
        if length(law.columns) != 1 || lifting_of[only(law.columns)] === nothing
            return UncertaintyBlock(law.distribution, kept[law.columns], law.stages)
        end
        j = only(law.columns)
        l = lifting_of[j]
        name = form.uncertainty_names[j-1]
        d = _named_lifting_error(() -> LiftedDistribution(law.distribution, l), name)
        lo, hi = _support_box(law.distribution)
        W, hl = _hull_rows(l, lo, hi)
        rows, cols, values = SparseArrays.findnz(W)
        for (r, c, w) in zip(rows, cols, values)
            _push!(hull, length(h) + r, pieces[j][c] - 1, w)
        end
        append!(h, hl)
        return UncertaintyBlock(d, pieces[j], fill(only(law.stages), length(pieces[j])))
    end
    return laws, _sparse(hull, length(h), m̃), h
end

# The lifting map of `form`: its field `lifting`, or the identity map if nothing is lifted, so
# that `lift(map, ξ) == ξ`.
function _lifting_map(form::CanonicalForm)
    form.lifting === nothing || return form.lifting
    m = length(form.uncertainties)
    return LiftingMap(
        copy(form.uncertainties),
        copy(form.uncertainty_names),
        copy(form.uncertainty_stages),
        collect(1:1+m),
        LiftingBlock[],
        SparseArrays.spdiagm(0 => ones(1 + m)),
    )
end

# The uncertainties of the model that `form` was read from, and their names: those of its lifting
# map in a lifted form, whose own uncertainties are the lifted coordinates.
function _original_uncertainties(form::CanonicalForm)
    return form.lifting === nothing ? form.uncertainties : form.lifting.uncertainties
end

function _original_names(form::CanonicalForm)
    return form.lifting === nothing ? form.uncertainty_names : form.lifting.uncertainty_names
end

# The `AxisPiecewise` lifting of each uncertainty with breakpoints: a vector of breakpoints, or a
# number `n` of equally spaced breakpoints inside the support. No breakpoint means no lifting.
function _breakpoint_liftings(src, form::CanonicalForm, breakpoints)
    liftings = Dict{MOI.VariableIndex,AbstractLifting}()
    columns = Dict(v => 1 + j for (j, v) in enumerate(form.uncertainties))
    for (v, points) in breakpoints
        if !haskey(columns, v)
            error(
                "The variable $(_variable_name(src, v)) has breakpoints, but it is not an " *
                "uncertainty.",
            )
        elseif points isa Integer
            if points < 0
                error(
                    "The uncertainty $(_variable_name(src, v)) has $points breakpoints; the " *
                    "number of breakpoints must be nonnegative.",
                )
            end
            points == 0 && continue
            law = only(l for l in form.laws if columns[v] in l.columns)
            lo, hi = _support_box(law.distribution)
            k = findfirst(==(columns[v]), law.columns)
            if !(isfinite(lo[k]) && isfinite(hi[k]))
                error(
                    "The uncertainty $(_variable_name(src, v)) has $points breakpoints, but its " *
                    "support is unbounded.",
                )
            end
            points = collect(range(lo[k], hi[k]; length = points + 2)[2:end-1])
        end
        isempty(points) && continue
        liftings[v] = _named_lifting_error(_variable_name(src, v)) do
            return AxisPiecewise(points)
        end
    end
    return liftings
end

# Name the uncertainty in the `ArgumentError` of a lifting.
function _named_lifting_error(f::Function, name::String)
    try
        return f()
    catch err
        err isa ArgumentError || rethrow()
        error("The lifting of the uncertainty $name is invalid: $(err.msg)")
    end
end
