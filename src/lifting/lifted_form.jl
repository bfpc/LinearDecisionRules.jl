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
function lift(
    form::CanonicalForm,
    liftings::AbstractDict{MOI.VariableIndex,<:AbstractLifting},
    combinations::AbstractVector{<:Pair{Vector{MOI.VariableIndex},CombinationPiecewise}} = Pair{
        Vector{MOI.VariableIndex},
        CombinationPiecewise,
    }[],
)
    return _lift(form, liftings, combinations, Ref(0.0))
end

# `lift`, adding to `moment_seconds` the time of the moments of the lifted laws, which are computed
# when they are built. The time is not kept in the laws: the inner cache compares them field by
# field, and a time would differ at every `optimize!`.
function _lift(form::CanonicalForm, liftings, combinations, moment_seconds::Base.RefValue{Float64})
    if form.lifting !== nothing
        error("The canonical form is already lifted.")
    end
    m = length(form.uncertainties)
    lifting_of = _liftings_by_column(form, liftings)
    combination_inputs = _combination_columns(form, lifting_of, combinations)
    # The lifted coordinates `(j, k)`: column `j` of ξ itself if `k == 0`, else its `k`-th lifted
    # coordinate, and `(-c, k)` for the `k`-th piece of the combination `c`. A lifted coordinate is
    # in the largest stage of its inputs: the stage of `j`, or the last stage of the inputs of `c`.
    coordinates = Tuple{Int,Int}[]
    for j in 2:1+m
        l = lifting_of[j]
        if l === nothing
            push!(coordinates, (j, 0))
        else
            append!(coordinates, ((j, k) for k in 1:_num_lifted(l)))
        end
    end
    for (c, (_, l)) in enumerate(combinations)
        append!(coordinates, ((-c, k) for k in 1:_num_lifted(l)))
    end
    function stage_of(j)
        return j > 0 ? form.uncertainty_stages[j-1] :
               maximum(form.uncertainty_stages[i-1] for i in combination_inputs[-j])
    end
    stages = [stage_of(j) for (j, _) in coordinates]
    order = sortperm(stages; alg = Base.Sort.DEFAULT_STABLE)
    coordinates, stages = coordinates[order], stages[order]
    m̃ = length(coordinates)
    kept = zeros(Int, 1 + m)
    kept[1] = 1
    pieces = Dict{Int,Vector{Int}}()
    combination_pieces = [Int[] for _ in combinations]
    for (c, (j, k)) in enumerate(coordinates)
        if j < 0
            push!(combination_pieces[-j], 1 + c)
        elseif k == 0
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
    # The pieces of a combination are not in the retraction (see `CombinationPiecewise`).
    for (c, (_, l)) in enumerate(combinations)
        push!(blocks, LiftingBlock(l, combination_inputs[c], combination_pieces[c]))
    end
    R = _sparse(R, 1 + m, 1 + m̃)
    Rη = R[2:end, 2:end]
    laws, W_hull, h_hull = _lifted_laws(form, lifting_of, kept, pieces, m̃, moment_seconds)
    if !isempty(combinations)
        liftings_c = [l for (_, l) in combinations]
        laws, W_c, h_c = _combination_laws(
            form,
            laws,
            kept,
            liftings_c,
            combination_inputs,
            combination_pieces,
            m̃,
            moment_seconds,
        )
        W_hull, h_hull = vcat(W_hull, W_c), vcat(h_hull, h_c)
    end
    combination_names = [
        _combination_name(form, l, combination_inputs[c]) for (c, (_, l)) in enumerate(combinations)
    ]
    names = map(coordinates) do (j, k)
        j < 0 && return "$(combination_names[-j])[piece $(k + 1)]"
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
    # The variable of each lifted coordinate: that of its column of ξ, and for the pieces of a
    # combination, that of its input in the last stage (the variables only name the columns).
    function variable(j)
        j > 0 && return form.uncertainties[j-1]
        inputs = combination_inputs[-j]
        return form.uncertainties[inputs[argmax(form.uncertainty_stages[i-1] for i in inputs)]-1]
    end
    # The decisions, the matrices `A`, `P`, the constant `f` and the parameters do not change: the
    # first column of `R` is the first unit vector, so `B[:, 1]` is unchanged.
    return _replace(
        form;
        uncertainties = [variable(j) for (j, _) in coordinates],
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
function _lifted_laws(form::CanonicalForm, lifting_of, kept, pieces, m̃::Int, moment_seconds)
    hull = _Triplets()
    h = Float64[]
    laws = map(form.laws) do law
        if length(law.columns) != 1 || lifting_of[only(law.columns)] === nothing
            return UncertaintyBlock(law.distribution, kept[law.columns], law.stages)
        end
        j = only(law.columns)
        l = lifting_of[j]
        name = form.uncertainty_names[j-1]
        start = time_ns()
        d = _named_lifting_error(() -> LiftedDistribution(law.distribution, l), name)
        moment_seconds[] += _seconds_since(start)
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

# The columns of ξ of the inputs of each combination. An input must be an uncertainty that is not
# lifted itself, whose law is not conditioned by rejection.
function _combination_columns(form::CanonicalForm, lifting_of, combinations)
    column = Dict(v => 1 + j for (j, v) in enumerate(form.uncertainties))
    return map(combinations) do (inputs, l)
        if length(inputs) != _num_inputs(l)
            error("The lifting $l has $(_num_inputs(l)) inputs; got $(length(inputs)) variables.")
        elseif !allunique(inputs)
            error("The inputs of the lifting $l must be distinct.")
        end
        return map(inputs) do v
            if !haskey(column, v)
                error(
                    "The variable with index $(v.value) is in a combination with breakpoints, " *
                    "but it is not an uncertainty.",
                )
            end
            j = column[v]
            name = form.uncertainty_names[j-1]
            if lifting_of[j] !== nothing
                error(
                    "The uncertainty $name has breakpoints and is in a combination with " *
                    "breakpoints: an uncertainty in a combination cannot be lifted itself.",
                )
            end
            law = form.laws[findfirst(law -> j in law.columns, form.laws)]
            if law.distribution isa ConditionedDistribution
                error(
                    "The uncertainty $name is in a combination with breakpoints, but its law is " *
                    "conditioned by the constraints $(_constraint_names(law.distribution)) and " *
                    "estimated by rejection sampling: combinations of such laws are not supported.",
                )
            end
            return j
        end
    end
end

# The laws of the lifted form with combinations: the laws of the inputs of the combinations that
# share a law are merged with their pieces into one `_CombinationLaw`, in the place of the first of
# them; and the outer approximation of the hull of each combination, as rows `W η̃ ≥ h`.
function _combination_laws(form, laws, kept, liftings, inputs, pieces, m̃::Int, moment_seconds)
    law_of_column = Dict(j => i for (i, law) in enumerate(form.laws) for j in law.columns)
    groups = _UnionFind(length(form.laws))
    for columns in inputs
        _join!(groups, [law_of_column[j] for j in columns])
    end
    touched = Set(law_of_column[j] for columns in inputs for j in columns)
    component = Dict{Int,Vector{Int}}()
    for i in sort!(collect(touched))
        push!(get!(component, _root!(groups, i), Int[]), i)
    end
    hull = _Triplets()
    h = Float64[]
    merged = Dict{Int,UncertaintyBlock}()
    for members in sort!(collect(values(component)); by = first)
        combos =
            [c for (c, columns) in enumerate(inputs) if law_of_column[first(columns)] in members]
        originals = [form.laws[i] for i in members]
        # The position of each column of ξ in the coordinates `η` of the merged law.
        columns = reduce(vcat, [law.columns for law in originals])
        position = Dict(j => p for (p, j) in enumerate(columns))
        # A product of univariate laws is merged as its components, so that each is integrated or
        # enumerated on its own.
        names = [_combination_name(form, liftings[c], inputs[c]) for c in combos]
        start = time_ns()
        d = _named_lifting_error(join(names, " and "), "combination") do
            return _CombinationLaw(
                reduce(vcat, [_components(law.distribution) for law in originals]),
                liftings[combos],
                [[position[j] for j in inputs[c]] for c in combos],
            )
        end
        moment_seconds[] += _seconds_since(start)
        block_columns = vcat([kept[j] for j in columns], (pieces[c] for c in combos)...)
        block_stages = vcat(
            reduce(vcat, [law.stages for law in originals]),
            (
                fill(maximum(form.uncertainty_stages[j-1] for j in inputs[c]), length(pieces[c])) for c in combos
            )...,
        )
        merged[first(members)] = UncertaintyBlock(d, block_columns, block_stages)
        for (c, name, (s_min, s_max)) in zip(combos, names, d.ranges)
            W_η, W_p, rhs = _named_lifting_error(name, "combination") do
                return _hull_rows(liftings[c], s_min, s_max)
            end
            for r in axes(W_η, 1)
                row = length(h) + r
                for (i, j) in enumerate(inputs[c])
                    iszero(W_η[r, i]) || _push!(hull, row, kept[j] - 1, W_η[r, i])
                end
                for (k, column) in enumerate(pieces[c])
                    iszero(W_p[r, k]) || _push!(hull, row, column - 1, W_p[r, k])
                end
            end
            append!(h, rhs)
        end
    end
    out = UncertaintyBlock[]
    for (i, law) in enumerate(laws)
        if haskey(merged, i)
            push!(out, merged[i])
        elseif !(i in touched)
            push!(out, law)
        end
    end
    return out, _sparse(hull, length(h), m̃), h
end

# A name for the combination `l` of the columns `columns` of ξ, as `"(2 a + b - 1)"`.
function _combination_name(form::CanonicalForm, l::CombinationPiecewise, columns::Vector{Int})
    names = [form.uncertainty_names[j-1] for j in columns]
    return _combination_name(names, l.coefficients, l.constant)
end

function _combination_name(names::Vector{String}, coefficients, constant::Real)
    terms = String[]
    for (w, name) in zip(coefficients, names)
        sign = w < 0 ? "-" : "+"
        magnitude = abs(w) == 1 ? name : "$(_short(abs(w))) $name"
        push!(terms, isempty(terms) ? (w < 0 ? "-$magnitude" : magnitude) : "$sign $magnitude")
    end
    iszero(constant) || push!(terms, "$(constant < 0 ? "-" : "+") $(_short(abs(constant)))")
    return "(" * join(terms, " ") * ")"
end

_short(x::Float64) = isinteger(x) ? string(Int(x)) : string(x)

# The `CombinationPiecewise` liftings of the combinations with breakpoints of a model: a vector of
# breakpoints, or a number `n` of equally spaced breakpoints inside the range of the combination on
# the support. No breakpoint means no lifting. The terms of a variable are added up, and a
# combination must have an uncertainty with a nonzero coefficient.
function _combination_liftings(src, form::CanonicalForm, combinations)
    out = Pair{Vector{MOI.VariableIndex},CombinationPiecewise}[]
    uncertainty = Set(form.uncertainties)
    for (f, points) in combinations
        coefficient = Dict{MOI.VariableIndex,Float64}()
        for term in f.terms
            coefficient[term.variable] = get(coefficient, term.variable, 0.0) + term.coefficient
        end
        filter!(p -> !iszero(p.second), coefficient)
        inputs = sort!(collect(keys(coefficient)); by = v -> v.value)
        for v in inputs
            if !(v in uncertainty)
                error(
                    "The variable $(_variable_name(src, v)) is in a combination with " *
                    "breakpoints, but it is not an uncertainty.",
                )
            end
        end
        if isempty(inputs)
            error(
                "A combination with breakpoints must have an uncertainty with a nonzero coefficient.",
            )
        end
        a = [coefficient[v] for v in inputs]
        if points isa Integer
            if points < 0
                error(
                    "A combination has $points breakpoints; the number of breakpoints must be " *
                    "nonnegative.",
                )
            end
            points == 0 && continue
            s_min, s_max = _form_combination_range(form, inputs, a, f.constant)
            points = collect(range(s_min, s_max; length = points + 2)[2:end-1])
        end
        isempty(points) && continue
        name = _combination_name([_variable_name(src, v) for v in inputs], a, f.constant)
        lifting = _named_lifting_error(name, "combination") do
            return CombinationPiecewise(a, points; constant = f.constant)
        end
        push!(out, inputs => lifting)
    end
    return out
end

# The range of `c + a ⋅ η[inputs]` on the support of the laws of `form`.
function _form_combination_range(form::CanonicalForm, inputs, a, c)
    column = Dict(v => 1 + j for (j, v) in enumerate(form.uncertainties))
    columns = [column[v] for v in inputs]
    members = unique([findfirst(law -> j in law.columns, form.laws) for j in columns])
    laws = reduce(vcat, [_components(form.laws[i].distribution) for i in members])
    all_columns = reduce(vcat, [form.laws[i].columns for i in members])
    position = Dict(j => p for (p, j) in enumerate(all_columns))
    return _combination_range(laws, _offsets(laws), [position[j] for j in columns], a, c)
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

function _original_stages(form::CanonicalForm)
    return form.lifting === nothing ? form.uncertainty_stages : form.lifting.uncertainty_stages
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

# Name the uncertainty, or the combination (`what = "combination"`), in the `ArgumentError` of a
# lifting.
function _named_lifting_error(f::Function, name::String, what::String = "uncertainty")
    try
        return f()
    catch err
        err isa ArgumentError || rethrow()
        error("The lifting of the $what $name is invalid: $(err.msg)")
    end
end
