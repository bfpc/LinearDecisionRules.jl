# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The joint law of independent laws and of the pieces of the `CombinationPiecewise` liftings of
# their uncertainties. A lifting couples the laws of its inputs, which may be in several stages, so
# they are merged into one law with its pieces; liftings that share a law share the merged law.

"""
    _CombinationLaw

The law of `(η, p₁, …, p_L)`, where `η` gathers the uncertainties of independent `laws` and `pₗ`
are the pieces of the `l`-th [`CombinationPiecewise`](@ref) lifting, whose inputs are the
coordinates `inputs[l]` of `η`. Its mean and covariance are computed once, on construction. When
the laws are discrete with finitely many atoms or uniform, they are exact, from the laws of sums
of the terms of the combinations (integrals of dimension one, at most two for two liftings);
otherwise, by enumeration of the atoms of the discrete laws and nested quadrature (QuadGK, split at
the breakpoints) over the continuous univariate ones. Other laws are not supported yet.

`minimum` and `maximum` are the box of the laws and of the pieces, and `rand` draws each law and
lifts. It is a nominal `ContinuousMultivariateDistribution`, as [`LiftedDistribution`](@ref).
"""
struct _CombinationLaw <: Distributions.ContinuousMultivariateDistribution
    laws::Vector{Distributions.Distribution}
    # Law `k` has the coordinates `offsets[k] .+ (1:_dimension(laws[k]))` of `η`.
    offsets::Vector{Int}
    liftings::Vector{CombinationPiecewise}
    inputs::Vector{Vector{Int}}
    # The range `[s_min, s_max]` of the combination of each lifting on the support.
    ranges::Vector{Tuple{Float64,Float64}}
    lower::Vector{Float64}
    upper::Vector{Float64}
    mean::Vector{Float64}
    covariance::Matrix{Float64}
end

# The independent laws of a law, in the order of its coordinates: the univariate laws of a product,
# or the law itself.
_components(d::Distributions.Distribution) = Distributions.Distribution[d]

_components(d::Distributions.Product) = Distributions.Distribution[c for c in d.v]

function _components(d::Distributions.ProductDistribution{1,0})
    return Distributions.Distribution[c for c in d.dists]
end

function _offsets(laws::Vector{<:Distributions.Distribution})
    return cumsum(Int[0; [_dimension(d) for d in laws[1:end-1]]])
end

# The laws of the coordinates `positions` of `η`, by index.
function _laws_of(offsets::Vector{Int}, laws, positions)
    return sort!(unique([findlast(<(p), offsets) for p in positions]))
end

# The coefficients of `a ⋅ η[positions]` on the coordinates of the law `k`.
function _local_coefficients(laws, offsets::Vector{Int}, k::Int, positions, a)
    local_a = zeros(_dimension(laws[k]))
    for (p, w) in zip(positions, a)
        i = p - offsets[k]
        1 <= i <= length(local_a) && (local_a[i] += w)
    end
    return local_a
end

# The range of `c + a ⋅ η[positions]` on the support of the laws: on the atoms of a law with
# finitely many, on its box otherwise.
function _combination_range(
    laws::Vector{<:Distributions.Distribution},
    offsets::Vector{Int},
    positions::Vector{Int},
    a::Vector{Float64},
    c::Float64,
)
    s_min, s_max = c, c
    for k in _laws_of(offsets, laws, positions)
        d = laws[k]
        local_a = _local_coefficients(laws, offsets, k, positions, a)
        atoms = _finite_atoms(d)
        if atoms === nothing
            lo, hi = _support_box(d)
            s_min +=
                sum(min(w * l, w * u) for (w, l, u) in zip(local_a, lo, hi) if w != 0; init = 0.0)
            s_max +=
                sum(max(w * l, w * u) for (w, l, u) in zip(local_a, lo, hi) if w != 0; init = 0.0)
        else
            values = local_a' * atoms[1]
            s_min += minimum(values)
            s_max += maximum(values)
        end
    end
    return s_min, s_max
end

function _CombinationLaw(
    laws::Vector{<:Distributions.Distribution},
    liftings::Vector{CombinationPiecewise},
    inputs::Vector{Vector{Int}},
)
    laws = Vector{Distributions.Distribution}(laws)
    offsets = _offsets(laws)
    for d in laws
        if !(_finite_atoms(d) !== nothing || d isa Distributions.ContinuousUnivariateDistribution)
            throw(
                ArgumentError(
                    "The law $d of an input of a combination must be discrete with finitely " *
                    "many atoms, or continuous and univariate.",
                ),
            )
        end
        lo, hi = _support_box(d)
        if !(all(isfinite, lo) && all(isfinite, hi))
            throw(ArgumentError("The law $d of an input of a combination must be bounded."))
        end
    end
    ranges = [
        _combination_range(
            laws,
            offsets,
            inputs[l],
            liftings[l].coefficients,
            liftings[l].constant,
        ) for l in eachindex(liftings)
    ]
    m = offsets[end] + _dimension(laws[end])
    lower, upper = Float64[], Float64[]
    for d in laws
        lo, hi = _support_box(d)
        append!(lower, lo)
        append!(upper, hi)
    end
    for (l, (s_min, s_max)) in zip(liftings, ranges)
        lo, hi = _lifted_bounds(l, s_min, s_max)
        append!(lower, lo)
        append!(upper, hi)
    end
    mean, covariance = _combination_moments(laws, offsets, m, liftings, inputs)
    return _CombinationLaw(laws, offsets, liftings, inputs, ranges, lower, upper, mean, covariance)
end

# The coordinates of the pieces of each lifting, after the `m` coordinates of `η`.
function _piece_positions(m::Int, liftings::Vector{CombinationPiecewise})
    sizes = [_num_lifted(l) for l in liftings]
    starts = m .+ cumsum([0; sizes[1:end-1]])
    return [s .+ (1:n) for (s, n) in zip(starts, sizes)]
end

# With `quadrature = true`, every moment is by enumeration and nested quadrature, which the tests
# compare with the laws of sums.
function _combination_moments(laws, offsets, m, liftings, inputs; quadrature::Bool = false)
    pieces = _piece_positions(m, liftings)
    n = m + sum(_num_lifted, liftings)
    μ = zeros(n)
    Σ = zeros(n, n)
    for (k, d) in enumerate(laws)
        r = offsets[k] .+ (1:_dimension(d))
        μ[r] = _law_mean(d)
        Σ[r, r] = _law_covariance(d)
    end
    lifted_laws = [_laws_of(offsets, laws, inputs[l]) for l in eachindex(liftings)]
    # One lifting: E[p], E[p ηⱼᵀ] for the coordinates j of its laws, and E[p pᵀ].
    for (l, lifting) in enumerate(liftings)
        ks = lifted_laws[l]
        J = reduce(vcat, [collect(offsets[k] .+ (1:_dimension(laws[k]))) for k in ks])
        np, nj = _num_lifted(lifting), length(J)
        hinges = quadrature ? nothing : _hinge_moments(laws, offsets, ks, lifting, inputs[l])
        if hinges === nothing
            values = _expectation(laws, offsets, ks, [l], liftings, inputs) do η
                p = lift(lifting, view(η, inputs[l]))
                return vcat(p, vec(p * view(η, J)'), vec(p * p'))
            end
            Ep = values[1:np]
            Epη = reshape(values[np.+(1:np*nj)], np, nj)
            Epp = reshape(values[np+np*nj+1:end], np, np)
        else
            Eh, Ehη, Ehh = hinges
            D = _pieces_of_hinges(np)
            Ep, Epη, Epp = D * Eh, D * Ehη, D * Ehh * D'
        end
        μ[pieces[l]] = Ep
        Σ[pieces[l], J] = Epη - Ep * μ[J]'
        Σ[J, pieces[l]] = Σ[pieces[l], J]'
        Σ[pieces[l], pieces[l]] = Epp - Ep * Ep'
    end
    # Two liftings that share a law are correlated; the others are independent.
    for l in eachindex(liftings), r in l+1:length(liftings)
        isempty(intersect(lifted_laws[l], lifted_laws[r])) && continue
        nl, nr = _num_lifted(liftings[l]), _num_lifted(liftings[r])
        cross = nothing
        if !quadrature
            cross = _hinge_cross_moments(laws, offsets, lifted_laws, liftings, inputs, l, r)
        end
        if cross === nothing
            ks = sort!(union(lifted_laws[l], lifted_laws[r]))
            values = _expectation(laws, offsets, ks, [l, r], liftings, inputs) do η
                p = lift(liftings[l], view(η, inputs[l]))
                q = lift(liftings[r], view(η, inputs[r]))
                return vec(p * q')
            end
            C = reshape(values, nl, nr)
        else
            C = _pieces_of_hinges(nl) * cross * _pieces_of_hinges(nr)'
        end
        C -= μ[pieces[l]] * μ[pieces[r]]'
        Σ[pieces[l], pieces[r]] = C
        Σ[pieces[r], pieces[l]] = C'
    end
    return μ, _symmetric(Σ)
end

# The moments by laws of sums (`docs/math/lifted-moments.tex`) are those of the hinges
# `hᵢ = (s - bᵢ)₊` of the combination `s` at its breakpoints, and the pieces are `p = D h`:
# `pᵢ = hᵢ - hᵢ₊₁` for `i < n` and `pₙ = hₙ`.
function _pieces_of_hinges(n::Int)
    D = Matrix{Float64}(LinearAlgebra.I, n, n)
    for i in 1:n-1
        D[i, i+1] = -1.0
    end
    return D
end

# The term `a ⋅ η_K` of the law `d` in a sum, with the coefficients `a` of its coordinates, or
# `nothing` for a law that is neither discrete with finitely many atoms nor uniform.
function _sum_term(d::Distributions.Distribution, a::Vector{Float64})
    atoms = _finite_atoms(d)
    if atoms !== nothing
        return _DiscreteTerm(vec(a' * atoms[1]), atoms[2])
    elseif d isa Distributions.Uniform
        lo, hi = a[1] * d.a, a[1] * d.b
        return _UniformTerm(min(lo, hi), abs(hi - lo))
    end
    return nothing
end

# `E[h]`, `E[h ηᵀ]` for the coordinates of the laws `ks` of the lifting, in order, and `E[h hᵀ]`
# (`prop:one-combination`), or `nothing` unless every law of `ks` has a term.
function _hinge_moments(laws, offsets, ks, lifting::CombinationPiecewise, positions)
    coefficients =
        [_local_coefficients(laws, offsets, k, positions, lifting.coefficients) for k in ks]
    terms = [_sum_term(laws[k], a) for (k, a) in zip(ks, coefficients)]
    any(isnothing, terms) && return nothing
    b, c = lifting.breakpoints, lifting.constant
    s = _SumLaw(c, terms)
    π1 = [_stop_loss(s, 1, x) for x in b]
    π2 = [_stop_loss(s, 2, x) for x in b]
    # For b ≤ b′, (s - b)₊ (s - b′)₊ = (s - b′)₊² + (b′ - b) (s - b′)₊.
    Ehh =
        [π2[max(i, j)] + abs(b[i] - b[j]) * π1[max(i, j)] for i in eachindex(b), j in eachindex(b)]
    blocks = map(eachindex(ks)) do t
        rest = _SumLaw(c, terms[[1:t-1; t+1:end]])
        return _hinge_coordinate_moments(laws[ks[t]], terms[t], coefficients[t], rest, b)
    end
    return π1, reduce(hcat, blocks), Ehh
end

# `E[hᵢ η_m]` for the coordinates `m` of the law `d` of the term `term`, with `rest` the law of the
# rest of the combination, independent of `d`: `E[η_m π₁(bᵢ - a ⋅ η_K)]` for the stop-loss
# transform `π₁` of `rest`, a sum over the atoms of `d`.
function _hinge_coordinate_moments(d, term::_DiscreteTerm, ::Vector{Float64}, rest, b)
    A, p = _finite_atoms(d)
    E = zeros(length(b), size(A, 1))
    for (k, v) in enumerate(term.values), (i, x) in enumerate(b)
        E[i, :] .+= (p[k] * _stop_loss(rest, 1, x - v)) .* view(A, :, k)
    end
    return E
end

# For a uniform `d`, an integral over `d` for each atom `z` of the discrete part of `rest`, whose
# integrand is a polynomial between the knots where `bᵢ - a x - z` crosses those of the uniform
# part.
function _hinge_coordinate_moments(
    d::Distributions.Uniform,
    ::_UniformTerm,
    coefficients::Vector{Float64},
    rest,
    b,
)
    a = coefficients[1]
    E = zeros(length(b), 1)
    lo, hi = Float64(d.a), Float64(d.b)
    degree = length(rest.widths) + 2
    for (z, q) in zip(rest.atoms, rest.probabilities), (i, x) in enumerate(b)
        shift = x - rest.offset - z
        knots = [(shift - k) / a for k in rest.knots]
        g(y) = y * _uniform_stop_loss(rest, 1, shift - a * y)
        E[i, 1] += q * _piecewise_integral(g, lo, hi, knots, degree) / (hi - lo)
    end
    return E
end

# `E[hᵢ(s) h′ⱼ(s′)]` for the liftings `l` and `r`, which share laws (`prop:two-combinations`): with
# `s = A + Y` and `s′ = A′ + Y′`, where `A`, `A′` are the terms of the shared laws, an expectation
# of `π₁^Y(bᵢ - A) π₁^{Y′}(b′ⱼ - A′)` over the shared laws: a sum over the joint atoms of the
# discrete ones and an integral over the uniform ones, whose terms in `A′` must be `λ` times those
# in `A`. `nothing` for other laws, or uniform laws that are not shared that way.
function _hinge_cross_moments(laws, offsets, lifted_laws, liftings, inputs, l::Int, r::Int)
    ks_l, ks_r = lifted_laws[l], lifted_laws[r]
    shared = intersect(ks_l, ks_r)
    coefficients(k, t) = _local_coefficients(laws, offsets, k, inputs[t], liftings[t].coefficients)
    all(
        k -> _sum_term(laws[k], coefficients(k, k in ks_l ? l : r)) !== nothing,
        union(ks_l, ks_r),
    ) || return nothing
    function rest(t, ks)
        return _SumLaw(
            liftings[t].constant,
            [_sum_term(laws[k], coefficients(k, t)) for k in setdiff(ks, shared)],
        )
    end
    Y, Y′ = rest(l, ks_l), rest(r, ks_r)
    uniform = [k for k in shared if _finite_atoms(laws[k]) === nothing]
    a = Float64[coefficients(k, l)[1] for k in uniform]
    a′ = Float64[coefficients(k, r)[1] for k in uniform]
    (any(iszero, a) || any(iszero, a′)) && return nothing
    λ = isempty(uniform) ? 0.0 : a′[1] / a[1]
    isapprox(a′, λ .* a; rtol = 1e-12) || return nothing
    # The values of `(A, A′)` on the joint atoms of the shared discrete laws.
    joint = Dict((0.0, 0.0) => 1.0)
    for k in shared
        k in uniform && continue
        A, p = _finite_atoms(laws[k])
        count = Float64(length(joint)) * length(p)
        count > _MAX_COMBINATION_ATOMS && _too_many_atoms(count)
        u, v = vec(coefficients(k, l)' * A), vec(coefficients(k, r)' * A)
        next = Dict{Tuple{Float64,Float64},Float64}()
        for ((x, y), q) in joint, j in eachindex(p)
            key = (x + u[j] + 0.0, y + v[j] + 0.0)
            next[key] = get(next, key, 0.0) + q * p[j]
        end
        joint = next
    end
    b, b′ = liftings[l].breakpoints, liftings[r].breakpoints
    E = zeros(length(b), length(b′))
    if isempty(uniform)
        for ((x, y), q) in joint, (i, β) in enumerate(b), (j, β′) in enumerate(b′)
            E[i, j] += q * _stop_loss(Y, 1, β - x) * _stop_loss(Y′, 1, β′ - y)
        end
        return E
    end
    # The uniform part of `A` is `U = U.offset + τ`, with the density of the sum of the uniform
    # laws for `τ`, and that of `A′` is `λ U`.
    U = _SumLaw(0.0, [_sum_term(laws[k], [w]) for (k, w) in zip(uniform, a)])
    degree = length(U.widths) - 1 + length(Y.widths) + length(Y′.widths) + 2
    knots_Y, knots_Y′ = _stop_loss_knots(Y), _stop_loss_knots(Y′)
    for ((x, y), q) in joint, (i, β) in enumerate(b), (j, β′) in enumerate(b′)
        knots = [U.knots; [β - x - U.offset - k for k in knots_Y]]
        append!(knots, [(β′ - y - k) / λ - U.offset for k in knots_Y′])
        function g(τ)
            return _uniform_density(U, τ) *
                   _stop_loss(Y, 1, β - x - U.offset - τ) *
                   _stop_loss(Y′, 1, β′ - y - λ * (U.offset + τ))
        end
        E[i, j] += q * _piecewise_integral(g, 0.0, U.width, knots, degree)
    end
    return E
end

# `E[f(η)]` over the independent laws `laws[ks]`, where `f` reads the coordinates of these laws in a
# full vector `η`. The discrete laws are enumerated first; the continuous ones are integrated in
# turn, each on the subintervals between the points where the combination of a lifting in `active`
# crosses a breakpoint, for the values of the outer coordinates and every vertex of the box of the
# inner ones: the integrand is then smooth on each subinterval.
function _expectation(f, laws, offsets, ks, active, liftings, inputs)
    discrete = [k for k in ks if _finite_atoms(laws[k]) !== nothing]
    continuous = [k for k in ks if !(k in discrete)]
    count = prod(Float64(length(_finite_atoms(laws[k])[2])) for k in discrete; init = 1.0)
    count > _MAX_COMBINATION_ATOMS && _too_many_atoms(count)
    η = fill(NaN, offsets[end] + _dimension(laws[end]))
    atoms = [_finite_atoms(laws[k]) for k in discrete]
    total = nothing
    for index in CartesianIndices(Tuple(length(p) for (_, p) in atoms))
        probability = 1.0
        for (a, k) in enumerate(discrete)
            A, p = atoms[a]
            η[offsets[k].+(1:_dimension(laws[k]))] = view(A, :, index[a])
            probability *= p[index[a]]
        end
        value = probability .* _integrate(f, η, laws, offsets, continuous, active, liftings, inputs)
        total = total === nothing ? value : total .+ value
    end
    return total
end

function _integrate(f, η, laws, offsets, continuous, active, liftings, inputs)
    isempty(continuous) && return f(η)
    k, inner = continuous[1], continuous[2:end]
    i = offsets[k] + 1
    d = laws[k]
    lo, hi = minimum(d), maximum(d)
    splits = Float64[]
    inner_positions = [offsets[j] + 1 for j in inner]
    for l in active
        lifting = liftings[l]
        a_i = sum((w for (p, w) in zip(inputs[l], lifting.coefficients) if p == i); init = 0.0)
        iszero(a_i) && continue
        # The part of the combination fixed by the outer coordinates, and the coefficients of the
        # inner ones.
        fixed, free = lifting.constant, Tuple{Int,Float64}[]
        for (p, w) in zip(inputs[l], lifting.coefficients)
            if p in inner_positions
                push!(free, (p, w))
            elseif p != i
                fixed += w * η[p]
            end
        end
        for vertex in Iterators.product(([0, 1] for _ in free)...), b in lifting.breakpoints
            rest = fixed
            for ((p, w), v) in zip(free, vertex)
                dj = laws[findlast(<(p), offsets)]
                rest += w * (v == 0 ? minimum(dj) : maximum(dj))
            end
            x = (b - rest) / a_i
            lo < x < hi && push!(splits, x)
        end
    end
    points = [lo; sort!(unique!(splits)); hi]
    value, _ = QuadGK.quadgk(points...; rtol = 1e-10) do x
        ηx = copy(η)
        ηx[i] = x
        return Distributions.pdf(d, x) .*
               _integrate(f, ηx, laws, offsets, inner, active, liftings, inputs)
    end
    return value
end

Base.length(d::_CombinationLaw) = length(d.mean)

Base.eltype(::Type{<:_CombinationLaw}) = Float64

Base.minimum(d::_CombinationLaw) = copy(d.lower)

Base.maximum(d::_CombinationLaw) = copy(d.upper)

Distributions.mean(d::_CombinationLaw) = copy(d.mean)

Distributions.cov(d::_CombinationLaw) = copy(d.covariance)

Distributions.var(d::_CombinationLaw) = LinearAlgebra.diag(d.covariance)

# The number of coordinates of `η`, before the pieces.
_num_original(d::_CombinationLaw) = d.offsets[end] + _dimension(d.laws[end])

# The bounds of the laws, and those of the pieces that the convex-hull rows do not imply (the lower
# bound of the last piece of each lifting; see `_hull_rows` of `CombinationPiecewise`).
function _box_rows(d::_CombinationLaw, lifted_box::Bool)
    lower, upper = Float64[], Float64[]
    for law in d.laws
        lo, hi = _box_rows(law, lifted_box)
        append!(lower, lo)
        append!(upper, hi)
    end
    for (l, (s_min, s_max)) in zip(d.liftings, d.ranges)
        lo, hi = _lifted_bounds(l, s_min, s_max)
        if !lifted_box
            lo, hi = fill(-Inf, length(lo)), fill(Inf, length(hi))
            lo[end] = 0.0
        end
        append!(lower, lo)
        append!(upper, hi)
    end
    return lower, upper
end

# The rows of the support of each law, on its coordinates.
function support_constraints(d::_CombinationLaw)
    n = length(d)
    blocks = [(support_constraints(law), d.offsets[k]) for (k, law) in enumerate(d.laws)]
    W = reduce(vcat, [_embed_rows(rows.W, offset, n) for (rows, offset) in blocks])
    h = reduce(vcat, [Vector{Float64}(rows.h) for (rows, _) in blocks])
    linear = sum(
        sum(MOI.dimension(c) for c in rows.cones if c isa MOI.Nonnegatives; init = 0) for
        (rows, _) in blocks
    )
    cones = _SupportCone[MOI.Nonnegatives(linear)]
    # The rows of the laws, in order: their linear rows first, as the cones of each law are.
    order = Int[]
    first = 0
    for (rows, _) in blocks
        offset = 0
        for c in rows.cones
            dim = MOI.dimension(c)
            c isa MOI.Nonnegatives && append!(order, first + offset .+ (1:dim))
            offset += dim
        end
        first += offset
    end
    first = 0
    for (rows, _) in blocks
        offset = 0
        for c in rows.cones
            dim = MOI.dimension(c)
            if c isa MOI.SecondOrderCone
                append!(order, first + offset .+ (1:dim))
                push!(cones, c)
            end
            offset += dim
        end
        first += offset
    end
    return (W = W[order, :], h = h[order], cones = cones)
end

function _embed_rows(W::AbstractMatrix, offset::Int, n::Int)
    E = zeros(size(W, 1), n)
    E[:, offset.+(1:size(W, 2))] = W
    return E
end

# The dual bound needs the conditional expectations of the later coordinates given the earlier
# ones to be linear (`rem:lce` of `docs/math`); `stages` holds the stages of the coordinates of
# `η`, then those of the pieces, each in the last stage of its inputs. From the stage of a
# combination on, its pieces are functions of the history, so they add nothing to the condition,
# which is then that of each law (the laws are independent): each law answers for itself. Before
# that stage, the conditional expectation of a piece is a nonlinear function of the earlier
# coordinates of the laws of its inputs, or a constant if these laws have none. So the condition:
# - fails for a combination of several stages (which `_check_combinations_across_stages` refuses);
# - is unknown when a law of the inputs of a combination has a coordinate in an earlier stage,
#   or when a law does not know its own;
# - fails when a law fails its own, and holds otherwise.
function _linear_conditional_expectations(d::_CombinationLaw, stages::AbstractVector)
    coordinates(k) = d.offsets[k] .+ (1:_dimension(d.laws[k]))
    η_stages = stages[1:(d.offsets[end]+_dimension(d.laws[end]))]
    result = true
    for inputs in d.inputs
        allequal(η_stages[inputs]) || return false
        t = η_stages[first(inputs)]
        for k in eachindex(d.laws)
            law = coordinates(k)
            if any(in(inputs), law) && any(<(t), η_stages[law])
                result = nothing
            end
        end
    end
    for k in eachindex(d.laws)
        value = _linear_conditional_expectations(d.laws[k], η_stages[coordinates(k)])
        value === false && return false
        value === nothing && (result = nothing)
    end
    return result
end

function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::_CombinationLaw,
    x::AbstractVector{<:Real},
)
    m = _num_original(d)
    for (k, law) in enumerate(d.laws)
        x[d.offsets[k].+(1:_dimension(law))] .= rand(rng, law)
    end
    for (l, positions) in enumerate(_piece_positions(m, d.liftings))
        x[positions] .= lift(d.liftings[l], view(x, d.inputs[l]))
    end
    return x
end

function Base.show(io::IO, d::_CombinationLaw)
    print(io, "_CombinationLaw(", d.laws, ", ", d.liftings, ")")
    return
end
