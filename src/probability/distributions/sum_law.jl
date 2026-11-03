# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The law of a sum of independent discrete and uniform terms, and its stop-loss transforms
# `E[(S - t)₊^q]`, `q = 1, 2`, exactly: the moments of `CombinationPiecewise` liftings over such
# laws (`docs/math/lifted-moments.tex`).

# The largest number of outcomes that the moments of combination liftings enumerate: the atoms of
# the law of a sum of discrete terms, or the joint atoms of the discrete laws that two liftings
# share (and, by quadrature, the joint atoms of the discrete laws of the inputs).
const _MAX_COMBINATION_ATOMS = 10^7

function _too_many_atoms(count::Real)
    return error(
        "The moments of a combination lifting enumerate $(Int(count)) outcomes, more than " *
        "$_MAX_COMBINATION_ATOMS: use fewer uncertainties with finitely many atoms in " *
        "combinations that share them.",
    )
end

# A term `a ⋅ η_K` of a sum, for one law `K`: discrete with finitely many values, or uniform on
# `[lower, lower + width]`.
struct _DiscreteTerm
    values::Vector{Float64}
    probabilities::Vector{Float64}
end

struct _UniformTerm
    lower::Float64
    width::Float64
end

# The law of `S = constant + Σ terms` for independent terms, built by `_SumLaw(constant, terms)`,
# as `S = offset + Z + Y`: `Z` is discrete, the convolution of the discrete terms with the atoms of
# equal values merged, and `Y = X₁ + … + X_d` is a sum of independent uniform laws `Xᵢ` on
# `[0, wᵢ]`, whose stop-loss transforms are piecewise polynomials with the knots
# `w_S = Σ_{i ∈ S} wᵢ` (`prop:uniform-sum` of `docs/math/lifted-moments.tex`).
struct _SumLaw
    offset::Float64
    # The atoms of `Z`, sorted, and their probabilities.
    atoms::Vector{Float64}
    probabilities::Vector{Float64}
    # The widths `wᵢ` of the uniform terms, `W = Σ wᵢ`, and the distinct sums `w_S`, sorted, with
    # the sum of `(-1)^|S|` over the subsets `S` of each; then the same in `BigFloat`, exactly.
    widths::Vector{Float64}
    width::Float64
    knots::Vector{Float64}
    signs::Vector{Float64}
    exact_knots::Vector{BigFloat}
    exact_signs::Vector{BigFloat}
end

const _EXACT_PRECISION = 256

function _SumLaw(constant::Float64, terms::AbstractVector)
    offset = constant
    atoms = Dict(0.0 => 1.0)
    widths = Float64[]
    for term in terms
        if term isa _UniformTerm
            offset += term.lower
            term.width > 0 && push!(widths, term.width)
            continue
        end
        count = Float64(length(atoms)) * length(term.values)
        count > _MAX_COMBINATION_ATOMS && _too_many_atoms(count)
        next = Dict{Float64,Float64}()
        for (z, p) in atoms, (v, q) in zip(term.values, term.probabilities)
            # `+ 0.0` turns `-0.0` into `0.0`, so that they are one key.
            key = z + v + 0.0
            next[key] = get(next, key, 0.0) + p * q
        end
        atoms = next
    end
    values = sort!(collect(keys(atoms)))
    knots, signs = _subset_sums(Float64, widths)
    exact_knots, exact_signs = setprecision(BigFloat, _EXACT_PRECISION) do
        return _subset_sums(BigFloat, widths)
    end
    return _SumLaw(
        offset,
        values,
        [atoms[z] for z in values],
        widths,
        foldl(+, widths; init = 0.0),
        knots,
        signs,
        exact_knots,
        exact_signs,
    )
end

# The distinct sums `w_S` of the subsets of `widths`, sorted, and the sum of `(-1)^|S|` over the
# subsets of each (zero sums of signs dropped), in the arithmetic of `T`.
function _subset_sums(::Type{T}, widths::Vector{Float64}) where {T}
    sums = Dict{T,T}(zero(T) => one(T))
    for w in widths
        next = copy(sums)
        for (k, c) in sums
            key = k + T(w)
            next[key] = get(next, key, zero(T)) - c
        end
        sums = next
    end
    filter!(entry -> !iszero(entry.second), sums)
    knots = sort!(collect(keys(sums)))
    return knots, [sums[k] for k in knots]
end

# The closed form of the uniform part `Y` of `law` at `τ`: `E[(τ - Y)₊^m]` for `m ≥ 0`, and the
# density of `Y` for `m = -1`,
#     m! / (m + d)! / ∏ wᵢ  Σ_S (-1)^|S| (τ - w_S)₊^(m + d),
# with `(m + d)! / m!` read as `(d - 1)!` for `m = -1`. The sum cancels when its terms are large
# against its value, as when the widths differ by orders of magnitude: it is then computed again
# in `BigFloat`, with the knots exact.
function _uniform_closed_form(law::_SumLaw, m::Int, τ::Float64)
    value, magnitude = _uniform_closed_form(Float64, law.knots, law.signs, law.widths, m, τ)
    if magnitude > 256 * abs(value)
        value = setprecision(BigFloat, _EXACT_PRECISION) do
            knots, signs = law.exact_knots, law.exact_signs
            return first(_uniform_closed_form(BigFloat, knots, signs, law.widths, m, τ))
        end
    end
    return Float64(value)
end

# The value and the sum of the absolute values of its terms. The terms are written relative to
# `W = Σ wᵢ`, so that the powers do not overflow.
function _uniform_closed_form(::Type{T}, knots, signs, widths, m::Int, τ::Float64) where {T}
    d = length(widths)
    W = foldl(+, (T(w) for w in widths); init = zero(T))
    scale = W^m
    for (i, w) in enumerate(widths)
        scale *= W / (max(m + i, 1) * T(w))
    end
    x = T(τ)
    total, magnitude = zero(T), zero(T)
    for (k, c) in zip(knots, signs)
        k < x || break
        term = c * ((x - k) / W)^(m + d)
        total += term
        magnitude += abs(term)
    end
    return scale * total, scale * magnitude
end

# `E[(Y - t)₊^q]`, `q = 1, 2`, for the uniform part `Y` of `law`, from the end of the range of `Y`
# nearer to `t`, where the closed form has fewer terms: `(Y - t)₊ = (W - t - (W - Y))₊` with
# `W - Y` distributed as `Y`, and `(Y - t)₊ = (Y - t) + (t - Y)₊`,
# `(Y - t)₊² = (Y - t)² - (t - Y)₊²`.
function _uniform_stop_loss(law::_SumLaw, q::Int, t::Float64)
    isempty(law.widths) && return max(-t, 0.0)^q
    W = law.width
    t >= W && return 0.0
    t >= W / 2 && return _uniform_closed_form(law, q, W - t)
    if q == 1
        return W / 2 - t + _uniform_closed_form(law, 1, t)
    end
    variance = sum(abs2, law.widths) / 12
    return variance + (W / 2 - t)^2 - _uniform_closed_form(law, 2, t)
end

# The density of the uniform part `Y` of `law` at `τ`, which is symmetric around `W / 2`.
function _uniform_density(law::_SumLaw, τ::Float64)
    W = law.width
    0 < τ < W || return 0.0
    return _uniform_closed_form(law, -1, min(τ, W - τ))
end

# `E[(S - t)₊^q]` for `S ~ law` and `q = 1, 2`: a sum over the atoms of `Z` of the stop-loss
# transforms of `Y`.
function _stop_loss(law::_SumLaw, q::Int, t::Float64)
    total = 0.0
    for (z, p) in zip(law.atoms, law.probabilities)
        u = t - law.offset - z
        u >= law.width && continue
        total += p * _uniform_stop_loss(law, q, u)
    end
    return total
end

# The knots of `t ↦ E[(S - t)₊^q]`, the values `offset + z + w_S`.
function _stop_loss_knots(law::_SumLaw)
    return [law.offset + z + k for z in law.atoms for k in law.knots]
end

# `∫_lo^hi g(x) dx` for `g` a polynomial of degree at most `degree` between consecutive `knots`:
# Gauss–Legendre on each piece, exact up to rounding.
function _piecewise_integral(g, lo::Float64, hi::Float64, knots, degree::Int)
    nodes, weights = QuadGK.gauss(cld(degree + 1, 2))
    points = sort!(unique!([lo; [k for k in knots if lo < k < hi]; hi]))
    total = 0.0
    for i in 1:length(points)-1
        half, middle = (points[i+1] - points[i]) / 2, (points[i+1] + points[i]) / 2
        for (x, w) in zip(nodes, weights)
            total += w * half * g(middle + half * x)
        end
    end
    return total
end
