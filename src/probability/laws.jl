# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The interface of a law, with the fallbacks for any `Distributions.Distribution`: its support (the
# box, `support_constraints`, the choice of representation, the membership test of a point), its
# atoms, its moments and the trait of the dual bound. The distributions of the package only add
# methods.

_dimension(::Distributions.UnivariateDistribution) = 1
_dimension(d::Distributions.MultivariateDistribution) = length(d)
_dimension(::Distributions.Distribution) = 0

# The pieces of the cone of a support block: the nonnegative orthant and second-order cones.
const _SupportCone = Union{MOI.Nonnegatives,MOI.SecondOrderCone}

# The bounds `(lower, upper)` of the support of a law, or `nothing` if the law does not define
# them. `applicable(minimum, d)` cannot tell: every multivariate law falls back on
# `Base.minimum(itr)`, which throws a `MethodError` for laws without bounds, such as `Dirichlet`.
function _support_bounds(d::Distributions.UnivariateDistribution)
    return Float64[minimum(d)], Float64[maximum(d)]
end

function _support_bounds(d::Distributions.MultivariateDistribution)
    try
        return Vector{Float64}(minimum(d)), Vector{Float64}(maximum(d))
    catch err
        err isa MethodError || rethrow()
        return nothing
    end
end

# The box of the support of a law; a bound that the law does not define is infinite, which keeps
# Ξ a superset of the support.
function _support_box(d::Distributions.Distribution)
    bounds = _support_bounds(d)
    if bounds === nothing
        n = _dimension(d)
        return fill(-Inf, n), fill(Inf, n)
    end
    return bounds
end

"""
    support_constraints(d::Distributions.Distribution) -> NamedTuple

Return the rows that describe the support of the law `d` beyond its box (`minimum` and
`maximum`), as `(W = W, h = h, cones = cones)`: the outcomes `η` of `d` satisfy `W η - h ∈ K`,
where `K = cones[1] × cones[2] × …` is a product of `MOI.Nonnegatives` and `MOI.SecondOrderCone`s
whose dimensions add up to `length(h)`, and `W` has one column per component of `d`.

The optimizer adds these rows to the support Ξ, so the formulations see them:
the robust counterparts of [`PrimalLDR`](@ref) hold on them, and the cone constraints of
[`DualLDR`](@ref) use them. They must hold on the support of `d`, so that the primal rule is
feasible and the dual bound valid; a tighter description gives a better rule and a tighter
bound. [`simulate`](@ref) and [`evaluate`](@ref) check given scenarios against them.

The default has no rows. Add a method for a law of your own whose support is not its box, as
[`ConfidenceMvNormal`](@ref) does for its ellipsoid (a second-order cone row, or a box along its
principal axes, by its `support`). A [`ConditionedDistribution`](@ref) gives the rows of its laws.

## Example

```julia
d = ConfidenceMvNormal([100.0, 80.0], [100.0 20.0; 20.0 64.0], 0.95; support = :ellipsoid)
rows = support_constraints(d)
rows.cones  # [MOI.SecondOrderCone(3)]
```
"""
function support_constraints(d::Distributions.Distribution)
    return _no_support_constraints(_dimension(d))
end

function _no_support_constraints(n::Int)
    return (W = zeros(0, n), h = Float64[], cones = _SupportCone[])
end

# Whether the representation of the support of `d` is chosen by `canonicalize` (`support = :auto`),
# and the law with the representation chosen, with second-order cones if `conic`.
_has_auto_support(::Distributions.Distribution) = false

_resolve_support(d::Distributions.Distribution, ::Bool) = d

# How far `v` is from `cone`, in the units of `v`: 0 if `v ∈ cone`.
function _cone_violation(::MOI.SecondOrderCone, v::AbstractVector)
    return max(0.0, LinearAlgebra.norm(view(v, 2:length(v))) - v[1])
end

_cone_violation(::MOI.Nonnegatives, v::AbstractVector) = max(0.0, -minimum(v; init = 0.0))

# The one rule for a point in a set `W x - h ∈ K`, such as the support Ξ: whether `v = W x - h` is
# outside `cone` beyond `tolerance` relative to the size of `h`, that is, `tolerance (1 + |h_r|)`
# for each row of `MOI.Nonnegatives` and `tolerance (1 + ‖h‖₂)` for a second-order cone.
# `evaluate` and `simulate` use the tolerance of the user, and the conditioning by constraints
# tagged `Truncate()` uses `_CONDITIONING_TOLERANCE`.
function _outside_cone(::MOI.Nonnegatives, v::AbstractVector, h::AbstractVector, tolerance::Real)
    return any(r -> v[r] < -tolerance * (1 + abs(h[r])), eachindex(v, h))
end

function _outside_cone(
    cone::MOI.SecondOrderCone,
    v::AbstractVector,
    h::AbstractVector,
    tolerance::Real,
)
    return _cone_violation(cone, v) > tolerance * (1 + LinearAlgebra.norm(h))
end

# The same for the product of `cones`, with `v` and `h` stacked in their order.
function _outside_cone(cones::AbstractVector, v::AbstractVector, h::AbstractVector, tolerance::Real)
    offset = 0
    for cone in cones
        r = offset .+ (1:MOI.dimension(cone))
        offset += MOI.dimension(cone)
        _outside_cone(cone, view(v, r), view(h, r), tolerance) && return true
    end
    return false
end

# Small enough to be rounding only, so that the atoms of a discrete law on the boundary of the set
# are kept.
const _CONDITIONING_TOLERANCE = 1e-9

# The atoms of a law with finitely many outcomes, as the columns of a matrix, and their
# probabilities, all positive; `nothing` for another law.
_finite_atoms(::Distributions.Distribution) = nothing

function _finite_atoms(d::Distributions.DiscreteUnivariateDistribution)
    # `support` throws for the laws with an infinite support, such as `Poisson`.
    (lo,), (hi,) = _support_box(d)
    isfinite(lo) && isfinite(hi) || return nothing
    atoms = try
        Distributions.support(d)
    catch err
        err isa MethodError || rethrow()
        nothing
    end
    atoms isa AbstractVector || return nothing
    positive = [a for a in atoms if Distributions.pdf(d, a) > 0]
    p = [Float64(Distributions.pdf(d, a)) for a in positive]
    return reshape(Vector{Float64}(positive), 1, :), p
end

_law_mean(d::Distributions.UnivariateDistribution) = [Float64(Distributions.mean(d))]
_law_mean(d::Distributions.MultivariateDistribution) = Vector{Float64}(Distributions.mean(d))

_law_covariance(d::Distributions.UnivariateDistribution) = fill(Float64(Distributions.var(d)), 1, 1)
_law_covariance(d::Distributions.MultivariateDistribution) = Matrix{Float64}(Distributions.cov(d))

# Whether the conditional expectation of the components of `d` in later stages, given those in
# earlier stages, is affine in the latter, when its components are in the stages `stages`: `true`,
# `false`, or `nothing` when it is not known. Separate laws are independent, so this is, law by
# law, condition (i) of the dual bound (Kuhn, Wiesemann and Georghiou, 2011, Section 4), which
# independent stages satisfy. A law in one stage has it trivially.
function _linear_conditional_expectations(d::Distributions.Distribution, stages::AbstractVector)
    return allequal(stages) ? true : nothing
end

_linear_conditional_expectations(::Distributions.UnivariateDistribution, ::AbstractVector) = true

# The components of a product of univariate laws are independent.
function _linear_conditional_expectations(
    ::Union{Distributions.Product,Distributions.ProductDistribution{1,0}},
    ::AbstractVector,
)
    return true
end

# The law of some components of a normal vector given the others is normal, with an affine mean.
_linear_conditional_expectations(::Distributions.AbstractMvNormal, ::AbstractVector) = true

const _ContinuousTruncated =
    Distributions.Truncated{<:Distributions.ContinuousUnivariateDistribution}

_law_mean(d::_ContinuousTruncated) = [_truncated_moments(d)[1]]

_law_covariance(d::_ContinuousTruncated) = fill(_truncated_moments(d)[2], 1, 1)

# Distributions has the moments of few truncated laws, such as the normal one; the others are by
# quadrature of the density of the truncated law.
function _truncated_moments(d::_ContinuousTruncated)
    try
        return Float64(Distributions.mean(d)), Float64(Distributions.var(d))
    catch err
        err isa MethodError || rethrow()
    end
    lo, hi = minimum(d), maximum(d)
    mean, _ = QuadGK.quadgk(x -> x * Distributions.pdf(d, x), lo, hi; rtol = 1e-12)
    var, _ = QuadGK.quadgk(x -> (x - mean)^2 * Distributions.pdf(d, x), lo, hi; rtol = 1e-12)
    return mean, var
end
