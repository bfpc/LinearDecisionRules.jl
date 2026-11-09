# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    LinearDecisionRules.Experimental

Features whose interface may change between minor versions:

 * [`Experimental.set_breakpoints`](@ref): breakpoints on a combination of uncertainties, with the
   [`CombinationPiecewise`](@ref) lifting.
 * [`Experimental.RescalingMode`](@ref): other ways to rescale the uncertainties, for
   [`RescaleUncertainties`](@ref)`()`.
"""
module Experimental

import JuMP
import MathOptInterface as MOI

import ..LinearDecisionRules:
    AxisPiecewise,
    CanonicalForm,
    CombinationBreakPoints,
    CombinationPiecewise,
    LiftedDistribution,
    Moments,
    Optimizer,
    RescaleUncertainties,
    _Rescaling,
    _rescaling,
    _support_box

"""
    Experimental.set_breakpoints(model::JuMP.Model, combination, points)

Set breakpoints on the affine `combination` of uncertainties of `model`, such as the total inflow
`sum(w[:, t])` of a stage: at `optimize!`, the pieces of the combination between the breakpoints
become coordinates of the rules, in addition to the uncertainties (see
[`CombinationPiecewise`](@ref)). The rules are then affine in the uncertainties plus piecewise
affine in the combination.

`points` is a vector of interior breakpoints of the combination, or an integer `n` for `n` equally
spaced breakpoints inside its range on the support. `nothing`, `0` or an empty vector remove the
breakpoints of the combination. Setting breakpoints on the same combination again replaces them.

The pieces are in the last stage of the uncertainties of the combination. The laws of these
uncertainties must be discrete with finitely many atoms, or continuous and univariate, and an
uncertainty in a combination cannot have [`BreakPoints`](@ref LinearDecisionRules.BreakPoints)
itself. With uncertainties of several stages, the conditional expectations of the pieces given the
earlier uncertainties are not linear, so [`DualLDR`](@ref LinearDecisionRules.DualLDR) lets the
rules of the earlier decisions use the later uncertainties and pieces of the combination: its
bound is certified, and at least the bound without the combination, but it may improve little on
it.

## Example

```julia
import LinearDecisionRules as LDR
@variable(model, w[1:4, t = 1:T] in Uncertainty(law), Stage(t))
for t in 1:T
    LDR.Experimental.set_breakpoints(model, sum(w[:, t]), 5)
end
```
"""
function set_breakpoints(model::JuMP.Model, combination::JuMP.AbstractJuMPScalar, points)
    f = convert(MOI.ScalarAffineFunction{Float64}, JuMP.moi_function(combination))
    f = MOI.Utilities.canonical(f)
    if !(points === nothing || points isa Integer || points isa AbstractVector{<:Real})
        throw(
            ArgumentError(
                "The breakpoints must be an integer (their number), a vector of numbers or " *
                "`nothing`; got $(repr(points)).",
            ),
        )
    end
    src = JuMP.backend(model)
    current = if CombinationBreakPoints() in MOI.get(src, MOI.ListOfModelAttributesSet())
        something(MOI.get(src, CombinationBreakPoints()), [])
    else
        []
    end
    entries =
        Pair{MOI.ScalarAffineFunction{Float64},Any}[g => p for (g, p) in current if !_same(g, f)]
    if !(
        points === nothing ||
        (points isa Integer && points == 0) ||
        (points isa AbstractVector && isempty(points))
    )
        push!(entries, f => (points isa Integer ? Int(points) : Vector{Float64}(points)))
    end
    MOI.set(src, CombinationBreakPoints(), entries)
    return
end

# Whether two canonical affine functions are the same combination.
function _same(f::MOI.ScalarAffineFunction{Float64}, g::MOI.ScalarAffineFunction{Float64})
    return f.constant == g.constant &&
           [(t.variable, t.coefficient) for t in f.terms] == [(t.variable, t.coefficient) for t in g.terms]
end

"""
    Experimental.RescalingMode

An enum of the ways to rescale the uncertainties, for
[`RescaleUncertainties`](@ref LinearDecisionRules.RescaleUncertainties)`()` in place of `true`
or `false`. Each uncertainty `η` is written `η = c + s ζ`; the modes differ in `c` and `s`:

 * `PER_PIECE`: what `true` does. `c` is the center and `s` the half-width of the box of the law
   of `η`, so that `ζ ∈ [-1, 1]`, and each piece of an uncertainty with
   [`BreakPoints`](@ref LinearDecisionRules.BreakPoints) is rescaled on its own interval, so that
   it is `-1` below its breakpoint.
 * `BEFORE_LIFTING`: as `PER_PIECE`, except for the pieces, which are rescaled as if the
   uncertainty were rescaled to `[-1, 1]` and then lifted: they are `0` below their breakpoint.
 * `HYBRID`: as `PER_PIECE` for the first piece, and each other piece in `[0, 1]`.
 * `SCALE_ONLY`: no centering, `c = 0`, and `s` the largest magnitude of the box of the law of
   `η` (of each piece, with breakpoints), so that `ζ` is in `[-1, 1]` but not centered.

A law without a bounded box gets its mean and standard deviation, or, with `SCALE_ONLY`, `c = 0`
and its root mean square. The rules, the values and the bounds are the same with every mode
(only the scaling of the inner problems changes), and `MOI.get` returns the mode that was set.
The modes are compared in `benchmark/scaling`: which one is fastest depends on the problem and
on the solver.

## Example

```julia
import LinearDecisionRules as LDR
set_attribute(model, LDR.RescaleUncertainties(), LDR.Experimental.BEFORE_LIFTING)
```
"""
@enum RescalingMode PER_PIECE BEFORE_LIFTING HYBRID SCALE_ONLY

function MOI.set(o::Optimizer, ::RescaleUncertainties, mode::RescalingMode)
    o.rescale = mode
    return
end

function _rescaling(form::CanonicalForm, mom::Moments, mode::RescalingMode)
    if mode == SCALE_ONLY
        return _scale_only(form, mom)
    end
    r = _rescaling(form, mom)
    if mode != PER_PIECE
        for law in form.laws
            if law.distribution isa LiftedDistribution{<:Any,AxisPiecewise}
                lo, hi = _support_box(law.distribution)
                _rescale_pieces!(r.center, r.scale, law.columns .- 1, lo, hi, mode)
            end
        end
    end
    return r
end

# The pieces of one uncertainty, the columns `η` of η, with the boxes `lo` and `hi`.
function _rescale_pieces!(center, scale, η, lo, hi, mode::RescalingMode)
    if !(all(isfinite, lo) && all(isfinite, hi))
        return
    elseif mode == BEFORE_LIFTING
        # The box of the uncertainty is [lo₁, hi₁ + Σᵢ₌₂ hiᵢ], the pieces after the first in [0, Δᵢ].
        s = (hi[1] - lo[1] + sum(hi[2:end])) / 2
        center[η[1]] = lo[1] + s
        center[η[2:end]] .= 0.0
        scale[η] .= s
    elseif mode == HYBRID
        center[η[2:end]] .= 0.0
        scale[η[2:end]] .= hi[2:end]
    end
    return
end

# No centering: each uncertainty divided by the largest magnitude of its box, or by its root mean
# square if the box is not bounded, or kept if that is zero or not finite.
function _scale_only(form::CanonicalForm, mom::Moments)
    m = length(form.uncertainties)
    scale = ones(m)
    for (law, block) in zip(form.laws, mom.blocks)
        lo, hi = _support_box(law.distribution)
        for (k, j) in enumerate(law.columns)
            size = max(abs(lo[k]), abs(hi[k]))
            if !(isfinite(size) && size > 0)
                size = sqrt(mom.mean[j]^2 + block.covariance[k, k])
            end
            if isfinite(size) && size > 0
                scale[j-1] = size
            end
        end
    end
    return _Rescaling(zeros(m), scale)
end

# The keyword `public` needs Julia 1.11.
if VERSION >= v"1.11"
    eval(
        Expr(
            :public,
            :set_breakpoints,
            :RescalingMode,
            :PER_PIECE,
            :BEFORE_LIFTING,
            :HYBRID,
            :SCALE_ONLY,
        ),
    )
end

end # module Experimental
