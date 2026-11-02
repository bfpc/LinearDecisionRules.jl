# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    LinearDecisionRules.Experimental

Features whose interface may change between minor versions (decision 9 of the design notes):

 * [`Experimental.set_breakpoints`](@ref): breakpoints on a combination of uncertainties, with the
   [`CombinationPiecewise`](@ref) lifting.
"""
module Experimental

import JuMP
import MathOptInterface as MOI

import ..LinearDecisionRules: CombinationBreakPoints, CombinationPiecewise

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
uncertainty in a combination cannot have [`BreakPoints`](@ref) itself. With uncertainties of
several stages, the combination is not supported by [`DualLDR`](@ref): the dual bound needs the
conditional expectations of the later stages given the earlier ones to be linear, which the pieces
are not.

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

end # module Experimental
