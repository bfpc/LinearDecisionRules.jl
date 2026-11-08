# Experimental

```@meta
CurrentModule = LinearDecisionRules
```

The `Experimental` submodule holds features that work and are tested, but whose interface may still change in a minor version. Try them, and tell us how they behave on your problems!

```@docs
Experimental
```

## Breakpoints on combinations of uncertainties

[`BreakPoints`](@ref) make a rule piecewise affine in one uncertainty. Sometimes the decisions react to a *combination* of uncertainties instead: the total demand served from one stock, the total inflow of a stage, the inflow of a reservoir over two stages. [`Experimental.set_breakpoints`](@ref) puts breakpoints on such a combination, and the rules become affine in the uncertainties plus piecewise affine in the combination.

Here, two products are sold from one stock, so the sales depend on the total demand:

```@example combination
using JuMP, HiGHS, Distributions, LinearDecisionRules
import LinearDecisionRules as LDR

function pooled_newsvendor()
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, d[1:2] in Uncertainty(Uniform(80, 120)), Stage(2))
    @variable(model, buy >= 0, Stage(1))
    @variable(model, sell[1:2] >= 0, Stage(2))
    @constraint(model, sell .<= d)
    @constraint(model, sum(sell) <= buy)
    @objective(model, Max, -10buy + 15sum(sell))
    return model
end

linear = pooled_newsvendor()
optimize!(linear)
objective_value(linear), objective_bound(linear)
```

With 3 breakpoints on the total demand, equally spaced on its range `[160, 240]`, the rule improves and the bound tightens around the optimal value, about 908.9:

```@example combination
model = pooled_newsvendor()
d = model[:d]
LDR.Experimental.set_breakpoints(model, d[1] + d[2], 3)
optimize!(model)
objective_value(model), objective_bound(model)
```

The pieces of the combination appear in the rules, as a [`CombinationTerm`](@ref):

```@example combination
decision_rule(model[:sell][1])
```

### What to expect

- **The primal rule** is always a feasible policy, with any combination.
- **The dual bound** is always certified (see [`is_certified`](@ref)), but it is as tight as without breakpoints on the combination only when the uncertainties of the combination are all in one stage and their laws have no uncertainty in an earlier stage. Otherwise, as for a combination over several stages such as the demands of two consecutive periods, the pieces have conditional expectations that are not linear in the earlier uncertainties, and [`DualLDR`](@ref) lets the rules of the earlier decisions use the later uncertainties and pieces of the combination. The bound stays valid, and is at least the bound without the combination, but it may improve little on it: the combination helps the primal rule more than the bound.
- **Laws:** the uncertainties of a combination must have laws that are discrete with finitely many atoms, or continuous and univariate (products of them are fine), and they cannot have [`BreakPoints`](@ref) themselves.
- **Few inputs work best.** The support of the pieces is an outer approximation of their convex hull, which gets looser as the combination has more uncertainties. On the hydrothermal benchmark, breakpoints on the inflow of each reservoir over two stages (2 inflows) cut the cost of the primal rule by 40% to 70%, while those on the total inflow of a stage (4 inflows) helped much less. On the inventory benchmark, breakpoints on the cumulative demand did not help at all.
- **Moments are fast for discrete and uniform laws.** They are exact, from the law of the sum, and take milliseconds even for many uniform uncertainties: the closed form for `d` uniform laws has `2^d` terms, which stays small up to a dozen or so. The discrete laws are enumerated, up to 10⁷ joint atoms; beyond that, `optimize!` stops with an error that says to use fewer such uncertainties. Other continuous laws are integrated over every continuous uncertainty of the combination, and so are uniform laws shared by two combinations whose coefficients in one are not proportional to those in the other (such as `d[1] + d[2]` and `d[1] - d[2]`), so combinations of more than 3 of them can take minutes before the solve starts; [`time_breakdown`](@ref) tells how long the moments took.
- **Chains of combinations** that share uncertainties, such as the pairs `(t - 1, t)` for every stage, merge their laws into one support block, which makes larger problems.

The math notes (a PDF linked from the [developer documentation](developer/index.md)) describe the lifting, its support and its moments in detail.

### Reference

```@docs
Experimental.set_breakpoints
CombinationPiecewise
CombinationTerm
CombinationBreakPoints
```
