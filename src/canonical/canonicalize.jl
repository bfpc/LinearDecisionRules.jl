# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# `canonicalize`: the reader, the transforms of the laws, the checks and the liftings, in order.

"""
    canonicalize(
        src::MOI.ModelLike,
        info::ModelInfo;
        conic_support::Bool = true,
    ) -> CanonicalForm

Read the model `src` and its metadata `info` into a [`CanonicalForm`](@ref).

`src` is only read, so it can be `JuMP.backend(model)` or any MOI model. JuMP parameters
(`VariableIndex`-in-`Parameter`) alone in a row or in the objective are kept symbolic in the
constant column of `B` and in `f` (see [`ParameterTerms`](@ref)), and the inner problems carry
them; elsewhere, multiplied by a variable or in a row tagged `ValidInequality()` or `Truncate()`,
they are replaced by their values.

The number of stages is `T = max(2, largest declared stage)`; unmarked variables are in stage `T`,
and a single warning lists them when `T > 2`.

The rows tagged `Truncate()` condition the law on the set `C` where they hold, and bound Ξ:

 * a univariate law whose rows involve it alone, such as `100 ≤ d ≤ 110`, is truncated exactly:
   `Distributions.truncated(d, lo, hi)` for a continuous law, or the atoms in `[lo, hi]`, as a
   `DiscreteNonParametric`, for a discrete law with a finite support. Its rows are then dropped,
   since the box of the truncated law replaces them;
 * otherwise, the declarations coupled by rows tagged `Truncate()` (connected components, by
   union–find) share one law, in the place of the first. When all their laws have finitely many
   atoms, and their product at most a million, it is exact: a [`MvDiscreteNonParametric`](@ref)
   on the atoms of the product that satisfy the rows, with their probabilities renormalized (a
   declared [`ConditionedDistribution`](@ref) of such laws is replaced in the same way). Otherwise
   it is a `ConditionedDistribution`, whose moments are estimated by rejection sampling (see
   [`moments`](@ref)). The rows stay in Ξ.

Constraints in `MOI.SecondOrderCone` describe Ξ only: they must be over uncertainties and tagged
`ValidInequality()` or `Truncate()`, and they become [`ConicRow`](@ref)s. A law conditioned by such
a constraint is estimated by rejection sampling, unless its laws have finitely many atoms.

Last, the uncertainties with breakpoints (`info.breakpoints`) and the combinations of uncertainties
with breakpoints (`info.combinations`, from `Experimental.set_breakpoints`) are lifted (see
[`lift`](@ref)): the result is then a lifted form, whose field `lifting` maps the original
uncertainties to its columns.

A law whose representation of the support is chosen automatically, such as a
[`ConfidenceMvNormal`](@ref) with `support = :auto`, gets second-order cone rows (see
[`support_constraints`](@ref)) when `conic_support` is `true` and no decision is integer, and
linear rows otherwise; the optimizer sets `conic_support` to whether its inner optimizer supports
second-order cone constraints.

Errors name the variables and constraints involved, for: unsupported constraint types, uncertainty-
only rows not tagged `ValidInequality()` or `Truncate()`, second-order cone constraints with
decisions or without a tag, tagged equalities (Ξ must be full-dimensional), a truncation of
probability zero, bounds or integrality on an uncertainty, integer decisions after stage 1,
products of uncertainties and decisions in constraints (random recourse), quadratic constraints,
and equality rows with an uncertainty that none of their decisions observe.

## Example

```julia
model = Model()
@variable(model, buy >= 0)
@variable(model, sell >= 0)
@variable(model, d)
@constraint(model, sell <= buy)
@constraint(model, sell <= d)
@objective(model, Max, -10buy + 15sell)
import LinearDecisionRules as LDR
info = LDR.ModelInfo(;
    uncertainties = [LDR.UncertaintyDeclaration(Uniform(80, 120), [index(d)])],
    decision_stages = Dict(index(buy) => 1),
)
form = LDR.canonicalize(backend(model), info)
```

For an `LDRModel`, [`model_info`](@ref) reads `info` from the tags of the model.
"""
function canonicalize(src::MOI.ModelLike, info::ModelInfo; conic_support::Bool = true)
    return _canonicalize(src, info, Ref(0.0); conic_support)
end

# `canonicalize`, adding to `moment_seconds` the time of the moments of the lifted laws (see
# `_lift`).
function _canonicalize(
    src::MOI.ModelLike,
    info::ModelInfo,
    moment_seconds::Base.RefValue{Float64};
    conic_support::Bool = true,
)
    variables = MOI.get(src, MOI.ListOfVariableIndices())
    parameters, values = _parameter_list(src)
    parameter = Dict(zip(parameters, values))
    entries = _check_declarations(src, info, variables, parameter)
    layout = _layout(src, info, variables, parameters, values, entries)
    constraints = _read_constraints(src, info, layout)
    sense, objective = _read_objective(src, layout)
    form = _canonical_form(src, layout, constraints, sense, objective)
    form = _condition(src, _resolve_representations(form, conic_support))
    _check_valid_rows_at_mean(src, form)
    _check_observable_equalities(src, form)
    liftings = _breakpoint_liftings(src, form, info.breakpoints)
    combinations = _combination_liftings(src, form, info.combinations)
    isempty(liftings) && isempty(combinations) && return form
    return _lift(form, liftings, combinations, moment_seconds)
end
