# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    LiftedTerm(lifting::AbstractLifting, coefficients::Vector{Float64})

The term `coefficients ⋅ lift(lifting, d)` of a [`DecisionRule`](@ref) in an uncertainty `d` with
breakpoints. For an [`AxisPiecewise`](@ref) lifting, `coefficients[k]` is the slope of the rule in
`d` on the `k`-th piece, `[b_{k-1}, b_k]`, between the breakpoints (and the bounds of the
support). Call it with the value of `d`.
"""
struct LiftedTerm
    lifting::AbstractLifting
    coefficients::Vector{Float64}
end

(term::LiftedTerm)(η::Real) = LinearAlgebra.dot(term.coefficients, lift(term.lifting, η))

"""
    DecisionRule(
        constant::Float64,
        coefficients::Dict{JuMP.VariableRef,Float64},
        lifted::Dict{JuMP.VariableRef,LiftedTerm} = Dict{JuMP.VariableRef,LiftedTerm}(),
    )

The decision rule `x(η) = constant + Σ coefficients[d] * d + Σ lifted[d](d)`, over the
uncertainties `d` of the model. Build it with [`decision_rule`](@ref).

An uncertainty with [`BreakPoints`](@ref) is in `lifted`, with the coefficients of the rule on
its pieces (see [`LiftedTerm`](@ref)), so the rule is piecewise affine in it; the others are in
`coefficients`.

Call it with a scenario of the uncertainties of the model: a `Dict` of the value of each
uncertainty, or pairs.

## Example

```julia
rule = decision_rule(sell)
rule(Dict(d => 90.0))
rule(d => 90.0)
rule.lifted[d].coefficients  # the slope on each piece, if d has breakpoints
```
"""
struct DecisionRule
    constant::Float64
    coefficients::Dict{JuMP.VariableRef,Float64}
    lifted::Dict{JuMP.VariableRef,LiftedTerm}
end

function DecisionRule(constant::Float64, coefficients::Dict{JuMP.VariableRef,Float64})
    return DecisionRule(constant, coefficients, Dict{JuMP.VariableRef,LiftedTerm}())
end

function (rule::DecisionRule)(scenario::AbstractDict)
    value = rule.constant
    for (d, coefficient) in rule.coefficients
        value += coefficient * scenario[d]
    end
    for (d, term) in rule.lifted
        value += term(scenario[d])
    end
    return value
end

(rule::DecisionRule)(scenario::Pair...) = rule(Dict(scenario))

_by_index(terms) = sort!(collect(terms); by = p -> JuMP.index(p[1]).value)

function Base.show(io::IO, rule::DecisionRule)
    print(io, "DecisionRule: ", rule.constant)
    for (d, coefficient) in _by_index(rule.coefficients)
        print(io, coefficient < 0 ? " - " : " + ", abs(coefficient), " ", d)
    end
    for (d, term) in _by_index(rule.lifted)
        print(io, " + ", term.coefficients, " ⋅ lift(", term.lifting, ", ", d, ")")
    end
    return
end

"""
    decision_rule(x::JuMP.VariableRef; formulation::AbstractFormulation = PrimalLDR())

The decision rule of the variable `x` in the last solution of `formulation`, as a
[`DecisionRule`](@ref) over the uncertainties of the model. The rule of an uncertainty is itself,
and the rule of a parameter is its value.

The rule is affine in each uncertainty, except in those with [`BreakPoints`](@ref), in which it
is piecewise affine: the rule applies the lifting to the scenario.

The rule of [`DualLDR`](@ref) is not a policy: only its objective value is a bound.

## Example

```julia
optimize!(model)
rule = decision_rule(sell)
rule(Dict(d => 90.0))
```
"""
function decision_rule(x::JuMP.VariableRef; formulation::AbstractFormulation = PrimalLDR())
    model = JuMP.owner_model(x)
    rule = MOI.get(model, RuleCoefficients(formulation), x)
    uncertainties = MOI.get(model, UncertaintyVariables())
    if JuMP.index(x) in uncertainties
        return DecisionRule(0.0, Dict(x => 1.0))
    end
    map = MOI.get(model, UncertaintyLifting())
    # The column of ξ of each uncertainty that is not lifted.
    column(j) = map === nothing ? 1 + j : map.kept[1+j]
    coefficients = Dict{JuMP.VariableRef,Float64}(
        JuMP.VariableRef(model, v) => rule[column(j)] for
        (j, v) in enumerate(uncertainties) if column(j) > 0 && !iszero(rule[column(j)])
    )
    lifted = Dict{JuMP.VariableRef,LiftedTerm}()
    if map !== nothing
        for block in map.blocks
            c = rule[block.columns]
            all(iszero, c) && continue
            # `AxisPiecewise` lifts one uncertainty.
            d = JuMP.VariableRef(model, uncertainties[only(block.inputs)-1])
            lifted[d] = LiftedTerm(block.lifting, c .+ 0.0)
        end
    end
    return DecisionRule(rule[1] + 0.0, coefficients, lifted)
end

_a_variable(::Real) = nothing

_a_variable(x::JuMP.GenericVariableRef) = x

function _a_variable(expr::JuMP.GenericAffExpr)
    return isempty(expr.terms) ? nothing : first(keys(expr.terms))
end

function _a_variable(expr::JuMP.GenericQuadExpr)
    return isempty(expr.terms) ? _a_variable(expr.aff) : first(keys(expr.terms)).a
end

# `E[a ⋅ ξ  b ⋅ ξ]` with `E[ξ ξᵀ] = μ μᵀ + blockdiag(Σ)`.
function _product_moment(mom::Moments, a::Vector{Float64}, b::Vector{Float64})
    value = LinearAlgebra.dot(a, mom.mean) * LinearAlgebra.dot(b, mom.mean)
    for block in mom.blocks
        value += LinearAlgebra.dot(a[block.columns], block.covariance, b[block.columns])
    end
    return value
end

"""
    expected_value(expr; formulation::AbstractFormulation = PrimalLDR()) -> Float64

The exact expectation `E[expr]` of an affine or quadratic expression, for the decision rules of
`formulation`.

`value(expr)` plugs the expected value of each variable into `expr`, which is the same for an
affine expression but not for a product, such as `price * sell`, when `sell` depends on `price`.
The expectation of a product uses the second moments of the uncertainties.

## Example

```julia
optimize!(model)
expected_value(price * sell)
```
"""
function expected_value(
    expr::JuMP.AbstractJuMPScalar;
    formulation::AbstractFormulation = PrimalLDR(),
)
    x = _a_variable(expr)
    if x === nothing
        return Float64(JuMP.constant(expr))
    end
    model = JuMP.owner_model(x)
    mom = MOI.get(model, UncertaintyMoments())
    f = convert(MOI.ScalarQuadraticFunction{Float64}, JuMP.moi_function(expr))
    rules = Dict{MOI.VariableIndex,Vector{Float64}}()
    function rule(v)
        return get!(rules, v) do
            return MOI.get(model, RuleCoefficients(formulation), JuMP.VariableRef(model, v))
        end
    end
    value = f.constant
    for term in f.affine_terms
        value += term.coefficient * LinearAlgebra.dot(rule(term.variable), mom.mean)
    end
    for term in f.quadratic_terms
        # MOI counts half of a diagonal term.
        coefficient = term.variable_1 == term.variable_2 ? term.coefficient / 2 : term.coefficient
        a, b = rule(term.variable_1), rule(term.variable_2)
        value += coefficient * _product_moment(mom, a, b)
    end
    return value
end

expected_value(x::Real; formulation::AbstractFormulation = PrimalLDR()) = Float64(x)
