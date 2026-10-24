# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    DecisionRule(constant::Float64, coefficients::Dict{JuMP.VariableRef,Float64})

The affine decision rule `x(η) = constant + Σ coefficients[d] * d`, over the uncertainties `d` of
the model. Build it with [`decision_rule`](@ref).

Call it with a scenario, a `Dict` of the value of each uncertainty, or with pairs.

## Example

```julia
rule = decision_rule(sell)
rule(Dict(d => 90.0))
rule(d => 90.0)
```
"""
struct DecisionRule
    constant::Float64
    coefficients::Dict{JuMP.VariableRef,Float64}
end

function (rule::DecisionRule)(scenario::AbstractDict)
    value = rule.constant
    for (d, coefficient) in rule.coefficients
        value += coefficient * scenario[d]
    end
    return value
end

(rule::DecisionRule)(scenario::Pair...) = rule(Dict(scenario))

function Base.show(io::IO, rule::DecisionRule)
    print(io, "DecisionRule: ", rule.constant)
    for (d, coefficient) in sort!(collect(rule.coefficients); by = p -> JuMP.index(p[1]).value)
        print(io, coefficient < 0 ? " - " : " + ", abs(coefficient), " ", d)
    end
    return
end

"""
    decision_rule(x::JuMP.VariableRef; formulation::AbstractFormulation = PrimalLDR())

The decision rule of the variable `x` in the last solution of `formulation`, as a
[`DecisionRule`](@ref) over the uncertainties of the model. The rule of an uncertainty is itself,
and the rule of a parameter is its value.

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
    coefficients = Dict{JuMP.VariableRef,Float64}(
        JuMP.VariableRef(model, v) => rule[1+j] for
        (j, v) in enumerate(uncertainties) if !iszero(rule[1+j])
    )
    return DecisionRule(rule[1] + 0.0, coefficients)
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
