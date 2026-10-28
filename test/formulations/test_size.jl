# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestSize

using JuMP
using Test

import Clarabel
import LinearDecisionRules
import LinearDecisionRules: DualLDR, PrimalLDR, moments, solve_formulation

const LDR = LinearDecisionRules

include(joinpath(@__DIR__, "..", "oracles", "dense.jl"))
include(joinpath(@__DIR__, "..", "problems", "library.jl"))

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$name" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

# Inventory with a quadratic cost `0.01 Σ I_t² + 0.001 (Σ I_t)²`: the term `(Σ I_t)²` couples all
# the stocks, whose rules see up to T + 1 columns. The means of the decisions are variables, so
# that `(Xμ)ᵀP(Xμ)` has one term per nonzero of `P` and not one per pair of columns: for T = 16,
# 952 quadratic terms instead of 11 628, for 168 more nonzeros in the rows.
function quadratic_inventory(T::Int)
    p = Problems.inventory(T)
    I = [variable_by_name(p.model, "I[$t]") for t in 1:T]
    q = 0.01 * sum(I[t]^2 for t in 1:T) + 0.001 * sum(I)^2
    set_objective_function(p.model, objective_function(p.model) + q)
    return p
end

function test_quadratic_objective()
    T = 16
    form = Problems.form(quadratic_inventory(T))
    mom = moments(form)
    for f in (PrimalLDR(), DualLDR())
        model = Model()
        LDR._build!(model, f, form, mom)
        @test length(objective_function(model).terms) == 952
    end
    primal = solve_formulation(form, PrimalLDR(), mom, Clarabel.Optimizer)
    oracle = DenseOracle.solve(DenseOracle.build_primal, form, mom, Clarabel.Optimizer)
    @test isapprox(primal.objective_value, oracle.objective_value; rtol = 1e-6)
    dual = solve_formulation(form, DualLDR(), mom, Clarabel.Optimizer)
    oracle = DenseOracle.solve(DenseOracle.build_dual, form, mom, Clarabel.Optimizer)
    @test isapprox(dual.objective_value, oracle.objective_value; rtol = 1e-6)
    return
end

# Variables, affine constraints and their nonzeros.
function model_size(model::Model)
    nonzeros = 0
    for (F, S) in list_of_constraint_types(model)
        F <: AffExpr || continue
        for c in all_constraints(model, F, S)
            nonzeros += length(constraint_object(c).func.terms)
        end
    end
    constraints = num_constraints(model; count_variable_in_set_constraints = false)
    return (variables = num_variables(model), constraints = constraints, nonzeros = nonzeros)
end

# Inventory, T = 32 (33 stages, one block per stage, each a box). Measured: the block primal has
# 5 744 variables, 2 734 constraints and 13 101 nonzeros against 10 096, 5 408 and 21 807 for the
# dense one; the closed form on the boxes has one auxiliary per (row, box) instead of two
# multipliers, so 3 696 variables (as the hand-written model of `benchmark/timing.jl`), as many
# constraints and 12 109 nonzeros (one fewer per bound of an order). The structured dual has
# 3 948, 6 956 and 17 575 against 1 648, 9 376 and 215 855 (the dense dual has no variables for
# the slacks nor for `u`).
function test_inventory()
    form = Problems.form(Problems.inventory(32))
    mom = moments(form)
    closed, primal, dual = Model(), Model(), Model()
    LDR._build!(closed, PrimalLDR(), form, mom)
    LDR._build!(primal, PrimalLDR(; box_closed_form = false), form, mom)
    LDR._build!(dual, DualLDR(), form, mom)
    closed, primal, dual = model_size(closed), model_size(primal), model_size(dual)
    @test closed == (variables = 3696, constraints = 2734, nonzeros = 12109)
    dense_primal = model_size(first(DenseOracle.build_primal(form, mom)))
    dense_dual = model_size(first(DenseOracle.build_dual(form, mom)))
    @test primal == (variables = 5744, constraints = 2734, nonzeros = 13101)
    @test dense_primal == (variables = 10096, constraints = 5408, nonzeros = 21807)
    @test dual == (variables = 3948, constraints = 6956, nonzeros = 17575)
    @test dense_dual == (variables = 1648, constraints = 9376, nonzeros = 215855)
    return
end

end # module TestSize

TestSize.runtests()
