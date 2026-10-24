# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestSize

using JuMP
using Test

import LinearDecisionRules
import LinearDecisionRules: DualLDR, PrimalLDR, moments

const LDR = LinearDecisionRules

include(joinpath(@__DIR__, "..", "oracles", "dense.jl"))
include(joinpath(@__DIR__, "problems.jl"))

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

# Inventory, T = 32 (33 stages, one block per stage). Measured: the block primal has 5 744
# variables, 2 734 constraints and 13 101 nonzeros against 10 096, 5 408 and 21 807 for the dense
# one; the structured dual 3 948, 6 956 and 17 575 against 1 648, 9 376 and 215 855 (the dense
# dual has no variables for the slacks nor for `u`).
function test_inventory()
    form = Problems.form(Problems.inventory(32))
    mom = moments(form)
    primal, dual = Model(), Model()
    LDR._build!(primal, PrimalLDR(), form, mom)
    LDR._build!(dual, DualLDR(), form, mom)
    primal, dual = model_size(primal), model_size(dual)
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
