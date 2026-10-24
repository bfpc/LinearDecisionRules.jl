# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestDense

using JuMP
using Test

import HiGHS
import LinearDecisionRules: moments

include(joinpath(@__DIR__, "dense.jl"))
include(joinpath(@__DIR__, "..", "formulations", "problems.jl"))

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

# The oracle reproduces the known values (the integer newsvendor's dual is its MIP incumbent).
function test_known_values()
    for p in Problems.library()
        form = Problems.form(p)
        mom = moments(form)
        @testset "$(p.name)" begin
            primal = DenseOracle.solve(DenseOracle.build_primal, form, mom, HiGHS.Optimizer)
            dual = DenseOracle.solve(DenseOracle.build_dual, form, mom, HiGHS.Optimizer)
            @test primal.termination_status == MOI.OPTIMAL
            @test dual.termination_status == MOI.OPTIMAL
            isnan(p.primal) || @test Problems.matches(primal.objective_value, p.primal, p)
            isnan(p.dual) || @test Problems.matches(dual.objective_value, p.dual, p)
        end
    end
    return
end

# Without the row `E[s(ξ)] ≥ 0`, the dual of `Max 2x, x ≤ 3` would be unbounded.
function test_deterministic_dual()
    p = Problems.deterministic()
    form = Problems.form(p)
    dual = DenseOracle.solve(DenseOracle.build_dual, form, moments(form), HiGHS.Optimizer)
    @test dual.objective_value ≈ 6
    @test dual.X ≈ [3;;]
    return
end

end # module TestDense

TestDense.runtests()
