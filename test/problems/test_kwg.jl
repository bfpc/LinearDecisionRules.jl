# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestKWG

using JuMP
using Test

import LinearDecisionRules as LDR

# The inventory of Kuhn, Wiesemann and Georghiou (2011), section 5: see `kwg.jl`.
include(joinpath(@__DIR__, "kwg.jl"))

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

function _solve(T; kwargs...)
    model = kwg_model(T; formulations = [LDR.PrimalLDR(), LDR.DualLDR()], kwargs...)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test LDR.is_certified(model)
    return objective_bound(model), objective_value(model)
end

# The bounds of the paper's Table 1, printed with one decimal, to their last digit.
function test_paper_table_1()
    for (T, (lb, ub)) in enumerate(KWG_TABLE_1)
        bound, value = _solve(T)
        @test bound ≈ lb atol = 0.1
        @test value ≈ ub atol = 0.1
    end
    return
end

# Without the capacity of each factory over the horizon, the dual bounds are the static ones of
# Bodur and Luedtke (2022), "Two-stage linear decision rules for multi-stage stochastic
# programming", Math. Program. 191: 1972.4, 8664.4 and 22764.8.
function test_without_cumulative_capacity()
    for (T, lb) in ((2, 1972.4), (5, 8664.4), (10, 22764.8))
        bound, _ = _solve(T; cumulative = false)
        @test bound ≈ lb atol = 0.05
    end
    return
end

# Breakpoints tighten both sides; the values were measured with this implementation.
function test_breakpoints()
    bound, value = _solve(10)
    bound1, value1 = _solve(10; breakpoints = 1)
    bound3, value3 = _solve(10; breakpoints = 3)
    @test bound < bound1 < bound3 < value3 < value1 < value
    @test bound1 ≈ 22821.327516 rtol = 1e-7
    @test value1 ≈ 23334.058981 rtol = 1e-7
    @test bound3 ≈ 22842.635252 rtol = 1e-7
    @test value3 ≈ 23238.002955 rtol = 1e-7
    return
end

end # module TestKWG

TestKWG.runtests()
