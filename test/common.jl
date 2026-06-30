using Test
using Random
using Logging

using LinearDecisionRules

using JuMP
using Ipopt
using HiGHS
using Distributions
using LinearAlgebra

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end
