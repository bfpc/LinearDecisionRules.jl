# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestWholeModelChecks

using JuMP
using Test

import Distributions
import HiGHS
import LinearDecisionRules
import LinearDecisionRules:
    BreakPoints, DualLDR, FirstStage, LDRModel, MvDiscreteNonParametric, PrimalLDR, Uncertainty

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

# `min or max  q(x, y) + x + y` with `y >= d - x`, `d` uniform; `q` is given by the caller.
function quadratic_model(sense, q)
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x <= 2, FirstStage)
    @variable(model, 0 <= y <= 2)
    @variable(model, d in Uncertainty(Distributions.Uniform(0, 1)))
    @constraint(model, y >= d - x)
    @objective(model, sense, q(x, y) + x + y)
    return model
end

function form_of(model)
    info = LinearDecisionRules.model_info(backend(model))
    return LinearDecisionRules.canonicalize(backend(model), info)
end

# The dual bound needs `P ⪰ 0` for a minimization and `-P ⪰ 0` for a maximization.
function test_nonconvex_objective_warns()
    warning = (:warn, r"quadratic objective of this minimization is not convex.*decisions x, y")
    model = quadratic_model(MIN_SENSE, (x, y) -> x^2 - y^2)
    @test_logs warning match_mode = :any optimize!(model)
    warning = (:warn, r"of this maximization is not convex: the matrix -P .*decisions x, y")
    model = quadratic_model(MAX_SENSE, (x, y) -> x^2 + y^2)
    @test_logs warning match_mode = :any optimize!(model)
    # An indefinite coupling term, not visible on the diagonal.
    warning = (:warn, r"of this minimization is not convex: the matrix P .*decisions x, y")
    model = quadratic_model(MIN_SENSE, (x, y) -> x^2 + y^2 + 3x * y)
    @test_logs warning match_mode = :any optimize!(model)
    return
end

function test_convex_objective_does_not_warn()
    for (sense, q) in (
        (MIN_SENSE, (x, y) -> x^2 + y^2),
        (MAX_SENSE, (x, y) -> -x^2 - y^2),
        # Semidefinite and singular: (x - y)^2.
        (MIN_SENSE, (x, y) -> (x - y)^2),
        (MAX_SENSE, (x, y) -> -(x - y)^2),
        (MIN_SENSE, (x, y) -> 0.0 * x),
    )
        @test_logs optimize!(quadratic_model(sense, q))
    end
    return
end

function test_nonconvex_decisions_are_named()
    model = quadratic_model(MIN_SENSE, (x, y) -> x^2 - y^2)
    @test LinearDecisionRules._nonconvex_decisions(form_of(model)) == [1, 2]
    model = quadratic_model(MIN_SENSE, (x, y) -> y^2)
    @test LinearDecisionRules._nonconvex_decisions(form_of(model)) === nothing
    return
end

function discrete_model(law)
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    @variable(model, d[1:2] in Uncertainty(law))
    @constraint(model, sell <= buy)
    @constraint(model, sell <= d[1] + d[2])
    @objective(model, Max, sell - 0.5buy)
    return model
end

# A discrete law with fewer atoms than dimensions + 1 has a singular covariance.
function test_singular_law_warns()
    # Two atoms in two dimensions: the covariance has rank 1.
    law = MvDiscreteNonParametric([[1.0, 2.0], [2.0, 3.0]], [0.5, 0.5])
    warning = (:warn, r"second-moment matrix.*singular.*d\[1\], d\[2\] \(rank 1 of 2\)")
    model = discrete_model(law)
    @test_logs warning match_mode = :any optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    # The primal is exact, and the bound is not certified.
    @test [r.certified for r in unsafe_backend(model).results] == [true, false]
    @test endswith(
        raw_status(model, DualLDR()),
        "[not certified: the second-moment matrix is singular]",
    )
    return
end

function test_nonsingular_law_does_not_warn()
    law = MvDiscreteNonParametric([[1.0, 2.0], [2.0, 3.0], [2.5, 1.0]], [0.3, 0.3, 0.4])
    @test_logs optimize!(discrete_model(law))
    return
end

# A constant coordinate has zero variance.
function test_constant_coordinate_is_singular()
    mom = LinearDecisionRules.Moments(
        [1.0, 0.5, 2.0],
        [LinearDecisionRules.CovarianceBlock([2, 3], [0.0 0.0; 0.0 1.0])],
    )
    blocks = LinearDecisionRules._singular_blocks(mom)
    @test length(blocks) == 1 && blocks[1][2] == 1
    # Independent of the units of the uncertainties.
    for scale in (1e-6, 1.0, 1e6)
        C = scale^2 .* [1.0 0.5; 0.5 1.0]
        mom = LinearDecisionRules.Moments(
            [1.0, 0.0, 0.0],
            [LinearDecisionRules.CovarianceBlock([2, 3], C)],
        )
        @test isempty(LinearDecisionRules._singular_blocks(mom))
    end
    return
end

# A lifted discrete law needs more atoms than pieces: 3 atoms on 3 pieces span a plane in the space
# of the lifted coordinates, and the warning names them.
function test_lifted_law_with_few_atoms_warns()
    model = LDRModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell >= 0)
    law = Distributions.DiscreteNonParametric([80.0, 100.0, 120.0], [0.3, 0.4, 0.3])
    @variable(model, d in Uncertainty(law))
    @constraint(model, sell <= buy)
    @constraint(model, sell <= d)
    @objective(model, Max, sell - 0.5buy)
    set_attribute(d, BreakPoints(), [90.0, 110.0])
    @test_logs (:warn, r"singular.*\(rank 2 of 3\)") match_mode = :any optimize!(model)
    # With one breakpoint, 3 atoms are enough.
    set_attribute(d, BreakPoints(), [90.0])
    @test_logs optimize!(model)
    return
end

end # module TestWholeModelChecks

TestWholeModelChecks.runtests()
