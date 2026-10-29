# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The interface of a law and its fallbacks for any `Distributions.Distribution`; the methods of the
# laws of the package are tested with each law.

module TestLaws

using Test

import Distributions
import LinearDecisionRules as LDR
import MathOptInterface as MOI

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

# A bound that a law does not define is infinite, so that Ξ stays a superset of its support.
function test_support_box()
    @test LDR._support_box(Distributions.Uniform(80, 120)) == ([80.0], [120.0])
    @test LDR._support_box(Distributions.Normal()) == ([-Inf], [Inf])
    # `Dirichlet` has no `minimum`: the box of its 3 components is unbounded.
    @test LDR._support_box(Distributions.Dirichlet(3, 1.0)) == (fill(-Inf, 3), fill(Inf, 3))
    return
end

# Other laws have no rows beyond their box.
function test_support_constraints_fallback()
    @test isempty(LDR.support_constraints(Distributions.Uniform(0, 1)).h)
    rows = LDR.support_constraints(Distributions.MvNormal([0.0, 0.0], [1.0 0.5; 0.5 1.0]))
    @test size(rows.W) == (0, 2) && isempty(rows.cones)
    @test !LDR._has_auto_representation(Distributions.Uniform(0, 1))
    return
end

# The tolerance of a point in a set is relative to the constant of each row, or of each cone.
function test_outside_cone()
    outside = LDR._outside_cone
    # Two rows `v ≥ 0` with `h = (0, 100)`: the second one may be violated by 101 × 1e-3.
    @test !outside(MOI.Nonnegatives(2), [0.0, -0.1], [0.0, 100.0], 1e-3)
    @test outside(MOI.Nonnegatives(2), [-0.01, 0.0], [0.0, 100.0], 1e-3)
    # A cone: `‖(3, 4)‖ - 4.9 = 0.1` against `1e-2 (1 + ‖h‖)`.
    cone = MOI.SecondOrderCone(3)
    @test !outside(cone, [4.9, 3.0, 4.0], [0.0, 0.0, 9.0], 1e-2)
    @test outside(cone, [4.9, 3.0, 4.0], [0.0, 0.0, 1.0], 1e-2)
    # A product of cones, in their order.
    @test outside([MOI.Nonnegatives(1), cone], [0.0, 4.9, 3.0, 4.0], [0.0, 0.0, 0.0, 1.0], 1e-2)
    return
end

# The atoms of positive probability of a discrete law with a finite support, and none otherwise.
function test_finite_atoms()
    atoms, p = LDR._finite_atoms(Distributions.Categorical([0.5, 0.0, 0.5]))
    @test atoms == [1.0 3.0] && p == [0.5, 0.5]
    @test LDR._finite_atoms(Distributions.Poisson(3.0)) === nothing
    @test LDR._finite_atoms(Distributions.Normal()) === nothing
    return
end

# The moments of a truncated continuous law, from Distributions or by quadrature.
function test_truncated_moments()
    half_normal = Distributions.truncated(Distributions.Normal(), 0.0, Inf)
    @test only(LDR._law_mean(half_normal)) ≈ sqrt(2 / π)
    @test only(LDR._law_covariance(half_normal)) ≈ 1 - 2 / π
    uniform = Distributions.truncated(Distributions.Uniform(0, 10), 2, 4)
    @test only(LDR._law_mean(uniform)) ≈ 3
    @test only(LDR._law_covariance(uniform)) ≈ 4 / 12
    return
end

end # module TestLaws

TestLaws.runtests()
