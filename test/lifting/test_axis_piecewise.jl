# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestAxisPiecewise

using Test

import LinearDecisionRules
import LinearDecisionRules: AbstractLifting, AxisPiecewise, lift, retraction
import Random

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

const LIFTINGS = [
    AxisPiecewise([100.0]),
    AxisPiecewise([110, 90, 100]),
    AxisPiecewise(collect(82.5:2.5:117.5)),
    AxisPiecewise([760 / 7]),
]

function test_construction()
    l = AxisPiecewise([110, 90, 100])
    @test l isa AbstractLifting
    @test l.breakpoints == [90.0, 100.0, 110.0]
    @test l == AxisPiecewise([90.0, 100.0, 110.0])
    @test hash(l) == hash(AxisPiecewise([90.0, 100.0, 110.0]))
    @test sprint(show, l) == "AxisPiecewise([90.0, 100.0, 110.0])"
    @test_throws ArgumentError AxisPiecewise(Float64[])
    @test_throws ArgumentError AxisPiecewise([1.0, NaN])
    @test_throws ArgumentError AxisPiecewise([1.0, Inf])
    @test_throws ArgumentError AxisPiecewise([1.0, 2.0, 1.0])
    return
end

# The pieces of the previous version (`UnivariatePieceWise._rand!`): the first one includes the
# minimum of the support.
function test_pieces()
    l = AxisPiecewise([90, 100, 110])
    @test lift(l, 85.0) == [85, 0, 0, 0]
    @test lift(l, 90.0) == [90, 0, 0, 0]
    @test lift(l, 104.0) == [90, 10, 4, 0]
    @test lift(l, 110.0) == [90, 10, 10, 0]
    @test lift(l, 118) == [90, 10, 10, 8]
    @test lift(l, [104.0]) == lift(l, 104.0)
    @test_throws DimensionMismatch lift(l, [1.0, 2.0])
    @test lift(AxisPiecewise([100]), 80) == [80, 0]
    return
end

function test_retraction_of_lift()
    rng = Random.Xoshiro(1)
    for l in LIFTINGS
        R = retraction(l)
        @test R isa LinearDecisionRules.SparseArrays.SparseMatrixCSC{Float64,Int}
        @test size(R) == (1, length(l.breakpoints) + 1)
        # Also outside the support: L extrapolates the first and the last pieces.
        samples = [60 .+ 80 .* rand(rng, 200); l.breakpoints]
        @test all(R * lift(l, η) ≈ [η] for η in samples)
    end
    return
end

# The lifted outcomes of the support satisfy the box and the hull rows.
function test_hull_rows_hold()
    rng = Random.Xoshiro(2)
    lower, upper = [80.0], [120.0]
    for l in LIFTINGS
        lo, hi = LinearDecisionRules._lifted_bounds(l, lower, upper)
        W, h = LinearDecisionRules._hull_rows(l, lower, upper)
        @test size(W) == (length(l.breakpoints), length(l.breakpoints) + 1)
        lifted = [lift(l, η) for η in [80 .+ 40 .* rand(rng, 500); 80.0; 120.0; l.breakpoints]]
        @test all(all(lo .- 1e-9 .<= η̃ .<= hi .+ 1e-9) for η̃ in lifted)
        @test all(all(W * η̃ .>= h .- 1e-9) for η̃ in lifted)
        # Every hull row is tight at both ends of the support.
        for η in (80.0, 120.0)
            @test W * lift(l, η) ≈ h atol = 1e-9
        end
    end
    return
end

# The hull separates the lifted path from the chords that cut its corners.
function test_hull_rows_cut()
    l = AxisPiecewise([100.0])
    W, h = LinearDecisionRules._hull_rows(l, [80.0], [120.0])
    @test W == [20.0 -20.0]
    @test h == [1600.0]
    lo, hi = LinearDecisionRules._lifted_bounds(l, [80.0], [120.0])
    @test lo == [80, 0] && hi == [100, 20]
    # `(80, 20)` is in the box, but not in the convex hull of the path (80,0)-(100,0)-(100,20).
    @test only(W * [80.0, 20.0]) < only(h)
    return
end

function test_breakpoints_outside_support()
    for points in ([80.0], [120.0], [70.0, 100.0], [100.0, 130.0])
        l = AxisPiecewise(points)
        @test_throws ArgumentError LinearDecisionRules._hull_rows(l, [80.0], [120.0])
        @test_throws ArgumentError LinearDecisionRules._lifted_bounds(l, [80.0], [120.0])
    end
    return
end

end # module TestAxisPiecewise

TestAxisPiecewise.runtests()
