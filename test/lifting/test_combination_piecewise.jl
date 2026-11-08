# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestCombinationPiecewise

using JuMP
using Test

import Distributions
import HiGHS
import LinearAlgebra
import LinearDecisionRules
import LinearDecisionRules:
    CombinationPiecewise,
    DualLDR,
    ModelInfo,
    MvDiscreteNonParametric,
    PrimalLDR,
    canonicalize,
    lift,
    moments,
    retraction,
    support_blocks

const LDR = LinearDecisionRules

include(joinpath(@__DIR__, "..", "problems", "library.jl"))
include(joinpath(@__DIR__, "..", "oracles", "dense.jl"))

include(joinpath(@__DIR__, "..", "common.jl"))
import .TestCommon: error_message

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

# The problem `p` with breakpoints on combinations, given as `expr => points` over its variables.
function with_combinations(p::Problems.Problem, combinations::Pair...)
    info = ModelInfo(;
        uncertainties = p.info.uncertainties,
        decision_stages = p.info.decision_stages,
        valid_rows = p.info.valid_rows,
        breakpoints = p.info.breakpoints,
        combinations = [
            convert(MOI.ScalarAffineFunction{Float64}, moi_function(f)) => points for
            (f, points) in combinations
        ],
    )
    return Problems.Problem(p.name, "", p.model, info, NaN, NaN)
end

function test_lift()
    l = CombinationPiecewise([1.0, 2.0], [10.0, 20.0, 30.0]; constant = -5.0)
    @test LDR._num_inputs(l) == 2 && LDR._num_lifted(l) == 3
    # s = -5 + 3 + 2 × 7 = 12: pieces (min(s, 10), 2, 0, 0) without the first.
    @test lift(l, [3.0, 7.0]) == [2.0, 0.0, 0.0]
    @test lift(l, [10.0, 20.0]) == [10.0, 10.0, 15.0]
    @test lift(l, [0.0, 0.0]) == zeros(3)
    @test iszero(retraction(l)) && size(retraction(l)) == (2, 3)
    @test LDR._keeps_inputs(l) && !LDR._keeps_inputs(LDR.AxisPiecewise([1.0]))
    @test l == CombinationPiecewise([1, 2], [30, 10, 20]; constant = -5)
    @test hash(l) == hash(CombinationPiecewise([1, 2], [30, 10, 20]; constant = -5))
    @test sprint(show, l) == "CombinationPiecewise([1.0, 2.0], [10.0, 20.0, 30.0]; constant = -5.0)"
    @test_throws DimensionMismatch lift(l, [1.0])
    @test_throws ArgumentError CombinationPiecewise([0.0, 0.0], [1.0])
    @test_throws ArgumentError CombinationPiecewise([1.0], Float64[])
    @test_throws ArgumentError CombinationPiecewise([1.0], [1.0, 1.0])
    @test_throws ArgumentError CombinationPiecewise([Inf], [1.0])
    return
end

# The rows of the outer approximation hold at lifted points of the support, and at the vertices of
# the lifted segment, where the combination is at a breakpoint.
function test_hull_rows_hold()
    l = CombinationPiecewise([1.0, -2.0], [-20.0, 0.0, 15.0]; constant = 3.0)
    lo, hi = [0.0, 0.0], [40.0, 30.0]
    s_min, s_max = 3.0 - 60.0, 3.0 + 40.0
    W_η, W_p, h = LDR._hull_rows(l, s_min, s_max)
    @test size(W_η) == (4, 2) && size(W_p) == (4, 3)
    holds(η) = all(W_η * η + W_p * lift(l, η) .>= h .- 1e-9)
    for x in range(lo[1], hi[1]; length = 41), y in range(lo[2], hi[2]; length = 31)
        @test holds([x, y])
    end
    # The pieces of a point violate the rows if they do not come from the combination.
    η = [20.0, 5.0]
    @test !all(W_η * η + W_p * (lift(l, η) .+ [1.0, 0.0, 0.0]) .>= h .- 1e-9)
    lo_p, hi_p = LDR._lifted_bounds(l, s_min, s_max)
    @test lo_p == zeros(3) && hi_p == [20.0, 15.0, 28.0]
    @test_throws ArgumentError LDR._hull_rows(l, -10.0, s_max)
    return
end

# `E[f(η)]` by the midpoint rule on a grid of `n` points per uniform coordinate.
function grid_expectation(f, lo, hi, n)
    h = (hi .- lo) ./ n
    total = nothing
    for index in CartesianIndices(Tuple(fill(n, length(lo))))
        η = [lo[i] + (index[i] - 0.5) * h[i] for i in eachindex(lo)]
        v = f(η)
        total = total === nothing ? v : total .+ v
    end
    return total ./ n^length(lo)
end

function grid_moments(lo, hi, liftings, inputs, n)
    m = length(lo)
    full(η) = vcat(η, [lift(l, η[i]) for (l, i) in zip(liftings, inputs)]...)
    μ = grid_expectation(full, lo, hi, n)
    M = grid_expectation(η -> (x = full(η); x * x'), lo, hi, n)
    return μ, M - μ * μ'
end

function test_moments_uniform()
    laws = [Distributions.Uniform(0, 40), Distributions.Uniform(0, 30)]
    l = CombinationPiecewise([1.0, -2.0], [-20.0, 0.0, 15.0]; constant = 3.0)
    d = LDR._CombinationLaw(laws, [l], [[1, 2]])
    @test length(d) == 5
    @test d.ranges == [(-57.0, 43.0)]
    μ, Σ = grid_moments([0.0, 0.0], [40.0, 30.0], [l], [[1, 2]], 1000)
    @test isapprox(d.mean, μ; rtol = 1e-5, atol = 1e-4)
    @test isapprox(d.covariance, Σ; rtol = 1e-5, atol = 1e-3)
    @test d.covariance[1:2, 1:2] ≈ [400/3 0; 0 75]
    return
end

# Two liftings that share the middle uncertainty are correlated, and one on another law is not.
function test_moments_overlapping()
    laws = [Distributions.Uniform(0, 10) for _ in 1:4]
    l1 = CombinationPiecewise([1.0, 1.0], [8.0, 12.0])
    l2 = CombinationPiecewise([1.0, 2.0], [15.0])
    l3 = CombinationPiecewise([1.0], [5.0])
    inputs = [[1, 2], [2, 3], [4]]
    d = LDR._CombinationLaw(laws, [l1, l2, l3], inputs)
    # The coordinates of the first three laws and of `l1`, `l2` (columns 5 to 7), on a 3-D grid.
    μ, Σ = grid_moments(zeros(3), fill(10.0, 3), [l1, l2], inputs[1:2], 200)
    keep = [1, 2, 3, 5, 6, 7]
    @test isapprox(d.mean[keep], μ; rtol = 1e-4, atol = 1e-4)
    @test isapprox(d.covariance[keep, keep], Σ; rtol = 1e-4, atol = 1e-3)
    # `l3` is `max(η₄ - 5, 0)`, independent of the others.
    @test d.mean[8] ≈ 1.25
    @test d.covariance[8, 8] ≈ 125 / 30 - 1.25^2
    # E[η₄ max(η₄ - 5, 0)] - E[η₄] E[max(η₄ - 5, 0)] = 125 / 12 - 25 / 4.
    @test d.covariance[4, 8] ≈ 125 / 12 - 25 / 4
    @test iszero(d.covariance[5:7, 8]) && iszero(d.covariance[8, 1:3])
    @test !iszero(d.covariance[5:6, 7])
    return
end

# A joint discrete law and a discrete univariate law, by plain enumeration of their atoms.
function test_moments_discrete()
    atoms = [[80.0, 30.0], [100.0, 60.0], [120.0, 40.0], [90.0, 70.0], [110.0, 20.0], [95.0, 45.0]]
    p = [0.1, 0.2, 0.3, 0.15, 0.15, 0.1]
    joint = MvDiscreteNonParametric(atoms, p)
    single = Distributions.DiscreteNonParametric([1.0, 2.0, 4.0], [0.5, 0.3, 0.2])
    l = CombinationPiecewise([1.0, 1.0, 10.0], [130.0, 150.0])
    d = LDR._CombinationLaw([joint, single], [l], [[1, 2, 3]])
    μ, M = zeros(5), zeros(5, 5)
    for (a, q) in zip(atoms, p), (x, r) in zip([1.0, 2.0, 4.0], [0.5, 0.3, 0.2])
        η = [a; x]
        v = [η; lift(l, η)]
        μ .+= q * r .* v
        M .+= q * r .* (v * v')
    end
    @test d.mean ≈ μ
    @test d.covariance ≈ M - μ * μ' atol = 1e-9
    # On the atoms of the joint law, the sum of its two coordinates ranges over [110, 160].
    @test d.ranges == [(110.0 + 10.0, 160.0 + 40.0)]
    return
end

# A discrete law and a uniform one in one combination: the uniform is integrated for each atom.
function test_moments_mixed()
    single = Distributions.DiscreteNonParametric([0.0, 5.0], [0.4, 0.6])
    uniform = Distributions.Uniform(0, 10)
    l = CombinationPiecewise([1.0, 1.0], [6.0, 9.0])
    d = LDR._CombinationLaw([single, uniform], [l], [[1, 2]])
    μ, M = zeros(4), zeros(4, 4)
    for (x, q) in zip([0.0, 5.0], [0.4, 0.6])
        μx, Σx = grid_moments(
            [0.0],
            [10.0],
            [CombinationPiecewise([1.0], [6.0 - x, 9.0 - x])],
            [[1]],
            20_000,
        )
        # The pieces of `x + y` are those of `y` with the breakpoints shifted by `x`.
        Mx = Σx + μx * μx'
        v = [x; μx]
        Vx = [x^2 x*μx'; x*μx Mx]
        μ .+= q .* v
        M .+= q .* Vx
    end
    @test isapprox(d.mean, μ; atol = 1e-6)
    @test isapprox(d.covariance, M - μ * μ'; atol = 1e-5)
    return
end

# The moments by laws of sums against those by enumeration and nested quadrature.
function both_moments(laws, liftings, inputs)
    laws = Vector{Distributions.Distribution}(laws)
    offsets = LDR._offsets(laws)
    m = offsets[end] + LDR._dimension(laws[end])
    sums = LDR._combination_moments(laws, offsets, m, liftings, inputs)
    quadrature = LDR._combination_moments(laws, offsets, m, liftings, inputs; quadrature = true)
    return sums, quadrature
end

function test_moments_by_laws_of_sums()
    U = Distributions.Uniform
    atoms = [[80.0, 30.0], [100.0, 60.0], [120.0, 40.0], [90.0, 70.0], [110.0, 20.0], [95.0, 45.0]]
    joint = MvDiscreteNonParametric(atoms, [0.1, 0.2, 0.3, 0.15, 0.15, 0.1])
    single = Distributions.DiscreteNonParametric([0.0, 1.0, 3.0], [0.2, 0.5, 0.3])
    pair(a, b; constant = 0.0) = CombinationPiecewise(a, b; constant)
    cases = [
        # Uniform laws, with a negative coefficient and a constant.
        (
            [U(0, 1), U(0, 2), U(1, 4)],
            [pair([1.0, 2.0, -1.0], [0.0, 1.5, 3.0]; constant = 0.5)],
            [[1, 2, 3]],
        ),
        # Widths of very different sizes, whose closed form is computed in BigFloat.
        ([U(0, 1e-3), U(0, 1e3)], [pair([1.0, 1.0], [400.0, 999.9995])], [[1, 2]]),
        # One coordinate of a joint law, whose other coordinate is not an input, and a uniform law.
        ([joint, U(0, 10)], [pair([1.0, 1.0], [95.0, 110.0])], [[1, 3]]),
        # Pairs that share one uniform law, and a lifting of another law.
        (
            [U(0, 10) for _ in 1:4],
            [pair([1.0, 1.0], [8.0, 12.0]), pair([1.0, 2.0], [15.0]), pair([1.0], [5.0])],
            [[1, 2], [2, 3], [4]],
        ),
        # Cumulative sums, which share two uniform laws with the same coefficients.
        (
            [U(0, 1), U(0, 2), U(0, 3)],
            [pair([1.0, 1.0], [1.0, 2.0]), pair([1.0, 1.0, 1.0], [2.5, 4.0])],
            [[1, 2], [1, 2, 3]],
        ),
        # Liftings that share two uniform laws with other coefficients: their covariance is by
        # quadrature.
        (
            [U(0, 1), U(0, 2), U(0, 3)],
            [pair([1.0, 2.0], [1.5]), pair([1.0, 1.0, 1.0], [3.0])],
            [[1, 2], [1, 2, 3]],
        ),
        # Liftings that share a joint law, with a discrete law and a uniform one.
        (
            [single, joint, U(0, 5)],
            [pair([1.0, 1.0], [100.0]), pair([1.0, 1.0, 1.0], [60.0])],
            [[1, 2], [3, 4, 2]],
        ),
    ]
    for (laws, liftings, inputs) in cases
        (μ, Σ), (μq, Σq) = both_moments(laws, liftings, inputs)
        @test isapprox(μ, μq; rtol = 1e-10, atol = 1e-10)
        @test isapprox(Σ, Σq; rtol = 1e-10, atol = 1e-9 * maximum(abs, Σq))
    end
    return
end

# Eight uniform demands, which nested quadrature would take hours for: the covariance of the
# pieces with the total `s = Σ ηⱼ`, from the integrals over each demand, agrees with the one from
# the law of `s` alone, since `E[s (s - b)₊] = E[(s - b)₊²] + b E[(s - b)₊]`.
function test_moments_many_uniforms()
    laws = [Distributions.Uniform(80 + 5i, 120 + 3i) for i in 1:8]
    l = CombinationPiecewise(ones(8), [850.0, 880.0, 910.0])
    d = LDR._CombinationLaw(laws, [l], [collect(1:8)])
    terms = [LDR._UniformTerm(Float64(u.a), Float64(u.b - u.a)) for u in laws]
    s = LDR._SumLaw(0.0, terms)
    Esh = [LDR._stop_loss(s, 2, x) + x * LDR._stop_loss(s, 1, x) for x in l.breakpoints]
    expected = LDR._pieces_of_hinges(3) * Esh - d.mean[9:11] * sum(d.mean[1:8])
    @test vec(sum(d.covariance[9:11, 1:8]; dims = 2)) ≈ expected rtol = 1e-10
    @test LinearAlgebra.isposdef(LinearAlgebra.Symmetric(d.covariance))
    return
end

function test_law_errors()
    l = CombinationPiecewise([1.0, 1.0], [1.0])
    normal = Distributions.MvNormal(zeros(2), [1.0 0; 0 1])
    @test_throws ArgumentError LDR._CombinationLaw([normal], [l], [[1, 2]])
    unbounded = Distributions.Normal()
    @test_throws ArgumentError LDR._CombinationLaw(
        [unbounded, Distributions.Uniform(0, 1)],
        [l],
        [[1, 2]],
    )
    return
end

# Draws of the law lift the draws of its laws, and their mean is close to the mean.
function test_rand()
    laws = [Distributions.Uniform(0, 10), Distributions.Uniform(0, 10)]
    l = CombinationPiecewise([1.0, 1.0], [8.0, 12.0])
    d = LDR._CombinationLaw(laws, [l], [[1, 2]])
    rng = LDR.Random.Xoshiro(1)
    X = reduce(hcat, [rand(rng, d) for _ in 1:20_000])
    @test all(X[3:4, s] == lift(l, X[1:2, s]) for s in axes(X, 2))
    @test isapprox(vec(sum(X; dims = 2)) ./ size(X, 2), d.mean; atol = 0.1)
    @test minimum(d) == [0.0, 0.0, 0.0, 0.0] && maximum(d) == [10.0, 10.0, 4.0, 8.0]
    return
end

# The form of the two-product newsvendor with breakpoints on the total demand.
function test_lifted_form_structure()
    p = Problems.newsvendor_two_products()
    demand = p.model[:demand]
    form = Problems.form(with_combinations(p, demand[1] + demand[2] => [180.0, 200.0, 220.0]))
    names = ["demand[1]", "demand[2]"]
    pieces = ["(demand[1] + demand[2])[piece $k]" for k in 2:4]
    @test form.uncertainty_names == [names; pieces]
    @test form.uncertainty_stages == fill(2, 5)
    law = only(form.laws)
    @test law.distribution isa LDR._CombinationLaw
    @test law.columns == 2:6
    block = only(form.lifting.blocks)
    @test block.inputs == [2, 3] && block.columns == [4, 5, 6]
    @test form.lifting.kept == [1, 2, 3]
    @test form.lifting.R == [LinearAlgebra.I(3) zeros(3, 3)]
    # The rows of the data are those of the original form, with zero columns for the pieces.
    original = Problems.form(p)
    @test form.B_u == [original.B_u zeros(size(original.B_u, 1), 3)]
    # The outer approximation: 4 rows of the hull; the bounds of the pieces with `lifted_box`, or
    # only the lower bound of the last piece without it.
    for (lifted_box, rows) in ((true, 4 + 4 + 6), (false, 4 + 4 + 1))
        blocks = support_blocks(form; lifted_box)
        @test length(blocks) == 1
        @test length(only(blocks).h) == rows
        @test count(==(:hull), only(blocks).roles) == 4
    end
    # The pieces of a combination of one uncertainty with breakpoints too are refused.
    both = with_combinations(p, demand[1] + demand[2] => 2)
    both = Problems.Problem(
        "",
        "",
        p.model,
        ModelInfo(;
            uncertainties = [
                Problems.declare(Distributions.Uniform(80, 120), demand[1]),
                Problems.declare(Distributions.Uniform(80, 120), demand[2]),
            ],
            decision_stages = p.info.decision_stages,
            breakpoints = Dict{MOI.VariableIndex,Union{Int,Vector{Float64}}}(index(demand[1]) => 1),
            combinations = both.info.combinations,
        ),
        NaN,
        NaN,
    )
    @test occursin("cannot be lifted itself", error_message(() -> Problems.form(both)))
    return
end

# A combination of one uncertainty spans the same rules as its `AxisPiecewise` lifting, and the
# outer approximation of its hull is exact, so both bounds are those of the breakpoints.
function test_single_input_is_axis()
    p = Problems.newsvendor()
    demand = p.model[:demand]
    for (k, (points, primal, dual)) in enumerate(Problems.NEWSVENDOR_BREAKPOINTS)
        q = with_combinations(p, 1.0 * demand => points)
        @test Problems.solve(q, PrimalLDR()).objective_value ≈ primal
        @test Problems.solve(q, DualLDR()).objective_value ≈ dual
        # A scaled and shifted combination has the same pieces.
        r = with_combinations(p, 2.0 * demand - 10.0 => 2 .* points .- 10)
        @test Problems.solve(r, PrimalLDR()).objective_value ≈ primal
        @test Problems.solve(r, DualLDR()).objective_value ≈ dual
    end
    return
end

# The structured formulations of a lifted form match the dense ones.
function test_dense_oracle()
    p = Problems.newsvendor_two_products()
    demand = p.model[:demand]
    form = Problems.form(with_combinations(p, demand[1] + 2demand[2] => 3))
    mom = moments(form)
    for (f, builder) in
        ((PrimalLDR(), DenseOracle.build_primal), (DualLDR(), DenseOracle.build_dual))
        structured = Problems.solve(form, f).objective_value
        oracle = DenseOracle.solve(builder, form, mom, HiGHS.Optimizer)
        @test structured ≈ oracle.objective_value rtol = 1e-7
    end
    return
end

# Two demands served from one stock: the best sales are `min(buy, d₁ + d₂)`, piecewise affine in
# the total demand, whose law is triangular on [160, 240].
function pooled_newsvendor()
    model = Model()
    @variable(model, buy >= 0)
    @variable(model, sell[1:2] >= 0)
    @variable(model, demand[1:2])
    @constraint(model, sell .<= demand)
    @constraint(model, sum(sell) <= buy)
    @objective(model, Max, -10buy + 15sum(sell))
    info = ModelInfo(;
        uncertainties = [Problems.declare(Distributions.Uniform(80, 120), demand[i]) for i in 1:2],
        decision_stages = Problems.stages(buy => 1),
    )
    return Problems.Problem("pooled_newsvendor", "", model, info, 800, 2900 / 3)
end

# The optimal stock is the critical quantile 1/3 of the total demand, `b = 160 + √(3200/3)`, where
# `E[min(b, d₁ + d₂)] = b - (b - 160)³ / 9600`.
const POOLED_STOCK = 160 + sqrt(3200 / 3)

const POOLED_OPTIMUM = -10POOLED_STOCK + 15 * (POOLED_STOCK - (POOLED_STOCK - 160)^3 / 9600)

# Breakpoints on the total demand tighten both bounds of the linear rules, nested breakpoints
# more; with a breakpoint at the optimal stock, the primal rule is optimal.
function test_bounds_tighten()
    p = pooled_newsvendor()
    demand = p.model[:demand]
    total = demand[1] + demand[2]
    nested = ([], [200.0], [180.0, 200.0, 220.0], collect(170.0:10.0:230.0))
    values = map(nested) do points
        q = isempty(points) ? p : with_combinations(p, total => points)
        return Problems.solve(q, PrimalLDR()).objective_value,
        Problems.solve(q, DualLDR()).objective_value
    end
    primal, dual = collect(first.(values)), collect(last.(values))
    @test primal ≈ [800, 900, 900, 907.8125]
    @test dual ≈ [2900 / 3, 950, 918.75, 912.8472222222224]
    @test all(primal .<= POOLED_OPTIMUM .+ 1e-7) && all(dual .>= POOLED_OPTIMUM .- 1e-7)
    q = with_combinations(p, total => [POOLED_STOCK])
    @test Problems.solve(q, PrimalLDR()).objective_value ≈ POOLED_OPTIMUM
    @test Problems.solve(q, DualLDR()).objective_value ≈ 954.5213797275389
    return
end

function test_errors()
    p = Problems.newsvendor_two_products()
    demand, buy = p.model[:demand], p.model[:buy]
    form(f, points) = Problems.form(with_combinations(p, f => points))
    @test occursin("not an uncertainty", error_message(() -> form(demand[1] + buy[1], 2)))
    @test occursin("nonzero coefficient", error_message(() -> form(0.0 * demand[1] + 1.0, 2)))
    # The errors of the lifting, of its law and of its hull name the combination.
    msg = error_message(() -> form(demand[1] + demand[2], [150.0]))
    @test occursin("The lifting of the combination (demand[1] + demand[2]) is invalid", msg)
    @test occursin("strictly inside the range", msg)
    msg = error_message(() -> form(demand[1] + demand[2], [120.0, 120.0]))
    @test occursin("The lifting of the combination (demand[1] + demand[2]) is invalid", msg)
    @test occursin("must be distinct", msg)
    @test occursin("nonnegative", error_message(() -> form(demand[1] + demand[2], -1)))
    # No breakpoint, no lifting.
    @test Problems.form(with_combinations(p, demand[1] + demand[2] => 0)).lifting === nothing
    return
end

# `(η₁, η₂, η₃)` with `η₁, η₂ ∈ {0, 1}` and `η₃ = η₁ + η₂ ± 1`: the conditional expectation of `η₃`
# given `η₁, η₂` is `η₁ + η₂`, and given `η₁` alone it is `η₁ + 1/2`.
const NOISY_SUM = MvDiscreteNonParametric(
    [[a, b, a + b + e] for a in 0.0:1.0 for b in 0.0:1.0 for e in (-1.0, 1.0)],
    fill(1 / 8, 8),
)

# The trait that the dual bound reads, for the merged law of the inputs of combinations: `stages`
# are those of the coordinates of the laws, and each piece is in the stage of its inputs.
function test_linear_conditional_expectations()
    function lce(d, stages)
        return LDR._linear_conditional_expectations(
            d,
            [
                stages
                [stages[first(c)] for (c, l) in zip(d.inputs, d.liftings) for _ in l.breakpoints]
            ],
        )
    end
    l = CombinationPiecewise([1.0, 1.0], [1.0])
    # A combination of the stage-2 coordinates of a law over stages 2 and 3 that has the property:
    # the pieces are known from stage 2 on, and nothing comes before them.
    d = LDR._CombinationLaw([NOISY_SUM], [l], [[1, 2]])
    @test lce(d, [2, 2, 3]) === true
    @test lce(d, [2, 2, 2]) === true
    # A law that fails it fails it with the pieces.
    square = MvDiscreteNonParametric(
        [[a, b, (a - b)^2 + e] for a in 0.0:1.0 for b in 0.0:1.0 for e in (-1.0, 1.0)],
        fill(1 / 8, 8),
    )
    @test lce(LDR._CombinationLaw([square], [l], [[1, 2]]), [2, 2, 3]) === false
    # A combination of the stage-3 coordinates of a law with a coordinate in stage 2: the expectation
    # of the pieces given `η₁` is not known to be affine.
    d = LDR._CombinationLaw([NOISY_SUM], [CombinationPiecewise([1.0, 1.0], [1.5])], [[2, 3]])
    @test lce(d, [2, 3, 3]) === nothing
    # The same with independent univariate laws (the components of a product): a constant.
    u = Distributions.Uniform(0, 1)
    d = LDR._CombinationLaw([u, u, u], [l], [[2, 3]])
    @test lce(d, [2, 3, 3]) === true
    # A combination of several stages.
    d = LDR._CombinationLaw([u, u], [l], [[1, 2]])
    @test lce(d, [2, 3]) === false
    return
end

end # module TestCombinationPiecewise

TestCombinationPiecewise.runtests()
