module DistributionsTests

include("common.jl")

### MvDiscreteNonParametric
function test_mv_discrete_params()
    d = LinearDecisionRules.MvDiscreteNonParametric(
        [[1.0, 2.0], [3.0, 4.0]],
        [0.4, 0.6],
    )
    @test Base.eltype(typeof(d)) == Float64
    xs, ps = Distributions.params(d)
    @test length(xs) == 2
    @test sum(ps) ≈ 1.0
    return nothing
end

function test_mv_discrete_stats()
    d = LinearDecisionRules.MvDiscreteNonParametric(
        [[3.0, 1.0], [1.0, 3.0]],
        [0.5, 0.5],
    )
    @test Distributions.minimum(d) == [1.0, 1.0]
    @test Distributions.var(d) == LinearAlgebra.diag(Distributions.cov(d))
    @test Distributions.insupport(d, [3.0, 1.0])
    @test !Distributions.insupport(d, [0.0, 0.0])

    # rand must return one of the two support points
    rng = Random.Xoshiro(42)
    for _ in 1:20
        x = rand(rng, d)
        @test x == [3.0, 1.0] || x == [1.0, 3.0]
    end

    # Integer eltype
    d_int = LinearDecisionRules.MvDiscreteNonParametric(
        [[1, 2], [3, 4], [5, 6]],
        [0.2, 0.3, 0.5],
    )
    @test Base.eltype(typeof(d_int)) == Int
    @test Distributions.length(d_int) == 2
    @test Distributions.minimum(d_int) == [1, 2]
    @test Distributions.maximum(d_int) == [5, 6]
    @test Distributions.insupport(d_int, [3, 4])
    @test !Distributions.insupport(d_int, [2, 3])

    rng2 = Random.Xoshiro(123)
    for _ in 1:20
        x = rand(rng2, d_int)
        @test x == [1, 2] || x == [3, 4] || x == [5, 6]
    end

    # Int32 eltype
    d_i32 = LinearDecisionRules.MvDiscreteNonParametric(
        [Int32[10, 20], Int32[30, 40]],
        [0.5, 0.5],
    )
    @test Base.eltype(typeof(d_i32)) == Int32
    @test Distributions.minimum(d_i32) == Int32[10, 20]
    @test Distributions.maximum(d_i32) == Int32[30, 40]

    rng3 = Random.Xoshiro(7)
    for _ in 1:20
        x = rand(rng3, d_i32)
        @test x == Int32[10, 20] || x == Int32[30, 40]
    end

    return nothing
end


### Piecewise linear lift of one-dimensional distributions
function test_univariate_piecewise_params()
    # Distributions.params should return original params plus break_points
    d = LinearDecisionRules.UnivariatePieceWise(Uniform(0.0, 1.0), [0.3, 0.7])
    ps = Distributions.params(d)
    @test length(ps) == 3  # (a, b) from Uniform + break_points
    @test ps[end] == [0.3, 0.7]
    return nothing
end

function test_univariate_piecewise_errors()
    @test_throws ArgumentError LinearDecisionRules.UnivariatePieceWise(
        Normal(0, 1),
        [0.0],
    )
    @test_throws ArgumentError LinearDecisionRules.UnivariatePieceWise(
        Uniform(0, 1),
        [-Inf],
    )
    d = LinearDecisionRules.UnivariatePieceWise(Uniform(0, 1), [0.3, 0.7])
    @test LinearDecisionRules._original(d) isa Uniform
    return nothing
end

function test_breakpoint_integer_on_vector_uncert()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    @variable(
        m,
        v[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = product_distribution([Uniform(0, 1), Uniform(0, 1)]),
        )
    )
    @test_throws ErrorException set_attribute(
        v[1],
        LinearDecisionRules.BreakPoints(),
        2,
    )
    return nothing
end

function test_univariate_piecewise_more_errors()
    # eltype mismatch: Float64 distribution, Float32 breakpoints
    @test_throws ArgumentError LinearDecisionRules.UnivariatePieceWise(
        Uniform(0.0, 1.0),
        Float32[0.3f0],
    )
    # max is +Inf (Exponential: min=0, max=Inf)
    @test_throws ArgumentError LinearDecisionRules.UnivariatePieceWise(
        Exponential(1),
        [0.5],
    )
    # breakpoint at +Inf
    @test_throws ArgumentError LinearDecisionRules.UnivariatePieceWise(
        Uniform(0, 1),
        [Inf],
    )
    return nothing
end

function test_univariate_piecewise_constructor_errors()
    o_dist = Distributions.Uniform(1.0, 4.0)

    # break point at or above upper bound
    @test_throws ArgumentError LinearDecisionRules.UnivariatePieceWise(
        o_dist,
        [3.5, 4.5],
    )
    # break point at or below lower bound
    @test_throws ArgumentError LinearDecisionRules.UnivariatePieceWise(
        o_dist,
        [0.5, 2.0],
    )
    # empty break_points
    @test_throws ArgumentError LinearDecisionRules.UnivariatePieceWise(
        o_dist,
        Float64[],
    )
    # NaN break point
    @test_throws ArgumentError LinearDecisionRules.UnivariatePieceWise(
        o_dist,
        [NaN],
    )

    return nothing
end

function test_piecewise_distribution()
    o_dist = Distributions.Uniform(1.0, 4.0)

    p_dist = LinearDecisionRules.UnivariatePieceWise(o_dist, [2.0, 3.0])

    len = length(p_dist)

    nu_mean = Distributions.mean(p_dist)
    nu_cov = Distributions.cov(p_dist)

    rng = Random.MersenneTwister(123)

    N = 1_000_000

    mc_mean = zeros(len)
    mc_cov = zeros(len, len)
    x = zeros(len)
    for i in 1:N
        fill!(x, 0.0)
        Random.rand!(rng, p_dist, x)
        mc_mean .+= x
        mc_cov .+= x * x'
    end
    mc_mean ./= N
    mc_cov ./= N
    mc_cov .-= mc_mean * mc_mean'

    # @show nu_mean
    # @show mc_mean
    # @show nu_cov
    # @show mc_cov

    @test nu_mean ≈ mc_mean atol = 1e-2
    @test nu_cov ≈ mc_cov atol = 1e-2

    return
end

function test_break_points_getter()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(
        m,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1)),
    )

    # before setting: returns nothing
    @test get_attribute(demand, LinearDecisionRules.BreakPoints()) === nothing

    # after setting: returns the vector
    set_attribute(demand, LinearDecisionRules.BreakPoints(), [0.5])
    @test get_attribute(demand, LinearDecisionRules.BreakPoints()) == [0.5]

    # after clearing: returns nothing again
    set_attribute(demand, LinearDecisionRules.BreakPoints(), nothing)
    @test get_attribute(demand, LinearDecisionRules.BreakPoints()) === nothing

    return nothing
end

function test_break_points_errors_extended()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        s in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @variable(
        m,
        v[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = product_distribution([Uniform(0, 1), Uniform(0, 1)]),
        )
    )

    @test_throws ErrorException set_attribute(
        x,
        LinearDecisionRules.BreakPoints(),
        [0.5],
    )
    @test_throws ErrorException set_attribute(
        x,
        LinearDecisionRules.BreakPoints(),
        nothing,
    )
    @test_throws ErrorException set_attribute(
        x,
        LinearDecisionRules.BreakPoints(),
        2,
    )

    @test_throws ErrorException set_attribute(
        v[1],
        LinearDecisionRules.BreakPoints(),
        [0.5],
    )

    @test_throws ErrorException set_attribute(
        s,
        LinearDecisionRules.BreakPoints(),
        Float64[],
    )
    @test_throws ErrorException set_attribute(
        s,
        LinearDecisionRules.BreakPoints(),
        0,
    )
    return nothing
end

function test_breakpoints_getter_and_vector_nothing()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        s in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @variable(
        m,
        v[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = product_distribution([Uniform(0, 1), Uniform(0, 1)]),
        )
    )
    @test_throws ErrorException get_attribute(
        x,
        LinearDecisionRules.BreakPoints(),
    )
    @test_throws ErrorException get_attribute(
        v[1],
        LinearDecisionRules.BreakPoints(),
    )
    @test_throws ErrorException set_attribute(
        v[1],
        LinearDecisionRules.BreakPoints(),
        nothing,
    )
    return nothing
end

function test_get_decision_pwl_piece_errors()
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)
    @variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell >= 0)
    @variable(
        ldr,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(80.0, 120.0)),
    )
    @constraint(ldr, sell <= buy)
    @constraint(ldr, sell <= demand)
    @objective(ldr, Max, -10buy + 15sell)
    set_attribute(demand, LinearDecisionRules.BreakPoints(), 2)
    optimize!(ldr)

    @test_throws ErrorException LinearDecisionRules.get_decision(
        ldr,
        sell,
        demand,
    )
    @test_throws ErrorException LinearDecisionRules.get_decision(
        ldr,
        sell,
        demand;
        piece = 10,
    )
    return nothing
end

function test_vector_uncertainty_with_pwl()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, buy >= 0, LinearDecisionRules.FirstStage)
    @variable(m, sell >= 0)
    @variable(
        m,
        scalar_d in
        LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1)),
    )
    @variable(
        m,
        vec_d[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = product_distribution([Uniform(0, 1), Uniform(0, 1)]),
        )
    )
    @constraint(m, sell <= buy)
    @constraint(m, sell <= scalar_d + vec_d[1])
    @objective(m, Min, buy - sell)
    set_attribute(scalar_d, LinearDecisionRules.BreakPoints(), 2)
    optimize!(m)
    @test termination_status(m) == MOI.OPTIMAL
    return nothing
end



function test_newsvendor_piecewise()
    buy_cost = 10
    return_value = 8
    sell_value = 15

    demand_max = 120
    demand_min = 80
    demand_distr = Distributions.Uniform(demand_min, demand_max)

    # LDR

    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)
    # unset_silent(ldr)

    @variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell >= 0)
    @variable(ldr, ret >= 0)
    @variable(
        ldr,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = demand_distr)
    )

    @constraint(ldr, sell + ret <= buy)

    @constraint(ldr, sell <= demand)

    @objective(
        ldr,
        Max,
        -buy_cost * buy + return_value * ret + sell_value * sell
    )

    optimize!(ldr)

    ldr_p_obj = Float64[]
    push!(ldr_p_obj, objective_value(ldr))
    ldr_d_obj = Float64[]
    push!(ldr_d_obj, objective_value(ldr; dual = true))

    @test ldr_p_obj[] <= ldr_d_obj[] + 1e-6

    for n_intervals in 2:12
        set_attribute(
            demand,
            LinearDecisionRules.BreakPoints(),
            n_intervals - 1,
        )
        optimize!(ldr)
        push!(ldr_p_obj, objective_value(ldr))
        push!(ldr_d_obj, objective_value(ldr; dual = true))
        @test ldr_p_obj[end] <= ldr_d_obj[end] + 1e-6
        for j in 1:n_intervals÷2
            # Divisibility for LDR improvement
            if n_intervals % j != 0
                continue
            end
            @test ldr_p_obj[end] >= ldr_p_obj[j] - 1e-6
            @test ldr_d_obj[end] <= ldr_d_obj[j] + 1e-6
        end
    end

    return
end

### Confidence MVNormal
function test_confidence_mv_normal()
    # --- Unit tests on the distribution object ---
    μ = [1.0, 2.0]
    Σ = [1.0 0.5; 0.5 2.0]
    α = 0.90
    d = LinearDecisionRules.ConfidenceMvNormal(μ, Σ, α)

    @test Distributions.length(d) == 2
    @test Distributions.mean(d) ≈ μ

    # Covariance should be a positive scalar multiple of Σ
    cov_d = Distributions.cov(d)
    ratio = cov_d ./ Σ
    @test all(isapprox.(ratio, ratio[1, 1]; rtol = 1e-10))
    # Scaling must be < 1 (truncation reduces variance)
    @test ratio[1, 1] < 1.0
    @test ratio[1, 1] > 0.0

    # Bounds: μ_k ± ρ·√Σ_kk
    ρ = sqrt(Distributions.quantile(Distributions.Chisq(2), α))
    @test Distributions.minimum(d) ≈ μ - ρ .* sqrt.(LinearAlgebra.diag(Σ))
    @test Distributions.maximum(d) ≈ μ + ρ .* sqrt.(LinearAlgebra.diag(Σ))

    # insupport: centre is always in the ellipsoid
    @test Distributions.insupport(d, μ)
    # A point far outside should not be in support
    @test !Distributions.insupport(d, μ + [1000.0, 1000.0])

    # Sampling: empirical mean should be close to μ
    rng = Random.MersenneTwister(42)
    N = 50_000
    samples = [rand(rng, d) for _ in 1:N]
    emp_mean = sum(samples) / N
    @test norm(emp_mean - μ) < 0.05

    # --- 1-D case: compare with analytic truncated normal ---
    # For d=1, Σ=[σ²], α gives ρ = σ·z_{(1+α)/2}
    σ = 2.0
    d1 = LinearDecisionRules.ConfidenceMvNormal([0.0], [σ^2;;], 0.95)
    trunc_dist = truncated(Normal(0.0, σ), -d1.ρ * σ, d1.ρ * σ)
    @test Distributions.var(d1)[1] ≈ Distributions.var(trunc_dist) rtol = 1e-4
    @test Distributions.params(d)[1] == μ

    # --- Integration with LDR: 2-D newsvendor ---
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)
    set_attribute(ldr, LinearDecisionRules.SolveDual(), false)

    μ_d = [100.0, 80.0]
    Σ_d = [100.0 20.0; 20.0 64.0]
    dist_ldr = LinearDecisionRules.ConfidenceMvNormal(μ_d, Σ_d, 0.95)

    @variable(ldr, buy[1:2] >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell[1:2] >= 0)
    @variable(
        ldr,
        demand[1:2] in
        LinearDecisionRules.Uncertainty(; distribution = dist_ldr)
    )

    @constraint(ldr, [i = 1:2], sell[i] <= buy[i])
    @constraint(ldr, [i = 1:2], sell[i] <= demand[i])
    @objective(ldr, Max, sum(-10 * buy[i] + 15 * sell[i] for i in 1:2))

    optimize!(ldr)
    @test termination_status(ldr) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    return nothing
end

function test_confidence_mv_normal_bounds()
    # The ellipsoid in original space is {x : (x-μ)'Σ⁻¹(x-μ) ≤ ρ²}.
    # Via Σ = LL', this is the image of the ball B(0,ρ) under x = μ + Lz.
    # The bounding box is lb[k] = μ[k] - ρ√Σ_kk, ub[k] = μ[k] + ρ√Σ_kk.
    # We test two properties:
    #   1. Containment: every point in the ellipsoid lies within [lb, ub].
    #   2. Tightness:   each face of the box is touched by the ellipsoid.
    test_cases = [
        ([0.0, 0.0], [1.0 0.0; 0.0 1.0], 0.95),           # identity covariance
        ([100.0, 80.0], [100.0 40.0; 40.0 64.0], 0.95),    # correlated 2-D
        ([1.0, 2.0, 3.0], [2.0 0.5 0.1; 0.5 1.0 0.3; 0.1 0.3 1.5], 0.80),  # 3-D
    ]
    for (μ_tc, Σ_tc, α) in test_cases
        dist = LinearDecisionRules.ConfidenceMvNormal(μ_tc, Σ_tc, α)
        lb = Distributions.minimum(dist)
        ub = Distributions.maximum(dist)
        n = length(μ_tc)
        ρ = dist.ρ
        L = dist.L

        # --- Containment check 1: deterministic grid ---
        # Sweep a uniform grid over the unit ball in z-space (‖z‖ ≤ 1),
        # scale by ρ, then map to x = μ + L(ρz).  Every resulting x must
        # lie inside [lb, ub].
        grid = range(-1, 1; length = 30)
        violations = 0
        for z_unit in Iterators.product(fill(grid, n)...)
            z = collect(z_unit)
            norm(z) > 1 && continue   # keep only points inside the unit ball
            x = μ_tc + L * (ρ .* z)
            if any(x .< lb .- 1e-10) || any(x .> ub .+ 1e-10)
                violations += 1
            end
        end
        @test violations == 0

        # --- Containment check 2: random ellipsoid boundary ---
        # Draw a random unit vector, scale to ‖z‖ = ρ (boundary of the ball),
        # then map to x = μ + Lz.  Boundary points are the hardest cases for
        # the box constraint, so this is a strong probabilistic check.
        rng = Random.MersenneTwister(42)
        violations_sampled = 0
        for _ in 1:1000
            z = randn(rng, n)
            z .*= ρ / norm(z)   # scale to ellipsoid boundary
            x = μ_tc + L * z
            if any(x .< lb .- 1e-10) || any(x .> ub .+ 1e-10)
                violations_sampled += 1
            end
        end
        @test violations_sampled == 0

        # --- Containment check 3: samples from the distribution ---
        # rand(dist) uses rejection sampling internally (draw z ~ N(0,I),
        # accept if ‖z‖² ≤ ρ², return μ + Lz).  Samples are strictly inside
        # the ellipsoid, so they must also be inside [lb, ub].
        violations_dist = 0
        for _ in 1:1000
            x = rand(rng, dist)
            if any(x .< lb .- 1e-10) || any(x .> ub .+ 1e-10)
                violations_dist += 1
            end
        end
        @test violations_dist == 0

        # --- Tightness: each bound is achieved by an analytic tangent point ---
        # To show ub[k] = μ[k] + ρ√Σ_kk is tight we need a point on the
        # ellipsoid boundary whose k-th coordinate equals ub[k].
        #
        # Maximise eₖ'x = eₖ'(μ + Lz) subject to ‖z‖ = ρ.
        # The maximum of a linear function on a sphere is achieved in the
        # direction of the gradient, so z* = ρ · L'eₖ / ‖L'eₖ‖.
        #
        # The achieved value is:
        #   x*[k] = μ[k] + eₖ'L z* = μ[k] + ρ · eₖ'LL'eₖ / ‖L'eₖ‖
        #         = μ[k] + ρ · ‖L'eₖ‖ = μ[k] + ρ√Σ_kk = ub[k].
        for k in 1:n
            ek = zeros(n)
            ek[k] = 1.0
            Lt_ek = L' * ek                        # gradient direction in z-space
            z_star = ρ * Lt_ek / norm(Lt_ek)       # tangent point on ball boundary
            x_star = μ_tc + L * z_star             # map back to original space
            @test x_star[k] ≈ ub[k] rtol = 1e-10
        end
    end
    return nothing
end

function test_confidence_mv_normal_rotated_box()
    # The rotated (principal-axis) box gives another outer approximation of the
    # ellipsoidal support than the axis-aligned box.  This test verifies that:
    #   1. No rejection-sampling warning is emitted.
    #   2. Both primal and dual are solved successfully.
    #   3. For a correlated distribution in a problem with no cost uncertainty,
    #      the LDR bounds are at least as tight as those obtained with an
    #      uncorrelated distribution with the same marginals.
    μ = [100.0, 80.0]
    α = 0.95

    # --- (a) correlated demands ---
    Σ_corr = [100.0 40.0; 40.0 64.0]
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)
    dist = LinearDecisionRules.ConfidenceMvNormal(μ, Σ_corr, α)
    @variable(ldr, buy[1:2] >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell[1:2] >= 0)
    @variable(
        ldr,
        demand[1:2] in LinearDecisionRules.Uncertainty(; distribution = dist)
    )
    @constraint(ldr, [i = 1:2], sell[i] <= buy[i])
    @constraint(ldr, [i = 1:2], sell[i] <= demand[i])
    @objective(ldr, Max, sum(-10 * buy[i] + 15 * sell[i] for i in 1:2))

    # Verify no rejection-sampling warning is emitted.
    @test_logs min_level = Logging.Warn optimize!(ldr)

    @test termination_status(ldr; dual = false) in
          (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test termination_status(ldr; dual = true) in
          (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    obj_corr_primal = objective_value(ldr; dual = false)
    obj_corr_dual = objective_value(ldr; dual = true)

    # --- (b) uncorrelated demands (Σ diagonal) ---
    # For diagonal Σ, the rotated box coincides with the axis-aligned box, so
    # Wu_implied adds no new information beyond what lb/ub already encode.
    Σ_diag = [100.0 0.0; 0.0 64.0]
    ldr2 = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr2)
    dist2 = LinearDecisionRules.ConfidenceMvNormal(μ, Σ_diag, α)
    @variable(ldr2, buy2[1:2] >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr2, sell2[1:2] >= 0)
    @variable(
        ldr2,
        demand2[1:2] in LinearDecisionRules.Uncertainty(; distribution = dist2)
    )
    @constraint(ldr2, [i = 1:2], sell2[i] <= buy2[i])
    @constraint(ldr2, [i = 1:2], sell2[i] <= demand2[i])
    @objective(ldr2, Max, sum(-10 * buy2[i] + 15 * sell2[i] for i in 1:2))

    # Verify no rejection-sampling warning is emitted.
    @test_logs min_level = Logging.Warn optimize!(ldr2)

    @test termination_status(ldr2; dual = false) in
          (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test termination_status(ldr2; dual = true) in
          (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    obj_diag_primal = objective_value(ldr2; dual = false)
    obj_diag_dual = objective_value(ldr2; dual = true)

    # In this problem, the LDR objective function depends only on the means,
    # because costs/prices are fixed.  The rotated uncertainty results in a
    # tighter feasible set because its extra `_valid_constraints` are not
    # aligned with the coordinate axes, as it is the case for the
    # uncorrelated distribution.  Therefore, the correlated case results in
    # tighter bounds (in this particular instance, they are equal):
    @test isfinite(obj_diag_primal)
    @test isfinite(obj_diag_dual)
    @test obj_corr_primal <= obj_corr_dual + 1e-6
    @test obj_corr_dual >= obj_diag_dual - 1e-6

    return nothing
end

### Rejection sampling
function test_rejection_sampling_attributes()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)

    # defaults
    @test get_attribute(m, LinearDecisionRules.RejectionSamplingTimeLimit()) ==
          10.0
    @test get_attribute(m, LinearDecisionRules.RejectionSamplingSeed()) == 1234
    @test get_attribute(
        m,
        LinearDecisionRules.RejectionSamplingMaxIterations(),
    ) == 1000
    @test get_attribute(
        m,
        LinearDecisionRules.RejectionSamplingWarnAttempts(),
    ) == 1000

    # round-trip set/get
    set_attribute(m, LinearDecisionRules.RejectionSamplingTimeLimit(), 5.0)
    set_attribute(m, LinearDecisionRules.RejectionSamplingSeed(), 42)
    set_attribute(m, LinearDecisionRules.RejectionSamplingMaxIterations(), 500)
    set_attribute(m, LinearDecisionRules.RejectionSamplingWarnAttempts(), 200)

    @test get_attribute(m, LinearDecisionRules.RejectionSamplingTimeLimit()) ==
          5.0
    @test get_attribute(m, LinearDecisionRules.RejectionSamplingSeed()) == 42
    @test get_attribute(
        m,
        LinearDecisionRules.RejectionSamplingMaxIterations(),
    ) == 500
    @test get_attribute(
        m,
        LinearDecisionRules.RejectionSamplingWarnAttempts(),
    ) == 200

    return nothing
end

function test_mixed_rejection_sampling()
    # d1 constrained (rejection sampling) + d2 free: mixed grouped/ungrouped uncertainties
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m,
        d1 in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @variable(
        m,
        d2 in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @constraint(m, d1 <= 0.8)
    @objective(m, Min, x)
    @test_logs (:warn, "Rejection sampling required") begin
      optimize!(m)
    end
    @test termination_status(m) == MOI.OPTIMAL

    # free vector distribution alongside constrained scalar
    m2 = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m2)
    @variable(m2, x2 >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m2,
        d3 in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @variable(
        m2,
        vec_d[1:2] in LinearDecisionRules.Uncertainty(;
            distribution = product_distribution([Uniform(0, 1), Uniform(0, 1)]),
        )
    )
    @constraint(m2, d3 <= 0.8)
    @objective(m2, Min, x2)
    @test_logs (:warn, "Rejection sampling required") begin
      optimize!(m2)
    end
    @test termination_status(m2) == MOI.OPTIMAL
    return nothing
end

function test_rejection_sampling_warnings()
    # warn fires when rejection sampling exceeds max attempts
    m1 = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m1)
    @variable(m1, x >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m1,
        d in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @constraint(m1, d <= 0.5)
    @objective(m1, Min, x)
    set_attribute(m1, LinearDecisionRules.RejectionSamplingWarnAttempts(), 0)
    @test_logs (:warn, "Rejection sampling required") (:warn, "Rejection sampling: cannot find a valid sample after 1 attempts") match_mode=:any begin
      optimize!(m1)
    end
    @test termination_status(m1) == MOI.OPTIMAL

    # warn fires when _attempts reaches warn_attempts threshold
    # d1+d2 >= 0.01 is always satisfied by Uniform(0,1), so first attempt triggers the warn
    m2 = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m2)
    @variable(m2, x2 >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m2,
        d1 in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @variable(
        m2,
        d2 in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @constraint(m2, d1 + d2 >= 0.01)
    @objective(m2, Min, x2)
    set_attribute(m2, LinearDecisionRules.RejectionSamplingWarnAttempts(), 1)
    @test_logs (:warn, "Rejection sampling required") (:warn, "Rejection sampling took too long") match_mode=:any begin
      optimize!(m2)
    end
    @test termination_status(m2) == MOI.OPTIMAL

    # time limit warn fires when RejectionSamplingTimeLimit is 0.0
    m3 = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m3)
    @variable(m3, x3 >= 0, LinearDecisionRules.FirstStage)
    @variable(
        m3,
        d3 in LinearDecisionRules.Uncertainty(; distribution = Uniform(0, 1))
    )
    @constraint(m3, d3 <= 0.5)
    @objective(m3, Min, x3)
    set_attribute(m3, LinearDecisionRules.RejectionSamplingTimeLimit(), 0.0)
    @test_logs (:warn, "Rejection sampling required") (:warn, "Rejection sampling reached time limit, estimation may be inaccurate") begin
      optimize!(m3)
    end
    @test termination_status(m3) == MOI.OPTIMAL
    return nothing
end

end # module

DistributionsTests.runtests()
