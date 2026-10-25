# Core tests: plain JuMP models + hand-built ModelInfo.
# Run from the repository root: julia --project=spike spike/core/test_core.jl
include(joinpath(@__DIR__, "SpikeCore.jl"))
using .SpikeCore
using Distributions, HiGHS, JuMP, Test
const MOI = JuMP.MOI

idx(v) = JuMP.index(v)
udecl(dist, vars...; stages = fill(nothing, length(vars))) =
    UncertaintyDecl(dist, collect(idx.(vars)), Vector{Union{Int,Nothing}}(stages))
both(c) = (solve_primal(c, HiGHS.Optimizer), solve_dual(c, HiGHS.Optimizer))

function newsvendor(; integer = false, price = false)
    m = Model()
    buy = integer ? @variable(m, buy >= 0, Int) : @variable(m, buy >= 0)
    @variable(m, sell >= 0)
    @variable(m, ret >= 0)
    @variable(m, d)
    @constraint(m, sell + ret <= buy)
    @constraint(m, sell <= d)
    decls = [udecl(Uniform(80, 120), d)]
    if price
        @variable(m, p)
        push!(decls, udecl(Uniform(13, 17), p))
        @objective(m, Max, -10buy + 8ret + p * sell)
    else
        @objective(m, Max, -10buy + 8ret + 15sell)
    end
    info = ModelInfo(uncertainties = decls, decision_stage = Dict(idx(buy) => 1))
    return m, info
end

@testset "SpikeCore" begin
    @testset "1. newsvendor" begin
        m, info = newsvendor()
        c = canonicalize(backend(m), info)
        @test c.T == 2
        pr, du = both(c)
        @test pr.termination_status == MOI.OPTIMAL
        @test pr.primal_status == MOI.FEASIBLE_POINT
        @test pr.objective_value ≈ 460
        @test du.objective_value ≈ 1460 / 3
        b, s, d = idx(m[:buy]), idx(m[:sell]), idx(m[:d])
        k, coefs = rule(c, pr, b)
        @test k ≈ 120 && isempty(coefs)
        k, coefs = rule(c, pr, s)
        @test abs(k) < 1e-6 && coefs[d] ≈ 1 && length(coefs) == 1
        @test expected_decision(c, pr, s) ≈ 100
        @test expected_decision(c, pr, d) ≈ 100
        @test mean_xi(c) ≈ [1, 100]
        @test second_moment(c) ≈ [1 100; 100 100^2+40^2/12]
        @test decisions(c) == [b, s, idx(m[:ret])] && uncertainties(c) == [d]
        obj = MOI.get(backend(m), MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
        @test expected_value(c, pr, obj) ≈ pr.objective_value
        @test expected_value(c, du, obj) ≈ du.objective_value
    end

    @testset "bare MOI model as src" begin
        m, info = newsvendor()
        dest = MOI.Utilities.Model{Float64}()
        map = MOI.copy_to(dest, backend(m))
        info2 = ModelInfo(uncertainties = [UncertaintyDecl(Uniform(80, 120), [map[idx(m[:d])]], [nothing])],
            decision_stage = Dict(map[idx(m[:buy])] => 1))
        pr, du = both(canonicalize(dest, info2))
        @test pr.objective_value ≈ 460 && du.objective_value ≈ 1460 / 3
    end

    @testset "2. integer first stage" begin
        m, info = newsvendor(integer = true)
        c = canonicalize(backend(m), info)
        @test c.integers == [1] && !any(c.mask[1, 2:end])
        pr, du = both(c)
        @test pr.objective_value ≈ 460
        @test pr.X[1, 1] ≈ 120
    end

    @testset "3. vector uncertainty" begin
        m = Model()
        @variable(m, buy[1:2] >= 0)
        @variable(m, sell[1:2] >= 0)
        @variable(m, ret[1:2] >= 0)
        @variable(m, d[1:2])
        @constraint(m, sell .+ ret .<= buy)
        @constraint(m, sell .<= d)
        @objective(m, Max, sum(-10buy[i] + 8ret[i] + 15sell[i] for i in 1:2))
        dist = product_distribution([Uniform(80, 120), Uniform(80, 120)])
        info = ModelInfo(uncertainties = [udecl(dist, d...)], decision_stage = Dict(idx.(buy) .=> 1))
        c = canonicalize(backend(m), info)
        pr, du = both(c)
        @test pr.objective_value ≈ 920
        @test du.objective_value ≈ 2 * 1460 / 3
        _, coefs = rule(c, pr, idx(sell[1]))
        @test !haskey(coefs, idx(d[2])) && coefs[idx(d[1])] ≈ 1
        @test second_moment(c)[2, 3] ≈ 100^2   # independent components
    end

    @testset "4. three-stage newsvendor" begin
        m = Model()
        @variable(m, buy >= 0)
        @variable(m, sell2 >= 0); @variable(m, ret2 >= 0); @variable(m, buy2 >= 0)
        @variable(m, demand2); @variable(m, buy_cost2)
        @variable(m, sell3 >= 0); @variable(m, ret3 >= 0)
        @variable(m, demand3)
        @constraint(m, sell2 + ret2 <= buy + buy2)
        @constraint(m, sell2 <= demand2)
        @constraint(m, sell2 + ret2 + sell3 + ret3 <= buy + buy2)
        @constraint(m, sell3 <= demand3)
        @objective(m, Max, -10buy - buy_cost2 * buy2 + 8ret2 + 15sell2 + 8ret3 + 15sell3)
        stages = Dict(idx(buy) => 1, idx(sell2) => 2, idx(ret2) => 2, idx(buy2) => 2, idx(sell3) => 3,
            idx(ret3) => 3)
        unc = [udecl(Uniform(80, 120), demand2; stages = [2]), udecl(Uniform(8, 14), buy_cost2; stages = [2]),
            udecl(Uniform(80, 120), demand3; stages = [3])]
        c = @test_logs canonicalize(backend(m), ModelInfo(uncertainties = unc, decision_stage = stages))
        @test c.T == 3
        pr, du = both(c)
        @test pr.objective_value ≈ 940
        @test du.objective_value ≈ 1000
        for v in (sell2, ret2, buy2), res in (pr, du)
            @test !haskey(rule(c, res, idx(v))[2], idx(demand3))
            @test res.X[findfirst(==(idx(v)), decisions(c)), 4] == 0.0
        end
        @test isempty(rule(c, pr, idx(buy))[2])
        # unmarked variables in a 3-stage model: warning naming them, assigned Stage(3)
        delete!(stages, idx(ret3))
        c2 = @test_logs (:warn, r"Stage\(3\).*ret3") canonicalize(backend(m),
            ModelInfo(uncertainties = unc, decision_stage = stages))
        @test c2.decision_stage[findfirst(==(idx(ret3)), decisions(c2))] == 3
    end

    @testset "5. random price" begin
        m, info = newsvendor(price = true)
        c = canonicalize(backend(m), info)
        pr, du = both(c)
        @test pr.objective_value ≈ 460
        @test du.objective_value ≈ 486.6666666666665
        @test isempty(rule(c, pr, idx(m[:buy]))[2])
        obj = MOI.get(backend(m), MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}())
        @test expected_value(c, pr, obj) ≈ pr.objective_value
        @test expected_value(c, du, obj) ≈ du.objective_value
        # E[p * sell] with sell = d: independent, so = 15 * 100; differs from E[p]E[sell] only when correlated
        psell = moi_function(m[:p] * m[:sell])
        @test expected_value(c, pr, psell) ≈ 1500
        d2 = moi_function(m[:d] * m[:d])       # MOI stores d² as 2.0 d*d
        @test expected_value(c, pr, d2) ≈ 100^2 + 40^2 / 12
        sell2 = moi_function(m[:sell]^2)
        @test expected_value(c, pr, sell2) ≈ 100^2 + 40^2 / 12 atol = 1e-4
    end

    @testset "quadratic objective conventions" begin
        for (st, expected) in ((1, 40^2 / 12), (2, 0.0))
            m = Model()
            @variable(m, x)
            @variable(m, d)
            @objective(m, Min, (x - d)^2 + 2x * d - 2x * d)
            info = ModelInfo(uncertainties = [udecl(Uniform(80, 120), d)], decision_stage = Dict(idx(x) => st))
            c = canonicalize(backend(m), info)
            pr, du = both(c)
            @test pr.objective_value ≈ expected atol = 1e-5
            @test du.objective_value ≈ expected atol = 1e-5
            @test pr.X[1, :] ≈ (st == 1 ? [100, 0] : [0, 1]) atol = 1e-4
        end
        # Max sense with a concave quadratic objective stays Max
        m = Model()
        @variable(m, x)
        @objective(m, Max, -x^2 + 4x + 1)
        pr = solve_primal(canonicalize(backend(m), ModelInfo()), HiGHS.Optimizer)
        @test pr.objective_value ≈ 5 && isapprox(pr.X[1, 1], 2; atol = 1e-5)
    end

    @testset "6. errors" begin
        m, info = newsvendor()
        @constraint(m, m[:d] <= 115)
        err = try canonicalize(backend(m), info); "" catch e; sprint(showerror, e) end
        @test occursin("Truncate()", err) && occursin("ValidInequality()", err)
        ci = all_constraints(m, AffExpr, MOI.LessThan{Float64})[end]
        info_t = ModelInfo(uncertainties = info.uncertainties, decision_stage = info.decision_stage,
            truncate_rows = Set{MOI.ConstraintIndex}([index(ci)]))
        @test_throws r"Truncate\(\).*not in the spike" canonicalize(backend(m), info_t)

        m, info = newsvendor()
        set_lower_bound(m[:d], 0)
        @test_throws r"d is an uncertainty" canonicalize(backend(m), info)

        m, info = newsvendor()
        @variable(m, z, Int)
        @test_throws r"z is in Stage\(2\)" canonicalize(backend(m), info)

        m, info = newsvendor()
        @constraint(m, m[:sell]^2 <= 1)
        @test_throws r"Quadratic constraints" canonicalize(backend(m), info)

        m, info = newsvendor()
        @constraint(m, m[:d] * m[:sell] <= 1000)
        @test_throws r"random recourse" canonicalize(backend(m), info)

        m, info = newsvendor()
        @constraint(m, [m[:buy], m[:sell], m[:ret]] in SecondOrderCone())
        @test_throws r"Unsupported constraint type.*SecondOrderCone" canonicalize(backend(m), info)
    end

    @testset "is_metadata_constraint" begin
        m, info = newsvendor()
        @constraint(m, [m[:buy], m[:sell], m[:ret]] in MOI.Nonnegatives(3))
        SpikeCore.is_metadata_constraint(::Type{MOI.VectorOfVariables}, ::Type{MOI.Nonnegatives}) = true
        @test solve_primal(canonicalize(backend(m), info), HiGHS.Optimizer).objective_value ≈ 460
    end

    @testset "7. ValidInequality" begin
        function two(valid)
            m = Model()
            @variable(m, buy >= 0)
            @variable(m, sell[1:2] >= 0)
            @variable(m, ret >= 0)
            @variable(m, d[1:2])
            @constraint(m, sell[1] + sell[2] + ret <= buy)
            @constraint(m, sell .<= d)
            @objective(m, Max, -10buy + 8ret + 15sum(sell))
            rows = Set{MOI.ConstraintIndex}()
            valid && push!(rows, index(@constraint(m, d[1] + d[2] <= 230)))
            info = ModelInfo(uncertainties = [udecl(Uniform(80, 120), d[1]), udecl(Uniform(80, 120), d[2])],
                decision_stage = Dict(idx(buy) => 1), valid_rows = rows)
            return canonicalize(backend(m), info)
        end
        c0, c1 = two(false), two(true)
        @test size(c1.W, 1) == size(c0.W, 1) + 1
        p0, p1 = solve_primal(c0, HiGHS.Optimizer), solve_primal(c1, HiGHS.Optimizer)
        @test p0.objective_value ≈ 920   # buy = 240, sell = d
        @test p1.objective_value > p0.objective_value + 1e-3
    end
end
