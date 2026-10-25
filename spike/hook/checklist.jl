# Design C checklist: plain JuMP.Model + MOI sets/attributes + optimize hook.
# Run from the repository root: julia --project=spike spike/hook/checklist.jl
include(joinpath(@__DIR__, "LDRHook.jl"))
using .LDRHook
using Aqua, Distributions, HiGHS, JuMP, Random, Test, TOML
const MOI = JuMP.MOI

errmsg(f) = try f(); "" catch e; sprint(showerror, e) end
note(item, msg) = println("NOTE [$item] ", msg)

function newsvendor(; integer = false, silent = true, kwargs...)
    m = LDRModel(HiGHS.Optimizer; kwargs...)
    silent && set_silent(m)
    if integer
        @variable(m, buy >= 0, Int, Stage(1))
    else
        @variable(m, buy >= 0, Stage(1))
    end
    @variable(m, d in Uncertainty(Uniform(80, 120)))
    @variable(m, sell >= 0)
    @variable(m, ret >= 0)
    @constraint(m, c_stock, sell + ret <= buy)
    @constraint(m, c_demand, sell <= d)
    @objective(m, Max, -10buy + 8ret + 15sell)
    return m
end

# Two markets sharing one stock: the valid inequality d1 + d2 <= 230 binds.
function two_markets(; tag = nothing)
    m = LDRModel(HiGHS.Optimizer); set_silent(m)
    @variable(m, buy >= 0, Stage(1))
    @variable(m, sell[1:2] >= 0)
    @variable(m, ret >= 0)
    @variable(m, d[1:2] in Uncertainty(Uniform(80, 120)))      # two independent declarations
    @constraint(m, sell[1] + sell[2] + ret <= buy)
    @constraint(m, sell .<= d)
    @objective(m, Max, -10buy + 8ret + 15sum(sell))
    tag === :valid && @constraint(m, d[1] + d[2] <= 230, ValidInequality())
    tag === :untagged && @constraint(m, d[1] + d[2] <= 230)
    tag === :truncate && @constraint(m, d[1] + d[2] <= 230, Truncate())
    return m
end

@testset "Design C: hook" begin
    @testset "1. newsvendor" begin
        m = newsvendor()
        @test termination_status(m) == MOI.OPTIMIZE_NOT_CALLED
        optimize!(m)
        @test termination_status(m) == MOI.OPTIMAL
        @test primal_status(m) == MOI.FEASIBLE_POINT
        @test objective_value(m) ≈ 460
        @test objective_bound(m) ≈ 1460 / 3
        @test relative_gap(m) ≈ (1460 / 3 - 460) / 460
        buy, sell, d, ret = m[:buy], m[:sell], m[:d], m[:ret]
        rb = decision_rule(buy)
        @test rb.constant ≈ 120 && isempty(rb.coefficients)
        rs = decision_rule(sell)
        @test abs(rs.constant) < 1e-6 && rs.coefficients[d] ≈ 1 && length(rs.coefficients) == 1
        @test rs(90.0) ≈ 90 && rs(d => 95.0) ≈ 95 && rs(Dict(d => 85.0)) ≈ 85
        @test value(buy) ≈ 120
        @test value(sell) ≈ 100
        @test value(d) ≈ 100
        @test value(ret) ≈ 20
        @test expected_value(objective_function(m)) ≈ 460
        @test expected_value(objective_function(m); dual = true) ≈ 1460 / 3
        @test value(objective_function(m)) ≈ 460        # JuMP's `value` on an expression = objective at E[.]
    end

    @testset "2. integer first stage" begin
        m = newsvendor(integer = true)
        optimize!(m)
        @test objective_value(m) ≈ 460
        @test value(m[:buy]) ≈ 120
        @test is_integer(m[:buy]) && stage(m[:buy]) == 1
    end

    @testset "3. vector uncertainty" begin
        m = LDRModel(HiGHS.Optimizer); set_silent(m)
        @variable(m, buy[1:2] >= 0, Stage(1))
        @variable(m, sell[1:2] >= 0)
        @variable(m, ret[1:2] >= 0)
        @variable(m, d[1:2] in Uncertainty(product_distribution([Uniform(80, 120), Uniform(80, 120)])))
        @constraint(m, sell .+ ret .<= buy)
        @constraint(m, sell .<= d)
        @objective(m, Max, sum(-10buy[i] + 8ret[i] + 15sell[i] for i in 1:2))
        optimize!(m)
        @test objective_value(m) ≈ 920
        @test objective_bound(m) ≈ 2 * 1460 / 3
        r = decision_rule(sell[1])
        @test !haskey(r.coefficients, d[2]) && r.coefficients[d[1]] ≈ 1
        @test value.(d) ≈ [100, 100]
        @test uncertainty_distribution(d[2]) isa Distributions.MultivariateDistribution
        # a vector uncertainty with Stage(t) on the whole declaration
        m2 = LDRModel(HiGHS.Optimizer)
        @variable(m2, e[1:2] in Uncertainty(product_distribution([Uniform(0, 1), Uniform(0, 1)])), Stage(2))
        @test stage.(e) == [2, 2]
        # deleting one component of a joint declaration is refused by MOI (cleanly, at `delete`)
        note(3, "delete(m, d[1]) -> " * first(split(errmsg(() -> delete(m2, e[1])), '\n')))
    end

    @testset "4. containers and stages (3-stage newsvendor)" begin
        m = LDRModel(HiGHS.Optimizer); set_silent(m)
        D = Uniform(80, 120)
        @variable(m, buy[t = 1:2] >= 0, Stage(t))
        @variable(m, sell[t = 2:3] >= 0, Stage(t))
        @variable(m, ret[t = 2:3] >= 0, Stage(t))
        @variable(m, demand[t = 2:3] in Uncertainty(D), Stage(t))
        @variable(m, cost2 in Uncertainty(Uniform(8, 14)), Stage(2))
        @constraint(m, sell[2] + ret[2] <= buy[1] + buy[2])
        @constraint(m, sell[2] <= demand[2])
        @constraint(m, sell[2] + ret[2] + sell[3] + ret[3] <= buy[1] + buy[2])
        @constraint(m, sell[3] <= demand[3])
        @objective(m, Max, -10buy[1] - cost2 * buy[2] + 8ret[2] + 15sell[2] + 8ret[3] + 15sell[3])
        @test_logs optimize!(m)                   # every variable is marked: no warning
        @test objective_value(m) ≈ 940
        @test objective_bound(m) ≈ 1000
        for v in (sell[2], ret[2], buy[2])
            @test !haskey(decision_rule(v).coefficients, demand[3])
            @test !haskey(decision_rule(v; dual = true).coefficients, demand[3])
        end
        @test isempty(decision_rule(buy[1]).coefficients)
        @test stage(buy[1]) == 1 && stage(buy[2]) == 2 && stage(sell[3]) == 3 && stage(demand[3]) == 3
        @test stage(cost2) == 2
        # scalar uncertainty with Stage(3), stage helpers
        m2 = LDRModel(HiGHS.Optimizer)
        @variable(m2, d in Uncertainty(D), Stage(3))
        @variable(m2, x)
        @test stage(d) == 3 && stage(x) === nothing
        set_stage(x, 2); @test stage(x) == 2
        set_stage(x, 3); @test stage(x) == 3
        @test uncertainty_distribution(d) == D && uncertainty_distribution(x) === nothing
        # an unmarked variable in a 3-stage model: a warning naming it (from the core)
        @variable(m, extra_unmarked >= 0)
        @test_logs (:warn, r"Stage\(3\).*extra_unmarked") optimize!(m)
        @test objective_value(m) ≈ 940
    end

    @testset "5. print" begin
        m = newsvendor()
        s = sprint(print, m)
        println("---- print(model) ----\n", s, "----------------------")
        @test occursin("Stage(1)", s) && occursin("Uncertainty(Uniform", s) && occursin("buy", s)
        @test occursin("d in Uncertainty", s)
    end

    @testset "6. delete" begin
        m = newsvendor()
        @variable(m, extra >= 0, Stage(1))
        @constraint(m, cx, extra + m[:sell] <= 1000)
        optimize!(m)
        @test objective_value(m) ≈ 460
        nc = num_constraints(m; count_variable_in_set_constraints = true)
        delete(m, cx); delete(m, extra)
        @test num_constraints(m; count_variable_in_set_constraints = true) == nc - 3   # cx, extra >= 0, extra in Stage(1)
        @test termination_status(m) == MOI.OPTIMIZE_NOT_CALLED
        optimize!(m)
        @test objective_value(m) ≈ 460
        r = decision_rule(m[:sell])
        @test abs(r.constant) < 1e-6 && r.coefficients[m[:d]] ≈ 1 && length(r.coefficients) == 1
        @test objective_bound(m) ≈ 1460 / 3
        # deleting an uncertainty variable removes its declaration; the model is then valid again
        m2 = newsvendor()
        @variable(m2, d2 in Uncertainty(Uniform(0, 1)))
        delete(m2, d2)
        optimize!(m2)
        @test objective_value(m2) ≈ 460
    end

    @testset "7. modification" begin
        m = newsvendor()
        @constraint(m, cap, m[:buy] <= 200)
        optimize!(m)
        @test objective_value(m) ≈ 460
        set_normalized_rhs(cap, 100)
        @test termination_status(m) == MOI.OPTIMIZE_NOT_CALLED     # results are invalidated by the modification
        @test_throws Exception decision_rule(m[:sell])
        @test_throws Exception value(m[:sell])
        @test !has_values(m)
        optimize!(m)
        ref = newsvendor()
        @constraint(ref, ref[:buy] <= 100)
        optimize!(ref)
        @test objective_value(m) ≈ objective_value(ref)
        @test objective_bound(m) ≈ objective_bound(ref)
        @test objective_value(m) < 460 - 1
        note(7, "after set_normalized_rhs: termination_status == OPTIMIZE_NOT_CALLED, value(x) errors; " *
                "invalidation done by CachingOptimizer + MockOptimizer(modify_allowed=false)")
    end

    @testset "8. results API" begin
        m = newsvendor()
        optimize!(m)
        @test primal_status(m; result = 1) == MOI.FEASIBLE_POINT && primal_status(m; result = 2) == MOI.FEASIBLE_POINT
        @test result_count(m) == 2
        @test occursin("OPTIMAL", sprint(show, solution_summary(m))) && raw_status(m) == "LDR primal: OPTIMAL; dual: OPTIMAL"
        @test solve_time(m) > 0
        @test objective_value(m; result = 1) ≈ 460 && objective_value(m; result = 2) ≈ 1460 / 3
        @test value(m[:sell]; result = 2) ≈ 100
        @test decision_rule(m[:sell]; dual = true).coefficients[m[:d]] ≈ 1
        @test objective_bound(m) ≈ objective_value(m; result = 2)
        @test relative_gap(m) > 0
        # primal disabled / dual disabled
        set_attribute(m, Formulations(), :primal)
        optimize!(m)
        @test objective_value(m) ≈ 460
        @test primal_status(m; result = 2) == MOI.NO_SOLUTION
        note(8, "dual disabled: objective_bound(m) = $(objective_bound(m)), relative_gap = $(relative_gap(m))")
        msg2 = errmsg(() -> decision_rule(m[:sell]; dual = true))
        @test occursin("dual formulation is disabled", msg2)
        msg3 = errmsg(() -> value(m[:sell]; result = 2))
        note(8, "dual disabled: value(x; result=2) -> " * first(split(msg3, '\n')))
        set_attribute(m, Formulations(), :dual)
        optimize!(m)
        @test primal_status(m; result = 1) == MOI.NO_SOLUTION
        @test objective_value(m; result = 2) ≈ 1460 / 3
        @test occursin("primal formulation is disabled", errmsg(() -> decision_rule(m[:sell])))
        note(8, "primal disabled: termination_status=$(termination_status(m)), objective_value(m)=$(objective_value(m))")
        set_attribute(m, Formulations(), :both)
        # random price: E[p * z] with z(ξ) = p
        mp = LDRModel(HiGHS.Optimizer); set_silent(mp)
        @variable(mp, buy >= 0, Stage(1))
        @variable(mp, d in Uncertainty(Uniform(80, 120)))
        @variable(mp, p in Uncertainty(Uniform(13, 17)))
        @variable(mp, sell >= 0)
        @variable(mp, ret >= 0)
        @variable(mp, z)
        @constraint(mp, sell + ret <= buy)
        @constraint(mp, sell <= d)
        @constraint(mp, z == p)
        @objective(mp, Max, -10buy + 8ret + p * sell)
        optimize!(mp)
        @test objective_value(mp) ≈ 460
        @test objective_bound(mp) ≈ 486.6666666666665
        exact = expected_value(p * z)
        @test exact ≈ 15^2 + 4^2 / 12
        @test !isapprox(exact, value(p) * value(z); atol = 0.5)
        @test value(p) * value(z) ≈ 225
        Random.seed!(1234)
        N = 400_000
        ps, ds = rand(Uniform(13, 17), N), rand(Uniform(80, 120), N)
        rz, rs = decision_rule(z), decision_rule(mp[:sell])
        @test isapprox(sum(ps .* rz.(ps)) / N, exact; rtol = 0.01)           # z(p) = p
        sampled = sum(ps[i] * rs(Dict(mp[:p] => ps[i], mp[:d] => ds[i])) for i in 1:N) / N
        @test isapprox(sampled, expected_value(p * mp[:sell]); rtol = 0.01)
        @test expected_value(p * mp[:sell]) ≈ 1500
        @test expected_value(p^2) ≈ 15^2 + 4^2 / 12
        @test expected_value(objective_function(mp)) ≈ objective_value(mp)
        # a user-called set_optimizer replaces the results holder; the hook puts it back
        set_optimizer(m, HiGHS.Optimizer)
        @test termination_status(m) == MOI.OPTIMIZE_NOT_CALLED
        optimize!(m)
        @test objective_value(m) ≈ 460 && objective_bound(m) ≈ 1460 / 3
        @test errmsg(() -> optimize!(m; foo = 1)) != ""
    end

    @testset "9. Ξ roles" begin
        p0 = two_markets()
        optimize!(p0)
        @test objective_value(p0) ≈ 920
        p1 = two_markets(tag = :valid)
        optimize!(p1)
        @test objective_value(p1) > objective_value(p0) + 1e-3
        @test constraint_tag(first(all_constraints(p1, AffExpr, MOI.LessThan{Float64}))) === nothing
        @test any(c -> constraint_tag(c) isa ValidInequality, all_constraints(p1, AffExpr, MOI.LessThan{Float64}))
        msg = errmsg(() -> optimize!(two_markets(tag = :untagged)))
        println("untagged uncertainty-only row -> ", msg)
        @test occursin("Truncate()", msg) && occursin("ValidInequality()", msg)
        msg = errmsg(() -> optimize!(two_markets(tag = :truncate)))
        @test occursin("Truncate()", msg) && occursin("not in the spike", msg)
        @test termination_status(p1) == MOI.OPTIMAL       # a failed optimize! on another model does not touch p1
    end

    @testset "10. attributes" begin
        m = newsvendor()
        @variable(m, other)
        d = m[:d]
        @test get_attribute(d, BreakPoints()) === nothing
        set_attribute(d, BreakPoints(), [100.0])
        @test get_attribute(d, BreakPoints()) == [100.0]
        delete(m, other)
        @test get_attribute(d, BreakPoints()) == [100.0]
        optimize!(m)
        @test get_attribute(d, BreakPoints()) == [100.0]       # still there after the results optimizer was filled
        @test objective_value(m) ≈ 460
        # another variable is not affected
        @test get_attribute(m[:sell], BreakPoints()) === nothing
    end

    @testset "11. unsupported input" begin
        m = newsvendor(); @constraint(m, m[:sell]^2 <= 1)
        msg = errmsg(() -> optimize!(m))
        println("sell^2 <= 1 -> ", msg)
        @test occursin("Quadratic constraints", msg)
        @test termination_status(m) == MOI.OPTIMIZE_NOT_CALLED
        m = newsvendor(); @constraint(m, m[:d] * m[:sell] <= 1000)
        @test occursin("random recourse", errmsg(() -> optimize!(m)))
        m = newsvendor(); @constraint(m, [m[:buy], m[:sell], m[:ret]] in SecondOrderCone())
        msg = errmsg(() -> optimize!(m))
        println("SOC -> ", msg)
        @test occursin("Unsupported constraint type", msg) && occursin("SecondOrderCone", msg)
        # constraint types a solver would bridge reach the core under their original type, never as bridged
        m = newsvendor(); @constraint(m, [m[:sell], m[:ret]] in MOI.Nonnegatives(2))
        msg = errmsg(() -> optimize!(m))
        println("Nonnegatives -> ", msg)
        @test occursin("Nonnegatives", msg)
        # Interval constraints (what bridges would split) are native
        m = newsvendor(); @constraint(m, 1 <= m[:sell] + m[:ret] <= 400)
        optimize!(m); @test termination_status(m) == MOI.OPTIMAL
        # bridges on: metadata sets still reach the core, no auxiliary variables appear
        for bridges in (false, true)
            mb = newsvendor(bridges = bridges)
            @constraint(mb, 1 <= mb[:sell] + mb[:ret] <= 400)
            optimize!(mb)
            @test objective_value(mb) ≈ 460
            @test length(MOI.get(JuMP.backend(mb), MOI.ListOfVariableIndices())) == num_variables(mb) == 4
            @test mb.ext[:LDR].canonical |> c -> length(c.decisions) + length(c.uncertainties) == 4
        end
    end

    @testset "12. extension probes" begin
        @testset "12a. JuMP parameters" begin
            m = newsvendor()
            @variable(m, par in Parameter(5.0))
            @constraint(m, m[:buy] <= 20 * par)
            msg = errmsg(() -> optimize!(m))
            note("12a", "optimize! with Parameter -> " * first(split(msg, '\n')))
            @test_broken isempty(msg)
        end
        @testset "12b. third-party builder" begin
            # a third-party package only needs JuMP + the tag types; typed on the plain JuMP.Model
            function build!(m::JuMP.Model)
                @variable(m, extra_ret >= 0)
                @constraint(m, extra_ret <= 5)
                return m
            end
            m = newsvendor()
            build!(m)
            optimize!(m)
            @test termination_status(m) == MOI.OPTIMAL && objective_value(m) ≈ 460
            note("12b", "works; but `build!(m::JuMP.Model)` cannot dispatch on LDR-ness (no LDRModel type): " *
                        "it must test `haskey(m.ext, :LDR)`; and tags (`Stage`) are accepted on any JuMP.Model")
            plain = Model()
            @variable(plain, x >= 0, Stage(1))
            @test stage(x) == 1
        end
        @testset "12c. constraint tag Stage(2)" begin
            m = newsvendor()
            c = @constraint(m, m[:buy] + m[:sell] <= 500, Stage(2))
            @test constraint_tag(c) == Stage(2)
            cn = @constraint(m, c_named, m[:buy] + m[:ret] <= 500, Stage(3))
            @test constraint_tag(cn) == Stage(3)
            optimize!(m)
            @test objective_value(m) ≈ 460          # the tag has no meaning yet; must not break the solve
            delete(m, c)
            @test constraint_tag(cn) == Stage(3)
            cs = @constraint(m, [i = 1:2], m[:buy] <= 400 + i, Stage(2))
            @test all(constraint_tag.(cs) .== Ref(Stage(2)))
            cb = @constraint(m, [m[:buy], m[:sell]] .<= [400, 300], Stage(2))      # broadcast form
            @test all(constraint_tag.(cb) .== Ref(Stage(2)))
            optimize!(m); @test objective_value(m) ≈ 460
        end
        @testset "12d. copy_model" begin
            m = newsvendor()
            set_attribute(m[:d], BreakPoints(), [100.0])
            c = @constraint(m, m[:buy] + m[:sell] <= 500, Stage(2))
            m2, ref = copy_model(m)
            @test stage(ref[m[:buy]]) == 1
            @test uncertainty_distribution(ref[m[:d]]) == Uniform(80, 120)
            @test get_attribute(ref[m[:d]], BreakPoints()) == [100.0]
            @test constraint_tag(ref[c]) == Stage(2)
            set_silent(m2)
            optimize!(m2)
            @test objective_value(m2) ≈ 460 && objective_bound(m2) ≈ 1460 / 3
            @test haskey(m2.ext, :LDR) && m2.ext[:LDR] !== m.ext[:LDR]
            # delete, then copy
            junk = @variable(m, lower_bound = 0)          # anonymous: JuMP keeps deleted *named* variables in object_dictionary and copy_model then fails
            set_stage(junk, 1); delete(m, junk)
            m3, ref3 = copy_model(m)
            optimize!(m3); @test objective_value(m3) ≈ 460
        end
    end

    @testset "Aqua" begin
        # Aqua needs a loadable package (it analyses in a clean subprocess): stage a copy of LDRHook.jl (+ core) as a
        # throw-away package in a temp dir and run Aqua on that copy.
        root = mktempdir()
        spike = dirname(@__DIR__)
        pkgdir = joinpath(root, "LDRHook")
        mkpath(joinpath(pkgdir, "src")); mkpath(joinpath(pkgdir, "core"))
        cp(joinpath(@__DIR__, "LDRHook.jl"), joinpath(pkgdir, "src", "LDRHook.jl"))
        cp(joinpath(spike, "core", "SpikeCore.jl"), joinpath(pkgdir, "core", "SpikeCore.jl"))
        deps = TOML.parsefile(joinpath(spike, "Project.toml"))["deps"]
        uuid = "7b0f8a6e-0c1d-4c52-9f0e-1d8b2b0f5a11"
        open(joinpath(pkgdir, "Project.toml"), "w") do io
            TOML.print(io, Dict("name" => "LDRHook", "uuid" => uuid, "version" => "0.0.1",
                "deps" => Dict(k => deps[k] for k in ("Distributions", "JuMP", "MathOptInterface", "LinearAlgebra"))))
        end
        push!(LOAD_PATH, root)
        pkg = Base.require(Base.PkgId(Base.UUID(uuid), "LDRHook"))
        Aqua.test_piracies(pkg)
        Aqua.test_ambiguities(pkg)
        Aqua.test_undefined_exports(pkg)
        filter!(!=(root), LOAD_PATH)
    end

    @testset "measurements" begin
        # methods that LDRHook adds to functions of other modules
        counts = Dict{String,Int}()
        for M in (JuMP, MOI, MOI.Utilities, Base, LDRHook.SpikeCore)
            for n in names(M; all = true)
                isdefined(M, n) || continue
                f = getfield(M, n)
                f isa Function || continue
                k = count(mt -> mt.module === LDRHook, methods(f))
                k > 0 && (counts["$M.$n"] = k)
            end
        end
        println("methods added to external functions: ", sum(values(counts)))
        for (k, v) in sort(collect(counts)); println("   ", k, " => ", v); end
        src = readlines(joinpath(@__DIR__, "LDRHook.jl"))
        in_doc = false
        loc = 0
        for l in src
            s = strip(l)
            if count("\"\"\"", s) == 1
                in_doc = !in_doc
                continue
            end
            (in_doc || isempty(s) || startswith(s, "#")) && continue
            loc += 1
        end
        println("non-blank non-comment, non-docstring lines in LDRHook.jl: ", loc)
    end
end
