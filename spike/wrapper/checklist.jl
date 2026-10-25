# Design A checklist. Run from the repository root:  julia --project=spike spike/wrapper/checklist.jl
include(joinpath(@__DIR__, "..", "core", "SpikeCore.jl"))
include(joinpath(@__DIR__, "LDRWrapper.jl"))
using .LDRWrapper
using Aqua, Distributions, HiGHS, JuMP, Random, Statistics, Test
const MOI = JuMP.MOI

err_msg(f) = try f(); "" catch e; sprint(showerror, e) end
silent(m) = (set_silent(m); m)

function newsvendor(; integer = false, valid = false)
    m = silent(LDRModel(HiGHS.Optimizer))
    integer ? @variable(m, buy >= 0, Int, Stage(1)) : @variable(m, buy >= 0, Stage(1))
    @variable(m, d in Uncertainty(Uniform(80, 120)))
    @variable(m, sell >= 0)
    @variable(m, ret >= 0)
    @constraint(m, sell + ret <= buy)
    @constraint(m, sell <= d)
    @objective(m, Max, -10buy + 8ret + 15sell)
    return m
end

# one stock shared by two markets, so that d1 + d2 <= 230 binds (core note (i))
function shared_stock(; tag = nothing)
    m = silent(LDRModel(HiGHS.Optimizer))
    @variable(m, buy >= 0, Stage(1))
    @variable(m, sell[1:2] >= 0)
    @variable(m, ret >= 0)
    @variable(m, d1 in Uncertainty(Uniform(80, 120)))
    @variable(m, d2 in Uncertainty(Uniform(80, 120)))
    @constraint(m, sell[1] + sell[2] + ret <= buy)
    @constraint(m, sell[1] <= d1)
    @constraint(m, sell[2] <= d2)
    @objective(m, Max, -10buy + 8ret + 15(sell[1] + sell[2]))
    tag === nothing || @constraint(m, d1 + d2 <= 230, tag)
    return m
end

@testset "Design A: LDRModel wrapper" begin
    @testset "1. newsvendor" begin
        m = newsvendor()
        optimize!(m)
        @test termination_status(m) == MOI.OPTIMAL
        @test primal_status(m) == MOI.FEASIBLE_POINT
        @test objective_value(m) ≈ 460
        @test objective_bound(m) ≈ 1460 / 3
        @test objective_value(m; formulation = :dual) ≈ 1460 / 3
        @test relative_gap(m) ≈ (1460 / 3 - 460) / 460
        buy, sell, d = m[:buy], m[:sell], m[:d]
        rb = decision_rule(buy)
        @test rb.constant ≈ 120 && isempty(rb.coefficients)
        rs = decision_rule(sell)
        @test abs(rs.constant) < 1e-6 && rs.coefficients[d] ≈ 1 && length(rs.coefficients) == 1
        @test rs(90.0) ≈ 90 && rs(d => 110.0) ≈ 110 && rb(90.0) ≈ 120
        @test decision_rule(d).coefficients == Dict(d => 1.0)
        @test value(buy) ≈ 120 && value(sell) ≈ 100 && value(d) ≈ 100
        @test value(-10buy + 8m[:ret] + 15sell) ≈ objective_value(m)
        @test expected_value(-10buy + 8m[:ret] + 15sell) ≈ objective_value(m)
        @test is_solved_and_feasible(m)
        println("   item 1 rule: sell = ", rs, ";  buy = ", rb)
    end

    @testset "2. integer first stage" begin
        m = newsvendor(integer = true)
        @test is_integer(m[:buy])
        optimize!(m)
        @test objective_value(m) ≈ 460
        @test value(m[:buy]) ≈ 120
        println("   item 2: integer first stage, dual bound = ", objective_bound(m), " (continuous: 1460/3)")
        @test objective_bound(m) >= objective_value(m)
    end

    @testset "3. vector uncertainty" begin
        m = silent(LDRModel(HiGHS.Optimizer))
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
        @test r([90.0, 70.0]) ≈ 90
        @test err_msg(() -> delete(m, d[1])) != ""        # one component of a vector uncertainty cannot go alone
    end

    @testset "4. containers and stages (3-stage newsvendor)" begin
        m = silent(LDRModel(HiGHS.Optimizer))
        @variable(m, buy[t = 1:2] >= 0, Stage(t))
        @variable(m, sell[t = 2:3] >= 0, Stage(t))
        @variable(m, ret[t = 2:3] >= 0, Stage(t))
        @variable(m, demand[t = 2:3] in Uncertainty(Uniform(80, 120)), Stage(t))
        @variable(m, cost2 in Uncertainty(Uniform(8, 14)), Stage(2))
        @constraint(m, sell[2] + ret[2] <= buy[1] + buy[2])
        @constraint(m, sell[2] <= demand[2])
        @constraint(m, sell[2] + ret[2] + sell[3] + ret[3] <= buy[1] + buy[2])
        @constraint(m, sell[3] <= demand[3])
        @objective(m, Max, -10buy[1] - cost2 * buy[2] + 8ret[2] + 15sell[2] + 8ret[3] + 15sell[3])
        @test stage(buy[1]) == 1 && stage(sell[3]) == 3 && stage(demand[3]) == 3 && stage(cost2) == 2
        @test_logs optimize!(m)                          # everything is marked: no warning
        @test objective_value(m) ≈ 940
        @test objective_bound(m) ≈ 1000
        for f in (:primal, :dual), v in (sell[2], ret[2], buy[2])
            @test !haskey(decision_rule(v; formulation = f).coefficients, demand[3])
        end
        @test isempty(decision_rule(buy[1]).coefficients)
        # scalar declaration with Stage after `in`, stage / set_stage
        @variable(m, d in Uncertainty(Uniform(0, 1)), Stage(3))
        @test stage(d) == 3
        set_stage(d, 2); @test stage(d) == 2
        x = @variable(m, x_free)
        @test stage(x) === nothing
        set_stage(x, 3); @test stage(x) == 3
        @test_throws ArgumentError set_stage(x, 0)
        @test_throws ArgumentError Stage(0)
        # unmarked variable in a 3-stage model: the core warns, naming it
        @test_logs (:warn, r"Stage\(3\).*x_free") match_mode = :any (set_stage(x, 3); delete!(m.stage, x.index); optimize!(m))
    end

    @testset "5. print" begin
        m = shared_stock(tag = ValidInequality())
        set_stage(m[:sell][1], 2)
        s = sprint(print, m)
        println("--- print(model) output ---\n", s, "---------------------------")
        @test occursin("d1 ~ Uniform", s) && occursin("[ValidInequality]", s) && occursin("buy: 1", s)
        @test occursin("sell[1]: 2", s) && occursin("sell[2]: unmarked", s)
        @test occursin("Max", s) && occursin("Subject to", s)
        @test !isempty(sprint(show, m))
    end

    @testset "6. delete" begin
        m = newsvendor()
        optimize!(m)
        @test objective_value(m) ≈ 460
        @variable(m, extra >= 0)
        @constraint(m, c_extra, extra + m[:sell] <= 1000)
        optimize!(m)
        @test objective_value(m) ≈ 460
        @test isfinite(value(extra))                   # free in the model: any feasible rule
        nv = num_variables(m)
        delete(m, c_extra)
        delete(m, extra)
        @test !is_valid(m, extra) && !is_valid(m, c_extra) && num_variables(m) == nv - 1
        @test termination_status(m) == MOI.OPTIMIZE_NOT_CALLED      # results dropped by the modification
        @test err_msg(() -> value(m[:buy])) != ""
        optimize!(m)
        @test objective_value(m) ≈ 460
        r = decision_rule(m[:sell])
        @test r.coefficients[m[:d]] ≈ 1 && abs(r.constant) < 1e-6
        # deleting a variable that still appears in a constraint removes the term, like JuMP
        @variable(m, extra2 >= 0)
        @constraint(m, extra2 + m[:sell] <= 1000)
        delete(m, extra2)
        optimize!(m)
        @test objective_value(m) ≈ 460
        # deleting an uncertainty declaration
        @variable(m, e in Uncertainty(Uniform(0, 1)))
        optimize!(m)
        @test length(decision_rule(m[:sell]).uncertainties) == 2
        delete(m, e)
        optimize!(m)
        @test length(decision_rule(m[:sell]).uncertainties) == 1 && objective_value(m) ≈ 460
    end

    @testset "7. modification (set_normalized_rhs)" begin
        m = newsvendor()
        @constraint(m, cap, m[:buy] <= 200)
        optimize!(m)
        @test objective_value(m) ≈ 460
        set_normalized_rhs(cap, 100)
        @test normalized_rhs(cap) == 100
        optimize!(m)
        ref = newsvendor()
        @test occursin("belongs", err_msg(() -> @constraint(ref, m[:buy] <= 100)))   # variable of another model
        @constraint(ref, ref[:buy] <= 100)
        optimize!(ref)
        @test objective_value(m) ≈ objective_value(ref)
        @test objective_value(m) < 460 - 1
        @test objective_bound(m) ≈ objective_bound(ref)
    end

    @testset "8. results API" begin
        m = newsvendor()
        @test termination_status(m) == MOI.OPTIMIZE_NOT_CALLED && primal_status(m) == MOI.NO_SOLUTION
        @test err_msg(() -> objective_value(m)) != ""
        optimize!(m)
        for f in (:primal, :dual)
            @test termination_status(m; formulation = f) == MOI.OPTIMAL
            @test primal_status(m; formulation = f) == MOI.FEASIBLE_POINT
            @test has_values(m; formulation = f)
        end
        @test objective_value(m; formulation = :primal) ≈ 460 && objective_value(m; formulation = :dual) ≈ 1460 / 3
        @test value(m[:sell]; formulation = :dual) ≈ 100
        @test_throws ArgumentError termination_status(m; formulation = :sampled)
        # disabled formulation
        set_attribute(m, SolveDual(), false)
        @test get_attribute(m, SolveDual()) == false
        optimize!(m)
        @test termination_status(m; formulation = :dual) == MOI.OPTIMIZE_NOT_CALLED
        @test primal_status(m; formulation = :dual) == MOI.NO_SOLUTION
        msg = err_msg(() -> objective_value(m; formulation = :dual))
        @test occursin("disabled", msg) && occursin("SolveDual", msg)
        @test occursin("disabled", err_msg(() -> objective_bound(m)))
        @test objective_value(m) ≈ 460
        set_attribute(m, SolvePrimal(), false)
        @test occursin("All formulations are disabled", err_msg(() -> optimize!(m)))
        set_attribute(m, SolvePrimal(), true); set_attribute(m, SolveDual(), true)
        # expected_value is exact and differs from value(p) * value(q) when q depends on p
        m = silent(LDRModel(HiGHS.Optimizer))
        @variable(m, buy >= 0, Stage(1))
        @variable(m, d in Uncertainty(Uniform(80, 120)))
        @variable(m, p in Uncertainty(Uniform(10, 20)))
        @variable(m, sell >= 0)
        @variable(m, ret >= 0)
        @variable(m, q)
        @constraint(m, sell + ret <= buy)
        @constraint(m, sell <= d)
        @constraint(m, q == 2p)
        @objective(m, Max, -10buy + 8ret + p * sell)
        optimize!(m)
        @test expected_value(p * sell) ≈ 1500
        @test value(p) * value(sell) ≈ 1500
        e = expected_value(p * q)
        @test e ≈ 2 * (15^2 + 100 / 12)                                   # 2 E[p²]
        @test !isapprox(e, value(p) * value(q); rtol = 1e-2)
        r = decision_rule(q)
        Random.seed!(7)
        ps = rand(Uniform(10, 20), 100_000)
        sampled = mean(v * r(p => v) for v in ps)
        @test isapprox(sampled, e; rtol = 1e-2)
        @test expected_value(objective_function(m)) ≈ objective_value(m)
        @test expected_value(objective_function(m); formulation = :dual) ≈ objective_value(m; formulation = :dual)
        println("   item 8: E[p q] = ", e, ", value(p)*value(q) = ", value(p) * value(q), ", sampled = ", sampled)
    end

    @testset "9. Ξ roles" begin
        m0, m1 = shared_stock(), shared_stock(tag = ValidInequality())
        optimize!(m0); optimize!(m1)
        @test objective_value(m0) ≈ 920
        @test objective_value(m1) > objective_value(m0) + 1e-3
        println("   item 9: primal without = ", objective_value(m0), ", with ValidInequality = ", objective_value(m1))
        # untagged uncertainty-only row: error at declaration, naming both tags
        m = shared_stock()
        msg = err_msg(() -> @constraint(m, m[:d1] + m[:d2] <= 230))
        @test occursin("Truncate()", msg) && occursin("ValidInequality()", msg)
        println("   item 9 untagged message: ", msg)
        # a tagged row that involves decisions is rejected
        @test occursin("only involve uncertainties", err_msg(() -> @constraint(m, m[:d1] + m[:buy] <= 230, ValidInequality())))
        # Truncate() is accepted at declaration; the core refuses at optimize!
        mt = shared_stock(tag = Truncate())
        msg = err_msg(() -> optimize!(mt))
        @test occursin("Truncate()", msg) && occursin("not in the spike", msg)
        # the tag is on the row and survives set_normalized_rhs
        c = @constraint(m, m[:d1] + m[:d2] <= 230, ValidInequality())
        set_normalized_rhs(c, 220)
        optimize!(m)
        @test objective_value(m) > 0
    end

    @testset "10. attributes" begin
        m = newsvendor()
        d = m[:d]
        @test get_attribute(d, BreakPoints()) === nothing
        set_attribute(d, BreakPoints(), [100.0])
        @test get_attribute(d, BreakPoints()) == [100.0]
        @variable(m, extra >= 0)
        delete(m, extra)
        @test get_attribute(d, BreakPoints()) == [100.0]
        @test occursin("scalar uncertainties", err_msg(() -> set_attribute(m[:sell], BreakPoints(), [1.0])))
        set_attribute(d, BreakPoints(), nothing)
        @test get_attribute(d, BreakPoints()) === nothing
    end

    @testset "11. unsupported input" begin
        m = newsvendor()
        n0 = num_constraints(m; count_variable_in_set_constraints = true)
        msg = err_msg(() -> @constraint(m, m[:sell]^2 <= 1))
        @test occursin("quadratic", msg) && occursin("not supported", msg)
        println("   item 11 messages:\n     quadratic: ", msg)
        msg2 = err_msg(() -> @constraint(m, m[:d] * m[:sell] <= 1000))
        @test occursin("uncertainty × decision", msg2)
        msg3 = err_msg(() -> @constraint(m, [m[:buy], m[:sell], m[:ret]] in SecondOrderCone()))
        @test occursin("SecondOrderCone", msg3) || occursin("only supports affine", msg3)
        msg4 = err_msg(() -> set_lower_bound(m[:d], 0))
        @test occursin("uncertainty", msg4)
        msg5 = err_msg(() -> @variable(m, z >= 0, Uncertainty(Uniform(0, 1))))   # not `in`: ScalarVariable + non-set arg
        @test msg5 != ""
        msg6 = err_msg(() -> @variable(m, w in Uncertainty(Normal(0, 1))))
        @test occursin("unbounded", msg6)
        msg7 = err_msg(() -> @constraint(m, m[:sell] in MOI.Integer()))
        @test msg7 == ""     # integrality on a decision is accepted; stage restriction is enforced at optimize!
        msg8 = err_msg(() -> @variable(m, w2 in Uncertainty(Uniform(0, 1)), Int))
        @test occursin("integer", msg8)
        println("     recourse: ", msg2, "\n     SOC: ", msg3, "\n     bound on d: ", msg4, "\n     unbounded: ", msg6)
        @test num_constraints(m; count_variable_in_set_constraints = true) == n0 + 1  # only the accepted Integer row
        # no bridge variables ever reach the core: the backend has exactly the user's variables, no bridges
        m = newsvendor()
        optimize!(m)
        @test MOI.get(backend(m), MOI.NumberOfVariables()) == num_variables(m) == 4
        @test !JuMP.bridge_constraints(inner_model(m))
        @test length(m.solution.canonical.decisions) + length(m.solution.canonical.uncertainties) == num_variables(m)
        # core errors that depend on the whole model still surface at optimize!
        mi = newsvendor()
        @variable(mi, zi, Int)
        @test occursin("zi is in Stage(2)", err_msg(() -> optimize!(mi)))
    end

    @testset "12a. probe: JuMP Parameter" begin
        m = newsvendor()
        r = try
            @variable(m, cap in Parameter(150.0))
            @constraint(m, pcap, m[:buy] <= cap)
            optimize!(m)
            v1 = objective_value(m)
            set_parameter_value(cap, 100.0)
            optimize!(m)
            (v1, objective_value(m), parameter_value(cap))
        catch e
            sprint(showerror, e)
        end
        println("   12a result: ", r)
        ref = newsvendor(); @constraint(ref, ref[:buy] <= 100); optimize!(ref)
        @test r isa Tuple && r[1] ≈ 460 && r[2] ≈ objective_value(ref) && r[3] == 100.0
    end

    @testset "12b. probe: third-party builder typed build!(::JuMP.Model)" begin
        function build!(model::JuMP.Model)
            @variable(model, extra_stock >= 0)
            @constraint(model, extra_stock <= 10)
            @objective(model, Max, objective_function(model) + 2extra_stock)
            return extra_stock
        end
        m = newsvendor()
        msg = err_msg(() -> build!(m))
        println("   12b: build!(::LDRModel) -> ", first(msg, 80), " ...")
        @test_broken msg == ""                      # a typed ::JuMP.Model builder cannot take an LDRModel
        @test occursin("MethodError", msg)
        # Escape hatch: hand the inner model to the builder. Its variables/rows go to the inner JuMP.Model, are read
        # by optimize! (unmarked decisions), and LDRVariable(m, vref) wraps them for stages/values.
        inner = inner_model(m)
        # the builder reads and sets the objective of the *inner* model, where LDR variables do not exist: it sees
        # the same MOI objective through inner VariableRefs
        extra = build!(inner)
        x = LDRVariable(m, extra)
        optimize!(m)
        println("   12b: build!(inner_model(m)) works; objective = ", objective_value(m), ", value(extra_stock) = ", value(x))
        @test objective_value(m) ≈ 480 && value(x) ≈ 10
        # bypassing the wrapper leaves earlier results visible until the next optimize!/wrapper mutation:
        set_upper_bound(extra, 5.0)                    # a write on the inner JuMP.Model, invisible to the wrapper
        @test termination_status(m) == MOI.OPTIMAL     # stale, not invalidated (the wrapper cannot see the write)
        set_stage(x, 1)
        @test termination_status(m) == MOI.OPTIMIZE_NOT_CALLED
    end

    @testset "12c. probe: tag on a constraint" begin
        m = newsvendor()
        c = @constraint(m, m[:sell] + m[:ret] <= m[:buy], Stage(2))
        @test stage(c) == 2
        set_stage(c, 3); @test stage(c) == 3
        optimize!(m)
        @test objective_value(m) ≈ 460          # the tag is stored and ignored by the solve
        @test stage(@constraint(m, m[:sell] <= 1000)) === nothing
        @test occursin("[Stage(3)]", sprint(print, m))
    end

    @testset "12d. probe: copy_model" begin
        m = shared_stock(tag = ValidInequality())
        set_attribute(m[:d1], BreakPoints(), [100.0])
        set_stage(m[:sell][1], 2)
        c, rmap = copy_model(m)
        @test c isa LDRModel && c !== m
        @test stage(c[:buy]) == 1 && stage(c[:sell][1]) == 2 && stage(c[:sell][2]) === nothing
        @test get_attribute(c[:d1], BreakPoints()) == [100.0]
        @test c[:buy] != m[:buy] && rmap[m[:buy]] == c[:buy]
        silent(c); optimize!(c); optimize!(m)
        @test objective_value(c) ≈ objective_value(m)
        @test objective_bound(c) ≈ objective_bound(m)
        # independence: modifying the copy does not touch the original
        set_stage(c[:buy], 2)
        @test stage(m[:buy]) == 1
    end

    @testset "13. other JuMP API (informational)" begin
        m = newsvendor()
        buy, sell, ret, d = m[:buy], m[:sell], m[:ret], m[:d]
        @expression(m, profit, -10buy + 8ret + 15sell)
        @test profit isa AffExpr || profit isa LDRWrapper.LDRAffExpr
        @constraints(m, begin
            cap2, buy <= 500
            sell <= 1000
        end)
        @variables(m, begin
            z1 >= 0
            z2[1:2] >= 0
        end)
        @variable(m, w[i in 1:3, j in 1:3; i != j] >= 0)
        @variable(m, anon)
        @constraint(m, 0 <= z1 <= 5)
        @constraint(m, z2 .<= 3)
        @constraint(m, sum(z2) + sum(w) <= 7)
        @test num_constraints(m; count_variable_in_set_constraints = false) == 8
        @test length(all_constraints(m; include_variable_in_set_constraints = false)) == 8
        @test variable_by_name(m, "sell") == sell && constraint_by_name(m, "cap2") == cap2
        @test constraint_object(cap2).set == MOI.LessThan(500.0) && normalized_rhs(cap2) == 500
        @test is_valid(m, sell) && owner_model(sell) === m && index(sell) isa MOI.VariableIndex
        @test Dict(sell => 1)[sell] == 1 && sell != ret
        set_normalized_coefficient(m[:cap2], buy, 2.0)     # generic JuMP code writing through MOI; results go stale
        @test normalized_coefficient(cap2, buy) == 2.0
        optimize!(m)
        @test objective_value(m) ≈ 460
        set_normalized_coefficient(m[:cap2], buy, 3.0)
        @test termination_status(m) == MOI.OPTIMIZE_NOT_CALLED
        set_optimizer(m, HiGHS.Optimizer); set_attribute(m, "time_limit", 60.0)
        optimize!(m)
        @test objective_value(m) ≈ 460
        # things a user may call that do not work (recorded, not asserted)
        m = newsvendor(); optimize!(m)
        tried = Dict(
            "solution_summary(m)" => () -> solution_summary(m),
            "solve_time(m)" => () -> solve_time(m),
            "raw_status(m)" => () -> raw_status(m),
            "result_count(m)" => () -> result_count(m),
            "set_time_limit_sec(m, 1.0)" => () -> set_time_limit_sec(m, 1.0),
            "isempty(m) / empty!(m)" => () -> empty!(m),
            "relax_integrality(m)" => () -> relax_integrality(m),
            "write_to_file(m, \"x.mps\")" => () -> write_to_file(m, "x.mps"),
            "unsafe_backend(m)" => () -> unsafe_backend(m),
            "set_objective_coefficient(m, x, 1.0)" => () -> set_objective_coefficient(m, m[:sell], 1.0),
            "LowerBoundRef(x)" => () -> LowerBoundRef(m[:buy]),
            "value(constraint)" => () -> value(first(all_constraints(m; include_variable_in_set_constraints = false))),
            "dual(constraint)" => () -> dual(first(all_constraints(m; include_variable_in_set_constraints = false))),
            "all_constraints(m, AffExpr, S)" => () -> all_constraints(m, AffExpr, MOI.LessThan{Float64}),
            "@NLconstraint(m, ...)" => () -> @NLconstraint(m, sin(m[:sell]) <= 1),
        )
        for (k, f) in sort!(collect(tried); by = first)
            println("   does not work: ", rpad(k, 40), first(replace(err_msg(f), "\n" => " "), 90))
        end
    end

    @testset "Aqua" begin
        Aqua.test_piracies(LDRWrapper)
        # Aqua.test_ambiguities(LDRWrapper) needs a loadable package (it runs in a clean process and fails with
        # "Non-package (non-toplevel) module is not supported"); LDRWrapper is a plain module here, so run what Aqua
        # runs (Test.detect_ambiguities) over LDRWrapper and the modules whose functions it extends, keeping the
        # pairs that involve one of its methods.
        amb = filter(Test.detect_ambiguities(LDRWrapper, Base, JuMP, MOI, Distributions; recursive = true)) do (m1, m2)
            m1.module === LDRWrapper || m2.module === LDRWrapper
        end
        foreach(p -> println("   AMBIGUITY: ", p[1], "\n              ", p[2]), amb)
        @test isempty(amb)
    end
end

# ------------------------------------------------------------------ measurements (printed, not tested)
let
    src = readlines(joinpath(@__DIR__, "LDRWrapper.jl"))
    code = count(l -> (s = strip(l); !isempty(s) && !startswith(s, "#")), src)
    in_doc = false; doc_lines = 0
    for l in src
        if count("\"\"\"", l) == 1; in_doc = !in_doc; doc_lines += 1
        elseif in_doc; doc_lines += !isempty(strip(l)); end
    end
    added = Tuple{Module,Symbol,Int}[]
    for mod in (JuMP, MOI, Base, Base.Broadcast)
        for n in names(mod; all = true)
            isdefined(mod, n) || continue
            f = getfield(mod, n)
            f isa Function || continue
            k = count(mt -> mt.module === LDRWrapper, methods(f))
            k > 0 && push!(added, (mod, n, k))
        end
    end
    println("non-blank non-comment lines of LDRWrapper.jl: ", code, " (of which docstring lines: ", doc_lines, ")")
    println("methods added to JuMP/MOI/Base functions: ", sum(a[3] for a in added), " over ", length(added), " functions")
    println(join(("$(parentmodule(getfield(a[1], a[2])) === a[1] ? "" : "")$(a[1]).$(a[2]) x$(a[3])" for a in added), ", "))
end
