# Design B checklist (spike/SPIKE.md). Run from the repository root:
#     julia --project=spike spike/optimizer/checklist.jl
include(joinpath(@__DIR__, "LDROptimizer.jl"))
using .LDROptimizer
using Aqua, Distributions, HiGHS, JuMP, Random, Test
import TOML     # stdlib, only to write the throwaway Aqua package
const MOI = JuMP.MOI
const SpikeCore = LDROptimizer.SpikeCore

function ldr_model()
    m = LDRModel(HiGHS.Optimizer)
    set_silent(m)
    return m
end

function newsvendor(; integer = false)
    m = ldr_model()
    if integer
        @variable(m, buy >= 0, Int, Stage(1))
    else
        @variable(m, buy >= 0, Stage(1))
    end
    @variable(m, d in Uncertainty(Uniform(80, 120)))
    @variable(m, sell >= 0)
    @variable(m, ret >= 0)
    @constraint(m, sell + ret <= buy)
    @constraint(m, sell <= d)
    @objective(m, Max, -10buy + 8ret + 15sell)
    return m
end

errmsg(f) = try f(); "" catch e; sprint(showerror, e) end

const AQUA_UUID = Base.UUID("6d1c3b7e-0b8e-4f3e-9d43-5e1d2a7b9c01")
function aqua_package()
    root = mktempdir()
    dir = joinpath(root, "LDROptimizer")
    mkpath(joinpath(dir, "src"))
    deps = TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["deps"]
    keep = filter(p -> p.first in ("Distributions", "JuMP", "LinearAlgebra", "MathOptInterface"), deps)
    open(joinpath(dir, "Project.toml"), "w") do io
        println(io, "name = \"LDROptimizer\"\nuuid = \"$AQUA_UUID\"\n[deps]")
        foreach(((k, v),) -> println(io, "$k = \"$v\""), keep)
    end
    write(joinpath(dir, "src", "LDROptimizer.jl"),
        "Base.include(Base.__toplevel__, $(repr(joinpath(@__DIR__, "LDROptimizer.jl"))))\n")
    push!(LOAD_PATH, root)
    return Base.require(Base.PkgId(AQUA_UUID, "LDROptimizer"))
end

@testset "Design B: LDROptimizer" begin
    @testset "1. newsvendor" begin
        m = newsvendor()
        buy, sell, d = m[:buy], m[:sell], m[:d]
        optimize!(m)
        @test termination_status(m) == MOI.OPTIMAL
        @test primal_status(m) == MOI.FEASIBLE_POINT
        @test objective_value(m) ≈ 460
        @test objective_bound(m) ≈ 1460 / 3
        rb, rs = decision_rule(buy), decision_rule(sell)
        @test rb.constant ≈ 120 && isempty(rb.coefficients)
        @test abs(rs.constant) < 1e-6 && length(rs.coefficients) == 1 && rs.coefficients[d] ≈ 1
        @test rs(d => 90.0) ≈ 90 && rb(Dict(d => 90.0)) ≈ 120
        rd = decision_rule(d)                       # an uncertainty is its own rule
        @test rd.constant == 0.0 && rd.coefficients == Dict(d => 1.0)
        @test value(buy) ≈ 120
        @test value(sell) ≈ 100
        @test value(d) ≈ 100
        @test MOI.Utilities.state(backend(m)) == MOI.Utilities.EMPTY_OPTIMIZER   # copied = false: never attached
    end

    @testset "2. integer first stage" begin
        m = newsvendor(integer = true)
        @test is_integer(m[:buy]) && stage(m[:buy]) == 1
        optimize!(m)
        @test objective_value(m) ≈ 460
        @test value(m[:buy]) ≈ 120
    end

    @testset "3. vector uncertainty" begin
        m = ldr_model()
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
        # a vector declaration with a stage
        @variable(m, e[1:2] in Uncertainty(product_distribution([Uniform(0, 1), Uniform(0, 1)])), Stage(2))
        @test stage.(e) == [2, 2]
    end

    @testset "4. containers and stages (3-stage newsvendor)" begin
        m = ldr_model()
        @variable(m, b[t = 1:2] >= 0, Stage(t))              # b[1] = buy, b[2] = buy2
        @variable(m, sell[t = 2:3] >= 0, Stage(t))
        @variable(m, ret[t = 2:3] >= 0, Stage(t))
        @variable(m, demand[t = 2:3] in Uncertainty(Uniform(80, 120)), Stage(t))
        @variable(m, buy_cost2 in Uncertainty(Uniform(8, 14)), Stage(2))
        @constraint(m, sell[2] + ret[2] <= b[1] + b[2])
        @constraint(m, sell[2] <= demand[2])
        @constraint(m, sell[2] + ret[2] + sell[3] + ret[3] <= b[1] + b[2])
        @constraint(m, sell[3] <= demand[3])
        @objective(m, Max, -10b[1] - buy_cost2 * b[2] + 8ret[2] + 15sell[2] + 8ret[3] + 15sell[3])
        @test stage.(b) == [1, 2] && stage(sell[3]) == 3 && stage(demand[3]) == 3 && stage(buy_cost2) == 2
        @test_logs optimize!(m)                              # everything is marked: no warning
        @test objective_value(m) ≈ 940
        @test objective_bound(m) ≈ 1000
        for v in (sell[2], ret[2], b[2]), f in (PrimalLDR(), DualLDR())
            @test !haskey(decision_rule(v; formulation = f).coefficients, demand[3])
        end
        @test isempty(decision_rule(b[1]).coefficients)
        # set_stage: move buy2 to stage 1, then back
        set_stage(b[2], 1)
        @test stage(b[2]) == 1
        optimize!(m)
        @test isempty(decision_rule(b[2]).coefficients)
        set_stage(b[2], 2)
        optimize!(m)
        @test objective_value(m) ≈ 940
        # unmarked variable in a 3-stage model: warning naming it
        @variable(m, slack >= 0)
        @test_logs (:warn, r"Stage\(3\).*slack") optimize!(m)
        @test stage(slack) === nothing
    end

    @testset "5. print" begin
        m = newsvendor()
        str = sprint(print, m)
        println("---- print(model) ----\n", str, "\n----------------------")
        @test occursin("d in Uncertainty(Uniform", str) || occursin("d ∈ Uncertainty(Uniform", str)
        @test_broken occursin("Stage", str)       # stages are variable attributes: JuMP's print does not show them
        @test occursin("solver: LDROptimizer", sprint(show, m))
    end

    @testset "6. delete" begin
        m = newsvendor()
        sell, d = m[:sell], m[:d]
        @variable(m, extra >= 0)
        @constraint(m, c_extra, sell + extra <= 90)
        optimize!(m)
        @test objective_value(m) < 460 - 1
        delete(m, c_extra)
        delete(m, extra)
        optimize!(m)
        @test objective_value(m) ≈ 460
        r = decision_rule(sell)
        @test collect(keys(r.coefficients)) == [d] && r.coefficients[d] ≈ 1 && abs(r.constant) < 1e-6
        @test value(sell) ≈ 100
    end

    @testset "7. modification" begin
        m = newsvendor()
        @constraint(m, cap, m[:buy] <= 200)
        optimize!(m)
        @test objective_value(m) ≈ 460
        set_normalized_rhs(cap, 100)
        optimize!(m)
        m2 = newsvendor()
        @constraint(m2, m2[:buy] <= 100)
        optimize!(m2)
        @test objective_value(m) ≈ objective_value(m2)
        @test objective_value(m) < 460 - 1
        @test objective_bound(m) ≈ objective_bound(m2)
    end

    @testset "8. results API" begin
        m = newsvendor()
        sell = m[:sell]
        optimize!(m)
        @test get_attribute(m, FormulationAttribute(PrimalLDR(), MOI.TerminationStatus())) == MOI.OPTIMAL
        @test get_attribute(m, FormulationAttribute(DualLDR(), MOI.PrimalStatus())) == MOI.FEASIBLE_POINT
        @test get_attribute(m, FormulationAttribute(PrimalLDR(), MOI.ObjectiveValue())) ≈ 460
        @test get_attribute(m, FormulationAttribute(DualLDR(), MOI.ObjectiveValue())) ≈ 1460 / 3
        @test objective_bound(m) ≈ 1460 / 3
        @test relative_gap(m) ≈ (1460 / 3 - 460) / 460
        @test get_attribute(m, Formulations()) == [PrimalLDR(), DualLDR()]
        # fixed result slots: 1 = primal rule, 2 = sampled rule (not in the spike)
        @test result_count(m) == 1
        @test primal_status(m; result = 2) == MOI.NO_SOLUTION
        @test_throws MOI.ResultIndexBoundsError value(sell; result = 2)
        @test dual_status(m) == MOI.NO_SOLUTION
        @test solve_time(m) >= 0
        summary = sprint(show, solution_summary(m))
        @test occursin("objective_bound    : 4.86667e+02", summary) && occursin("relative_gap", summary)
        println("---- solution_summary ----\n", summary, "\n--------------------------")
        # the dual rule
        @test decision_rule(sell; formulation = DualLDR()).coefficients[m[:d]] ≈ 1 atol = 1e-6
        # disabled formulation
        set_attribute(m, Formulations(), [PrimalLDR()])
        optimize!(m)
        @test objective_value(m) ≈ 460
        @test get_attribute(m, FormulationAttribute(DualLDR(), MOI.PrimalStatus())) == MOI.NO_SOLUTION
        @test get_attribute(m, FormulationAttribute(DualLDR(), MOI.TerminationStatus())) == MOI.OPTIMIZE_NOT_CALLED
        @test_throws MOI.GetAttributeNotAllowed objective_bound(m)
        @test occursin("DualLDR() was not solved", errmsg(() -> objective_bound(m)))
        @test occursin("DualLDR() was not solved", errmsg(() -> decision_rule(sell; formulation = DualLDR())))
        set_attribute(m, Formulations(), [DualLDR()])
        optimize!(m)
        @test termination_status(m) == MOI.OPTIMAL && objective_bound(m) ≈ 1460 / 3
        @test primal_status(m) == MOI.NO_SOLUTION && result_count(m) == 0
        @test_throws MOI.ResultIndexBoundsError value(sell)
        @test_throws MOI.ResultIndexBoundsError objective_value(m)
        @test_throws ArgumentError set_attribute(m, Formulations(), [])
        # Silent, TimeLimitSec and InnerOptimizer are forwarded
        created = Any[]
        factory = () -> (o = HiGHS.Optimizer(); push!(created, o); o)
        set_attribute(m, InnerOptimizer(), factory)
        set_attribute(m, Formulations(), (PrimalLDR(), DualLDR()))
        set_time_limit_sec(m, 10.0)
        optimize!(m)
        @test get_attribute(m, InnerOptimizer()) === factory
        @test length(created) == 2 && all(o -> MOI.get(o, MOI.TimeLimitSec()) == 10.0 && MOI.get(o, MOI.Silent()), created)
        @test time_limit_sec(m) == 10.0 && get_attribute(m, MOI.Silent())
        @test solver_name(m) == "LDROptimizer"
        # INFEASIBLE only on the dual's evidence
        m = newsvendor()
        @constraint(m, m[:sell] >= m[:d] - 10)               # infeasible in truth at d > 105
        @constraint(m, m[:sell] <= 95)
        optimize!(m)
        @test get_attribute(m, FormulationAttribute(PrimalLDR(), MOI.TerminationStatus())) == MOI.INFEASIBLE
        @test get_attribute(m, FormulationAttribute(DualLDR(), MOI.TerminationStatus())) == MOI.INFEASIBLE
        @test termination_status(m) == MOI.INFEASIBLE        # the dual proves it
        set_attribute(m, Formulations(), [PrimalLDR()])
        optimize!(m)
        @test termination_status(m) == MOI.LOCALLY_INFEASIBLE && primal_status(m) == MOI.NO_SOLUTION
        # feasible in truth (y = max(d1, d2)), but no affine rule exists: y = d1 + d2 is forced, violating y ≤ 1
        m = ldr_model()
        @variable(m, d[1:2] in Uncertainty(product_distribution([Uniform(0, 1), Uniform(0, 1)])))
        @variable(m, y)
        @constraint(m, y .>= d)
        @constraint(m, y <= d[1] + d[2])
        @constraint(m, y <= 1)
        @objective(m, Min, y)
        optimize!(m)
        @test get_attribute(m, FormulationAttribute(PrimalLDR(), MOI.TerminationStatus())) == MOI.INFEASIBLE
        @test get_attribute(m, FormulationAttribute(DualLDR(), MOI.TerminationStatus())) == MOI.OPTIMAL
        @test termination_status(m) == MOI.LOCALLY_INFEASIBLE && result_count(m) == 0
        @test isfinite(objective_bound(m))
        # expected_value with a random price correlated with the decision
        m = ldr_model()
        @variable(m, d in Uncertainty(Uniform(80, 120)))
        @variable(m, p in Uncertainty(Uniform(13, 17)))
        @variable(m, sell >= 0)
        @constraint(m, sell <= d)
        @objective(m, Max, p * sell - 15sell)           # sell only when the price is high: sell depends on p
        optimize!(m)
        @test objective_value(m) ≈ 20 * 16 / 12 atol = 1e-6
        ev = expected_value(p * sell)
        @test abs(ev - value(p) * value(sell)) > 1
        @test expected_value(p * sell - 15sell) ≈ objective_value(m)
        @test expected_value(2.0 + sell) ≈ 2 + value(sell)
        r = decision_rule(sell)
        rng = Random.MersenneTwister(1)
        est = sum(1:200_000) do _
            dv, pv = rand(rng, Uniform(80, 120)), rand(rng, Uniform(13, 17))
            pv * r(d => dv, p => pv)
        end / 200_000
        @test abs(est - ev) / abs(ev) < 0.01
        @info "expected_value(p * sell) = $ev, value(p) * value(sell) = $(value(p) * value(sell)), sampled $est"
    end

    @testset "9. Ξ roles" begin
        function two(tag)
            m = ldr_model()
            @variable(m, buy >= 0, Stage(1))
            @variable(m, sell[1:2] >= 0)
            @variable(m, ret >= 0)
            @variable(m, d[1:2] in Uncertainty(Uniform(80, 120)))
            @constraint(m, sell[1] + sell[2] + ret <= buy)
            @constraint(m, sell .<= d)
            @objective(m, Max, -10buy + 8ret + 15sum(sell))
            tag === :valid && @constraint(m, xi, d[1] + d[2] <= 230, ValidInequality())
            tag === :untagged && @constraint(m, xi, d[1] + d[2] <= 230)
            tag === :truncate && @constraint(m, xi, d[1] + d[2] <= 230, Truncate())
            return m
        end
        m0, m1 = two(:none), two(:valid)
        optimize!(m0)
        optimize!(m1)
        @test objective_value(m0) ≈ 920
        @test objective_value(m1) > objective_value(m0) + 1e-3
        @test constraint_tags(m1[:xi]) == [ValidInequality()]
        err = errmsg(() -> optimize!(two(:untagged)))
        @test occursin("Truncate()", err) && occursin("ValidInequality()", err)
        @test occursin("not in the spike", errmsg(() -> optimize!(two(:truncate))))
        m = two(:untagged)
        @test termination_status(m) == MOI.OPTIMIZE_NOT_CALLED    # error raised at optimize!, not at declaration
    end

    @testset "10. attributes" begin
        m = newsvendor()
        d = m[:d]
        set_attribute(d, BreakPoints(), [100.0])
        @test get_attribute(d, BreakPoints()) == [100.0]
        @variable(m, z)
        delete(m, z)
        @test get_attribute(d, BreakPoints()) == [100.0]
        optimize!(m)
        @test objective_value(m) ≈ 460
        @test get_attribute(d, BreakPoints()) == [100.0]
        @test get_attribute(m[:sell], BreakPoints()) === nothing
    end

    @testset "11. unsupported input" begin
        m = newsvendor()
        err = errmsg(() -> @constraint(m, m[:sell]^2 <= 1))
        println("---- sell^2 <= 1 ----\n", err, "\n---------------------")
        @test occursin("ScalarQuadraticFunction", err) && occursin("not supported", err)
        @test occursin("not supported", errmsg(() -> @constraint(m, m[:d] * m[:sell] <= 1000)))
        @test occursin("not supported", errmsg(() -> @constraint(m, [m[:buy], m[:sell], m[:ret]] in SecondOrderCone())))
        @test num_constraints(m; count_variable_in_set_constraints = true) == 6   # nothing was added
        @constraint(m, 0 <= m[:sell] + m[:ret] <= 1000)        # Interval: read by the core, no bridge
        optimize!(m)
        @test objective_value(m) ≈ 460
        c = unsafe_backend(m).canonical
        @test length(SpikeCore.decisions(c)) + length(SpikeCore.uncertainties(c)) == num_variables(m)
        @test backend(m).optimizer isa LDROptimizer.Optimizer   # no bridge layer at all
    end

    @testset "12a. probe: JuMP parameters" begin
        m = newsvendor()
        @variable(m, q in Parameter(150.0))      # accepted: JuMP does not check supports_add_constrained_variable
        @constraint(m, m[:buy] <= q)
        err = errmsg(() -> optimize!(m))
        println("---- Parameter ----\n", err, "\n-------------------")
        @test_broken isempty(err)                # the core rejects VariableIndex-in-Parameter
        set_parameter_value(q, 100.0)            # stored in the cache; would be re-read at the next optimize!
        @test parameter_value(q) == 100.0
        @test_broken isempty(errmsg(() -> optimize!(m)))
    end

    @testset "12b. probe: typed third-party builder" begin
        function build!(m::JuMP.Model)          # knows nothing about LDR
            @variable(m, extra_buy >= 0)
            @constraint(m, extra_cap, extra_buy <= 10)
            return m
        end
        m = newsvendor()
        build!(m)
        set_stage(m[:extra_buy], 1)
        @constraint(m, m[:sell] + m[:ret] <= m[:buy] + m[:extra_buy])
        optimize!(m)
        @test objective_value(m) ≈ 460
        @test value(m[:buy]) ≈ 120 && stage(m[:extra_buy]) == 1
        @test m isa JuMP.Model                  # the LDR model *is* a JuMP.Model: typed builders just work
    end

    @testset "12c. probe: Stage tag on a constraint" begin
        m = newsvendor()
        @constraint(m, c2, m[:sell] >= 0, Stage(2))
        @constraint(m, c3, m[:sell] + m[:ret] <= 500, ValidInequality())
        # two tags: rejected by JuMP's macro at expansion ("Cannot specify more than one additional positional argument")
        @test_throws Exception @macroexpand @constraint(m, m[:sell] <= 500, Stage(2), ValidInequality())
        @test stage(c2) == 2 && constraint_tags(c2) == [Stage(2)]
        c4 = @constraint(m, m[:sell] <= 1e9)
        @test stage(c4) === nothing && isempty(constraint_tags(c4))
        @test get_attribute(c2, ConstraintTags()) == [Stage(2)]
        @test occursin("involves decisions", errmsg(() -> optimize!(m)))   # ValidInequality on a decision row
        delete(m, c3)
        optimize!(m)                                                         # no meaning yet: ignored
        @test objective_value(m) ≈ 460
    end

    @testset "12d. probe: copy_model" begin
        m = newsvendor()
        set_attribute(m[:d], BreakPoints(), [100.0])
        @constraint(m, tagged, m[:sell] <= 1000, Stage(2))
        m2, refmap = copy_model(m)
        set_optimizer(m2, () -> LDROptimizer.Optimizer(HiGHS.Optimizer); add_bridges = false)
        set_silent(m2)
        @test stage(m2[:buy]) == 1
        @test get_attribute(m2[:d], BreakPoints()) == [100.0]
        @test constraint_tags(refmap[tagged]) == [Stage(2)]
        optimize!(m2)
        @test objective_value(m2) ≈ 460 && objective_bound(m2) ≈ 1460 / 3
    end

    @testset "Aqua" begin
        # Aqua refuses a non-package module (`Main.LDROptimizer`), so load the same file as a throwaway package
        # from a temporary implicit environment (only for Aqua; the tests above use the included module).
        @test_throws r"Non-package" Aqua.test_ambiguities(LDROptimizer)
        P = aqua_package()
        Aqua.test_piracies(P)
        Aqua.test_ambiguities(P)
    end
end
