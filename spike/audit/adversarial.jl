# Adversarial probes, identical for the three designs (audit).
#   julia --project=spike spike/audit/adversarial.jl          # A, B, C each in a fresh process, then the math check
#   julia --project=spike spike/audit/adversarial.jl A|B|C|math
# Each probe prints one line:  PROBE <id> | <design> | <PASS|FAIL|INFO> | <details>
# PASS/FAIL are for probes with an expected outcome; INFO records behaviour where no single answer is "right".
const SPIKE = dirname(@__DIR__)
using Distributions, HiGHS, JuMP, Test
const MOI = JuMP.MOI

const KEY = isempty(ARGS) ? "" : ARGS[1]
if KEY == "A"
    include(joinpath(SPIKE, "core", "SpikeCore.jl"))
    include(joinpath(SPIKE, "wrapper", "LDRWrapper.jl"))
    const X = LDRWrapper
elseif KEY == "B"
    include(joinpath(SPIKE, "optimizer", "LDROptimizer.jl"))
    const X = LDROptimizer
elseif KEY == "C"
    include(joinpath(SPIKE, "hook", "LDRHook.jl"))
    const X = LDRHook
elseif KEY == "math"
    include(joinpath(SPIKE, "core", "SpikeCore.jl"))
end

clean(s) = first(replace(s, r"\s+" => " "), 260)
errmsg(f) = try f(); "" catch e; clean(sprint(showerror, e)) end
function probe(f, id)
    status, detail = try
        f()
    catch e
        ("FAIL", "uncaught: " * clean(sprint(showerror, e)))
    end
    println("PROBE $id | $KEY | $status | $detail")
end
check(ok::Bool, detail) = (ok ? "PASS" : "FAIL", detail)
r3(x) = x isa Real ? round(x; digits = 3) : x
tryval(f) = try r3(f()) catch e; "ERR(" * first(clean(sprint(showerror, e)), 90) * ")" end

# ---------------------------------------------------------------- per-design adapters (results API differs)
newmodel(; factory = HiGHS.Optimizer) = (m = X.LDRModel(factory); set_silent(m); m)
dualobj(m) = KEY == "A" ? objective_value(m; formulation = :dual) :
             KEY == "B" ? get_attribute(m, X.FormulationAttribute(X.DualLDR(), MOI.ObjectiveValue())) :
             objective_value(m; result = 2)
status_line(m) = "ts=$(tryval(() -> termination_status(m))) ps=$(tryval(() -> primal_status(m))) " *
                 "obj=$(tryval(() -> objective_value(m))) bound=$(tryval(() -> objective_bound(m)))"

function newsvendor(; stage1 = true, integer = false, m = newmodel())
    if stage1
        integer ? @variable(m, buy >= 0, Int, X.Stage(1)) : @variable(m, buy >= 0, X.Stage(1))
    else
        @variable(m, buy >= 0)
    end
    @variable(m, d in X.Uncertainty(Uniform(80, 120)))
    @variable(m, sell >= 0)
    @variable(m, ret >= 0)
    @constraint(m, c_stock, sell + ret <= buy)
    @constraint(m, c_dem, sell <= d)
    @objective(m, Max, -10buy + 8ret + 15sell)
    return m
end

function run_probes()
    # (a) delete an uncertainty variable, then optimize
    probe("a1 delete unused uncertainty") do
        m = newsvendor(); @variable(m, e in X.Uncertainty(Uniform(0, 1)))
        optimize!(m); delete(m, e); optimize!(m)
        check(objective_value(m) ≈ 460, status_line(m))
    end
    probe("a2 delete used uncertainty d (sell<=d becomes sell<=0)") do
        m = newsvendor(); optimize!(m)
        msg = errmsg(() -> delete(m, m[:d]))
        msg == "" || return ("FAIL", "delete errored: " * msg)
        msg2 = errmsg(() -> optimize!(m))
        msg2 == "" || return ("FAIL", "optimize! errored: " * msg2)
        check(isapprox(objective_value(m), 0; atol = 1e-6), status_line(m) * "; constraint now: " *
              tryval(() -> string(constraint_object(m[:c_dem]).func)))
    end
    probe("a3 delete one component of a vector uncertainty") do
        m = newmodel()
        @variable(m, e[1:2] in X.Uncertainty(product_distribution([Uniform(0, 1), Uniform(0, 1)])))
        @variable(m, x >= 0); @constraint(m, x >= e[1] + e[2]); @objective(m, Min, x)
        msg = errmsg(() -> delete(m, e[1]))
        msg2 = msg == "" ? errmsg(() -> optimize!(m)) : "(not reached)"
        ("INFO", "delete: " * (msg == "" ? "accepted" : msg) * " | optimize!: " * (msg2 == "" ? "ok " * status_line(m) : msg2))
    end
    probe("a4 delete all components of a vector uncertainty at once") do
        m = newmodel()
        @variable(m, e[1:2] in X.Uncertainty(product_distribution([Uniform(0, 1), Uniform(0, 1)])))
        @variable(m, x >= 0); @constraint(m, x >= 1); @objective(m, Min, x)
        msg = errmsg(() -> delete(m, e))
        msg == "" || return ("FAIL", "delete(m, e) errored: " * msg)
        optimize!(m)
        check(objective_value(m) ≈ 1, status_line(m))
    end

    # (b) optimize twice, no changes
    probe("b optimize! twice (inner models built per call)") do
        n = Ref(0)
        m = newsvendor(m = newmodel(factory = () -> (n[] += 1; HiGHS.Optimizer())))
        optimize!(m); v1, b1, k1 = objective_value(m), objective_bound(m), n[]
        t = @elapsed optimize!(m); v2, b2 = objective_value(m), objective_bound(m)
        check(v1 == v2 && b1 == b2, "same results=$(v1 == v2 && b1 == b2); inner optimizers created: $k1 after 1st, $(n[]) after 2nd " *
                                     "(2nd optimize! $(round(t * 1e3; digits = 1)) ms) -> $(n[] > k1 ? "recomputed, no caching" : "cached")")
    end

    # (c) no Stage marks at all
    probe("c no Stage marks (two-stage default)") do
        m = newsvendor(stage1 = false)
        logs = Test.collect_test_logs(() -> optimize!(m))[1]
        r = X.decision_rule(m[:buy])
        check(objective_value(m) ≈ 500 && isempty(logs), status_line(m) * "; buy rule = $(r.constant) + $(collect(values(r.coefficients))) d; " *
              "warnings=$(length(logs)) (expected: buy is Stage(2) and sees d, primal 500)")
    end

    # (d) @expression mixing uncertainties and decisions
    probe("d @expression with uncertainties in constraint and objective") do
        m = newmodel()
        @variable(m, buy >= 0, X.Stage(1)); @variable(m, d in X.Uncertainty(Uniform(80, 120)))
        @variable(m, sell >= 0); @variable(m, ret >= 0)
        @expression(m, short, d - sell)
        @expression(m, profit, -10buy + 8ret + 15sell + 0 * d)
        @constraint(m, sell + ret <= buy); @constraint(m, short >= 0)
        @objective(m, Max, profit + 2 * short - 2d + 2sell)
        optimize!(m)
        check(objective_value(m) ≈ 460 && X.expected_value(profit) ≈ 460,
              status_line(m) * "; E[profit]=$(tryval(() -> X.expected_value(profit))) value(profit)=$(tryval(() -> value(profit)))")
    end

    # (e) fix on a decision, fix on an uncertainty
    probe("e1 fix(buy, 100) then optimize") do
        m = newsvendor(); fix(m[:buy], 100; force = true); optimize!(m)
        ("INFO", status_line(m) * "; value(buy)=$(tryval(() -> value(m[:buy])))")
    end
    probe("e2 fix(d, 100) on an uncertainty") do
        m = newsvendor()
        msg = errmsg(() -> fix(m[:d], 100))
        msg2 = msg == "" ? errmsg(() -> optimize!(m)) : "(not reached)"
        where = msg != "" ? "at fix" : msg2 != "" ? "at optimize!" : "NEVER"
        check(where != "NEVER", "error $where: " * (msg != "" ? msg : msg2))
    end

    # (f) change the objective after a solve
    probe("f1 set_objective_coefficient(m, sell, 20) after solve, re-solve") do
        m = newsvendor(); optimize!(m)
        msg = errmsg(() -> set_objective_coefficient(m, m[:sell], 20.0))
        msg == "" || return ("FAIL", "set_objective_coefficient: " * msg)
        stale = tryval(() -> termination_status(m))
        optimize!(m)
        ref = newsvendor(); @objective(ref, Max, -10ref[:buy] + 8ref[:ret] + 20ref[:sell]); optimize!(ref)
        check(objective_value(m) ≈ objective_value(ref) && objective_bound(m) ≈ objective_bound(ref),
              "status right after modification=$stale; new obj=$(r3(objective_value(m))) ref=$(r3(objective_value(ref)))")
    end
    probe("f2 new @objective after solve, re-solve") do
        m = newsvendor(); optimize!(m)
        @objective(m, Max, -10m[:buy] + 8m[:ret] + 20m[:sell])
        stale = tryval(() -> termination_status(m)); optimize!(m)
        ref = newsvendor(); @objective(ref, Max, -10ref[:buy] + 8ref[:ret] + 20ref[:sell]); optimize!(ref)
        check(objective_value(m) ≈ objective_value(ref), "status right after @objective=$stale; obj=$(r3(objective_value(m))) ref=$(r3(objective_value(ref)))")
    end
    probe("f3 set_objective_sense(m, MIN_SENSE) after solve") do
        m = newsvendor(); optimize!(m); set_objective_sense(m, MIN_SENSE)
        stale = tryval(() -> termination_status(m)); optimize!(m)
        ("INFO", "status right after=$stale; " * status_line(m))
    end

    # (g) set_silent / set_time_limit_sec after a solve
    for (label, op) in (("set_silent", m -> set_silent(m)), ("unset_silent", m -> unset_silent(m)),
                        ("set_time_limit_sec(10)", m -> set_time_limit_sec(m, 10.0)),
                        ("set_attribute(\"presolve\",\"off\")", m -> set_attribute(m, "presolve", "off")))
        probe("g $label after a solve: do results survive?") do
            m = newsvendor(); optimize!(m)
            msg = errmsg(() -> op(m))
            msg == "" || return ("FAIL", "$label errored: " * msg)
            survived = errmsg(() -> value(m[:buy])) == "" && termination_status(m) == MOI.OPTIMAL
            check(survived, "after $label: " * status_line(m) * " value(buy)=$(tryval(() -> value(m[:buy])))")
        end
    end

    # (h) names in error messages
    probe("h1 untagged uncertainty-only row: message names the row?") do
        m = newsvendor(); @variable(m, d2 in X.Uncertainty(Uniform(80, 120)))
        msg = errmsg(() -> @constraint(m, xi_row, m[:d] + d2 <= 230))
        when = msg == "" ? "optimize!" : "declaration"
        msg == "" && (msg = errmsg(() -> optimize!(m)))
        check(occursin("xi_row", msg) || occursin("d2", msg), "at $when: " * msg)
    end
    probe("h2 bound on an uncertainty: message names the variable?") do
        m = newsvendor()
        msg = errmsg(() -> set_lower_bound(m[:d], 90)); when = "set_lower_bound"
        msg == "" && (when = "optimize!"; msg = errmsg(() -> optimize!(m)))
        check(occursin("d", msg) && msg != "", "at $when: " * msg)
    end
    probe("h3 integer decision in stage 2: message names the variable?") do
        m = newsvendor(); set_integer(m[:sell])
        msg = errmsg(() -> optimize!(m))
        check(occursin("sell", msg), msg)
    end
    probe("h4 random recourse d*sell: message names the constraint?") do
        m = newsvendor()
        msg = errmsg(() -> @constraint(m, rr_row, m[:d] * m[:sell] <= 1000)); when = "declaration"
        msg == "" && (when = "optimize!"; msg = errmsg(() -> optimize!(m)))
        check(occursin("rr_row", msg), "at $when: " * msg)
    end

    # (i) inner solver fails
    probe("i1 unbounded primal LDR (Max x, no upper bound)") do
        m = newmodel(); @variable(m, d in X.Uncertainty(Uniform(0, 1))); @variable(m, x >= 0)
        @constraint(m, x >= d); @objective(m, Max, x)
        msg = errmsg(() -> optimize!(m))
        ("INFO", msg == "" ? status_line(m) * " has_values=$(tryval(() -> has_values(m))) value(x)=$(tryval(() -> value(x)))" : "optimize! errored: " * msg)
    end
    probe("i2 infeasible (sell >= d - 10, sell <= 95)") do
        m = newsvendor(); @constraint(m, m[:sell] >= m[:d] - 10); @constraint(m, m[:sell] <= 95)
        msg = errmsg(() -> optimize!(m))
        ("INFO", msg == "" ? status_line(m) * " value(sell)=$(tryval(() -> value(m[:sell])))" : "optimize! errored: " * msg)
    end
    probe("i3 primal LDR infeasible, truth feasible (y >= max(d1,d2), y <= 1)") do
        m = newmodel()
        @variable(m, e[1:2] in X.Uncertainty(product_distribution([Uniform(0, 1), Uniform(0, 1)])))
        @variable(m, y); @constraint(m, y .>= e); @constraint(m, y <= e[1] + e[2]); @constraint(m, y <= 1)
        @objective(m, Min, y)
        msg = errmsg(() -> optimize!(m))
        ("INFO", msg == "" ? status_line(m) * " dual obj=$(tryval(() -> dualobj(m)))" : "optimize! errored: " * msg)
    end

    # (j) user forgets LDRModel
    probe("j1 set_optimizer(LDR model, HiGHS.Optimizer), then optimize!") do
        m = newsvendor()
        msg = errmsg(() -> set_optimizer(m, HiGHS.Optimizer))
        msg2 = errmsg(() -> optimize!(m))
        ("INFO", "set_optimizer: " * (msg == "" ? "accepted" : msg) * " | optimize!: " * (msg2 == "" ? status_line(m) : msg2))
    end
    probe("j2 plain Model(HiGHS.Optimizer) with the design's tags") do
        m = Model(HiGHS.Optimizer); set_silent(m)
        msg = errmsg(() -> newsvendor(m = m))
        msg == "" || return ("INFO", "error at declaration: " * msg)
        msg2 = errmsg(() -> optimize!(m))
        ("INFO", "declaration accepted | optimize!: " * (msg2 == "" ? status_line(m) * " (SILENT WRONG ANSWER?)" : msg2))
    end

    # (k) other first-hour things
    probe("k1 value(p*sell) vs expected_value(p*sell), sell depends on p") do
        m = newmodel(); @variable(m, d in X.Uncertainty(Uniform(80, 120))); @variable(m, p in X.Uncertainty(Uniform(13, 17)))
        @variable(m, sell >= 0); @constraint(m, sell <= d); @objective(m, Max, p * sell - 15sell); optimize!(m)
        ("INFO", "value(p*sell)=$(tryval(() -> value(p * sell))) expected_value(p*sell)=$(tryval(() -> X.expected_value(p * sell))) " *
                 "value(objective_function)=$(tryval(() -> value(objective_function(m)))) objective_value=$(tryval(() -> objective_value(m)))")
    end
    probe("k16 solver attributes reach the inner solver (raw \"presolve\" and time limit)") do
        made = Any[]
        m = newsvendor(m = newmodel(factory = () -> (o = HiGHS.Optimizer(); push!(made, o); o)))
        msg1 = errmsg(() -> set_attribute(m, "presolve", "off"))
        msg2 = errmsg(() -> set_time_limit_sec(m, 7.0))
        optimize!(m)
        raw = isempty(made) ? "none" : string(unique(MOI.get(o, MOI.RawOptimizerAttribute("presolve")) for o in made))
        tl = isempty(made) ? "none" : string(unique(MOI.get(o, MOI.TimeLimitSec()) for o in made))
        ok = msg1 == "" && msg2 == "" && raw == "[\"off\"]" && tl == "[7.0]"
        check(ok, "set_attribute: $(msg1 == "" ? "accepted" : first(msg1, 90)); set_time_limit_sec: $(msg2 == "" ? "accepted" : first(msg2, 90)); " *
                  "inner presolve=$raw inner time_limit=$tl")
    end
    probe("k19 is_solved_and_feasible") do
        m = newsvendor(); optimize!(m)
        check(tryval(() -> is_solved_and_feasible(m)) == true, "is_solved_and_feasible=$(tryval(() -> is_solved_and_feasible(m)))")
    end
    probe("k20 result = 2 / second result slot") do
        m = newsvendor(); optimize!(m)
        ("INFO", "result_count=$(tryval(() -> result_count(m))) primal_status(result=2)=$(tryval(() -> primal_status(m; result = 2))) " *
                 "value(sell; result=2)=$(tryval(() -> value(m[:sell]; result = 2))) objective_value(result=2)=$(tryval(() -> objective_value(m; result = 2)))")
    end
    probe("k2 Min sense: bound on the correct side") do
        m = newsvendor(); @objective(m, Min, 10m[:buy] - 8m[:ret] - 15m[:sell]); optimize!(m)
        check(objective_value(m) ≈ -460 && objective_bound(m) ≈ -1460 / 3,
              status_line(m) * " relative_gap=$(tryval(() -> relative_gap(m)))")
    end
    probe("k3 solution_summary") do
        m = newsvendor(); optimize!(m)
        msg = errmsg(() -> sprint(show, solution_summary(m)))
        check(msg == "", msg == "" ? "ok" : msg)
    end
    probe("k4 deterministic model (no uncertainty)") do
        m = newmodel(); @variable(m, 0 <= x <= 3); @objective(m, Max, 2x)
        msg = errmsg(() -> optimize!(m))
        check(msg == "" && objective_value(m) ≈ 6, msg == "" ? status_line(m) : msg)
    end
    probe("k5 empty model") do
        m = newmodel()
        msg = errmsg(() -> optimize!(m))
        ("INFO", msg == "" ? status_line(m) : "optimize! errored: " * msg)
    end
    probe("k6 unbounded distribution Normal(0,1)") do
        m = newmodel()
        msg = errmsg(() -> @variable(m, w in X.Uncertainty(Normal(0, 1)))); when = "declaration"
        if msg == ""
            @variable(m, x >= 0); @constraint(m, x >= m[:w]); @objective(m, Min, x)
            when = "optimize!"; msg = errmsg(() -> optimize!(m))
            msg == "" && (msg = "no error; " * status_line(m))
        end
        ("INFO", "at $when: " * msg)
    end
    probe("k7 anonymous uncertainty (set = Uncertainty(...))") do
        m = newmodel()
        msg = errmsg(() -> (w = @variable(m, set = X.Uncertainty(Uniform(0, 1))); @variable(m, x >= 0);
                            @constraint(m, x >= w); @objective(m, Min, x); optimize!(m)))
        check(msg == "" && objective_value(m) ≈ 0.5, msg == "" ? status_line(m) : msg)
    end
    probe("k8 integer first stage: dual bound (item 2)") do
        m = newsvendor(integer = true); optimize!(m)
        ("INFO", status_line(m) * " dual obj=$(tryval(() -> dualobj(m)))")
    end
    probe("k9 relax_integrality") do
        m = newsvendor(integer = true)
        msg = errmsg(() -> relax_integrality(m))
        msg == "" || return ("FAIL", msg)
        optimize!(m)
        check(objective_bound(m) ≈ 1460 / 3, status_line(m))
    end
    probe("k10 dual(constraint) / has_duals") do
        m = newsvendor(); optimize!(m)
        ("INFO", "has_duals=$(tryval(() -> has_duals(m))) dual(c_stock)=$(tryval(() -> dual(m[:c_stock])))")
    end
    probe("k11 write_to_file(model, .mof.json)") do
        m = newsvendor()
        f = joinpath(mktempdir(), "m.mof.json")
        msg = errmsg(() -> write_to_file(m, f))
        ("INFO", msg == "" ? "written, $(filesize(f)) bytes" : msg)
    end
    probe("k12 stage of a decision later than every uncertainty (Stage(5), no stage-5 uncertainty)") do
        m = newsvendor(); X.set_stage(m[:ret], 5)
        logs = Test.collect_test_logs(() -> optimize!(m))[1]
        ("INFO", status_line(m) * " warnings=$(length(logs)): " * join((clean(string(l.message)) for l in logs), " / "))
    end
    probe("k13 value(d) for an uncertainty; num_variables counts uncertainties") do
        m = newsvendor(); optimize!(m)
        check(value(m[:d]) ≈ 100, "value(d)=$(tryval(() -> value(m[:d]))) num_variables=$(num_variables(m)) all_variables=$(length(all_variables(m)))")
    end
    probe("k14 @variable(m, d in Uncertainty(...)) with bounds (d >= 0)") do
        m = newmodel()
        msg = errmsg(() -> @variable(m, w >= 0, set = X.Uncertainty(Uniform(0, 1)))); when = "declaration"
        if msg == ""
            @variable(m, x >= 0); @constraint(m, x >= m[:w]); @objective(m, Min, x)
            when = "optimize!"; msg = errmsg(() -> optimize!(m))
        end
        check(msg != "", "error at $when: " * msg)
    end
    probe("k15 copy_model after a solve, then solve the copy") do
        m = newsvendor(); optimize!(m)
        c, _ = copy_model(m)
        KEY == "B" && set_optimizer(c, () -> X.Optimizer(HiGHS.Optimizer); add_bridges = false)
        set_silent(c)
        msg = errmsg(() -> optimize!(c))
        check(msg == "" && objective_value(c) ≈ 460, msg == "" ? status_line(c) : msg)
    end
end

# ---------------------------------------------------------------- math check for item 2 (design-independent)
function math_check()
    function dual_with_buy(b; integer = false)
        m = Model()
        @variable(m, buy >= 0); integer && set_integer(buy)
        b === nothing || fix(buy, b; force = true)
        @variable(m, sell >= 0); @variable(m, ret >= 0); @variable(m, d)
        @constraint(m, sell + ret <= buy); @constraint(m, sell <= d)
        @objective(m, Max, -10buy + 8ret + 15sell)
        info = SpikeCore.ModelInfo(uncertainties = [SpikeCore.UncertaintyDecl(Uniform(80, 120), [index(d)], [nothing])],
            decision_stage = Dict(index(buy) => 1))
        c = SpikeCore.canonicalize(backend(m), info)
        du = SpikeCore.solve_dual(c, HiGHS.Optimizer)
        return du.objective_value, du.X[1, 1]
    end
    cont = dual_with_buy(nothing)
    int = dual_with_buy(nothing; integer = true)
    println("MATH dual LDR, continuous buy: obj=$(cont[1]) buy=$(cont[2])")
    println("MATH dual LDR, integer buy:    obj=$(int[1]) buy=$(int[2])")
    for b in (floor(cont[2]), ceil(cont[2]))
        println("MATH dual LDR with buy fixed at $b: obj=$(dual_with_buy(b)[1])")
    end
    # deterministic model through the core (no uncertainty): Max 2x, 0 <= x <= 3
    m = Model(); @variable(m, 0 <= x <= 3); @objective(m, Max, 2x)
    c = SpikeCore.canonicalize(backend(m), SpikeCore.ModelInfo())
    pr, du = SpikeCore.solve_primal(c, HiGHS.Optimizer), SpikeCore.solve_dual(c, HiGHS.Optimizer)
    println("MATH deterministic Max 2x, x in [0,3]: primal $(pr.termination_status) $(pr.objective_value); " *
            "dual $(du.termination_status) $(du.primal_status) $(du.objective_value) X=$(du.X)")
    # unbounded: Max x, x >= d, d ~ U(0,1)
    m = Model(); @variable(m, x >= 0); @variable(m, d); @constraint(m, x >= d); @objective(m, Max, x)
    c = SpikeCore.canonicalize(backend(m), SpikeCore.ModelInfo(uncertainties = [SpikeCore.UncertaintyDecl(Uniform(0, 1), [index(d)], [nothing])]))
    pr, du = SpikeCore.solve_primal(c, HiGHS.Optimizer), SpikeCore.solve_dual(c, HiGHS.Optimizer)
    println("MATH unbounded Max x: primal $(pr.termination_status) $(pr.primal_status) obj=$(pr.objective_value); " *
            "dual $(du.termination_status) $(du.primal_status) obj=$(du.objective_value)")
    truth(b) = 5b - 7(b - 80)^2 / 80          # exact E[profit] for buy = b in [80, 120] (optimal recourse)
    bstar = 80 + 40 * 5 / 7
    println("MATH true optimum (continuous): buy=$(round(bstar; digits = 3)) value=$(round(truth(bstar); digits = 3)); " *
            "integer: buy=109 value=$(round(truth(109); digits = 3)), buy=108 value=$(round(truth(108); digits = 3))")
end

if KEY == ""
    for k in ("A", "B", "C", "math")
        run(`$(Base.julia_cmd()) --project=$(SPIKE) $(@__FILE__) $k`)
    end
elseif KEY == "math"
    math_check()
else
    run_probes()
end
