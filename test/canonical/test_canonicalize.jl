# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestCanonicalize

using JuMP
using Test

import Distributions
import LinearDecisionRules
import LinearDecisionRules: CanonicalForm, ModelInfo, UncertaintyDeclaration, canonicalize

include(joinpath(@__DIR__, "..", "common.jl"))
import .TestCommon: error_message

include(joinpath(@__DIR__, "..", "problems", "library.jl"))
import .Problems: declare

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

# Sets for `_is_metadata_constraint`: one tagged as metadata, one not.
struct MetadataTag <: MOI.AbstractScalarSet end
struct OtherTag <: MOI.AbstractScalarSet end
LinearDecisionRules._is_metadata_constraint(::Type{MOI.VariableIndex}, ::Type{MetadataTag}) = true

# The model and the `ModelInfo` of the newsvendor of the problem library.
function newsvendor(; integer = false)
    p = Problems.newsvendor(; integer)
    return p.model, p.info
end

function test_newsvendor()
    model, info = newsvendor()
    form = canonicalize(backend(model), info)
    @test form isa CanonicalForm
    @test form.decisions == index.([model[:buy], model[:sell], model[:ret]])
    @test form.decision_names == ["buy", "sell", "ret"]
    @test form.decision_stages == [1, 2, 2]
    @test form.uncertainties == [index(model[:demand])]
    @test form.uncertainty_names == ["demand"]
    @test form.uncertainty_stages == [2]
    @test form.num_stages == 2
    @test form.integer == falses(3)
    @test form.lower == [0, 0, 0]
    @test form.upper == [Inf, Inf, Inf]
    @test size(form.A_e) == (0, 3) && size(form.B_e) == (0, 2)
    @test Matrix(form.A_u) == [-1 1 1; 0 1 0]
    @test Matrix(form.B_u) == [0 0; 0 1]
    @test size(form.A_l) == (0, 3)
    @test form.u_constraints == index.(all_constraints(model, AffExpr, MOI.LessThan{Float64}))
    @test form.sense == MOI.MAX_SENSE
    @test iszero(form.P) && iszero(form.Q)
    @test Matrix(form.C) == [-10 0; 15 0; 8 0]
    @test form.d == [0] && form.f == 0
    @test length(form.laws) == 1
    @test form.laws[1].columns == [2] && form.laws[1].stages == [2]
    @test sprint(show, form) ==
          "CanonicalForm with 3 decisions, 1 uncertainties, 2 stages and 2 rows"
    return
end

function test_bare_moi_model()
    model, _ = newsvendor()
    src = MOI.Utilities.Model{Float64}()
    map = MOI.copy_to(src, backend(model))
    d = map[index(model[:demand])]
    info = ModelInfo(;
        uncertainties = [UncertaintyDeclaration(Distributions.Uniform(80, 120), [d])],
        decision_stages = Dict(map[index(model[:buy])] => 1),
    )
    form = canonicalize(src, info)
    @test form.decision_names == ["buy", "sell", "ret"]
    @test form.uncertainties == [d]
    @test Matrix(form.A_u) == [-1 1 1; 0 1 0]
    @test Matrix(form.B_u) == [0 0; 0 1]
    @test Matrix(form.C) == [-10 0; 15 0; 8 0]
    # Unnamed variables are printed as JuMP prints them.
    src = MOI.Utilities.Model{Float64}()
    x = MOI.add_variable(src)
    @test canonicalize(src, ModelInfo()).decision_names == ["_[$(x.value)]"]
    return
end

# The three-stage newsvendor, declared out of stage order to check that columns are sorted by stage.
function three_stage_newsvendor(; unmarked_ret3 = false)
    p = Problems.newsvendor_three_stages(; unmarked_ret3, out_of_order = true)
    return p.model, p.info
end

function test_three_stage_newsvendor()
    model, info = three_stage_newsvendor()
    form = @test_logs canonicalize(backend(model), info)
    @test form.num_stages == 3
    @test form.decision_names == ["buy", "sell2", "ret2", "buy2", "sell3", "ret3"]
    @test form.decision_stages == [1, 2, 2, 2, 3, 3]
    @test form.uncertainty_names == ["demand2", "buy_cost2", "demand3"]
    @test form.uncertainty_stages == [2, 2, 3]
    @test [law.columns for law in form.laws] == [[4], [2], [3]]
    @test [law.stages for law in form.laws] == [[3], [2], [2]]
    @test Matrix(form.A_u) == [
        -1 1 1 -1 0 0
        0 1 0 0 0 0
        -1 1 1 -1 1 1
        0 0 0 0 1 0
    ]
    @test Matrix(form.B_u) == [0 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 1]
    @test Matrix(form.C) == [-10 0 0 0; 15 0 0 0; 8 0 0 0; 0 0 -1 0; 15 0 0 0; 8 0 0 0]
    return
end

function test_unmarked_warning()
    model, info = three_stage_newsvendor(; unmarked_ret3 = true)
    form = @test_logs (:warn, r"Stage\(3\): ret3\.$") canonicalize(backend(model), info)
    @test form.decision_stages[6] == 3
    # An unmarked uncertainty is listed too, in the same warning.
    model, info = three_stage_newsvendor(; unmarked_ret3 = true)
    info.uncertainties[1] = declare(Distributions.Uniform(80, 120), model[:demand3])
    @test_logs (:warn, r"Stage\(3\): ret3, demand3\.$") canonicalize(backend(model), info)
    # Two stages: no warning.
    model, info = newsvendor()
    @test_logs canonicalize(backend(model), info)
    # Many unmarked variables are summarized.
    model = Model()
    @variable(model, x[1:12])
    @variable(model, y)
    info = ModelInfo(; decision_stages = Dict(index(y) => 3))
    @test_logs (:warn, r"x\[10\] and 2 more\.$") canonicalize(backend(model), info)
    return
end

function test_interval_rows()
    model = Model()
    @variable(model, x)
    @variable(model, d)
    @constraint(model, c, 10 <= 2x + d <= 50)
    info = ModelInfo(; uncertainties = [declare(Distributions.Uniform(0, 1), d)])
    form = canonicalize(backend(model), info)
    @test Matrix(form.A_u) == [2;;] && Matrix(form.B_u) == [50 -1]
    @test Matrix(form.A_l) == [2;;] && Matrix(form.B_l) == [10 -1]
    @test form.u_constraints == form.l_constraints == [index(c)]
    # A constant in the function moves to the right-hand side; an equal interval is an equality.
    src = MOI.Utilities.Model{Float64}()
    x, y = MOI.add_variables(src, 2)
    f = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -1.0], [x, y]), 5.0)
    ci = MOI.add_constraint(src, f, MOI.Interval(10.0, 50.0))
    MOI.add_constraint(src, 1.0 * x + 2.0, MOI.Interval(3.0, 3.0))
    info = ModelInfo(; uncertainties = [UncertaintyDeclaration(Distributions.Uniform(0, 1), [y])])
    form = canonicalize(src, info)
    @test Matrix(form.A_u) == [1;;] && Matrix(form.B_u) == [45 1]
    @test Matrix(form.A_l) == [1;;] && Matrix(form.B_l) == [5 1]
    @test form.l_constraints[1] == ci
    @test Matrix(form.A_e) == [1;;] && Matrix(form.B_e) == [1 0]
    return
end

# Each output of a vector row is a scalar row of the same constraint.
function test_vector_rows()
    model = Model()
    @variable(model, x[1:2])
    @variable(model, d)
    @constraint(model, le, [x[1] - d, 2x[2] + 1] in MOI.Nonpositives(2))
    @constraint(model, ge, [x[1] + x[2] - 3, x[2]] in MOI.Nonnegatives(2))
    @constraint(model, eq, [x[1] - x[2]] in MOI.Zeros(1))
    info = ModelInfo(; uncertainties = [declare(Distributions.Uniform(0, 1), d)])
    form = canonicalize(backend(model), info)
    @test Matrix(form.A_u) == [1 0; 0 2] && Matrix(form.B_u) == [0 1; -1 0]
    @test form.u_constraints == [index(le), index(le)]
    @test Matrix(form.A_l) == [1 1; 0 1] && Matrix(form.B_l) == [3 0; 0 0]
    @test form.l_constraints == [index(ge), index(ge)]
    @test Matrix(form.A_e) == [1 -1] && Matrix(form.B_e) == [0 0]
    @test form.e_constraints == [index(eq)]
    # An output with uncertainties only is an error naming the constraint.
    @constraint(model, bad, [x[1], d - 1] in MOI.Nonnegatives(2))
    msg = error_message(() -> canonicalize(backend(model), info))
    @test occursin("The constraint bad involves only uncertainties (d)", msg)
    return
end

function test_variable_bounds_and_integrality()
    model = Model()
    @variable(model, 1 <= x <= 4, Int)
    @variable(model, y, Bin)
    @variable(model, z == 3)
    @variable(model, 0 <= w)
    @constraint(model, x + y + z + w <= 10)
    stages = Dict(index(x) => 1, index(y) => 1)
    form = canonicalize(backend(model), ModelInfo(; decision_stages = stages))
    @test form.integer == [true, true, false, false]
    @test form.binary == [false, true, false, false]
    @test form.lower == [1, -Inf, 3, 0]
    @test form.upper == [4, Inf, 3, Inf]
    @test form.decision_stages == [1, 1, 2, 2]
    return
end

function test_parameters()
    model = Model()
    @variable(model, p in Parameter(3.0))
    @variable(model, x >= 0)
    @variable(model, d)
    @constraint(model, c1, x <= d + p)
    @constraint(model, c2, p * x + 2x >= 1)
    @constraint(model, c3, p * d + x <= 4)
    @constraint(model, c4, 2p >= 4)
    @objective(model, Min, p * x + p^2 + p * d + 2p + x)
    info = ModelInfo(;
        uncertainties = [declare(Distributions.Uniform(0, 1), d)],
        decision_stages = Dict(index(p) => 1),
    )
    form = canonicalize(backend(model), info)
    @test form.decisions == [index(x)]
    @test form.uncertainties == [index(d)]
    @test form.num_stages == 2
    @test Matrix(form.A_u) == [1; 1;;] && Matrix(form.B_u) == [3 1; 4 -3]
    @test form.u_constraints == index.([c1, c3])
    @test Matrix(form.A_l) == [5;;] && Matrix(form.B_l) == [1 0]
    @test Matrix(form.C) == [4 0]
    @test form.d == [3] && form.f == 9 + 6
    # `p` alone (in c1 and in `2p`) is kept symbolic; its products are values.
    terms = form.parameters
    @test terms.parameters == [index(p)] && terms.values == [3]
    @test terms.b_u == [0, 4] && Matrix(terms.Bp_u) == [1; 0;;]
    @test terms.b_l == [1] && Matrix(terms.Bp_l) == [0;;]
    @test isempty(terms.b_e) && size(terms.Bp_e) == (0, 1)
    @test terms.f == 9 && terms.fp == [2]
    @test form.B_u[:, 1] == terms.b_u + terms.Bp_u * terms.values
    @test form.f == terms.f + terms.fp' * terms.values
    # A violated parameter-only row is an error.
    set_parameter_value(p, 1.0)
    @test occursin(
        "c4 has no decisions nor uncertainties",
        error_message(() -> canonicalize(backend(model), info)),
    )
    return
end

# x'Px + x'Cξ + η'Qη + d'η + f at a point, to compare with MOI's evaluation.
function evaluate(form::CanonicalForm, x, η)
    ξ = [1; η]
    return x' * form.P * x + x' * form.C * ξ + η' * form.Q * η + form.d' * η + form.f
end

function test_quadratic_objective_conventions()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @variable(model, d)
    @variable(model, e)
    @variable(model, p in Parameter(2.0))
    @objective(
        model,
        Min,
        x^2 + 2x * y + 3y^2 - 4x * d + d^2 + d * e + 5d + 7 + 2x + p * x + p^2 + p * d + y * e,
    )
    info = ModelInfo(;
        uncertainties = [
            declare(
                Distributions.product_distribution([
                    Distributions.Uniform(0, 1),
                    Distributions.Uniform(0, 1),
                ]),
                d,
                e,
            ),
        ],
    )
    form = canonicalize(backend(model), info)
    @test Matrix(form.P) == [1 1; 1 3]
    @test Matrix(form.C) == [4 -4 0; 0 0 1]
    @test Matrix(form.Q) == [1 0.5; 0.5 0]
    @test form.d == [7, 0]
    @test form.f == 11
    @test form.P == form.P' && form.Q == form.Q'
    obj = MOI.get(backend(model), MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}())
    vals =
        Dict(index(x) => 1.5, index(y) => -2.0, index(d) => 0.25, index(e) => 3.0, index(p) => 2.0)
    @test MOI.Utilities.eval_variables(v -> vals[v], obj) ≈ evaluate(form, [1.5, -2.0], [0.25, 3.0])
    # (x - d)² in a Max problem: the sense is kept.
    model = Model()
    @variable(model, x)
    @variable(model, d)
    @objective(model, Max, -(x - d)^2)
    form = canonicalize(
        backend(model),
        ModelInfo(; uncertainties = [declare(Distributions.Uniform(0, 1), d)]),
    )
    @test form.sense == MOI.MAX_SENSE
    @test Matrix(form.P) == [-1;;] && Matrix(form.C) == [0 2] && Matrix(form.Q) == [-1;;]
    # Objectives that are a single variable, or absent.
    model = Model()
    @variable(model, x)
    @objective(model, Min, x)
    @test Matrix(canonicalize(backend(model), ModelInfo()).C) == [1;;]
    model = Model()
    @variable(model, x)
    form = canonicalize(backend(model), ModelInfo())
    @test form.sense == MOI.FEASIBILITY_SENSE && iszero(form.C) && form.f == 0
    return
end

function test_error_uncertainty_only_row()
    model, info = newsvendor()
    @constraint(model, cap, model[:demand] <= 115)
    msg = error_message(() -> canonicalize(backend(model), info))
    @test occursin("The constraint cap involves only uncertainties (demand)", msg)
    @test occursin("ValidInequality()", msg) && occursin("Truncate()", msg)
    # Unnamed constraints are printed.
    model, info = newsvendor()
    @constraint(model, model[:demand] <= 115)
    msg = error_message(() -> canonicalize(backend(model), info))
    @test occursin("The constraint `demand <= 115` involves only uncertainties (demand)", msg)
    return
end

function test_unnamed_constraints_are_printed()
    model = Model()
    @variable(model, x[1:2])
    @variable(model, d[1:2])
    @variable(model, anonymous)
    set_name(anonymous, "")
    src = backend(model)
    quadratic = @constraint(model, x[1]^2 - 2x[1] * d[1] + x[2] >= 1)
    vector = @constraint(model, [x[1] - 1, d[2]] in MOI.Nonnegatives(2))
    cases = (
        @constraint(model, d[1] + d[2] <= 190) => "`d[1] + d[2] <= 190`",
        @constraint(model, 1 <= 2x[1] - 0.5d[1] + 3 <= 4.5) => "`-2 <= 2 x[1] - 0.5 d[1] <= 1.5`",
        @constraint(model, x[1] == -d[2]) => "`x[1] + d[2] == 0`",
        quadratic => "`x[1]² - 2 x[1] * d[1] + x[2] >= 1`",
        vector => "`[x[1] - 1, d[2]] in Nonnegatives(2)`",
        @constraint(model, d[1] in MOI.LessThan(110.0)) => "`d[1] <= 110`",
        @constraint(model, anonymous >= 0) => "`_[$(index(anonymous).value)] >= 0`",
    )
    for (c, text) in cases
        @test LinearDecisionRules._constraint_name(src, index(c)) == text
    end
    return
end

function test_error_truncate()
    cases = (
        (m, d) -> @constraint(m, cap, d == 100),
        (m, d) -> @constraint(m, cap, d <= m[:buy]),
        (m, d) -> @constraint(m, cap, 100 <= d <= 90),
        (m, d) -> @constraint(m, cap, d >= 130),
    )
    texts = (
        "cap is tagged `Truncate()`, but it is an equality",
        "cap is tagged `Truncate()`, but it has decisions",
        "The constraints cap tagged `Truncate()` condition the law of the uncertainty demand on " *
        "[100.0, 90.0], whose probability is zero.",
        "on [130.0, 120.0], whose probability is zero",
    )
    for (add, text) in zip(cases, texts)
        model, info = newsvendor()
        cap = add(model, model[:demand])
        info = ModelInfo(;
            uncertainties = info.uncertainties,
            decision_stages = info.decision_stages,
            truncate_rows = Set{MOI.ConstraintIndex}([index(cap)]),
        )
        @test occursin(text, error_message(() -> canonicalize(backend(model), info)))
    end
    return
end

function test_valid_rows()
    model, info = newsvendor()
    cap = @constraint(model, cap, 85 <= model[:demand] <= 115)
    valid = Set{MOI.ConstraintIndex}([index(cap)])
    info = ModelInfo(; uncertainties = info.uncertainties, valid_rows = valid)
    form = canonicalize(backend(model), info)
    @test Matrix(form.W_valid) == [1; -1;;] && form.h_valid == [85, -115]
    @test form.valid_constraints == [index(cap), index(cap)]
    @test size(form.A_u, 1) == 2
    # Tagged rows with decisions, equalities, or violated at the mean are errors.
    model, _ = newsvendor()
    d, sell = model[:demand], model[:sell]
    tests = (
        :vx => @build_constraint(d + sell <= 115),
        :veq => @build_constraint(d == 100),
        :vmean => @build_constraint(d >= 101),
    )
    for (name, con) in tests
        ci = add_constraint(model, con, string(name))
        info = ModelInfo(;
            uncertainties = info.uncertainties,
            valid_rows = Set{MOI.ConstraintIndex}([index(ci)]),
        )
        msg = error_message(() -> canonicalize(backend(model), info))
        @test occursin("$name is tagged `ValidInequality()`", msg)
        @test occursin(name == :vx ? "decisions" : name == :veq ? "equality" : "mean", msg)
        delete(model, ci)
    end
    return
end

function test_error_bound_on_uncertainty()
    model, info = newsvendor()
    set_lower_bound(model[:demand], 0)
    msg = error_message(() -> canonicalize(backend(model), info))
    @test occursin("The uncertainty demand has a bound or integrality constraint", msg)
    model, info = newsvendor()
    set_integer(model[:demand])
    @test occursin(
        "The uncertainty demand",
        error_message(() -> canonicalize(backend(model), info)),
    )
    return
end

function test_error_integer_after_stage_1()
    model, info = newsvendor()
    @variable(model, z, Int)
    msg = error_message(() -> canonicalize(backend(model), info))
    @test occursin("The integer decision z is in Stage(2) (it is unmarked)", msg)
    model, info = newsvendor()
    @variable(model, z, Bin)
    info.decision_stages[index(z)] = 2
    msg = error_message(() -> canonicalize(backend(model), info))
    @test occursin("The integer decision z is in Stage(2);", msg)
    # In stage 1 it is fine.
    model, info = newsvendor(; integer = true)
    @test canonicalize(backend(model), info).integer == [true, false, false]
    return
end

function test_error_random_recourse()
    model, info = newsvendor()
    @constraint(model, rr, model[:demand] * model[:sell] <= 1000)
    msg = error_message(() -> canonicalize(backend(model), info))
    @test occursin(
        "rr multiplies the decision sell by the uncertainty demand (random recourse)",
        msg,
    )
    return
end

function test_error_quadratic_constraint()
    model, info = newsvendor()
    @constraint(model, q, model[:sell]^2 <= 1)
    msg = error_message(() -> canonicalize(backend(model), info))
    @test occursin("The constraint q has the product of sell and sell; quadratic constraints", msg)
    model, info = newsvendor()
    @constraint(model, q, model[:demand]^2 + model[:sell] <= 1)
    @test occursin(
        "product of demand and demand",
        error_message(() -> canonicalize(backend(model), info)),
    )
    return
end

function test_error_unsupported_constraint()
    model, info = newsvendor()
    @constraint(model, [model[:buy], model[:sell], model[:ret]] in MOI.ExponentialCone())
    msg = error_message(() -> canonicalize(backend(model), info))
    @test occursin("VectorOfVariables-in-MathOptInterface.ExponentialCone", msg)
    @test occursin("not supported", msg)
    return
end

function test_metadata_constraint()
    model, info = newsvendor()
    @constraint(model, model[:buy] in MetadataTag())
    @test canonicalize(backend(model), info).decision_names == ["buy", "sell", "ret"]
    @constraint(model, model[:buy] in OtherTag())
    @test occursin("OtherTag", error_message(() -> canonicalize(backend(model), info)))
    return
end

function test_error_unobservable_equality()
    model, info = newsvendor()
    @constraint(model, eq, model[:buy] == model[:demand])
    msg = error_message(() -> canonicalize(backend(model), info))
    @test occursin("The equality constraint eq has the uncertainty demand (Stage(2))", msg)
    @test !occursin("Integer", msg)
    # A stage-2 decision observes it.
    model, info = newsvendor()
    @constraint(model, eq, model[:buy] + model[:sell] == model[:demand])
    @test size(canonicalize(backend(model), info).A_e) == (1, 3)
    # An integer decision observes no uncertainty, even of stage 1.
    model, info = newsvendor(; integer = true)
    @constraint(model, eq, model[:buy] == model[:demand])
    info.uncertainties[1] = declare(Distributions.Uniform(80, 120), model[:demand]; stages = [1])
    msg = error_message(() -> canonicalize(backend(model), info))
    @test occursin("eq has the uncertainty demand (Stage(1))", msg)
    @test occursin("Integer decisions observe none.", msg)
    return
end

function test_error_declarations()
    model, info = newsvendor()
    d, buy = index(model[:demand]), index(model[:buy])
    bad = (
        ModelInfo(;
            uncertainties = [UncertaintyDeclaration(Distributions.Uniform(0, 1), [d, buy])],
        ) => "has dimension 1, but 2 variables",
        ModelInfo(;
            uncertainties = [info.uncertainties[1], info.uncertainties[1]],
        ) => "demand is declared as an uncertainty twice",
        ModelInfo(;
            uncertainties = info.uncertainties,
            decision_stages = Dict(d => 1),
        ) => "The uncertainty demand has a decision stage",
        ModelInfo(; decision_stages = Dict(buy => 0)) => "buy has Stage(0)",
        ModelInfo(;
            uncertainties = [declare(Distributions.Uniform(0, 1), model[:demand]; stages = [0])],
        ) => "demand has Stage(0)",
        ModelInfo(;
            uncertainties = [
                UncertaintyDeclaration(Distributions.Uniform(0, 1), [MOI.VariableIndex(99)]),
            ],
        ) => "The uncertainty with index 99 is not a variable of the model",
        ModelInfo(;
            decision_stages = Dict(MOI.VariableIndex(99) => 1),
        ) => "The variable with index 99 has a stage, but it is not in the model",
        ModelInfo(;
            breakpoints = Dict(d => [100.0]),
        ) => "The variable demand has breakpoints, but it is not an uncertainty",
        ModelInfo(;
            uncertainties = [UncertaintyDeclaration(Distributions.Wishart(3, [1.0 0; 0 1]), [d])],
        ) => "univariate or multivariate",
    )
    for (info, msg) in bad
        @test occursin(msg, error_message(() -> canonicalize(backend(model), info)))
    end
    return
end

function test_replace()
    model, info = newsvendor()
    form = canonicalize(backend(model), info)
    replaced = LinearDecisionRules._replace(form; f = 1.0, d = [2.0])
    @test replaced.f == 1.0 && replaced.d == [2.0]
    for name in fieldnames(CanonicalForm)
        name in (:f, :d) || @test getfield(replaced, name) === getfield(form, name)
    end
    # A misspelled keyword is an error, not a field left unchanged.
    @test_throws ArgumentError LinearDecisionRules._replace(form; lwas = form.laws)
    return
end

end # module TestCanonicalize

TestCanonicalize.runtests()
