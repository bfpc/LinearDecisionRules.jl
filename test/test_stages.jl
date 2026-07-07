module TestStages

include("common.jl")

Stage = LinearDecisionRules.Stage

function test_stage_uncertainty_forms()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)

    # keyword stage form
    @variable(
        m,
        d_kw in LinearDecisionRules.Uncertainty(;
            distribution = Uniform(0.0, 1.0),
            stage = 1,
        )
    )
    # attribute stage form
    @variable(
        m,
        d_attr in LinearDecisionRules.Uncertainty(; distribution = Uniform(0.0, 1.0))
    )
    set_attribute(d_attr, Stage(1))

    @test get_attribute(d_kw, Stage()) == 1
    @test get_attribute(d_attr, Stage()) == 1

    # @variable(m, x >= 0, Stage(1)) does not work
    @variable(m, x >= 0)
    set_attribute(x, Stage(1))
    @constraint(m, x >= d_kw + d_attr)
    @objective(m, Min, x)
    optimize!(m)
    @test termination_status(m) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    return nothing
end

function test_stage_constraint_validation()
    m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(m)
    @variable(
        m,
        d2 in LinearDecisionRules.Uncertainty(;
            distribution = Uniform(0.0, 1.0),
            stage = 2,
        )
    )
    @variable(m, x >= 0)
    # Maybe also add stage for x?
    @constraint(m, con_bad, x >= d2)
    set_attribute(con_bad, Stage(1))
    @objective(m, Min, x)
    @test_throws ErrorException optimize!(m)
    return nothing
end

function test_first_stage_no_warning_legacy_mode()
    @test_logs min_level = Logging.Warn begin
        m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
        set_silent(m)
        @variable(m, x >= 0, LinearDecisionRules.FirstStage)
        @variable(m, y >= 0)
        @variable(
            m,
            d in LinearDecisionRules.Uncertainty(; distribution = Uniform(0.0, 1.0))
        )
        @constraint(m, 2x + y >= 0.5 * d)
        @objective(m, Min, x + y)
        optimize!(m)
    end
    return nothing
end

function test_first_stage_mixed_with_stage_warns()
    @test_logs (:warn, r"Mixing FirstStage with Stage\(n\) is deprecated") begin
        m = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
        set_silent(m)
        @variable(m, x >= 0, LinearDecisionRules.FirstStage)
        @variable(m, y >= 0)
        set_attribute(y, Stage(2))
    end
    return nothing
end

end # TestStages module

TestStages.runtests()
