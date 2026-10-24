# The T-period inventory problem of benchmark/timing.jl, written for the 0.4 package (`build_new`)
# and for the previous version (`build_legacy`). Include after `using JuMP, HiGHS, Distributions`
# and after defining `LDR` (the package to use).
const LO, HI = 60.0, 100.0
const C_ORDER, H_HOLD, B_BACK = 1.0, 0.5, 3.0
const XMAX, I0 = 150.0, 0.0

# `declare_uncertainty(m, T)` and `set_stage(x, t)` are the only differences between the packages.
function _inventory!(m, T, d, x, I, y)
    @constraint(m, I[1] == I0 + x[1] - d[1])
    @constraint(m, [t = 2:T], I[t] == I[t-1] + x[t] - d[t])
    @constraint(m, [t = 1:T], y[t] >= H_HOLD * I[t])
    @constraint(m, [t = 1:T], y[t] >= -B_BACK * I[t])
    @objective(m, Min, sum(C_ORDER * x[t] + y[t] for t in 1:T))
    return m
end

function build_new(T, optimizer)
    m = LDR.LDRModel(optimizer)
    dist = Distributions.Uniform(LO, HI)
    @variable(m, d[t = 1:T] in LDR.Uncertainty(dist), LDR.Stage(t + 1))
    @variable(m, 0 <= x[t = 1:T] <= XMAX, LDR.Stage(t))
    @variable(m, I[t = 1:T], LDR.Stage(t + 1))
    @variable(m, y[t = 1:T], LDR.Stage(t + 1))
    return _inventory!(m, T, d, x, I, y)
end

function build_legacy(T, optimizer)
    m = LDR.LDRModel(optimizer)
    dist = Distributions.Uniform(LO, HI)
    @variable(m, d[t = 1:T] in LDR.Uncertainty(; distribution = dist, stage = t + 1))
    @variable(m, 0 <= x[t = 1:T] <= XMAX, LDR.Stage(t))
    @variable(m, I[t = 1:T], LDR.Stage(t + 1))
    @variable(m, y[t = 1:T], LDR.Stage(t + 1))
    return _inventory!(m, T, d, x, I, y)
end
