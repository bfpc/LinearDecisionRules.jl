using JuMP, LinearDecisionRules
using Ipopt, Distributions
using PyPlot

LDR = LinearDecisionRules

# Global parameters
demand = 0.5
inflow_dist = Uniform(0, 0.3)

# PWLF value function
begin
m = LDR.LDRModel(Ipopt.Optimizer)
set_silent(m)
@variable(m, vi in LDR.Uncertainty(distribution=Uniform(0, 1)))
@variable(m, inflow in LDR.Uncertainty(distribution=inflow_dist))
@variable(m, 0 <= vf <= 1)
@variable(m, gh >= 0.0)
@variable(m, 0 <= gt[1:3] <= 0.2)

@constraint(m, balance, vf == vi - gh + inflow)
@constraint(m, sum(gt) + gh == demand)

@objective(m, Min, gt[1] + 2gt[2] + 3gt[3] + vf^2/2 - vf + 0.5)

# Solve the primal LDR
set_attribute(m, LDR.SolvePrimal(), true)
set_attribute(m, LDR.SolveDual(), false)
set_attribute(vi, LDR.BreakPoints(), 5)
optimize!(m)

# Get the value function
VF = JuMP.Model()
@variable(VF, x)
@objective(VF, Min, 0.0)
LDR.set_parametric_objective!(VF, m, Dict(vi => x))

function eval_VF(V, x)
    fix(V[:x], x)
    JuMP.optimize!(V)
    return JuMP.objective_value(V)
end

set_optimizer(VF, Ipopt.Optimizer)
set_silent(VF)

xs = 0:0.01:1
ys = eval_VF.(VF, xs)
plt.figure()
plt.plot(xs, ys, label="PWLF Value Function")
end

# LDR value function
begin
m = LDR.LDRModel(Ipopt.Optimizer)
set_silent(m)
@variable(m, vi)
@variable(m, inflow in LDR.Uncertainty(distribution=inflow_dist))
@variable(m, 0 <= vf <= 1)
@variable(m, gh >= 0.0)
@variable(m, 0 <= gt[1:3] <= 0.2)

@constraint(m, balance, vf == vi - gh + inflow)
@constraint(m, sum(gt) + gh == demand)

@objective(m, Min, gt[1] + 2gt[2] + 3gt[3] + vf^2/2 - vf + 0.5)

# Solve the primal LDR
set_attribute(m, LDR.SolvePrimal(), true)
set_attribute(m, LDR.SolveDual(), false)
optimize!(m)

function eval_VF2(m, x)
    fix(m[:vi], x)
    JuMP.optimize!(m)
    return JuMP.objective_value(m)
end

xs = 0:0.01:1
ys = eval_VF2.(m, xs)
plt.plot(xs, ys, label="LDR Value Function")

end

# SAA value function
begin
Nscen = 100

m = Model(Ipopt.Optimizer)
set_silent(m)
@variable(m, vi)
@variable(m, 0 <= vf[1:Nscen] <= 1)
@variable(m, gh[1:Nscen] >= 0.0)
@variable(m, 0 <= gt[1:3, 1:Nscen] <= 0.2)
@variable(m, inflow[1:Nscen])
@constraint(m, [s=1:Nscen], vf[s] == vi - gh[s] + inflow[s])
@constraint(m, [s=1:Nscen], sum(gt[:,s]) + gh[s] == demand)
@objective(m, Min, (1/Nscen) * sum(gt[1,s] + 2gt[2,s] + 3gt[3,s] + vf[s]^2/2 - vf[s] + 0.5 for s=1:Nscen))

# Sample inflows
for s in 1:Nscen
    fix(inflow[s], rand(inflow_dist))
end

# Solve the SAA
optimize!(m)
function eval_VF3(m, x)
    fix(m[:vi], x)
    JuMP.optimize!(m)
    return JuMP.objective_value(m)
end

xs = 0:0.01:1
ys = eval_VF3.(m, xs)
plt.plot(xs, ys, label="SAA Value Function")
end

plt.legend()
plt.title("Value Function Comparison")
plt.savefig("1dtoy_value_function_comparison.pdf")
