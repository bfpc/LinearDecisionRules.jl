include(joinpath(@__DIR__, "recursion.jl"))

pwldr_models, pwl_vfs = hydro_thermal_rpwldr(; stages = 12, rees=1:1, subsys=1:1, stored_energy_breakpoints=15, inflow_breakpoints=15, stored_energy_dist = :uni);
sddp_models = hydro_thermal_sddp(; stages = 12, rees=1:1, subsys=1:1, inflow_dist = :tri, train=true, time_limit=20);

function eval_VF(V, x)
    fix.(V[:stored_energy], x)
    JuMP.optimize!(V)
    return JuMP.objective_value(V)
end

vf11 = SDDP.ValueFunction(sddp_models[11])

ys_pwl = eval_VF.(pwl_vfs[11], [[x] for x in 0:200])
ys_sddp = [SDDP.evaluate(vf11, Dict(Symbol("stored_energy[1]") => x))[1] for x in 0:200]

plt.plot(0:200, ys_pwl, label="PWLDR")
plt.plot(0:200, ys_sddp, label="SDDP")
plt.title("Comparing Value functions")
plt.xlabel("Stored Energy")
plt.ylabel("Cost-to-go")
plt.legend()
