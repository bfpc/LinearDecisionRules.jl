# get_decision(m, vf, [inflow=>0.0342])
# get_decision(m, vf, inflow=>0.0342, demand=>33)

"""
    get_decision(m, x, η)

Coefficient of η in the LDR of x
"""
function get_decision(model, x, η; dual = false)
    var_to_column = model.ext[:var_to_column] # type assert
    column_to_canonical = model.ext[:column_to_canonical] # type assert
    i = column_to_canonical[var_to_column[x]]
    j = column_to_canonical[var_to_column[η]]
    @assert haskey(model.cache_uncertainty, η)
    @assert !haskey(model.cache_uncertainty, x)
    @assert JuMP.is_valid(model, η)
    @assert JuMP.is_valid(model, x)
    if dual
        if !model.solve_dual
            error("SolveDual() is set to false, no result available.")
        end
        return value(model.dual_model[:X][i,j+1])
    end
    if !model.solve_primal
        error("SolvePrimal() is set to false, no result available.")
    end
    return value(model.primal_model[:X][i,j+1])
end

"""
    get_decision(m, x)

Constant term in the LDR of x
"""
function get_decision(model, x; dual = false)
    var_to_column = model.ext[:var_to_column] # type assert
    column_to_canonical = model.ext[:column_to_canonical] # type assert
    i = column_to_canonical[var_to_column[x]]
    @assert !haskey(model.cache_uncertainty, x)
    @assert JuMP.is_valid(model, x)
    if dual
        if !model.solve_dual
            error("SolveDual() is set to false, no result available.")
        end
        return value(model.dual_model[:X][i,1])
    end
    if !model.solve_primal
        error("SolvePrimal() is set to false, no result available.")
    end
    return value(model.primal_model[:X][i,1])
end

