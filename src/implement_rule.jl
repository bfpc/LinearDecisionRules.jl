"""
    get_decision(m, x, η; dual = false, piece = nothing)

Coefficient of η in the LDR of x
"""
function get_decision(model, x, η; dual = false, piece = nothing)
    @assert haskey(model.cache_model.uncertainty_to_distribution, η)
    @assert !haskey(model.cache_model.uncertainty_to_distribution, x)
    @assert JuMP.is_valid(model, η)
    @assert JuMP.is_valid(model, x)
    if !isempty(model.extended_variables) # then its is PWL
        x = model.map_cache_to_pwl[x]
        η = model.map_cache_to_pwl[η]
        if haskey(model.pwl_data, η)
            pieces = model.extended_variables[η]
            if piece === nothing
                error("piece index keword argument not provided for piecewise linear decision rule.")
            end
            if !(1 <= piece <= length(pieces))
                error("Piece index, $piece, out of bounds 1 to $(length(pieces))")
            end
            η = pieces[piece]
        end
    end

    var_to_column = model.ext[:_LDR_var_to_column] # type assert
    column_to_canonical = model.ext[:_LDR_column_to_canonical] # type assert
    i = column_to_canonical[var_to_column[x]]
    j = column_to_canonical[var_to_column[η]]
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
    get_decision(m, x; dual = false)

Constant term in the LDR of x
"""
function get_decision(model, x; dual = false)
    @assert !haskey(model.cache_model.uncertainty_to_distribution, x)
    @assert JuMP.is_valid(model, x)
    if !isempty(model.extended_variables) # then its is PWL
        x = model.map_cache_to_pwl[x]
    end
    var_to_column = model.ext[:_LDR_var_to_column] # type assert
    column_to_canonical = model.ext[:_LDR_column_to_canonical] # type assert
    i = column_to_canonical[var_to_column[x]]
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

