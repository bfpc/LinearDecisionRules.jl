function set_parametric_objective!(new_model, model, uncertainty_to_var::Dict)
    if haskey(new_model.ext, :_LDR_value_function_set)
        error("Value function already set in the new model.")
    end

    # validate input
    for (η, new_var) in uncertainty_to_var
        @assert JuMP.is_valid(model, η)
        @assert haskey(model.cache_model.uncertainty_to_distribution, η)
        @assert JuMP.is_valid(new_model, new_var)
    end

    ABC = model.ext[:_LDR_ABC]
    _M = model.ext[:_LDR_M]
    r = model.ext[:_LDR_r]::Float64
    # dim_x = size(ABC.Ae, 2)
    # dim_ξ = size(ABC.Be, 2)
    # dim_uncertainty = dim_ξ - 1
    # ξ = Vector{JuMP.GenericAffExpr{Float64,JuMP.GenericVariableRef{Float64}}}(
    #     undef,
    #     dim_ξ,
    # )
    M = similar(
        _M,
        JuMP.GenericQuadExpr{Float64,JuMP.GenericVariableRef{Float64}},
    )
    M .= _M
    ξ = Dict{Int,JuMP.GenericVariableRef{Float64}}()
    # for i in eachindex(ξ)
    #     ξ[i] = M[i, 1]
    # end

    var_to_column = model.ext[:_LDR_var_to_column] # type assert
    column_to_canonical = model.ext[:_LDR_column_to_canonical] # type assert

    added_variables = JuMP.VariableRef[]
    added_constraints = JuMP.ConstraintRef[]
    for (η, new_var) in uncertainty_to_var
        if !isempty(model.extended_variables) # then its is PWL
            η_cache = η
            η = model.map_cache_to_pwl[η_cache]
            if haskey(model.pwl_data, η_cache)
                break_points = model.pwl_data[η_cache]
                pieces = model.extended_variables[η]
                # 1 - now we need to create variables for each piece
                _lb = lower_bound.(pieces)
                _ub = upper_bound.(pieces)
                new_var_piece = JuMP.VariableRef[]
                for (i, piece) in enumerate(pieces)
                    # create variables in the new_model
                    # to associate with piece in the old_model (previosuly solved)
                    v = @variable(
                        new_model,
                        base_name = "$(new_var)_pwl_$i",
                        upper_bound = _ub[i],
                        lower_bound = _lb[i],
                    )
                    push!(new_var_piece, v)
                    push!(added_variables, v)
                    # map these new variable inthe objetctive
                    j = column_to_canonical[var_to_column[piece]]
                    ξ[j+1] = v
                end
                # 2- now we need to add constraints of the pwl rule
                con = @constraint(new_model, sum(new_var_piece) == new_var,)
                push!(added_constraints, con)
                _add_pwl_constraints(
                    new_model,
                    new_var_piece[1],
                    new_var_piece[2:end],
                    break_points,
                    _lb[1],
                    sum(_ub),
                    added_constraints,
                )
            else
                # non plw case
                j = column_to_canonical[var_to_column[η]]
                ξ[j+1] = new_var
            end
        else
            j = column_to_canonical[var_to_column[η]]
            ξ[j+1] = new_var
        end
    end

    I, J = size(_M)

    for i in 1:I
        ξ_i = get(ξ, i, _M[i, 1])
        for j in 1:J
            ξ_j = get(ξ, j, _M[j, 1])
            if !(ξ_i isa Number) || !(ξ_j isa Number)
                M[i, j] = ξ_i * ξ_j
            end
        end
    end

    X = value.(model.primal_model[:X])

    # display(_M)
    # display(X)
    # display(ABC.P)
    # display(ABC.C)
    # display(M)

    out =
        LinearAlgebra.tr((X' * ABC.P * X) * M) +
        LinearAlgebra.tr((ABC.C' * X) * M) +
        r

    func = JuMP.objective_function(new_model)
    JuMP.set_objective_function(new_model, func + out)

    new_model.ext[:_LDR_value_function_set] = true

    return out # value_function
end

#=
struct ValueFunction

end

vf(x):

vf(x): = min t  s.a.  B y + c t <= h - Ax

c é um vetor coluna

copiar para um modelo novo

=#
