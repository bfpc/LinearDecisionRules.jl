function _create_pwl_model(model::LDRModel)

    cache_model = model.cache_model
    pwl_model = model.pwl_model

    # hack for avoid .ext warning
    ext = copy(cache_model.model.ext)
    empty!(cache_model.model.ext)
    raw_pwl_model, map_cache_to_pwl = JuMP.copy_model(cache_model.model)
    cache_model.model.ext = ext

    model.map_cache_to_pwl = map_cache_to_pwl

    pwl_model.model = raw_pwl_model
    empty!(pwl_model.first_stage)
    empty!(pwl_model.vector_distributions)
    empty!(pwl_model.scalar_distributions)
    empty!(pwl_model.uncertainty_to_distribution)

    empty!(model.extended_variables)

    for var_cache in cache_model.first_stage
        var_pwl = model.map_cache_to_pwl[var_cache]
        push!(pwl_model.first_stage, var_pwl)
    end

    # TODO
    # this is empty for now
    # but will be used by ConfidenceMvNormal
    for con_cache in cache_model.uncertainty_valid_constraints
        con_pwl = model.map_cache_to_pwl[con_cache]
        push!(pwl_model.uncertainty_valid_constraints, con_pwl)
    end

    for (uncertainty_cache, (d_idx, v_idx)) in cache_model.uncertainty_to_distribution

        uncertainty_pwl = model.map_cache_to_pwl[uncertainty_cache]
        if v_idx > 0
            push!(
                pwl_model.vector_distributions,
                cache_model.vector_distributions[d_idx],
            )
            pwl_model.uncertainty_to_distribution[uncertainty_pwl] = (
                length(pwl_model.vector_distributions),
                v_idx,
            )
        else
            if haskey(model.pwl_data, uncertainty_cache)
                break_points = model.pwl_data[uncertainty_cache]
                pwl_dist = UnivariatePieceWise(
                    cache_model.scalar_distributions[d_idx],
                    break_points,
                )
                push!(
                    pwl_model.vector_distributions,
                    pwl_dist,
                )

                _lb = Distributions.minimum(pwl_dist)
                _ub = Distributions.maximum(pwl_dist)
                # set bounds of the first one
                # TODO the bounds here might need to be parametric
                set_upper_bound(uncertainty_pwl, _ub[1])
                set_lower_bound(uncertainty_pwl, _lb[1])
                new_vars = JuMP.VariableRef[]
                # add the others
                pwl_model.uncertainty_to_distribution[uncertainty_pwl] = (
                    length(pwl_model.vector_distributions),
                    1,
                )
                for i in 1:length(break_points)
                    # TODO the bounds here might need to be parametric
                    v = @variable(
                        pwl_model.model,
                        base_name="$(uncertainty_pwl)_pwl_$i",
                        upper_bound = _ub[i+1],
                        lower_bound = _lb[i+1],
                    )
                    push!(new_vars, v)
                    pwl_model.uncertainty_to_distribution[v] = (
                        length(pwl_model.vector_distributions),
                        i+1,
                    )
                end
                _add_pwl_vars_to_constraints(
                    pwl_model.model,
                    uncertainty_pwl,
                    new_vars,
                )
                _add_pwl_constraints(
                    pwl_model.model,
                    uncertainty_pwl,
                    new_vars,
                    break_points,
                    _lb[1],
                    sum(_ub),
                    pwl_model.uncertainty_valid_constraints,
                )
                model.extended_variables[uncertainty_pwl] = vcat(uncertainty_pwl, new_vars)
            else
                push!(
                    pwl_model.scalar_distributions,
                    cache_model.scalar_distributions[d_idx],
                )
                pwl_model.uncertainty_to_distribution[uncertainty_pwl] = (
                    length(pwl_model.scalar_distributions),
                    v_idx,
                )
            end
        end
    end

    return nothing
end

function _add_pwl_vars_to_constraints(
    model::Model,
    uncertainty::VariableRef,
    new_vars::Vector{VariableRef},
)
    N = length(new_vars)
    new_coefs = fill(0.0, N)
    # todo: do doulbe loop with abrrier for performance
    for con in all_constraints(model, include_variable_in_set_constraints=false)
        coef = normalized_coefficient(con, uncertainty)
        if !iszero(coef)
            fill!(new_coefs, coef)
            set_normalized_coefficient(fill(con, N), new_vars, new_coefs)
        end
    end
    return nothing
end

function _add_pwl_constraints(
    model::Model,
    uncertainty::VariableRef,
    new_vars::Vector{VariableRef},
    break_points::Vector{Float64},
    _min::Float64,
    _max::Float64,
    uncertainty_valid_constraints,
)
    if length(break_points) == 1
        con = @constraint(
            model,
            new_vars[1] * (break_points[1] - _min) <= (uncertainty - _min) * (_max - break_points[1])
        )
        push!(uncertainty_valid_constraints, con)
        return nothing
    end
    con = @constraint(
        model,
        new_vars[1] * (break_points[1] - _min) <= (uncertainty - _min) * (break_points[2] - break_points[1])
    )
    push!(uncertainty_valid_constraints, con)
    for i in 2:length(break_points)-1
        con = @constraint(
            model,
            new_vars[i] * (break_points[i] - break_points[i-1]) <= new_vars[i-1] * (break_points[i+1] - break_points[i])
        )
        push!(uncertainty_valid_constraints, con)
    end
    con = @constraint(
        model,
        new_vars[end] * (break_points[end] - break_points[end-1]) <= new_vars[end-1] * (_max - break_points[end])
    )
    push!(uncertainty_valid_constraints, con)
    return nothing
end