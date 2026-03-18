# The sampled (SAA) LDR model enforces constraints per-scenario
# instead of using dual multiplier matrices (Λ).
#
# min  (1/N) Σₛ [ ξₛ⊤ X⊤ P X ξₛ + C⊤ X ξₛ ] + r
# s.t. Ae X ξₛ = Be ξₛ                    ∀s
#      Au X ξₛ ≤ Bu ξₛ                    ∀s
#      Al X ξₛ ≥ Bl ξₛ                    ∀s
#      xl ≤ X ξₛ ≤ xu (where finite)      ∀s

function _solve_sampled_ldr(model)
    if haskey(model.sampled_model.ext, :_LDR_built_sampled)
        model.sampled_model = JuMP.Model()
    end
    model.sampled_model.ext[:_LDR_built_sampled] = true

    ABC = model.ext[:_LDR_ABC]
    Ξ = model.ext[:_LDR_scenarios]
    r = model.ext[:_LDR_r_sampled]::Float64

    first_stage_indices = model.ext[:_LDR_first_stage_indices]

    dim_x = size(ABC.Ae, 2)
    dim_ξ = size(ABC.Be, 2)
    N = size(Ξ, 2)

    # LDR coefficients X (same structure as primal/dual)
    @expression(model.sampled_model, X[1:dim_x, 1:dim_ξ], AffExpr(0.0))
    for i in 1:dim_x
        if i in first_stage_indices
            X[i, 1] = @variable(model.sampled_model, base_name = "X[$i,1]")
        else
            for j in 1:dim_ξ
                X[i, j] = @variable(model.sampled_model, base_name = "X[$i,$j]")
            end
        end
    end
    for i in ABC.bin
        for (var, _) in X[i, 1].terms
            set_binary(var)
        end
    end
    for i in ABC.int
        for (var, _) in X[i, 1].terms
            set_integer(var)
        end
    end

    # Per-scenario constraints
    for s in 1:N
        ξ = Ξ[:, s]

        # Equality constraints: Ae X ξ == Be ξ
        if size(ABC.Ae, 1) > 0
            @constraint(model.sampled_model, ABC.Ae * X * ξ .== ABC.Be * ξ)
        end

        # Upper inequality: Au X ξ <= Bu ξ
        if size(ABC.Au, 1) > 0
            @constraint(model.sampled_model, ABC.Au * X * ξ .<= ABC.Bu * ξ)
        end

        # Lower inequality: Al X ξ >= Bl ξ
        if size(ABC.Al, 1) > 0
            @constraint(model.sampled_model, ABC.Al * X * ξ .>= ABC.Bl * ξ)
        end

        # Variable upper bounds
        idxs_u = findall(isfinite, ABC.xu)
        if !isempty(idxs_u)
            @constraint(
                model.sampled_model,
                X[idxs_u, :] * ξ .<= ABC.xu[idxs_u]
            )
        end

        # Variable lower bounds
        idxs_l = findall(isfinite, ABC.xl)
        if !isempty(idxs_l)
            @constraint(
                model.sampled_model,
                X[idxs_l, :] * ξ .>= ABC.xl[idxs_l]
            )
        end
    end

    # Objective: use M̂ trace form when quadratic/cross-terms exist,
    # otherwise use simpler μ̂ form for linear objectives
    if haskey(model.ext, :_LDR_M_empirical)
        M̂ = model.ext[:_LDR_M_empirical]
        @expression(
            model.sampled_model,
            obj,
            LinearAlgebra.tr(X' * ABC.P * X * M̂) +
            LinearAlgebra.tr(ABC.C' * X * M̂) +
            r
        )
    else
        μ̂ = model.ext[:_LDR_μ_empirical]
        @expression(
            model.sampled_model,
            obj,
            LinearAlgebra.dot(ABC.C[:, 1], X * μ̂) + r
        )
    end

    if model.ext[:_LDR_sense] == MOI.MIN_SENSE
        @objective(model.sampled_model, Min, obj)
    else
        @objective(model.sampled_model, Max, obj)
    end

    set_optimizer(model.sampled_model, model.solver)
    if model.silent
        set_silent(model.sampled_model)
    else
        unset_silent(model.sampled_model)
    end

    # DEBUG: Print model details before solve on 32-bit
    if Sys.WORD_SIZE == 32
        println("DEBUG [solve_sampled] model built, about to optimize!")
        println("DEBUG [solve_sampled] num_variables=$(JuMP.num_variables(model.sampled_model))")
        println("DEBUG [solve_sampled] num_constraints=$(JuMP.num_constraints(model.sampled_model; count_variable_in_set_constraints=true))")
        println("DEBUG [solve_sampled] objective_sense=$(JuMP.objective_sense(model.sampled_model))")
        println("DEBUG [solve_sampled] objective_type=$(typeof(JuMP.objective_function(model.sampled_model)))")
        println("DEBUG [solve_sampled] free_memory=$(Sys.free_memory() / 1024^2) MB")
        lp_path = tempname() * ".lp"
        JuMP.write_to_file(model.sampled_model, lp_path)
        lp = read(lp_path, String)
        println("DEBUG [solve_sampled] LP file: $(length(lp)) bytes, $(count('\n', lp)) lines")
        println(lp)
        println("DEBUG [solve_sampled] JuMP model:")
        println(model.sampled_model)
        flush(stdout)
    end

    optimize!(model.sampled_model)

    return
end
