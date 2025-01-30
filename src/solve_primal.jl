# The LDR model "cancels out" ξ in the constraints, and uses the "trace trick"
# to factor E[ξ ξ⊤] in the objective function.
#
# min E[ ξ⊤ C⊤ X ξ + ξ⊤ X⊤ Q X ξ + r ]
# s.t. Ae X ξ = Be ξ
#      Au X ξ + Su ξ = Bu ξ
#      Al X ξ - Sl ξ = Bl ξ
#      I X ξ + Sxu ξ = Xu ξ
#      I X ξ - Sxl ξ = Xl ξ
#      Su ξ >= 0
#      Sl ξ >= 0
#      Sxu ξ >= 0
#      Sxl ξ >= 0

function _solve_primal_ldr(model)
    if haskey(model.primal_model.ext, :built_primal)
        # Lazy "flush"
        model.primal_model = JuMP.Model()
    end
    model.primal_model.ext[:built_primal] = true

    ABC = model.ext[:ABC]

    first_stage_indices = model.ext[:first_stage_indices]

    dim_x = size(ABC.Ae, 2)
    dim_ξ = size(ABC.Be, 2)
    dim_uncertainty = dim_ξ - 1
    # LDRs
    @expression(model.primal_model, X[1:dim_x, 1:dim_ξ], AffExpr(0.0))
    for i in 1:dim_x
        if i in first_stage_indices
            X[i, 1] = @variable(model.primal_model, base_name="X[$i,1]")
        else
            for j in 1:dim_ξ
                X[i, j] = @variable(model.primal_model, base_name="X[$i,$j]")
            end
        end
    end
    # @variable(model.primal_model, X[1:dim_x, 1:dim_ξ])
    @variable(model.primal_model, Su[1:size(ABC.Bu, 1), 1:dim_ξ])
    @variable(model.primal_model, Sl[1:size(ABC.Bl, 1), 1:dim_ξ])

    # Equality constraints
    @constraint(model.primal_model, ABC.Ae * X .== ABC.Be)

    # Inequality constraints
    # Uncertainty polyhedron: W ξ ≥ h from
    # Ξ = { ξ = (1, η) ∈ ℝ^m | Wu η ≤ hu, Wl η ≥ hl, lb ≤ η ≤ ub }
    nu = size(ABC.Wu, 1)
    nl = size(ABC.Wl, 1)
    nW = 2dim_ξ + nu + nl
    W = [ 1 zeros(1, dim_uncertainty);
         -1 zeros(1, dim_uncertainty);
         zeros(nu + nl + 2dim_uncertainty, 1) [-ABC.Wu; ABC.Wl; -SparseArrays.I(dim_uncertainty); SparseArrays.I(dim_uncertainty)]]
    h = [1; -1; -ABC.hu; ABC.hl; -ABC.ub; ABC.lb]

    @constraint(model.primal_model, ABC.Au * X .+ Su .== ABC.Bu)
    @variable(model.primal_model, ΛSu[1:size(ABC.Bu, 1),1:nW] >= 0)
    @constraint(model.primal_model, ΛSu * W .== Su)
    @constraint(model.primal_model, ΛSu * h .>= 0)

    @constraint(model.primal_model, ABC.Al * X .- Sl .== ABC.Bl)
    @variable(model.primal_model, ΛSl[1:size(ABC.Bl, 1),1:nW] >= 0)
    @constraint(model.primal_model, ΛSl * W .== Sl)
    @constraint(model.primal_model, ΛSl * h .>= 0)

    # Can only include rows where the bound is not +Inf
    idxs = findall(x -> x != Inf, ABC.xu)
    @variable(model.primal_model, Sxu[idxs, 1:dim_ξ])
    @constraint(model.primal_model, X[idxs,1] .+ Sxu[idxs,1] .== ABC.xu[idxs])
    @constraint(model.primal_model, X[idxs,2:end] .+ Sxu[idxs,2:end] .== 0)
    @variable(model.primal_model, ΛSxu[idxs,1:nW] >= 0)
    @constraint(model.primal_model, ΛSxu.data * W .== Sxu.data)
    @constraint(model.primal_model, ΛSxu.data * h .>= 0)

    # Can only include rows where the bound is not -Inf
    idxs = findall(x -> x != -Inf, ABC.xl)
    @variable(model.primal_model, Sxl[idxs, 1:dim_ξ])
    @constraint(model.primal_model, X[idxs,1] .- Sxl[idxs,1] .== ABC.xl[idxs])
    @constraint(model.primal_model, X[idxs,2:end] .- Sxl[idxs,2:end] .== 0)
    @variable(model.primal_model, ΛSxl[idxs,1:nW] >= 0)
    @constraint(model.primal_model, ΛSxl.data * W .== Sxl.data)
    @constraint(model.primal_model, ΛSxl.data * h .>= 0)

    @expression(model.primal_model, obj, tr(X' * ABC.P * X * ABC.M) + tr(ABC.C' * X * ABC.M) + ABC.r)

    if model.ext[:sense] == MOI.MIN_SENSE
        @objective(model.primal_model, Min, obj)
    else
        @objective(model.primal_model, Max, obj)
    end

    set_optimizer(model.primal_model, model.solver)
    if model.silent
        set_silent(model.primal_model)
    else
        unset_silent(model.primal_model)
    end
    optimize!(model.primal_model)

    return
end
