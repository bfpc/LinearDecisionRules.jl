
function _solve_dual_ldr(model)
    if haskey(model.dual_model.ext, :built_dual)
        # Lazy "flush"
        model.dual_model = JuMP.Model()
    end
    model.dual_model.ext[:built_dual] = true

    ABC = model.ext[:_LDR_ABC]
    M = model.ext[:_LDR_M]
    r = model.ext[:_LDR_r]::Float64

    first_stage_indices = model.ext[:_LDR_first_stage_indices]

    dim_x = size(ABC.Ae, 2)
    dim_ξ = size(ABC.Be, 2)
    dim_uncertainty = dim_ξ - 1
    # LDRs
    @expression(model.dual_model, X[1:dim_x, 1:dim_ξ], AffExpr(0.0))
    for i in 1:dim_x
        if i in first_stage_indices
            X[i, 1] = @variable(model.dual_model, base_name="X[$i,1]")
        else
            for j in 1:dim_ξ
                X[i, j] = @variable(model.dual_model, base_name="X[$i,$j]")
            end
        end
    end
    # @variable(model.dual_model, X[1:dim_x, 1:dim_ξ])
    @variable(model.dual_model, Su[1:size(ABC.Bu, 1), 1:dim_ξ])
    @variable(model.dual_model, Sl[1:size(ABC.Bl, 1), 1:dim_ξ])

    # Equality constraints
    @constraint(model.dual_model, ABC.Ae * X .== ABC.Be)

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

    # (W - h e1⊤)
    W2 = deepcopy(W)
    W2[:, 1] .-= h 

    # Constraints on slack matrices "S" are of the form
    # (W - h e1⊤) M Sᵗ ≥ 0
    # which are equivalent to S [(W - h e1⊤) M]ᵗ ≥ 0
    WMt = (W2 * M)'

    @constraint(model.dual_model, ABC.Au * X .+ Su .== ABC.Bu)
    @constraint(model.dual_model, Su * WMt .>= 0)

    @constraint(model.dual_model, ABC.Al * X .- Sl .== ABC.Bl)
    @constraint(model.dual_model, Sl * WMt .>= 0)

    # Can only include rows where the bound is not +Inf
    idxs = findall(x -> x != Inf, ABC.xu)
    @variable(model.dual_model, Sxu[idxs, 1:dim_ξ])
    @constraint(model.dual_model, X[idxs,1] .+ Sxu[idxs,1] .== ABC.xu[idxs])
    @constraint(model.dual_model, X[idxs,2:end] .+ Sxu[idxs,2:end] .== 0)
    @constraint(model.dual_model, Sxu.data * WMt .>= 0)

    # Can only include rows where the bound is not -Inf
    idxs = findall(x -> x != -Inf, ABC.xl)
    @variable(model.dual_model, Sxl[idxs, 1:dim_ξ])
    @constraint(model.dual_model, X[idxs,1] .- Sxl[idxs,1] .== ABC.xl[idxs])
    @constraint(model.dual_model, X[idxs,2:end] .- Sxl[idxs,2:end] .== 0)
    @constraint(model.dual_model, Sxl.data * WMt .>= 0)

    @expression(model.dual_model, obj, LinearAlgebra.tr(X' * ABC.P * X * M) + LinearAlgebra.tr(ABC.C' * X * M) + r)

    if model.ext[:_LDR_sense] == MOI.MIN_SENSE
        @objective(model.dual_model, Min, obj)
    else
        @objective(model.dual_model, Max, obj)
    end

    set_optimizer(model.dual_model, model.solver)
    if model.silent
        set_silent(model.dual_model)
    else
        unset_silent(model.dual_model)
    end
    optimize!(model.dual_model)

    return
end
