"""
base form is
min E[ c(ξ)⊤ x(ξ) + x(ξ)⊤ Q x(ξ) + r ]
s.t. Ae x(ξ) = be(ξ)
     Au x(ξ) ≤ bu(ξ)
     Al x(ξ) ≥ bl(ξ)
     I x(ξ) ≤ xu
     I x(ξ) ≥ xl
     ∀ ξ ∈ Ξ

where Ξ is a polytope described by
Ξ = { ξ ∈ ℝ^m | Wu ξ ≤ hu, Wl ξ ≥ hl, lb ≤ ξ ≤ ub }
(it cannot have a further affine constraint, since its linear span must be all of ℝ^m)

Introducing positive slack variables, we can write this as
min E[ c(ξ)⊤ x(ξ) + x(ξ)⊤ Q x(ξ) + r ]
s.t. Ae x(ξ) = be(ξ)
     Au x(ξ) + su(ξ) = bu(ξ)
     Al x(ξ) - sl(ξ) = bl(ξ)
     I x(ξ) + sxu(ξ) = xu
     I x(ξ) - sxl(ξ) = xl
     su(ξ) >= 0
     sl(ξ) >= 0
     sxu(ξ) >= 0
     sxl(ξ) >= 0

Assuming x(ξ) = X ξ, etc, and that the first coordinate of ξ is 1, we can write this as
min E[ ξ⊤ C⊤ X ξ + ξ⊤ X⊤ Q X ξ + r ]
s.t. Ae X ξ = Be ξ
     Au X ξ + Su ξ = Bu ξ
     Al X ξ - Sl ξ = Bl ξ
     I X ξ + Sxu ξ = Xu ξ
     I X ξ - Sxl ξ = Xl ξ
     Su ξ >= 0
     Sl ξ >= 0
     Sxu ξ >= 0
     Sxl ξ >= 0

# TODO generalize r
# TODO generalize A (hard needs SDP)
# Notation: Q or P for the quadratic term?
"""

function _variable_maps(data::MatrixData, uncertainty_variables, first_stage_variables)
    A = data.A
    m, n = size(A)

    is_uncertainty = falses(n)
    for i in eachindex(data.variables)
        if data.variables[i] in keys(uncertainty_variables)
            is_uncertainty[i] = true
        end
    end
    uncertainty_indices = findall(is_uncertainty)
    variable_indices = findall(!, is_uncertainty)

    column_to_canonical = zeros(Int64, n)
    for (i, col) in enumerate(uncertainty_indices)
        column_to_canonical[col] = i
    end
    for (i, col) in enumerate(variable_indices)
        column_to_canonical[col] = i
    end

    first_stage_indices = Set(findall(x -> x in first_stage_variables, data.variables))

    return uncertainty_indices, variable_indices, column_to_canonical, first_stage_indices
end

function _canonical(
    data::MatrixData,
    uncertainty_indices,
    variable_indices,
    uncertainty_to_distribution,
    scalar_distributions,
    vector_distributions,
)
    # look for rows of data.A (a sparse matrix) that are only of uncertainty
    

    @assert length(data.binaries) == 0
    @assert length(data.integers) == 0
    if data.sense == MOI.FEASIBILITY_SENSE
        error("Objective function sense is MOI.FEASIBILITY_SENSE, check your objective function.")
    end
    @assert length(data.variables) > 0
    @assert length(data.affine_constraints) +
        length(data.variable_constraints) > 0


    A = data.A
    rows = SparseArrays.rowvals(A)
    m, n = size(A)

    # Build matrix of E[ξ⊤ ξ] (the first index is the extra coordinate 1)
    # The first line has the means, the remaining the covariances + product of means

    # precompute the means and variances of the distributions
    scalar_means = [Distributions.mean(d) for d in scalar_distributions]
    vector_means = [Distributions.mean(d) for d in vector_distributions]
    scalar_vars = [Distributions.var(d) for d in scalar_distributions]
    vector_vars = [Distributions.cov(d) for d in vector_distributions]
    # vector_idxs is used to fill the covariance matrix
    vector_idxs = [zeros(Int, length(d)) for d in vector_distributions]
    dim_uncertainty = 1 + length(uncertainty_indices)
    μ = zeros(dim_uncertainty)
    μ[1] = 1
    M = zeros(dim_uncertainty, dim_uncertainty)
    # fill meand and M's diagonal
    # there are meny indices next:
    # - lin_idx is the index in the linearized vector (for the M matrix)
    # - var_idx is the index in the data.variables vector
    # - dist_idx is the index in the scalar or vector distributions
    # - inner_idx is the internal index of a variabel inside their distribution
    for (lin_idx, var_idx) in enumerate(uncertainty_indices)
        var = data.variables[var_idx]
        dist_idx, inner_idx = uncertainty_to_distribution[var]
        if inner_idx == 0 # scalar
            μ[1+lin_idx] = scalar_means[dist_idx]
            M[1+lin_idx, 1+lin_idx] = scalar_vars[dist_idx]
        else # vector
            μ[1+lin_idx] = vector_means[dist_idx][inner_idx]
            # this is used to fill the covariance matrix next
            vector_idxs[dist_idx][inner_idx] = lin_idx
        end
    end
    # fill variance blocks
    for (dist_idx, list) in enumerate(vector_idxs)
        for (inner_idx, lin_idx) in enumerate(list)
            for (inner_idx2, lin_idx2) in enumerate(list)
                M[1+lin_idx, 1+lin_idx2] = vector_vars[dist_idx][inner_idx, inner_idx2]
            end
        end
    end
    M .+= μ .* μ'

    # Process the rows of A
    has_variables = falses(m)
    for j = variable_indices
        for i in SparseArrays.nzrange(A, j)
            has_variables[rows[i]] = true
        end
        # VERY Suboptimal since tags the same row multiple times if it has multiple variables
    end
    uncertainty_rows = findall(!, has_variables)
    mixed_rows = findall(has_variables)

    mm = length(mixed_rows)
    equality_rows = Int[]
    sizehint!(equality_rows, mm÷2)
    upper_bound_rows = Int[]
    sizehint!(upper_bound_rows, mm÷2)
    lower_bound_rows = Int[]
    sizehint!(lower_bound_rows, mm÷2)
    interval_rows = Int[]
    sizehint!(interval_rows, mm÷20)
    for j in mixed_rows
        if data.b_lower[j] == data.b_upper[j]
            push!(equality_rows, j)
        elseif data.b_lower[j] == -Inf
            push!(upper_bound_rows, j)
        elseif data.b_upper[j] == Inf
            push!(lower_bound_rows, j)
        else
            push!(interval_rows, j)
        end
    end

    uu = length(uncertainty_rows)
    u_equality_rows = Int[]
    sizehint!(u_equality_rows, uu÷20)
    u_upper_bound_rows = Int[]
    sizehint!(u_upper_bound_rows, uu÷2)
    u_lower_bound_rows = Int[]
    sizehint!(u_lower_bound_rows, uu÷2)
    u_interval_rows = Int[]
    sizehint!(u_interval_rows, uu÷20)
    for j in uncertainty_rows
        if data.b_lower[j] == data.b_upper[j]
            push!(u_equality_rows, j)
        elseif data.b_lower[j] == -Inf
            push!(u_upper_bound_rows, j)
        elseif data.b_upper[j] == Inf
            push!(u_lower_bound_rows, j)
        else
            push!(u_interval_rows, j)
        end
    end

    if length(u_equality_rows) > 0
        @warn "equality constraint on uncertainty variable"
    end


    # Build the matrices (Ae, Au, Al) and vectors (be, bu, bl, xu, xl) for the problem
    #  Ae x(ξ) = be(ξ)
    #  Au x(ξ) ≤ bu(ξ)
    #  Al x(ξ) ≥ bl(ξ)
    #  I x(ξ) ≤ xu
    #  I x(ξ) ≥ xl
    Ae = A[equality_rows, variable_indices]
    be = data.b_lower[equality_rows]
    Be = [be -A[equality_rows, uncertainty_indices]]
    Au = A[upper_bound_rows, variable_indices]
    bu = data.b_upper[upper_bound_rows]
    Bu = [bu -A[upper_bound_rows, uncertainty_indices]]
    Al = A[lower_bound_rows, variable_indices]
    bl = data.b_lower[lower_bound_rows]
    Bl = [bl -A[lower_bound_rows, uncertainty_indices]]
    xu = data.x_upper[variable_indices]
    xl = data.x_lower[variable_indices]
    
    # Build the matrices (Wu, Wl) and vectors (hu, hl, lb, ub) for the uncertainty
    Wu = A[u_upper_bound_rows, uncertainty_indices]
    hu = data.b_upper[u_upper_bound_rows]
    Wl = A[u_lower_bound_rows, uncertainty_indices]
    hl = data.b_lower[u_lower_bound_rows]
    lb = data.x_lower[uncertainty_indices]
    ub = data.x_upper[uncertainty_indices]

    # Build the LDR matrices (Q, C) and constant r from the quadratic objective
    # [x η]⊤ Q [x η] + c⊤ [x η] + c_offset = x⊤ Q_11 x + 2 η⊤ Q_21 x + η⊤ Q_4 η + c_1⊤ x + c_2⊤ η + c_offset
    # We define ξ = [1; η], and pose the LDR  x = X ξ; taking the expectation over ξ, we get
    # E[ ξ⊤ X⊤ Q_11 X ξ + 2 η⊤ Q_21 X ξ + η⊤ Q_4 η + c_1⊤ X ξ + c_2⊤ η + c_offset ]
    # = Tr(X⊤ Q_11 X E[ξ ξ⊤]) + Tr([c_1⊤; Q_21] X E[ξ⊤ ξ]) + E[η⊤ Q_4 η] + c_1⊤ X E[ξ] + c_2⊤ E[η] + c_offset
    # = Tr(X⊤ Q_11 X M) + Tr([c_1⊤; Q_21] X M) + Tr([c_offset c_2⊤/2; c_2/2 Q_4] M)
    P = data.Q[variable_indices, variable_indices]
    C = [data.c[variable_indices] data.Q[variable_indices, uncertainty_indices]]
    # r = [data.c_offset 0.5 * data.c[uncertainty_indices]'; 0.5 * data.c[uncertainty_indices] data.Q[uncertainty_indices, uncertainty_indices]]
    r1 = data.c_offset
    r2 = data.c[uncertainty_indices]' * M[2:end, 1]
    r3 = sum(data.Q[uncertainty_indices, uncertainty_indices] .* M[2:end, 2:end])
    r = r1 + r2 + r3[1]

    return (Ae=Ae, Be=Be, Au=Au, Bu=Bu, Al=Al, Bl=Bl, xu=xu, xl=xl, Wu=Wu, hu=hu, Wl=Wl, hl=hl, lb=lb, ub=ub, M=M, P=P, C=C, r=r)
end

function _prepare_data(model)

    data = matrix_data(model.cache_model)
    var_to_column = Dict(vi => i for (i, vi) in enumerate(data.variables))
    model.ext[:var_to_column] = var_to_column
    uncertainty_indices, variable_indices, column_to_canonical, first_stage_indices = 
        _variable_maps(data, model.uncertainty_to_distribution, model.cache_first_stage)
    model.ext[:column_to_canonical] = column_to_canonical
    ABC = _canonical(
        data,
        uncertainty_indices,
        variable_indices,
        model.uncertainty_to_distribution,
        model.cache_scalar_distributions,
        model.cache_vector_distributions,
    )
    model.ext[:sense] = data.sense
    model.ext[:ABC] = ABC
    model.ext[:first_stage_indices] = first_stage_indices

    return nothing
end