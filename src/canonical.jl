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

function _variable_maps(data::MatrixData, uncertainty_variables)
    A = data.A
    # vals = nonzeros(A)
    m, n = size(A)
    # n_uncertainty = length(uncertainty_variables)

    is_uncertainty = falses(n)
    distributions = Distributions.Distribution[]
    for i in eachindex(data.variables)
        if data.variables[i] in keys(uncertainty_variables)
            is_uncertainty[i] = true
            push!(distributions, uncertainty_variables[data.variables[i]])
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

    return uncertainty_indices, variable_indices, column_to_canonical, distributions
end

function _canonical(data::MatrixData, uncertainty_indices, variable_indices, distributions)
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
    # vals = nonzeros(A)
    m, n = size(A)

    # Build matrix of E[ξ⊤ ξ] (the first index is the extra coordinate 1)
    # The first line has the means, the remaining the covariances + product of means
    dim_uncertainty = 1 + length(uncertainty_indices)
    vector_means = [1; Distributions.mean.(distributions)]
    M = zeros(dim_uncertainty, dim_uncertainty)
    for i in 2:dim_uncertainty
        M[i, i] = Distributions.var(distributions[i-1])
    end
    M .+= vector_means .* vector_means'

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
    bu = data.b_lower[upper_bound_rows]
    Bu = [bu -A[upper_bound_rows, uncertainty_indices]]
    Al = A[lower_bound_rows, variable_indices]
    bl = data.b_upper[lower_bound_rows]
    Bl = [bl -A[lower_bound_rows, uncertainty_indices]]
    xu = data.x_upper[variable_indices]
    xl = data.x_lower[variable_indices]
    
    # Build the matrices (Wu, Wl) and vectors (hu, hl, lb, ub) for the uncertainty
    Wu = A[u_upper_bound_rows, uncertainty_indices]
    hu = data.b_lower[u_upper_bound_rows]
    Wl = A[u_lower_bound_rows, uncertainty_indices]
    hl = data.b_upper[u_lower_bound_rows]
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
    # Ae, Be, Au, Bu, Al, Bl, xu, xl, Wu, hu, Wl, hl, lb, ub = _canonical(data, model.cache_uncertainty)
    uncertainty_indices, variable_indices, column_to_canonical, distributions = _variable_maps(data, model.cache_uncertainty)
    model.ext[:column_to_canonical] = column_to_canonical
    ABC = _canonical(data, uncertainty_indices, variable_indices, distributions)
    model.ext[:ABC] = ABC

    return nothing
end