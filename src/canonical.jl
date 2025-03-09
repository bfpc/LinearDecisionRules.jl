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

function _model_to_matrix(
    data::MatrixData,
    uncertainty_variables,
    first_stage_variables,
    uncertainty_valid_constraints,
)
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
    uncertainty_valid_indices = Set(findall(x -> x in uncertainty_valid_constraints, data.affine_constraints))

    return uncertainty_indices, variable_indices, column_to_canonical, first_stage_indices, uncertainty_valid_indices
end

function _second_moment_matrix(
    data::MatrixData,
    uncertainty_indices,
    uncertainty_to_distribution,
    scalar_distributions,
    vector_distributions,
    ABC,
)
    # Compute the second moment matrix M of the uncertainty
    # M = E[ξ⊤ ξ]
    # where ξ is a random vector with the following properties:
    # - lb ≤ ξ ≤ ub
    # - Wu ξ ≤ hu
    # - Wl ξ ≥ hl
    # - ξ is a vector of the random variables in uncertainty_indices
    # - the distribution of each variable is given by uncertainty_to_distribution
    # - scalar_distributions and vector_distributions are the distributions of the variables
    # - Wu, hu, Wl, hl are the matrices defining the polytope Ξ
    # - lb, ub are the lower and upper bounds of the variables
    # The second moment matrix M is such that
    # M[i, j] = E[ξ[i] ξ[j]]
    # where ξ[i] is the i-th element of ξ
    # The first element of ξ is 1 (the extra coordinate for the affine transformation

    Wu = ABC.Wu
    hu = ABC.hu
    Wl = ABC.Wl
    hl = ABC.hl
    lb = ABC.lb
    ub = ABC.ub

    # Build matrix M of E[ξ⊤ ξ] (the first index is the extra coordinate 1)
    # The first line has the means, the remaining the covariances + product of means

    all_groups, all_wu_rows_of_group, all_wl_rows_of_group = _compute_groups(Wu, Wl, uncertainty_indices, uncertainty_to_distribution, data)

    require_group_rejection_sampling = Set{Tuple{Int, Bool}}()
    for group in all_groups
        union!(require_group_rejection_sampling, group)
    end

    # vector_idxs is used to fill the covariance matrix
    vector_idxs = [zeros(Int, length(d)) for d in vector_distributions]
    scalar_idxs = zeros(Int, length(scalar_distributions))
    for (lin_idx, var_idx) in enumerate(uncertainty_indices)
        var = data.variables[var_idx]
        dist_idx, inner_idx = uncertainty_to_distribution[var]
        if inner_idx == 0 # scalar
            scalar_idxs[dist_idx] = lin_idx
        else # vector
            vector_idxs[dist_idx][inner_idx] = lin_idx
        end
    end

    if isempty(require_group_rejection_sampling)
        # some of these might use rejection sampling internally due to "boxed" or "truncated" distributions
        # precompute the means and variances of the distributions
        scalar_means = [Distributions.mean(d) for d in scalar_distributions]
        vector_means = [Distributions.mean(d) for d in vector_distributions]
        scalar_vars = [Distributions.var(d) for d in scalar_distributions]
        vector_vars = [Distributions.cov(d) for d in vector_distributions]
    else
        @warn "Rejection sampling required"
        scalar_means = zeros(length(scalar_distributions))
        vector_means = Vector{Vector{Float64}}(undef, length(vector_distributions))
        scalar_vars = zeros(length(scalar_distributions))
        vector_vars = [zeros(0, 0) for _ in vector_distributions]
        for (dist_idx, d) in enumerate(scalar_distributions)
            if !((dist_idx, false) in require_group_rejection_sampling)
                scalar_means[dist_idx] = Distributions.mean(d)
                scalar_vars[dist_idx] = Distributions.var(d)
            end
        end
        for (dist_idx, d) in enumerate(vector_distributions)
            if !((dist_idx, true) in require_group_rejection_sampling)
                vector_means[dist_idx] = Distributions.mean(d)
                vector_vars[dist_idx] = Distributions.cov(d)
            end
        end
    end

    dim_uncertainty = 1 + length(uncertainty_indices)
    μ = zeros(dim_uncertainty)
    μ[1] = 1
    M = zeros(dim_uncertainty, dim_uncertainty)
    # fill means and M's diagonal
    # there are many indices next:
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
            if length(vector_vars[dist_idx]) == 0
                continue
            end
            μ[1+lin_idx] = vector_means[dist_idx][inner_idx]
        end
    end
    # fill variance blocks
    for (dist_idx, list) in enumerate(vector_idxs)
        if length(vector_vars[dist_idx]) == 0
            continue
        end
        for (inner_idx, lin_idx) in enumerate(list)
            for (inner_idx2, lin_idx2) in enumerate(list)
                M[1+lin_idx, 1+lin_idx2] = vector_vars[dist_idx][inner_idx, inner_idx2]
            end
        end
    end

    # TODO: move this to a parameter
    time_per_estimation = 10.0
    seed = 1234
    max_iterations = 1000
    warn_attempts = 1000

    # fill non analytical blocks with rejection sampling
    candidate = zeros(length(uncertainty_indices))
    wu_m, wu_n = size(Wu)
    wl_m, wl_n = size(Wl)
    cache_wu_m = zeros(wu_m)
    cache_wl_m = zeros(wl_m)
    for i in eachindex(all_groups)
        group = all_groups[i]
        wu_rows = all_wu_rows_of_group[i]
        wl_rows = all_wl_rows_of_group[i]
        if isempty(group)
            continue
        end
        x = zeros(length(uncertainty_indices))
        x2 = zeros(length(uncertainty_indices), length(uncertainty_indices))
        n = 0
        initial_time = time()
        rng = Random.Xoshiro(seed)
        for i in 1:max_iterations
            _attempts = _sample_in_set!(
                rng, 
                candidate,
                cache_wu_m,
                cache_wl_m,
                group,
                wu_rows,
                wl_rows,
                scalar_distributions,
                vector_distributions,
                scalar_idxs,
                vector_idxs,
                lb,
                ub,
                Wu,
                hu,
                Wl,
                hl,
            )
            if _attempts == warn_attempts
                @warn "Rejection sampling took too long"
            end
            x .+= candidate
            # x2 .+= candidate * candidate'
            LinearAlgebra.mul!(x2, candidate, candidate', 1.0, 1.0)
            n += 1
            # TODO: add convergence check
            if time() - initial_time > time_per_estimation
                println("Estimation max time")
                break
            end
        end
        x ./= n
        x2 ./= n

        M[2:end, 2:end] .+= x2 .- (x .* x')
        μ[2:end] .+= x
    end
    M .+= μ .* μ'

    return M
end

function _canonical(
    data::MatrixData,
    uncertainty_indices,
    variable_indices,
    uncertainty_valid_indices,
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
    u_upper_bound_rows_v = Int[]
    sizehint!(u_upper_bound_rows_v, length(uncertainty_valid_indices))
    u_lower_bound_rows = Int[]
    sizehint!(u_lower_bound_rows, uu÷2)
    u_lower_bound_rows_v = Int[]
    sizehint!(u_lower_bound_rows_v, length(uncertainty_valid_indices))
    u_interval_rows = Int[]
    sizehint!(u_interval_rows, uu÷20)
    for j in uncertainty_rows
        if data.b_lower[j] == data.b_upper[j]
            push!(u_equality_rows, j)
        elseif data.b_lower[j] == -Inf
            if j in uncertainty_valid_indices
                push!(u_upper_bound_rows_v, j)
            else
                push!(u_upper_bound_rows, j)
            end
        elseif data.b_upper[j] == Inf
            if j in uncertainty_valid_indices
                push!(u_lower_bound_rows_v, j)
            else
                push!(u_lower_bound_rows, j)
            end
        else
            push!(u_interval_rows, j)
        end
    end

    if length(u_equality_rows) > 0
        @warn "pure equality constraint on uncertainty variables is not valid"
    end
    if length(u_interval_rows) > 0
        @warn "pure interval constraint on uncertainty variables is not valid"
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
    # same for valid constraints
    Wu_v = A[u_upper_bound_rows_v, uncertainty_indices]
    hu_v = data.b_upper[u_upper_bound_rows_v]
    Wl_v = A[u_lower_bound_rows_v, uncertainty_indices]
    hl_v = data.b_lower[u_lower_bound_rows_v]
    # bounds from variable input, from:
    # - distributions defition
    # - truncate for scalar distributions
    # - boxed for vector distributions
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
    Q = data.Q[uncertainty_indices, uncertainty_indices]
    d = data.c[uncertainty_indices]
    f = data.c_offset

    return (
        Ae=Ae,
        Be=Be,
        Au=Au,
        Bu=Bu,
        Al=Al,
        Bl=Bl,
        xu=xu,
        xl=xl,
        Wu=Wu,
        hu=hu,
        Wl=Wl,
        hl=hl,
        Wu_v=Wu_v,
        hu_v=hu_v,
        Wl_v=Wl_v,
        hl_v=hl_v,
        lb=lb,
        ub=ub,
        P=P,
        C=C,
        Q=Q,
        d=d,
        f=f,
    )
end

function _objective_constant(ABC, M)
    # r = [data.c_offset 0.5 * data.c[uncertainty_indices]'; 0.5 * data.c[uncertainty_indices] data.Q[uncertainty_indices, uncertainty_indices]]
    r1 = ABC.f
    r2 = ABC.d' * M[2:end, 1]
    r3 = sum(ABC.Q .* M[2:end, 2:end])
    r = r1 + r2 + r3[1]
    return r
end

function _prepare_data(model)

    stoch_model = if isempty(model.pwl_data)
        model.cache_model
    else
        model.pwl_model
    end
    data = matrix_data(stoch_model.model)
    var_to_column = Dict(vi => i for (i, vi) in enumerate(data.variables))
    model.ext[:_LDR_var_to_column] = var_to_column
    uncertainty_indices, variable_indices, column_to_canonical, first_stage_indices, uncertainty_valid_indices =
        _model_to_matrix(
            data,
            stoch_model.uncertainty_to_distribution,
            stoch_model.first_stage,
            stoch_model.uncertainty_valid_constraints,
        )
    model.ext[:_LDR_column_to_canonical] = column_to_canonical
    ABC = _canonical(
        data,
        uncertainty_indices,
        variable_indices,
        uncertainty_valid_indices,
    )
    M = _second_moment_matrix(
        data,
        uncertainty_indices,
        stoch_model.uncertainty_to_distribution,
        stoch_model.scalar_distributions,
        stoch_model.vector_distributions,
        ABC,
    )
    model.ext[:_LDR_r] = _objective_constant(ABC, M)
    model.ext[:_LDR_M] = M
    model.ext[:_LDR_sense] = data.sense
    model.ext[:_LDR_ABC] = ABC
    model.ext[:_LDR_first_stage_indices] = first_stage_indices

    return nothing
end

function _compute_groups(Wu, Wl, uncertainty_indices, uncertainty_to_distribution, data)
    # for each line of the matrices W, we check which random variables might be affected
    # we store the indices of the distributions that are affected
    # then we compute the groups that must be sampled together
    wu_rows = SparseArrays.rowvals(Wu)
    wu_m, wu_n = size(Wu)
    # wu_nz = SparseArrays.nonzeros(Wu)
    wu_groups = [Set{Tuple{Int, Bool}}() for _ in 1:wu_m]
    for col in 1:wu_n
        uncertainty_idx = uncertainty_indices[col]
        var = data.variables[uncertainty_idx]
        dist_idx, inner_idx = uncertainty_to_distribution[var]
        for r in SparseArrays.nzrange(Wu, col)
            row = wu_rows[r]
            push!(wu_groups[row], (dist_idx, inner_idx != 0))
        end
    end
    wl_rows = SparseArrays.rowvals(Wl)
    wl_m, wl_n = size(Wl)
    # wl_nz = SparseArrays.nonzeros(Wl)
    wl_groups = [Set{Tuple{Int, Bool}}() for _ in 1:wl_m]
    for col in 1:wl_n
        uncertainty_idx = uncertainty_indices[col]
        var = data.variables[uncertainty_idx]
        dist_idx, inner_idx = uncertainty_to_distribution[var]
        for r in SparseArrays.nzrange(Wl, col)
            row = wl_rows[r]
            push!(wl_groups[row], (dist_idx, inner_idx != 0))
        end
    end
    wl_row_idx = collect(1:wl_m)
    wu_row_idx = collect(1:wu_m)
    all_wl_rows_of_group = vcat([Set{Int}() for _ in 1:wu_m], [Set{Int}(i) for i in 1:wl_m])
    all_wu_rows_of_group = vcat([Set{Int}(i) for i in 1:wu_m], [Set{Int}() for _ in 1:wl_m])
    all_groups = vcat(wu_groups, wl_groups)

    # groupts that must be sampled together
    non_disjoint = true
    to_delete = Int[]
    while non_disjoint
        non_disjoint = false
        # merge groups with common elements
        for i in 1:length(all_groups)
            for j in i+1:length(all_groups)
                if !isdisjoint(all_groups[i], all_groups[j])
                    union!(all_groups[i], all_groups[j])
                    union!(all_wu_rows_of_group[i], all_wu_rows_of_group[j])
                    union!(all_wl_rows_of_group[i], all_wl_rows_of_group[j])
                    empty!(all_groups[j])
                    empty!(all_wu_rows_of_group[j])
                    empty!(all_wl_rows_of_group[j])
                    push!(to_delete, j)
                    non_disjoint = true
                end
            end
        end
        deleteat!(all_groups, to_delete)
        deleteat!(all_wu_rows_of_group, to_delete)
        deleteat!(all_wl_rows_of_group, to_delete)
        empty!(to_delete)
    end
    return all_groups, collect.(all_wu_rows_of_group), collect.(all_wl_rows_of_group)
end

function _sample_in_set!(rng, candidate, cache_wu_m, cache_wl_m, group, wu_rows, wl_rows, scalar_distributions, vector_distributions, scalar_idxs, vector_idxs, lb, ub, Wu, hu, Wl, hl)
    wu_m, wu_n = size(Wu)
    wl_m, wl_n = size(Wl)
    reject = true
    cont = 0
    while reject
        fill!(candidate, 0.0)
        cont += 1
        if cont > 1000
            println("Cannot sample a point in the set")
            break
        end
        reject = false
        for (dist_idx, is_vector) in group
            if is_vector
                dist = vector_distributions[dist_idx]
                list = vector_idxs[dist_idx]
                Random.rand!(rng, dist, @view candidate[list])
                for i in list
                    if !(lb[i] <= candidate[i] <= ub[i])
                        reject = true
                        break
                    end
                end
            else
                dist = scalar_distributions[dist_idx]
                val = Random.rand(rng, dist)
                i = scalar_idxs[dist_idx]
                candidate[i] = val
                if !(lb[i] < candidate[i] < ub[i])
                    reject = true
                    break
                end
            end
        end
        LinearAlgebra.mul!(cache_wu_m, Wu, candidate)
        for i in wu_rows
            if !(cache_wu_m[i] <= hu[i])
                reject = true
                break
            end
        end
        if reject
            continue
        end
        LinearAlgebra.mul!(cache_wl_m, Wl, candidate)
        for i in wl_rows
            if !(cache_wl_m[i] >= hl[i])
                reject = true
                break
            end
        end
    end
    return cont
end
