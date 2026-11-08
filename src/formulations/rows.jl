# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The parts that the formulations share: the rule variables, the rows and the objective.

# The rule of one decision: `variables[p]` is its coefficient on the column `columns[p]` of ξ, the
# columns sorted. They are a prefix `1:k`, except in a rule that `DualLDR` relaxes (the extra
# columns of `InformationStructure`); `prefix` says so, for the fast path of `_variable`.
struct _RuleRow
    columns::Vector{Int}
    variables::Vector{JuMP.VariableRef}
    prefix::Bool
end

function _RuleRow(columns::AbstractVector{Int}, variables::Vector{JuMP.VariableRef})
    return _RuleRow(collect(columns), variables, columns == 1:length(columns))
end

# The variable of the column `j` in the rule `x`, or `nothing` if the rule does not use it.
function _variable(x::_RuleRow, j::Int)
    if x.prefix
        return j <= length(x.variables) ? x.variables[j] : nothing
    end
    p = searchsortedfirst(x.columns, j)
    return p <= length(x.columns) && x.columns[p] == j ? x.variables[p] : nothing
end

_last_column(x::_RuleRow) = isempty(x.columns) ? 0 : x.columns[end]

# The rule variables: `X[i]` is the rule of decision `i`, with a variable for each column of ξ that
# it may use (see `_RuleRow`). A decision that observes only the constant column keeps its bounds as
# variable bounds; the bounds of the other decisions are rows (see `_inequalities`). A decision
# that is a parameter of the form (see `_with_parameter_decisions`) is a variable of `model` fixed
# to its value. It is not a JuMP `Parameter`: MOI's bridge of parameters, used by solvers without
# them (HiGHS), cannot change their value (MOI 1.54), so `set_parameter_value` drops the solver's
# copy of the model and every re-solve starts from scratch, while `fix` changes a bound in place
# and the solver keeps its basis (inventory, T = 64: 0 simplex iterations instead of 4999).
function _rule_variables!(model::JuMP.Model, form::CanonicalForm, observed::InformationStructure)
    X = Vector{_RuleRow}(undef, length(form.decisions))
    named = JuMP.set_string_names_on_creation(model)
    terms = form.parameters
    parameter_column = Dict(v => k for (k, v) in enumerate(terms.parameters))
    for i in eachindex(form.decisions)
        k = get(parameter_column, form.decisions[i], 0)
        if k > 0
            name = named ? form.decision_names[i] : ""
            variable = JuMP.@variable(model, base_name = name)
            JuMP.fix(variable, terms.values[k])
            X[i] = _RuleRow([1], [variable])
            continue
        end
        columns = visible_columns(observed, i)
        variables = map(columns) do j
            return JuMP.@variable(model, base_name = named ? "X[$i,$j]" : "")
        end
        X[i] = _RuleRow(columns, variables)
        constant = variables[1]
        if length(columns) == 1
            isfinite(form.lower[i]) && JuMP.set_lower_bound(constant, form.lower[i])
            isfinite(form.upper[i]) && JuMP.set_upper_bound(constant, form.upper[i])
        end
        if form.binary[i]
            JuMP.set_binary(constant)
        elseif form.integer[i]
            JuMP.set_integer(constant)
        end
    end
    return X
end

# One row `a ⋅ x(ξ) (≤, =, ≥) b ⋅ ξ`, and the columns of ξ on which it depends.
struct _LinearRow
    decisions::Vector{Int}
    coefficients::Vector{Float64}
    b::SparseArrays.SparseVector{Float64,Int}
    # `(a ⋅ X)[j]` can be nonzero for `j in 1:num_visible` and `j in extra`: the extra columns of
    # its decisions after `num_visible`, sorted (empty but in a relaxed dual).
    num_visible::Int
    extra::Vector{Int}
end

function _LinearRow(decisions, coefficients, b, num_visible::Int)
    return _LinearRow(decisions, coefficients, b, num_visible, Int[])
end

# Columns `j` where the row `a ⋅ X - b` is structurally nonzero: a column that one of its decisions
# may use, or a nonzero of `b`.
function _structural_columns(row::_LinearRow)
    later = filter(>(row.num_visible), SparseArrays.nonzeroinds(row.b))
    isempty(row.extra) && return vcat(1:row.num_visible, later)
    return vcat(1:row.num_visible, sort!(union(row.extra, later)))
end

function _is_deterministic(row::_LinearRow)
    return row.num_visible == 1 && isempty(row.extra) && all(<=(1), SparseArrays.nonzeroinds(row.b))
end

function _linear_rows(A::_SparseMatrix, B::_SparseMatrix, observed::InformationStructure)
    At, Bt = SparseArrays.sparse(transpose(A)), SparseArrays.sparse(transpose(B))
    return map(axes(A, 1)) do r
        range = SparseArrays.nzrange(At, r)
        decisions = SparseArrays.rowvals(At)[range]
        num_visible = maximum((observed.num_visible[i] for i in decisions); init = 1)
        extra = _row_extra(observed, decisions, num_visible)
        return _LinearRow(decisions, SparseArrays.nonzeros(At)[range], Bt[:, r], num_visible, extra)
    end
end

# The extra columns of the decisions of a row that are after its prefix `1:num_visible`.
function _row_extra(observed::InformationStructure, decisions, num_visible::Int)
    extra = Int[]
    for i in decisions, j in observed.extra[i]
        j > num_visible && push!(extra, j)
    end
    return sort!(unique!(extra))
end

# One inequality `s(ξ) = sign * (b ⋅ ξ - a ⋅ x(ξ)) ≥ 0`.
struct _Inequality
    sign::Float64
    row::_LinearRow
end

# The inequalities of `form`: the rows of `A_u` (sign 1) and `A_l` (sign -1), then the finite
# bounds of the decisions that observe some uncertainty.
function _inequalities(form::CanonicalForm, observed::InformationStructure)
    out = _Inequality[]
    append!(out, _Inequality.(1.0, _linear_rows(form.A_u, form.B_u, observed)))
    append!(out, _Inequality.(-1.0, _linear_rows(form.A_l, form.B_l, observed)))
    k = 1 + length(form.uncertainties)
    for (sign, bounds) in ((1.0, form.upper), (-1.0, form.lower)), i in eachindex(bounds)
        if isfinite(bounds[i]) && observed.num_visible[i] > 1
            b = SparseArrays.sparsevec([1], [bounds[i]], k)
            row = _LinearRow([i], [1.0], b, observed.num_visible[i], copy(observed.extra[i]))
            push!(out, _Inequality(sign, row))
        end
    end
    return out
end

# Add `coef * (a ⋅ X)[j]` to `expr`.
function _add_row_column!(expr::JuMP.AffExpr, X, row::_LinearRow, j::Int, coef::Float64)
    for (i, a) in zip(row.decisions, row.coefficients)
        x = _variable(X[i], j)
        x === nothing || JuMP.add_to_expression!(expr, coef * a, x)
    end
    return expr
end

# The equality rows hold for every ξ: match the coefficients of each column on which they depend.
function _add_equalities!(model::JuMP.Model, form::CanonicalForm, observed::InformationStructure, X)
    for row in _linear_rows(form.A_e, form.B_e, observed), j in _structural_columns(row)
        expr = _add_row_column!(JuMP.AffExpr(0.0), X, row, j, 1.0)
        JuMP.@constraint(model, expr == row.b[j])
    end
    return
end

# A deterministic inequality `sign * (b₁ - a ⋅ X[:, 1]) ≥ 0`.
function _add_deterministic!(model::JuMP.Model, X, ineq::_Inequality)
    expr = _add_row_column!(JuMP.AffExpr(ineq.sign * ineq.row.b[1]), X, ineq.row, 1, -ineq.sign)
    JuMP.@constraint(model, expr >= 0)
    return
end

# `E[η'Qη + d'η + f]`.
function _objective_constant(form::CanonicalForm, mom::Moments)
    μ = mom.mean[2:end]
    value = form.f + LinearAlgebra.dot(form.d, μ) + LinearAlgebra.dot(μ, form.Q, μ)
    for block in mom.blocks
        η = block.columns .- 1
        value += LinearAlgebra.dot(form.Q[η, η], block.covariance)
    end
    return value
end

# The objective `E[x'Px + x'Cξ] + E[η'Qη + d'η + f]` with `x = Xξ` and `E[ξξ'] = μμ' + Σ`, Σ block
# diagonal:
#   E[x'Px]  = (Xμ)'P(Xμ) + Σ_b tr(X_b' P X_b Σ_b),
#   E[x'Cξ] = ⟨C, XM⟩ = (Cμ)'(Xμ) + Σ_b ⟨C_b Σ_b, X_b⟩,
# where `X_b`, `C_b` are the columns of the covariance block `b`.
function _set_objective!(model::JuMP.Model, form::CanonicalForm, mom::Moments, X)
    if form.sense == MOI.FEASIBILITY_SENSE
        JuMP.set_objective_sense(model, MOI.FEASIBILITY_SENSE)
        return
    end
    μ = mom.mean
    obj = JuMP.QuadExpr(JuMP.AffExpr(_objective_constant(form, mom)))
    Cμ = form.C * μ
    for i in eachindex(X)
        iszero(Cμ[i]) && continue
        for (j, x) in zip(X[i].columns, X[i].variables)
            iszero(μ[j]) || JuMP.add_to_expression!(obj, Cμ[i] * μ[j], x)
        end
    end
    for block in mom.blocks
        Cb = form.C[:, block.columns]
        rows = unique!(sort(SparseArrays.rowvals(Cb)))
        isempty(rows) && continue
        CΣ = Matrix(Cb[rows, :]) * block.covariance
        for (r, i) in enumerate(rows), (c, j) in enumerate(block.columns)
            iszero(CΣ[r, c]) && continue
            x = _variable(X[i], j)
            x === nothing || JuMP.add_to_expression!(obj, CΣ[r, c], x)
        end
    end
    _add_quadratic_terms!(obj, model, form, mom, X)
    JuMP.set_objective(model, form.sense, isempty(obj.terms) ? obj.aff : obj)
    return
end

# `E[x'Px] = Σ_ik P_ik (E[x_i] E[x_k] + Σ_b X_ib' Σ_b X_kb)`, term by term: the columns of a block
# are visited in increasing order, so that those after the last column of a rule are skipped. The
# mean `E[x_i] = X_i ⋅ μ` of a decision is a variable `m_i` with one equality, so that the first
# term has one entry per nonzero of `P` instead of one per pair of columns.
function _add_quadratic_terms!(
    obj::JuMP.QuadExpr,
    model::JuMP.Model,
    form::CanonicalForm,
    mom::Moments,
    X,
)
    μ = mom.mean
    order = [sortperm(block.columns) for block in mom.blocks]
    first_column = [minimum(block.columns; init = typemax(Int)) for block in mom.blocks]
    I, K, V = SparseArrays.findnz(form.P)
    means = _decision_means!(model, X, μ, unique(vcat(I, K)))
    for (i, k, p) in zip(I, K, V)
        Xi, Xk = X[i], X[k]
        last_i, last_k = _last_column(Xi), _last_column(Xk)
        JuMP.add_to_expression!(obj, p * means[i], means[k])
        for (b, block) in enumerate(mom.blocks)
            first_column[b] <= min(last_i, last_k) || continue
            for a in order[b]
                ja = block.columns[a]
                ja <= last_i || break
                xa = _variable(Xi, ja)
                xa === nothing && continue
                for c in order[b]
                    jc = block.columns[c]
                    jc <= last_k || break
                    xc = _variable(Xk, jc)
                    xc === nothing && continue
                    s = p * block.covariance[a, c]
                    iszero(s) || JuMP.add_to_expression!(obj, s, xa, xc)
                end
            end
        end
    end
    return
end

# `E[x_i]` for the decisions `rows`: the expression `X_i ⋅ μ`, or a variable equal to it when it
# has more than one term.
function _decision_means!(model::JuMP.Model, X, μ::Vector{Float64}, rows)
    means = Dict{Int,JuMP.AffExpr}()
    for i in rows
        expr = JuMP.AffExpr(0.0)
        for (j, x) in zip(X[i].columns, X[i].variables)
            iszero(μ[j]) || JuMP.add_to_expression!(expr, μ[j], x)
        end
        if length(expr.terms) > 1
            m = JuMP.@variable(model)
            JuMP.@constraint(model, m == expr)
            means[i] = JuMP.AffExpr(0.0, m => 1.0)
        else
            means[i] = expr
        end
    end
    return means
end
