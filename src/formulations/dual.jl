# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    DualLDR(; attributes = [], lifted_box::Bool = false)

The dual linear decision rule: a relaxation whose objective value bounds the optimal value of the
stochastic program (an upper bound of a maximization, a lower bound of a minimization).

Its rule is not a policy. Each inequality `s(ξ) ≥ 0` is relaxed to `E[s(ξ) W'ξ] ∈ K`, where
`W'ξ ∈ K` describes Ξ, including the row `ξ₁ ≥ 0` (Kuhn, Wiesemann and Georghiou, 2011): `K` is
the nonnegative orthant of the linear rows of the [`SupportBlock`](@ref)s times a second-order
cone for each of their second-order cone rows, so the inner problem is then a second-order cone
program.

The rule of a decision uses the uncertainties that the decision observes, as in the primal, when
the conditional expectations of the later uncertainties given the earlier ones are linear (as for
independent stages). Separate declarations are independent, so this is a property of each law
whose components are in several stages: it holds for a product of univariate laws, a normal law
and a [`ConfidenceMvNormal`](@ref), and it is checked exactly for a
[`MvDiscreteNonParametric`](@ref). For a law that does not have it, such as the law of a
combination of uncertainties of several stages with breakpoints, or that is not known to, such as
a [`ConditionedDistribution`](@ref), the rules of the earlier decisions may also use the later
uncertainties of that law, except through their linearly predictable parts. The bound is then
still a bound, and certified, but weaker than with linear conditional expectations; the other laws
keep the usual information structure. With such a law, the rule of the dual uses uncertainties
that its decision does not observe.

`attributes` are `name => value` raw attributes of the inner optimizer, set after (so over) those
of the optimizer, for this formulation only. Formulations are compared by type: the attributes are
settings, not part of the identity.

The support of an uncertainty with `n` [`BreakPoints`](@ref) is described by `n + 2` rows: the
convex-hull rows of its pieces, and the two bounds of their box that these rows do not imply, as
in [`PrimalLDR`](@ref). `lifted_box = true` adds the other `2n` bounds of the box, which leaves the
support and the bound unchanged, with more constraints. Leaving them out was faster or as fast on
most benchmarks, but `true` was up to 3.3 times faster on some problems under HiGHS's default
simplex method (see [Performance](@ref performance)).

With HiGHS, IPX with crossover is often much faster than the default dual simplex on large duals,
especially with breakpoints:

```julia
DualLDR(; attributes = ["solver" => "ipx", "run_crossover" => "on"])
```

The interior points without crossover (`"run_crossover" => "off"`) are faster still when they
succeed, but they often stop at an `unknown` status, and their bound is then `NaN`; when they do
succeed, their bound can be a little invalid, since their point is only feasible up to a
tolerance. [Performance](@ref performance) compares the modes and the solvers.
"""
struct DualLDR <: AbstractFormulation
    attributes::Vector{Pair{String,Any}}
    lifted_box::Bool
end

function DualLDR(; attributes = Pair{String,Any}[], lifted_box::Bool = false)
    return DualLDR(_attribute_pairs(attributes), lifted_box)
end

# The formulations of a type are the same formulation whatever their attributes.
Base.:(==)(::DualLDR, ::DualLDR) = true

Base.hash(::DualLDR, h::UInt) = hash(:DualLDR, h)

function Base.show(io::IO, f::Union{PrimalLDR,DualLDR})
    settings = String[]
    if !isempty(f.attributes)
        pairs = ("$(repr(k)) => $(repr(v))" for (k, v) in f.attributes)
        push!(settings, "attributes = [$(join(pairs, ", "))]")
    end
    if f isa PrimalLDR && !f.box_closed_form
        push!(settings, "box_closed_form = false")
    end
    if f.lifted_box != typeof(f)().lifted_box
        push!(settings, "lifted_box = $(f.lifted_box)")
    end
    print(io, nameof(typeof(f)), "(", join(settings, ", "), ")")
    return
end

# The raw attributes that a formulation sets on its inner problem, over those of the optimizer.
_formulation_attributes(f::Union{PrimalLDR,DualLDR}) = f.attributes

capabilities(::DualLDR) = FormulationCapabilities(MOI.NO_SOLUTION, true)

# The rows `W'ξ = W_b η_b - h_b ξ₁ ∈ K_b` of one support block, seen through the moments:
# `E[s(ξ) W'ξ] = (S μ) (W'μ) + S Σ W'ᵀ` for the slack `s(ξ) = S ξ`.
struct _DualCone
    V::_SparseMatrix         # Σ W'ᵀ: one column per row of the block, one row per column of ξ
    Wμ::Vector{Float64}      # W'μ ∈ K_b, since the mean is in Ξ
    cones::Vector{_SupportCone}
    rows::Vector{UnitRange{Int}} # the rows of each piece of the cone
end

function _dual_cones(blocks::Vector{SupportBlock}, mom::Moments)
    Σ = _covariance_matrix(mom)
    return map(blocks) do block
        V = SparseArrays.dropzeros!(Σ[:, block.columns] * SparseArrays.sparse(transpose(block.W)))
        Wμ = block.W * mom.mean[block.columns] - block.h
        return _DualCone(V, Wμ, block.cones, _cone_rows(block))
    end
end

# With the slack rule `s(ξ) = S ξ`, `u = S μ`, the relaxation of `s(ξ) ≥ 0` on Ξ is
# `E[s(ξ) ξ₁] = u ≥ 0` and `E[s(ξ) W'ξ] = u (W'μ) + S Σ W'ᵀ ∈ K` for every support block, since
# `s(ξ) W'ξ ∈ K` on Ξ and K is a closed convex cone (Kuhn, Wiesemann and Georghiou, 2011,
# Proposition 3, for the nonnegative orthant). S is masked by the columns where the row is
# structurally nonzero, and Σ is block diagonal, so a piece of the cone with no masked column in
# its covariance block reduces to `u (W'μ)_piece ∈ K_piece`, implied by `u ≥ 0` and `W'μ ∈ K`.
function _add_dual_cone!(
    model::JuMP.Model,
    X,
    r::Int,
    ineq::_Inequality,
    μ::Vector{Float64},
    cones::Vector{_DualCone},
    cones_of::Vector{Vector{Int}},
)
    row, sign = ineq.row, ineq.sign
    S = Dict{Int,JuMP.VariableRef}()
    named = JuMP.set_string_names_on_creation(model)
    u = JuMP.@variable(model, lower_bound = 0.0, base_name = named ? "u[$r]" : "")
    mean = JuMP.AffExpr(0.0, u => 1.0)
    touched = Int[]
    for j in _structural_columns(row)
        S[j] = JuMP.@variable(model, base_name = named ? "S[$r,$j]" : "")
        # a ⋅ X_j + sign * S_j = b_j, that is, S_j = sign * (b_j - a ⋅ X_j).
        expr = _add_row_column!(JuMP.AffExpr(0.0), X, row, j, 1.0)
        JuMP.add_to_expression!(expr, sign, S[j])
        JuMP.@constraint(model, expr == row.b[j])
        iszero(μ[j]) || JuMP.add_to_expression!(mean, -μ[j], S[j])
        append!(touched, cones_of[j])
    end
    JuMP.@constraint(model, mean == 0)
    for b in unique!(sort!(touched))
        cone = cones[b]
        V = cone.V
        # The masked part `(S Σ W'ᵀ)_k` of the row `k` of the block.
        function masked(k)
            expr = JuMP.AffExpr(0.0)
            for p in SparseArrays.nzrange(V, k)
                s = get(S, SparseArrays.rowvals(V)[p], nothing)
                s === nothing || JuMP.add_to_expression!(expr, SparseArrays.nonzeros(V)[p], s)
            end
            return expr
        end
        for (piece, rows) in zip(cone.cones, cone.rows)
            if piece isa MOI.Nonnegatives
                for k in rows
                    expr = masked(k)
                    isempty(expr.terms) && continue
                    iszero(cone.Wμ[k]) || JuMP.add_to_expression!(expr, cone.Wμ[k], u)
                    JuMP.@constraint(model, expr >= 0)
                end
            else
                exprs = [masked(k) for k in rows]
                all(expr -> isempty(expr.terms), exprs) && continue
                for (expr, k) in zip(exprs, rows)
                    iszero(cone.Wμ[k]) || JuMP.add_to_expression!(expr, cone.Wμ[k], u)
                end
                JuMP.@constraint(model, exprs in JuMP.SecondOrderCone())
            end
        end
    end
    return
end

# The information structure of the dual, relaxed where condition (i) of `thm:dual-bound` fails or
# is not known (`prop:relaxed-mask` of `docs/math`). The bound needs `X = E[x ξᵀ] M⁻¹` to be an
# allowed rule for every nonanticipative policy `x`. For a decision `i` of stage `t` and a law
# whose later coordinates are not all linearly predictable at `t` (see `_predictable_coordinates`),
# that `X` may be nonzero on the later columns `f` of the law: they become extra columns of `i`.
# Each predictable column `j` of `f` still gives one equation, `E[x_i (ξ_j - E[ξ_j | history])] = 0`,
# which in `X` reads `X_{i,f} R[:, j] = 0`, with `R = Σ_ff - Σ_fp Σ_pp⁻¹ Σ_pf` the covariance of
# the residual of the linear prediction of `f` from the earlier coordinates `p` of the law: the
# equations returned, as `(i, f, R[:, predictable])`. The other laws keep the mask, since the laws
# are independent; the decisions that observe only the constant are never relaxed.
function _dual_information(form::CanonicalForm, mom::Moments)
    base = InformationStructure(form)
    extra = [Int[] for _ in form.decisions]
    equations = Tuple{Int,Vector{Int},Matrix{Float64}}[]
    relaxed = Dict{Tuple{Int,Int},Any}()
    for i in eachindex(form.decisions)
        base.num_visible[i] == 1 && continue
        t = form.decision_stages[i]
        for (l, (law, block)) in enumerate(zip(form.laws, mom.blocks))
            entry = get!(() -> _relaxed_law(law, block, t), relaxed, (l, t))
            entry === nothing && continue
            columns, R = entry
            append!(extra[i], columns)
            size(R, 2) > 0 && push!(equations, (i, columns, R))
        end
        sort!(extra[i])
    end
    observed = InformationStructure(form.decision_stages, base.column_stages, form.integer, extra)
    return observed, equations
end

# The later columns of `law` at stage `t` and the columns of `R` of their predictable ones, or
# `nothing` if they are all predictable, so that the mask holds.
function _relaxed_law(law::UncertaintyBlock, block::CovarianceBlock, t::Int)
    past = law.stages .<= t
    (all(past) || !any(past)) && return nothing
    predictable = _predictable_coordinates(law.distribution, law.stages, t)
    all(predictable) && return nothing
    @assert block.columns == law.columns
    f, p = findall(.!past), findall(past)
    Σ = block.covariance
    R = Σ[f, f] - Σ[f, p] * (LinearAlgebra.pinv(Σ[p, p]) * Σ[p, f])
    return law.columns[f], R[:, predictable[f]]
end

function _add_relaxed_equations!(model::JuMP.Model, X, equations)
    for (i, columns, R) in equations, c in axes(R, 2)
        expr = JuMP.AffExpr(0.0)
        for (m, j) in enumerate(columns)
            iszero(R[m, c]) || JuMP.add_to_expression!(expr, R[m, c], _variable(X[i], j))
        end
        isempty(expr.terms) || JuMP.@constraint(model, expr == 0)
    end
    return
end

function _build!(model::JuMP.Model, f::DualLDR, form::CanonicalForm, mom::Moments)
    observed, equations = _dual_information(form, mom)
    X = _rule_variables!(model, form, observed)
    _add_relaxed_equations!(model, X, equations)
    _add_equalities!(model, form, observed, X)
    cones = _dual_cones(support_blocks(form; f.lifted_box), mom)
    # The cones whose rows involve each column of ξ.
    cones_of = [Int[] for _ in mom.mean]
    for (b, cone) in enumerate(cones), j in unique(SparseArrays.rowvals(cone.V))
        push!(cones_of[j], b)
    end
    for (r, ineq) in enumerate(_inequalities(form, observed))
        if _is_deterministic(ineq.row)
            _add_deterministic!(model, X, ineq)
        else
            _add_dual_cone!(model, X, r, ineq, mom.mean, cones, cones_of)
        end
    end
    _set_objective!(model, form, mom, X)
    return X
end
