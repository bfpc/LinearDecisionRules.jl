# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    DualLDR(; attributes = [], lifted_box::Bool = true)

The dual linear decision rule: a relaxation whose objective value bounds the optimal value of the
stochastic program (an upper bound of a maximization, a lower bound of a minimization).

Its rule is not a policy. Each inequality `s(ξ) ≥ 0` is relaxed to `E[s(ξ) W'ξ] ∈ K`, where
`W'ξ ∈ K` describes Ξ, including the row `ξ₁ ≥ 0` (Kuhn, Wiesemann and Georghiou, 2011): `K` is
the nonnegative orthant of the linear rows of the [`SupportBlock`](@ref)s times a second-order
cone for each of their second-order cone rows, so the inner problem is then a second-order cone
program.

The bound needs the conditional expectations of the uncertainties of later stages, given those of
earlier stages, to be linear (as for independent stages). Separate declarations are independent,
so this is a property of each law whose components are in several stages: it holds for a product
of univariate laws, a normal law and a [`ConfidenceMvNormal`](@ref), and it is checked exactly
for a [`MvDiscreteNonParametric`](@ref); it is unknown for other laws, such as a
[`ConditionedDistribution`](@ref). Building the dual warns, and the bound is not certified, when
such a law does not have it or is not known to.

`attributes` are `name => value` raw attributes of the inner optimizer, set after (so over) those
of the optimizer, for this formulation only. Formulations are compared by type: the attributes are
settings, not part of the identity.

The support of an uncertainty with `n` [`BreakPoints`](@ref) is described by the convex-hull rows
of its pieces and the box of the pieces. `lifted_box = false` leaves out the `2n` bounds of the box
that the hull rows imply, which leaves the support and the bound unchanged, with fewer
constraints. On inventory problems with breakpoints, and on a hydro problem with 4 stages
(HiGHS 1.15, dual simplex), it was 8% to 47% faster, but 2.6 times slower on the hydro problem
with 8 stages, so it is not the default.

The dual needs its bound only, not a basis. With HiGHS, its interior point method without
crossover is often much faster on large problems:

```julia
DualLDR(; attributes = ["solver" => "ipm", "run_crossover" => "off"])
```

On inventory and hydro problems (HiGHS 1.15), it solved every dual within 7 seconds, where the
default dual simplex took 145 seconds on the hydro problem with T = 8. This is not the default, since the names are HiGHS's
and the best mode depends on the problem. The bound is then the dual objective value of the inner
linear program (see [`FormulationResult`](@ref)). Only a termination status `OPTIMAL` can be
trusted: HiGHS may stop an interior point without crossover at an `unknown` status (its HiPO
stagnates a little above the tolerance), with a wrong objective value or none; the termination
status is then `OTHER_ERROR` and the bound `NaN`. This happens on most of the small problems of the
test library, so keep the default for small models.
"""
struct DualLDR <: AbstractFormulation
    attributes::Vector{Pair{String,Any}}
    lifted_box::Bool
end

function DualLDR(; attributes = Pair{String,Any}[], lifted_box::Bool = true)
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

# The dual bound needs the conditional expectations of later stages given earlier ones to be
# linear, law by law (see `_linear_conditional_expectations`): the laws spanning several stages
# that do not have them, and those that are not known to, each as `"a, b (stages 2, 3)"`.
function _stage_dependent_laws(form::CanonicalForm)
    nonlinear, unknown = String[], String[]
    for law in form.laws
        allequal(law.stages) && continue
        value = _linear_conditional_expectations(law.distribution, law.stages)
        value === true && continue
        names = _name_list([form.uncertainty_names[j-1] for j in law.columns])
        stages = join(sort!(unique(law.stages)), ", ")
        push!(value === false ? nonlinear : unknown, "$names (stages $stages)")
    end
    return nonlinear, unknown
end

# A combination with breakpoints of uncertainties of several stages breaks the linear conditional
# expectations that the dual bound needs: its pieces are not affine in the earlier uncertainties
# (see `CombinationPiecewise`). `optimize!` checks it before solving any formulation.
function _check_combinations_across_stages(form::CanonicalForm, formulations)
    any(f -> f isa DualLDR, formulations) || return
    form.lifting === nothing && return
    map = form.lifting
    across = String[]
    for block in map.blocks
        _keeps_inputs(block.lifting) || continue
        stages = sort!(unique(map.uncertainty_stages[j-1] for j in block.inputs))
        length(stages) > 1 || continue
        names = _name_list([map.uncertainty_names[j-1] for j in block.inputs])
        push!(across, "$names (stages $(join(stages, ", ")))")
    end
    isempty(across) && return
    return error(
        "DualLDR() does not support breakpoints on a combination of uncertainties of several " *
        "stages, as for $(join(across, "; ")): the pieces of the combination are not affine in " *
        "the uncertainties of the earlier stages, so the dual would not be a bound. Use " *
        "PrimalLDR() or SampledLDR() for this model, or combinations within one stage.",
    )
end

function _warn_stage_dependence(form::CanonicalForm)
    nonlinear, unknown = _stage_dependent_laws(form)
    isempty(nonlinear) && isempty(unknown) && return
    parts = String[]
    isempty(nonlinear) || push!(parts, "fails for the law of $(join(nonlinear, "; "))")
    isempty(unknown) || push!(parts, "is not known for the law of $(join(unknown, "; "))")
    @warn(
        "The bound of DualLDR() is not certified: it needs the conditional expectations of the " *
        "uncertainties of later stages given those of earlier stages to be linear (as for " *
        "independent stages), which $(join(parts, ", and ")).",
    )
    return
end

function _build!(model::JuMP.Model, f::DualLDR, form::CanonicalForm, mom::Moments)
    _warn_stage_dependence(form)
    observed = InformationStructure(form)
    X = _rule_variables!(model, form, observed)
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
