"""
    SpikeCore

Shared core of the Phase 0 spike: reads a model through MOI, builds the canonical LDR form

    A x(ξ) (≤, =) B ξ  ∀ ξ ∈ Ξ = {ξ : W ξ ≥ h},   x(ξ) = X ξ,   ξ = (1, η)

and solves the dense primal (Kuhn–Wiesemann–Georghiou) and dual LDR approximations. See `spike/SPIKE.md`.
"""
module SpikeCore

import Distributions
import JuMP
import LinearAlgebra
import MathOptInterface as MOI

const VI = MOI.VariableIndex
const SAF = MOI.ScalarAffineFunction{Float64}
const SQF = MOI.ScalarQuadraticFunction{Float64}
const BoundSet = Union{MOI.LessThan{Float64},MOI.GreaterThan{Float64},MOI.EqualTo{Float64},MOI.Interval{Float64}}

export UncertaintyDecl, ModelInfo, Canonical, FormulationResult, canonicalize, solve_primal, solve_dual,
    decisions, uncertainties, mean_xi, second_moment, expected_decision, rule, expected_value

"""
    UncertaintyDecl(dist, vars, stages)

One uncertainty declaration: a `Univariate` distribution with one variable, or a `Multivariate` one with
`length(vars)` variables. `stages[k]` is the stage of `vars[k]` (`nothing` = unmarked). Separate declarations
are independent; inside one, `Distributions.cov(dist)` is used.
"""
struct UncertaintyDecl
    dist::Distributions.Distribution
    vars::Vector{VI}
    stages::Vector{Union{Int,Nothing}}
end

"""
    ModelInfo(; uncertainties, decision_stage, valid_rows, truncate_rows)

LDR metadata that the front end attaches to an MOI model. Variables not listed in any `UncertaintyDecl` are
decisions; `decision_stage` gives their stage (absent = unmarked). `valid_rows` are uncertainty-only rows
tagged `ValidInequality()` (they enter Ξ); `truncate_rows` are tagged `Truncate()` (not supported here).
"""
Base.@kwdef struct ModelInfo
    uncertainties::Vector{UncertaintyDecl} = UncertaintyDecl[]
    decision_stage::Dict{VI,Int} = Dict{VI,Int}()
    valid_rows::Set{MOI.ConstraintIndex} = Set{MOI.ConstraintIndex}()
    truncate_rows::Set{MOI.ConstraintIndex} = Set{MOI.ConstraintIndex}()
end

"""
    Canonical

Dense canonical form. With `n` decisions, `m` uncertainties and `k = 1 + m`:
`E X = F` (equalities, as identities in ξ), `G X ξ ≤ H ξ` on Ξ (all inequalities, including decision bounds),
Ξ = {ξ : W ξ ≥ h} (rows 1–2 encode ξ₁ = 1), `M = E[ξξᵀ]`, objective `tr(XᵀPXM) + tr(CᵀXM) + r` in `sense`.
`mask[i, j]` says whether `X[i, j]` is a free coefficient.
"""
struct Canonical
    decisions::Vector{VI}
    uncertainties::Vector{VI}
    decision_stage::Vector{Int}
    uncertainty_stage::Vector{Int}
    T::Int
    mask::BitMatrix
    integers::Vector{Int}
    binaries::Vector{Int}
    E::Matrix{Float64}
    F::Matrix{Float64}
    G::Matrix{Float64}
    H::Matrix{Float64}
    W::Matrix{Float64}
    h::Vector{Float64}
    mu::Vector{Float64}
    M::Matrix{Float64}
    P::Matrix{Float64}
    C::Matrix{Float64}
    r::Float64
    sense::MOI.OptimizationSense
end

"""
    FormulationResult

Result of one LDR formulation. `objective_value` is in the user's sense (a Max problem stays Max). `X` is
`n_decisions × (1 + n_uncertainties)`; masked entries are `0.0`; all entries are `NaN` without a primal solution.
"""
struct FormulationResult
    termination_status::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    objective_value::Float64
    X::Matrix{Float64}
end

"""
    is_metadata_constraint(F::Type, S::Type) -> Bool

Return `true` for constraint types that `canonicalize` must ignore (front-end metadata). Default `false`;
designs may add methods for their own set types.
"""
is_metadata_constraint(::Type, ::Type) = false

"Vector of decision variables in canonical order (rows of `X`)."
decisions(c::Canonical) = c.decisions
"Vector of uncertainty variables in canonical order (columns `2:end` of `X`)."
uncertainties(c::Canonical) = c.uncertainties
"`E[ξ]`, of length `1 + n_uncertainties` (first entry is 1)."
mean_xi(c::Canonical) = c.mu
"`M = E[ξξᵀ]`."
second_moment(c::Canonical) = c.M

_name(src, v::VI) = (s = try MOI.get(src, MOI.VariableName(), v) catch; "" end; isempty(s) ? string(v) : s)
_bounds(s::MOI.LessThan) = (-Inf, s.upper)
_bounds(s::MOI.GreaterThan) = (s.lower, Inf)
_bounds(s::MOI.EqualTo) = (s.value, s.value)
_bounds(s::MOI.Interval) = (s.lower, s.upper)

_box(d::Distributions.UnivariateDistribution) = ([Float64(minimum(d))], [Float64(maximum(d))])
function _box(d::Distributions.MultivariateDistribution)
    n = length(d)
    lo = try Float64.(minimum(d)) catch; fill(-Inf, n) end
    hi = try Float64.(maximum(d)) catch; fill(Inf, n) end
    return lo, hi
end
_moments(d::Distributions.UnivariateDistribution) = ([Distributions.mean(d)], fill(Distributions.var(d), 1, 1))
_moments(d::Distributions.MultivariateDistribution) = (Distributions.mean(d), Matrix(Distributions.cov(d)))

"""
    canonicalize(src::MOI.ModelLike, info::ModelInfo) -> Canonical

Read `src` with read-only MOI getters (works on `JuMP.backend(model)` or a bare MOI model) and build the
canonical LDR form. Errors on unsupported input; warns when `T > 2` and some variable is unmarked.
"""
function canonicalize(src::MOI.ModelLike, info::ModelInfo)
    all_vars = MOI.get(src, MOI.ListOfVariableIndices())
    in_model = Set(all_vars)
    # --- uncertainties: (declaration, component, stage) per variable
    unc_of = Dict{VI,Tuple{Int,Int,Union{Int,Nothing}}}()
    for (q, decl) in enumerate(info.uncertainties)
        nd = decl.dist isa Distributions.UnivariateDistribution ? 1 : length(decl.dist)
        length(decl.vars) == nd == length(decl.stages) ||
            error("UncertaintyDecl #$q: distribution has dimension $nd but $(length(decl.vars)) variables " *
                  "and $(length(decl.stages)) stages were given.")
        for (k, v) in enumerate(decl.vars)
            v in in_model || error("UncertaintyDecl #$q: variable $v is not in the model.")
            haskey(unc_of, v) && error("Variable $(_name(src, v)) is declared as an uncertainty twice.")
            unc_of[v] = (q, k, decl.stages[k])
        end
    end
    unc = [v for v in all_vars if haskey(unc_of, v)]
    dec = [v for v in all_vars if !haskey(unc_of, v)]
    for (v, _) in info.decision_stage
        haskey(unc_of, v) && error("Variable $(_name(src, v)) is an uncertainty but has a decision stage.")
    end
    n, m = length(dec), length(unc)
    k = 1 + m
    dcol = Dict(v => i for (i, v) in enumerate(dec))
    ucol = Dict(v => j for (j, v) in enumerate(unc))
    # --- stages
    declared = vcat(collect(Base.values(info.decision_stage)),
        Int[s for (_, _, s) in Base.values(unc_of) if s !== nothing])
    any(<(1), declared) && error("Stages must be ≥ 1; got $(minimum(declared)).")
    T = max(2, maximum(declared; init = 1))
    dstage = [get(info.decision_stage, v, T) for v in dec]
    ustage = [something(unc_of[v][3], T) for v in unc]
    if T > 2
        unmarked = [v for v in all_vars if haskey(unc_of, v) ? unc_of[v][3] === nothing : !haskey(info.decision_stage, v)]
        isempty(unmarked) || @warn "Model has $T stages; unmarked variables are assigned Stage($T): " *
                                   join((_name(src, v) for v in unmarked), ", ")
    end
    # --- constraints
    xl, xu = fill(-Inf, n), fill(Inf, n)
    integers, binaries = Int[], Int[]
    Arows = Tuple{Vector{Float64},Vector{Float64},Float64,Float64,MOI.ConstraintIndex}[]  # (a_dec, a_unc, lo, hi, ci)
    for (Fn, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        is_metadata_constraint(Fn, S) && continue
        cis = MOI.get(src, MOI.ListOfConstraintIndices{Fn,S}())
        if Fn == VI && (S <: BoundSet || S in (MOI.Integer, MOI.ZeroOne))
            for ci in cis
                v = MOI.get(src, MOI.ConstraintFunction(), ci)
                haskey(unc_of, v) && error("Variable $(_name(src, v)) is an uncertainty; it cannot have a bound or " *
                    "integrality constraint ($S). Its support comes from its distribution.")
                i = dcol[v]
                if S <: BoundSet
                    lo, hi = _bounds(MOI.get(src, MOI.ConstraintSet(), ci))
                    xl[i], xu[i] = max(xl[i], lo), min(xu[i], hi)
                else
                    dstage[i] == 1 || error("Integer/binary decision $(_name(src, v)) is in Stage($(dstage[i])); " *
                                            "integer decisions are only supported in Stage(1).")
                    push!(S == MOI.Integer ? integers : binaries, i)
                end
            end
        elseif Fn == SAF && S <: BoundSet
            for ci in cis
                f = MOI.get(src, MOI.ConstraintFunction(), ci)
                ad, au = zeros(n), zeros(m)
                for t in f.terms
                    haskey(dcol, t.variable) ? (ad[dcol[t.variable]] += t.coefficient) : (au[ucol[t.variable]] += t.coefficient)
                end
                lo, hi = _bounds(MOI.get(src, MOI.ConstraintSet(), ci)) .- f.constant
                push!(Arows, (ad, au, lo, hi, ci))
            end
        elseif Fn == SQF
            f = MOI.get(src, MOI.ConstraintFunction(), first(cis))
            if all(t -> haskey(unc_of, t.variable_1) != haskey(unc_of, t.variable_2), f.quadratic_terms)
                error("Constraint $(first(cis)) multiplies an uncertainty by a decision (random recourse); " *
                      "this is not supported.")
            end
            error("Quadratic constraints ($Fn-in-$S) are not supported by the LDR core.")
        else
            error("Unsupported constraint type in the LDR core: $Fn-in-$S.")
        end
    end
    # --- split rows: equalities, ≤ system (≥ rows negated, intervals split), Ξ rows
    Erows, Grows, Wrows = Tuple{Vector{Float64},Vector{Float64}}[], Tuple{Vector{Float64},Vector{Float64}}[],
        Tuple{Vector{Float64},Float64}[]
    for (ad, au, lo, hi, ci) in Arows
        if all(iszero, ad)
            ci in info.truncate_rows && error("Constraint $ci is tagged Truncate(), which is not in the spike.")
            ci in info.valid_rows || error("Constraint $ci involves only uncertainties. Tag it with " *
                "`ValidInequality()` (it describes the support Ξ) or `Truncate()` (it truncates the distribution).")
            lo == hi && error("ValidInequality constraint $ci is an equality; Ξ must be full-dimensional.")
            isfinite(lo) && push!(Wrows, (au, lo))          # au⋅η ≥ lo
            isfinite(hi) && push!(Wrows, (-au, -hi))        # −au⋅η ≥ −hi
            continue
        end
        (ci in info.valid_rows || ci in info.truncate_rows) &&
            error("Constraint $ci is tagged ValidInequality()/Truncate() but involves decisions.")
        if lo == hi
            push!(Erows, (ad, [lo; -au]))
        else
            isfinite(hi) && push!(Grows, (ad, [hi; -au]))     # ad⋅x ≤ hi − au⋅η
            isfinite(lo) && push!(Grows, (-ad, [-lo; au]))    # −ad⋅x ≤ −lo + au⋅η
        end
    end
    for i in 1:n
        e = zeros(n); e[i] = 1.0
        isfinite(xu[i]) && push!(Grows, (e, [xu[i]; zeros(m)]))
        isfinite(xl[i]) && push!(Grows, (-e, [-xl[i]; zeros(m)]))
    end
    _stack(rows, f, w) = isempty(rows) ? zeros(0, w) : permutedims(reduce(hcat, [f(r) for r in rows]))
    E, F = _stack(Erows, first, n), _stack(Erows, last, k)
    G, H = _stack(Grows, first, n), _stack(Grows, last, k)
    # --- Ξ = {ξ : W ξ ≥ h}: ξ₁ = 1, distribution box, valid rows
    lb, ub = fill(-Inf, m), fill(Inf, m)
    mu = [1.0; zeros(m)]
    M = zeros(k, k)
    for decl in info.uncertainties
        lo, hi = _box(decl.dist)
        mn, cv = _moments(decl.dist)
        idx = [ucol[v] for v in decl.vars]
        lb[idx], ub[idx], mu[1 .+ idx] = lo, hi, mn
        M[1 .+ idx, 1 .+ idx] = cv
    end
    M .+= mu * mu'
    Wl = [(begin e = zeros(m); e[j] = 1.0; e end, lb[j]) for j in 1:m if isfinite(lb[j])]
    Wu = [(begin e = zeros(m); e[j] = -1.0; e end, -ub[j]) for j in 1:m if isfinite(ub[j])]
    allW = vcat(Wl, Wu, Wrows)
    W = [1.0 zeros(1, m); -1.0 zeros(1, m); zeros(length(allW), 1) _stack(allW, first, m)]
    h = [1.0; -1.0; Float64[last(r) for r in allW]]
    # --- objective: x'Px + (c + 2 Q_du η)'x + η'Qη + d'η + f
    sense = MOI.get(src, MOI.ObjectiveSense())
    P, C, r = zeros(n, n), zeros(n, k), 0.0
    if sense != MOI.FEASIBILITY_SENSE
        Fo = MOI.get(src, MOI.ObjectiveFunctionType())
        Fo in (VI, SAF, SQF) || error("Unsupported objective function type in the LDR core: $Fo.")
        fobj = convert(SQF, MOI.get(src, MOI.ObjectiveFunction{Fo}()))
        Qfull = zeros(n + m, n + m)                   # f = z'Qz with z = [x; η]
        pos(v) = haskey(dcol, v) ? dcol[v] : n + ucol[v]
        for t in fobj.quadratic_terms                 # MOI: (c, a, a) is c/2 a²; (c, a, b) is c a b
            a, b = pos(t.variable_1), pos(t.variable_2)
            Qfull[a, b] += t.coefficient / 2
            a == b || (Qfull[b, a] += t.coefficient / 2)
        end
        cfull = zeros(n + m)
        for t in fobj.affine_terms
            cfull[pos(t.variable)] += t.coefficient
        end
        P = Qfull[1:n, 1:n]
        C = [cfull[1:n] 2 * Qfull[1:n, n+1:end]]
        r = fobj.constant + cfull[n+1:end]' * mu[2:end] + sum(Qfull[n+1:end, n+1:end] .* M[2:end, 2:end]; init = 0.0)
    end
    # --- stage mask
    mask = BitMatrix([j == 1 || ustage[j-1] <= dstage[i] for i in 1:n, j in 1:k])
    for i in vcat(integers, binaries)
        mask[i, 2:end] .= false
    end
    return Canonical(dec, unc, dstage, ustage, T, mask, integers, binaries, E, F, G, H, W, h, mu, M, P, C, r,
        sense == MOI.FEASIBILITY_SENSE ? MOI.MIN_SENSE : sense)
end

# ---------------------------------------------------------------- formulations

# Build the LDR model skeleton: rule variables, integrality, equalities, slacks S = H − G X, objective.
function _skeleton(c::Canonical, optimizer_factory, silent)
    model = JuMP.Model(optimizer_factory)
    silent && JuMP.set_silent(model)
    n, k = size(c.mask)
    X = Matrix{Union{Nothing,JuMP.VariableRef}}(nothing, n, k)
    for i in 1:n, j in 1:k
        c.mask[i, j] && (X[i, j] = JuMP.@variable(model, base_name = "X[$i,$j]"))
    end
    foreach(i -> JuMP.set_integer(X[i, 1]), c.integers)
    foreach(i -> JuMP.set_binary(X[i, 1]), c.binaries)
    # row expressions: (A X)[r, j] for a dense A, skipping zeros and masked entries
    function AX(A)
        out = [JuMP.AffExpr(0.0) for _ in 1:size(A, 1), _ in 1:k]
        for r in axes(A, 1), i in findall(!iszero, A[r, :]), j in 1:k
            X[i, j] === nothing || JuMP.add_to_expression!(out[r, j], A[r, i], X[i, j])
        end
        return out
    end
    EX = AX(c.E)
    JuMP.@constraint(model, [r in axes(EX, 1), j in 1:k], EX[r, j] == c.F[r, j])
    S = c.H .- AX(c.G)
    # objective: tr(XᵀPXM) + tr(CᵀXM) + r  =  Σ P[i,i'] M[j,j'] X[i,j] X[i',j'] + Σ (CM)[i,j] X[i,j] + r
    obj = JuMP.QuadExpr(JuMP.AffExpr(c.r))
    CM = c.C * c.M
    for i in 1:n, j in 1:k
        X[i, j] === nothing || iszero(CM[i, j]) || JuMP.add_to_expression!(obj, CM[i, j], X[i, j])
    end
    for I in findall(!iszero, c.P), J in findall(!iszero, c.M)
        a, b = X[I[1], J[1]], X[I[2], J[2]]
        (a === nothing || b === nothing) || JuMP.add_to_expression!(obj, c.P[I] * c.M[J], a, b)
    end
    JuMP.set_objective(model, c.sense, obj)
    return model, X, S
end

function _result(model, X)
    JuMP.optimize!(model)
    ts, ps = JuMP.termination_status(model), JuMP.primal_status(model)
    has = JuMP.has_values(model)
    Xv = [x === nothing ? 0.0 : (has ? JuMP.value(x) : NaN) for x in X]
    return FormulationResult(ts, ps, has ? JuMP.objective_value(model) : NaN, Xv)
end

"""
    solve_primal(c::Canonical, optimizer_factory; silent = true) -> FormulationResult

Primal LDR (conservative): each inequality row `s(ξ) = (H − G X)ξ ≥ 0` on Ξ is enforced via
`Λ ≥ 0, Λ W = s, Λ h ≥ 0`. Gives a feasible policy; its objective is a bound on the true one.
"""
function solve_primal(c::Canonical, optimizer_factory; silent = true)
    model, X, S = _skeleton(c, optimizer_factory, silent)
    nr, nW, k = size(S, 1), size(c.W, 1), size(c.W, 2)
    JuMP.@variable(model, Λ[1:nr, 1:nW] >= 0)
    for r in 1:nr
        for j in 1:k        # Λ W = s
            JuMP.@constraint(model, _dot(c.W[:, j], Λ[r, :]) == S[r, j])
        end
        JuMP.@constraint(model, _dot(c.h, Λ[r, :]) >= 0)
    end
    return _result(model, X)
end

"""
    solve_dual(c::Canonical, optimizer_factory; silent = true) -> FormulationResult

Dual LDR (progressive bound): each inequality row enforces `s M W'ᵀ ≥ 0` with `W' = W − h e₁ᵀ` without its
first two (zero) rows. Its objective bounds the optimal value from the other side.
"""
function solve_dual(c::Canonical, optimizer_factory; silent = true)
    model, X, S = _skeleton(c, optimizer_factory, silent)
    W2 = copy(c.W)
    W2[:, 1] .-= c.h
    MW = c.M * W2[3:end, :]'
    for r in axes(S, 1), l in axes(MW, 2)
        JuMP.@constraint(model, _dot(MW[:, l], S[r, :]) >= 0)
    end
    return _result(model, X)
end

# Σ a[l] * x[l] over the nonzeros of a, as an AffExpr.
function _dot(a::AbstractVector{Float64}, x::AbstractVector)
    e = JuMP.AffExpr(0.0)
    for l in eachindex(a)
        iszero(a[l]) || JuMP.add_to_expression!(e, a[l], x[l])
    end
    return e
end

# ---------------------------------------------------------------- results

# Coefficient vector a with v(ξ) = a ⋅ ξ.
function _coefs(c::Canonical, res::FormulationResult, v::VI)
    i = findfirst(==(v), c.decisions)
    i === nothing || return res.X[i, :]
    j = findfirst(==(v), c.uncertainties)
    j === nothing && error("Variable $v is not in the canonical model.")
    a = zeros(1 + length(c.uncertainties)); a[1+j] = 1.0
    return a
end

"""
    expected_decision(c, res, vi) -> Float64

`E[x_vi(ξ)] = X[i, :] ⋅ E[ξ]` for a decision; `E[η]` for an uncertainty.
"""
expected_decision(c::Canonical, res::FormulationResult, vi::VI) = LinearAlgebra.dot(_coefs(c, res, vi), c.mu)

"""
    rule(c, res, vi) -> (constant, Dict{MOI.VariableIndex,Float64})

The affine rule `x(η) = constant + Σ coef[u] u`, listing only nonzero coefficients. For an uncertainty `u`,
returns `(0.0, Dict(u => 1.0))`.
"""
function rule(c::Canonical, res::FormulationResult, vi::VI)
    a = _coefs(c, res, vi)
    return a[1], Dict(c.uncertainties[j] => a[1+j] for j in eachindex(c.uncertainties) if !iszero(a[1+j]))
end

"""
    expected_value(c, res, f) -> Float64

Exact `E[f(x(ξ), η)]` for an affine or quadratic MOI function `f` (or a `VariableIndex`), using `E[ξ]` and `M`.
"""
function expected_value(c::Canonical, res::FormulationResult, f::Union{VI,SAF,SQF})
    g = convert(SQF, f)
    val = g.constant
    for t in g.affine_terms
        val += t.coefficient * LinearAlgebra.dot(_coefs(c, res, t.variable), c.mu)
    end
    for t in g.quadratic_terms
        a, b = _coefs(c, res, t.variable_1), _coefs(c, res, t.variable_2)
        coef = t.variable_1 == t.variable_2 ? t.coefficient / 2 : t.coefficient
        val += coef * LinearAlgebra.dot(a, c.M, b)
    end
    return val
end

end # module
