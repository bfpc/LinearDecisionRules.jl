# Phase 0 spike, experiment dualnnz: size of the primal / dual LDR with the structure
# proposed in the new design, on the monolithic hydrothermal instance of
# examples/recursion/monolithic_vs_sddp.jl.
#
# Usage (from the repository root, Julia >= 1.11, environment spike/dualnnz which has the
# old package under `Pkg.develop`):
#   julia --project=spike/dualnnz spike/dualnnz/measure.jl validate   # objectives, small instances
#   julia --project=spike/dualnnz spike/dualnnz/measure.jl check      # formulas vs JuMP model sizes
#   julia --project=spike/dualnnz spike/dualnnz/measure.jl sizes      # writes spike/dualnnz/sizes.csv
#   julia --project=spike/dualnnz spike/dualnnz/measure.jl all        # the three, in this order
#
# What it does
#  * The instance is built with the model-construction code of the example (its source text is
#    sliced out of recursion.jl / monolithic_vs_sddp.jl, so SDDP and PyPlot are not needed).
#  * "Current" models: the code of src/solve_primal.jl and src/solve_dual.jl itself (for
#    `validate`, the real solve; for `check`, the same source text with the `set_optimizer` /
#    `optimize!` tail cut off, so that the JuMP model can be counted without solving it).
#  * "Structured" models (this file): M = mu mu' + Sigma (Sigma block diagonal, one block per
#    uncertainty declaration = one per stage here), S masked by history, u_r = S_r mu, one
#    `u_r >= 0` per row for all W' rows whose support block is not touched by S_r, full
#    entries only for the touched blocks; the block primal keeps Lambda only for touched
#    blocks. See README.md.
#  * Sizes are computed by pattern arithmetic (`counts`) on the canonical data; the same
#    formulas are checked against the real JuMP models on small instances and T = 12, k = 0.
#
# Conventions: `cons` = linear constraints (variable bounds excluded and reported in `bnds`);
# `nnz` = nonzero coefficients of the linear constraints (objective excluded; it is the same
# for all models: one row with the nonzero entries of X). Structural nonzeros: entries that
# JuMP stores even when numerically zero are counted as terms (see `check`).

using JuMP, LinearAlgebra, SparseArrays, Printf, Random, Statistics
import HiGHS, Distributions, LinearDecisionRules
const LDR = LinearDecisionRules
const MOI = JuMP.MOI
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const REC = joinpath(ROOT, "examples", "recursion")
const LDR_TIME_LIMIT = 600.0
const REES = 1:4
const SUBSYS = 1:5

# ---------------------------------------------------------------------------------------
# Instance (code of the example, sliced out of the sources)
# ---------------------------------------------------------------------------------------
let src = read(joinpath(REC, "recursion.jl"), String)
    a = first(findfirst("function hydro_thermal_data()", src))
    b = first(findfirst("function hydro_thermal_sddp(;", src))
    include_string(Main, src[a:b-1], "recursion.jl")
end
let src = read(joinpath(REC, "monolithic_vs_sddp.jl"), String)
    a = first(findfirst("month_of(t) =", src))
    b = first(findfirst("# Size of the reformulated", src))
    include_string(Main, src[a:b-1], "monolithic_vs_sddp.jl")
end

# The two "current" builders, i.e. the code of src/solve_{primal,dual}.jl without the solver tail.
function _load_current_builders()
    for (file, fname, new, marker) in (
        ("solve_primal.jl", "_solve_primal_ldr", "_build_primal_ldr", "set_optimizer(model.primal_model"),
        ("solve_dual.jl", "_solve_dual_ldr", "_build_dual_ldr", "set_optimizer(model.dual_model"),
    )
        src = read(joinpath(ROOT, "src", file), String)
        i = first(findfirst(marker, src))
        src = replace(src[1:i-1] * "\n    return nothing\nend\n", fname => new)
        Base.include_string(LDR, src, file)
    end
end
_load_current_builders()

# ---------------------------------------------------------------------------------------
# Canonical data
# ---------------------------------------------------------------------------------------
struct Canon
    ABC::Any
    M::Matrix{Float64}
    μ::Vector{Float64}           # E[ξ], ξ₁ = 1
    Σ::Matrix{Float64}           # M - μμ' (zero first row/column)
    r::Float64
    sense::MOI.OptimizationSense
    dstage::Vector{Int}
    ustage::Vector{Int}
    W::SparseMatrixCSC{Float64,Int}   # [e1; -e1; ...]  (nW × dimξ), as in solve_primal.jl
    h::Vector{Float64}
    W2::Matrix{Float64}          # W - h e1' without its first two rows ((nW-2) × dimξ)
    dimx::Int
    dimξ::Int
    patX::Matrix{Int}            # 1 where X[i,j] exists
end

function canon(m)
    LDR._prepare_data(m)
    ABC = m.ext[:_LDR_ABC]
    M = Matrix{Float64}(m.ext[:_LDR_M])
    dstage = m.ext[:_LDR_decision_stage_by_local_index]
    ustage = m.ext[:_LDR_uncertainty_stage_by_local_index]
    dimx, dimξ = size(ABC.Ae, 2), size(ABC.Be, 2)
    dimη = dimξ - 1
    Wu = vcat(ABC.Wu, ABC.Wu_v)
    hu = vcat(ABC.hu, ABC.hu_v)
    Wl = vcat(ABC.Wl, ABC.Wl_v)
    hl = vcat(ABC.hl, ABC.hl_v)
    nu, nl = size(Wu, 1), size(Wl, 1)
    W = [
        1 zeros(1, dimη)
        -1 zeros(1, dimη)
        zeros(nu + nl + 2dimη, 1) [-Wu; Wl; -SparseArrays.I(dimη); SparseArrays.I(dimη)]
    ]
    h = Float64[1; -1; -hu; hl; -ABC.ub; ABC.lb]
    W2 = Matrix(W)
    W2[:, 1] .-= h
    W2 = W2[3:end, :]
    μ = M[:, 1]
    @assert μ[1] ≈ 1
    Σ = M - μ * μ'
    patX = [Int(j == 1 || ustage[j-1] <= dstage[i]) for i in 1:dimx, j in 1:dimξ]
    return Canon(
        ABC, M, μ, Σ, m.ext[:_LDR_r]::Float64, m.ext[:_LDR_sense], dstage, ustage,
        sparse(W), h, W2, dimx, dimξ, patX,
    )
end

function spones(A)
    A = dropzeros(sparse(A))
    I, J, _ = findnz(A)
    return sparse(I, J, 1.0, size(A)...)
end

# Inequality row groups: A X + s S == B, S = slack rule (rows × dimξ).
function groups(c::Canon)
    ABC = c.ABC
    iu, il = findall(isfinite, ABC.xu), findall(isfinite, ABC.xl)
    E(idx) = sparse(1:length(idx), idx, 1.0, length(idx), c.dimx)
    Bx(v) = sparse([Float64.(v) zeros(length(v), c.dimξ - 1)])
    return [
        (name = :Su, A = sparse(ABC.Au), B = sparse(ABC.Bu), s = 1.0),
        (name = :Sl, A = sparse(ABC.Al), B = sparse(ABC.Bl), s = -1.0),
        (name = :Sxu, A = E(iu), B = Bx(ABC.xu[iu]), s = 1.0),
        (name = :Sxl, A = E(il), B = Bx(ABC.xl[il]), s = -1.0),
    ]
end

# Mask of S (rows × dimξ): S[r,j] can be nonzero iff the row has a variable whose rule uses ξ_j
# or B[r,j] ≠ 0 ("history" mask). nA[r,j] = number of terms of (A X)[r,j].
function group_mask(c::Canon, g)
    nA = spones(g.A) * c.patX
    mask = (nA .> 0) .| (Matrix(g.B) .!= 0)
    return mask, nA
end

# Blocks: block id of every column of ξ (0 for the constant column).
block_of_columns(c::Canon, mode::Symbol) =
    mode == :stage ? [0; c.ustage] : mode == :dense ? [0; ones(Int, c.dimξ - 1)] : error(mode)

# block of each row of W2 (the single block its uncertainty columns belong to)
function block_of_rows(c::Canon, blk)
    out = zeros(Int, size(c.W2, 1))
    for k in eachindex(out)
        bs = unique(blk[j] for j in 2:c.dimξ if c.W2[k, j] != 0)
        length(bs) == 1 || error("row $k of W' spans blocks $bs")
        out[k] = only(bs)
    end
    return out
end

# check that Σ is block diagonal for the partition
function check_blocks(c::Canon, blk)
    for j in 2:c.dimξ, i in 2:c.dimξ
        blk[i] != blk[j] && @assert abs(c.Σ[i, j]) <= 1e-9 * (1 + abs(c.M[i, j]))
    end
end

# ---------------------------------------------------------------------------------------
# Size formulas (pattern arithmetic, no model built)
# ---------------------------------------------------------------------------------------
function counts(c::Canon, mode::Symbol)
    blk = block_of_columns(c, mode)
    check_blocks(c, blk)
    brow = block_of_rows(c, blk)
    nb = maximum(blk)
    dimξ, nW2 = c.dimξ, size(c.W2, 1)
    wμ = c.W2 * c.μ
    @assert all(wμ .>= -1e-9) "mean outside Ξ"
    tol = 1e-9
    nrows_b = [count(==(b), brow) for b in 1:nb]             # W' rows per block
    ncols_b = [count(==(b), blk) for b in 1:nb]              # ξ columns per block
    nzμ_b = [count(k -> brow[k] == b && abs(wμ[k]) > tol, 1:nW2) for b in 1:nb]
    pos_b = nzμ_b
    nnzW_b = [count(!iszero, c.W2[brow.==b, 2:end]) for b in 1:nb]
    nh_b = [count(k -> brow[k] == b && c.h[k+2] != 0, 1:nW2) for b in 1:nb]
    nX = sum(c.patX)
    ABC = c.ABC
    neq = size(ABC.Ae, 1)
    eq_nnz = sum(spones(ABC.Ae) * c.patX)
    G = groups(c)
    R = 0
    cur_S_vars = 0           # R * dimξ
    eqS_nnz_cur = 0          # nnz of A X + s S == B (S full)
    nA_all = 0
    mask_size = 0            # Σ |mask_r|
    mask_cols_mu = 0
    nA_masked = 0
    cone_struct_cons = 0
    cone_struct_nnz = 0
    u_bounds = 0
    block_cons = 0           # block primal: Σ |cols(r)|
    block_lam = 0            # Σ Λ vars of block primal
    block_lam_nnz = 0
    block_coup_nnz = 0
    for g in G
        mask, nA = group_mask(c, g)
        Rg = size(mask, 1)
        R += Rg
        cur_S_vars += Rg * dimξ
        nA_all += sum(nA)
        eqS_nnz_cur += sum(nA) + Rg * dimξ
        mask_size += sum(mask)
        nA_masked += sum(nA)
        # per (row, block): number of masked columns
        Nrb = zeros(Int, Rg, nb)
        for j in 2:dimξ
            Nrb[:, blk[j]] .+= mask[:, j]
        end
        for r in 1:Rg
            collapse = false
            for b in 1:nb
                if Nrb[r, b] > 0
                    cone_struct_cons += nrows_b[b]
                    cone_struct_nnz += nrows_b[b] * Nrb[r, b] + nzμ_b[b]
                    block_cons += ncols_b[b]
                    block_lam += nrows_b[b]
                    block_lam_nnz += nnzW_b[b]
                    block_coup_nnz += nh_b[b]
                elseif pos_b[b] > 0
                    collapse = true
                end
            end
            u_bounds += collapse
        end
    end
    block_cons += R                 # constant column of every row
    block_lam += 2R                 # the two rows ±e1 of W
    block_lam_nnz += 2R
    block_coup_nnz += 2R
    nzWMt = count(!iszero, c.W2 * c.M)
    nW = nW2 + 2
    nnzW = nnz(c.W)
    # u_r equality: u_r - Σ_{j in mask_r} μ_j S_rj = 0
    u_eq_nnz = R + mask_size
    cur_primal = (
        vars = nX + R * nW, cons = neq * dimξ + R * dimξ + R, bnds = R * nW,
        nnz = eq_nnz + nA_all + R * nnzW + R * count(!iszero, c.h),
    )
    blk_primal = (
        vars = nX + block_lam, cons = neq * dimξ + block_cons + R, bnds = block_lam,
        nnz = eq_nnz + nA_masked_cols(c, G) + block_lam_nnz + block_coup_nnz,
    )
    cur_dual = (
        vars = nX + cur_S_vars, cons = neq * dimξ + R * dimξ + R * nW2, bnds = 0,
        nnz = eq_nnz + eqS_nnz_cur + R * nzWMt,
        nnz_dense = eq_nnz + eqS_nnz_cur + R * nW2 * dimξ,
        cone_nnz = R * nzWMt, cone_nnz_dense = R * nW2 * dimξ, cone_cons = R * nW2,
    )
    str_dual = (
        vars = nX + mask_size + R, cons = neq * dimξ + mask_size + R + cone_struct_cons + 0,
        bnds = u_bounds,
        nnz = eq_nnz + (nA_masked + mask_size) + u_eq_nnz + cone_struct_nnz,
        cone_nnz = cone_struct_nnz, cone_cons = cone_struct_cons,
    )
    # dual with masked S but the dense cone (no u, no collapse): intermediate step
    mask_dual_cone = mask_size * nW2
    return (; cur_primal, blk_primal, cur_dual, str_dual, R, nW2, dimξ, nX, mask_size, mask_dual_cone, nblocks = nb)
end

# nnz of the A X parts restricted to the columns kept by the block primal: those are always
# within the columns where X exists, which are a subset of the mask, so it equals Σ nA.
nA_masked_cols(c, G) = sum(sum(group_mask(c, g)[2]) for g in G)

# ---------------------------------------------------------------------------------------
# Size of a JuMP model
# ---------------------------------------------------------------------------------------
function jump_size(model)
    b = JuMP.backend(model)
    cons = bnds = terms = nz = 0
    for (F, S) in MOI.get(b, MOI.ListOfConstraintTypesPresent())
        for ci in MOI.get(b, MOI.ListOfConstraintIndices{F,S}())
            if F == MOI.VariableIndex
                bnds += 1
            elseif F == MOI.ScalarAffineFunction{Float64}
                f = MOI.get(b, MOI.ConstraintFunction(), ci)
                cons += 1
                terms += length(f.terms)
                nz += count(t -> t.coefficient != 0, f.terms)
            else
                error("unexpected constraint type $F")
            end
        end
    end
    return (vars = num_variables(model), cons = cons, bnds = bnds, terms = terms, nnz = nz)
end

# ---------------------------------------------------------------------------------------
# Structured models (JuMP)
# ---------------------------------------------------------------------------------------
function rule_variables!(model, c::Canon)
    X = [AffExpr(0.0) for _ in 1:c.dimx, _ in 1:c.dimξ]
    for i in 1:c.dimx, j in 1:c.dimξ
        c.patX[i, j] == 1 && add_to_expression!(X[i, j], 1.0, @variable(model))
    end
    @constraint(model, c.ABC.Ae * X .== c.ABC.Be)
    return X
end

function set_objective!(model, c::Canon, X)
    @assert iszero(c.ABC.P) "the spike objective assumes P = 0 (hydro instance)"
    CM = c.ABC.C * c.M
    obj = AffExpr(c.r)
    for i in 1:c.dimx, j in 1:c.dimξ
        c.patX[i, j] == 1 && add_to_expression!(obj, CM[i, j], X[i, j])
    end
    @objective(model, c.sense, obj)
end

# A_r X_{·j} as an AffExpr
function AX!(expr, At, X, r, j)
    for p in nzrange(At, r)
        add_to_expression!(expr, nonzeros(At)[p], X[rowvals(At)[p], j])
    end
    return expr
end

# JuMP silently drops a zero coefficient; `DROPPED` counts the structural terms that this
# happens to (zeros of mu or of Sigma inside a declared block), so the formulas can be checked.
const DROPPED = Ref(0)
function addterm!(expr, coef, v)
    coef == 0 && (DROPPED[] += 1)
    return add_to_expression!(expr, coef, v)
end

function build_structured_dual(c::Canon, mode::Symbol)
    blk = block_of_columns(c, mode)
    brow = block_of_rows(c, blk)
    nb = maximum(blk)
    wμ = c.W2 * c.μ
    V = c.Σ[2:end, 2:end] * c.W2[:, 2:end]'     # (dimη × nW2)
    model = Model()
    X = rule_variables!(model, c)
    for g in groups(c)
        mask, _ = group_mask(c, g)
        At = sparse(g.A')
        Rg = size(mask, 1)
        for r in 1:Rg
            S = Dict{Int,VariableRef}()
            for j in 1:c.dimξ
                mask[r, j] || continue
                S[j] = @variable(model)
                expr = AX!(AffExpr(0.0), At, X, r, j)
                add_to_expression!(expr, g.s, S[j])
                @constraint(model, expr == g.B[r, j])
            end
            # per block: masked columns
            cols_b = [Int[] for _ in 1:nb]
            for j in sort!(collect(keys(S)))
                j >= 2 && push!(cols_b[blk[j]], j)
            end
            collapse = any(b -> isempty(cols_b[b]) && any(k -> brow[k] == b && wμ[k] > 1e-9, eachindex(brow)), 1:nb)
            u = @variable(model, lower_bound = collapse ? 0.0 : -Inf)
            eu = AffExpr(0.0)
            add_to_expression!(eu, 1.0, u)
            for (j, s) in S
                addterm!(eu, -c.μ[j], s)
            end
            @constraint(model, eu == 0)
            for b in 1:nb
                isempty(cols_b[b]) && continue
                for k in eachindex(brow)
                    brow[k] == b || continue
                    e = AffExpr(0.0)
                    abs(wμ[k]) > 1e-9 && add_to_expression!(e, wμ[k], u)
                    for j in cols_b[b]
                        addterm!(e, V[j-1, k], S[j])
                    end
                    @constraint(model, e >= 0)
                end
            end
        end
    end
    set_objective!(model, c, X)
    return model
end

function build_block_primal(c::Canon, mode::Symbol)
    blk = block_of_columns(c, mode)
    brow = block_of_rows(c, blk)
    nb = maximum(blk)
    W2c = sparse(c.W2[:, 2:end])
    W2ct = sparse(W2c')                       # columns = rows k of W'
    model = Model()
    X = rule_variables!(model, c)
    rows_b = [findall(==(b), brow) for b in 1:nb]
    cols_b = [findall(==(b), blk) for b in 1:nb]
    for g in groups(c)
        mask, _ = group_mask(c, g)
        At = sparse(g.A')
        for r in 1:size(mask, 1)
            touched = [b for b in 1:nb if any(mask[r, j] for j in cols_b[b])]
            cols = vcat([1], reduce(vcat, cols_b[touched]; init = Int[]))
            λ1 = @variable(model, lower_bound = 0)
            λ2 = @variable(model, lower_bound = 0)
            exprs = Dict(j => AX!(AffExpr(0.0), At, X, r, j) for j in cols)
            add_to_expression!(exprs[1], g.s, λ1)
            add_to_expression!(exprs[1], -g.s, λ2)
            coupling = AffExpr(0.0)
            add_to_expression!(coupling, 1.0, λ1)
            add_to_expression!(coupling, -1.0, λ2)
            for b in touched, k in rows_b[b]
                Λ = @variable(model, lower_bound = 0)
                for p in nzrange(W2ct, k)
                    add_to_expression!(exprs[rowvals(W2ct)[p]+1], g.s * nonzeros(W2ct)[p], Λ)
                end
                hk = c.h[k+2]
                hk != 0 && add_to_expression!(coupling, hk, Λ)
            end
            for j in cols
                @constraint(model, exprs[j] == g.B[r, j])
            end
            @constraint(model, coupling >= 0)
        end
    end
    set_objective!(model, c, X)
    return model
end

# ---------------------------------------------------------------------------------------
# Instances and drivers
# ---------------------------------------------------------------------------------------
const DATA = Ref{Any}(nothing)
data() = (DATA[] === nothing && (DATA[] = hydro_thermal_data()); DATA[])

solve_jump(model) = (set_optimizer(model, HiGHS.Optimizer); set_silent(model);
    optimize!(model); (termination_status(model), objective_value(model)))

function reference_solves(m)
    out = Dict{Symbol,Float64}()
    for mode in (:primal, :dual)
        set_attribute(m, LDR.SolvePrimal(), mode == :primal)
        set_attribute(m, LDR.SolveDual(), mode == :dual)
        optimize!(m)
        @assert termination_status(m; dual = mode == :dual) == MOI.OPTIMAL
        out[mode] = objective_value(m; dual = mode == :dual)
    end
    return out
end

function validate(cases = [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0)])
    println("== validation: objectives of the current vs structured models (HiGHS)")
    @printf("%-7s %-6s %-5s %-18s %-18s %-9s | %-18s %-18s %-9s\n", "T,k", "mode", "dimξ",
        "primal current", "primal block", "rel.diff", "dual current", "dual struct.", "rel.diff")
    worst = 0.0
    for (T, k) in cases, mode in (:stage, :dense)
        m = build_monolith(data(), T, k)
        set_attribute(m, LDR.SolvePrimal(), true)
        set_attribute(m, LDR.SolveDual(), true)
        ref = reference_solves(m)      # also calls _prepare_data
        c = canon(m)
        st_p, o_p = solve_jump(build_block_primal(c, mode))
        st_d, o_d = solve_jump(build_structured_dual(c, mode))
        rp = abs(o_p - ref[:primal]) / max(1, abs(ref[:primal]))
        rd = abs(o_d - ref[:dual]) / max(1, abs(ref[:dual]))
        worst = max(worst, rp, rd)
        @printf("%-7s %-6s %-5d %-18.10f %-18.10f %-9.2e | %-18.10f %-18.10f %-9.2e  [%s,%s]\n",
            "$T,$k", mode, c.dimξ, ref[:primal], o_p, rp, ref[:dual], o_d, rd, st_p, st_d)
    end
    @printf("max relative difference: %.3e\n", worst)
    return worst
end

fmt(x) = @sprintf("%d", x)

function build_current_counts(m)
    cur = Dict{Symbol,Any}()
    slack = JuMP.get_attribute(m, LDR.SlackAsExpr())
    LDR._build_primal_ldr(m; slack_as_expr = slack)
    cur[:primal] = jump_size(m.primal_model)
    LDR._build_dual_ldr(m)
    cur[:dual] = jump_size(m.dual_model)
    m.primal_model = JuMP.Model()
    m.dual_model = JuMP.Model()
    GC.gc()
    return cur
end

function check(cases = [(2, 0), (3, 1), (12, 0)])
    println("== check: formulas vs actual JuMP model sizes (vars / linear cons / var bounds / terms / nonzero terms)")
    for (T, k) in cases
        m = build_monolith(data(), T, k)
        set_attribute(m, LDR.SolvePrimal(), true)
        set_attribute(m, LDR.SolveDual(), true)
        c = canon(m)
        n = counts(c, :stage)
        t = @elapsed cur = build_current_counts(m)
        println("T=$T k=$k  (current models built in $(round(t, digits = 1)) s)")
        # `dropped`: structural terms with a zero coefficient, which JuMP does not store
        show_cmp(name, f, a, dropped = 0) = @printf("  %-22s formula: vars=%d cons=%d bnds=%d nnz=%d | JuMP: vars=%d cons=%d bnds=%d nnz=%d (+%d dropped zeros) %s\n",
            name, f.vars, f.cons, f.bnds, f.nnz, a.vars, a.cons, a.bnds, a.nnz, dropped,
            (f.vars, f.cons, f.bnds, f.nnz) == (a.vars, a.cons, a.bnds, a.nnz + dropped) ? "OK" : "MISMATCH")
        show_cmp("current primal", n.cur_primal, cur[:primal])
        show_cmp("current dual", n.cur_dual, cur[:dual])
        @printf("    (current dual: dense-term formula %d, JuMP terms %d)\n", n.cur_dual.nnz_dense, cur[:dual].terms)
        for mode in (:stage, :dense)
            nn = counts(c, mode)
            mb = build_block_primal(c, mode)
            show_cmp("block primal [$mode]", nn.blk_primal, jump_size(mb))
            DROPPED[] = 0
            ms = build_structured_dual(c, mode)
            show_cmp("struct. dual [$mode]", nn.str_dual, jump_size(ms), DROPPED[])
        end
    end
end

function sizes(; Ts = (12, 24), ks = 0:3, path = joinpath(@__DIR__, "sizes.csv"))
    header = ["T", "k", "blocks", "xi_dim", "ineq_rows", "support_rows",
        "cur_primal_vars", "cur_primal_cons", "cur_primal_nnz",
        "blk_primal_vars", "blk_primal_cons", "blk_primal_nnz",
        "cur_dual_vars", "cur_dual_cons", "cur_dual_nnz", "cur_dual_cone_nnz",
        "str_dual_vars", "str_dual_cons", "str_dual_nnz", "str_dual_cone_nnz",
        "mask_only_dual_cone_nnz", "primal_nnz_ratio", "dual_nnz_ratio", "dual_cone_nnz_ratio"]
    rows = Vector{Any}[]
    for T in Ts, k in ks
        t = @elapsed (m = build_monolith(data(), T, k); set_attribute(m, LDR.SolvePrimal(), true);
            set_attribute(m, LDR.SolveDual(), true); c = canon(m))
        for mode in (:stage, :dense)
            n = counts(c, mode)
            row = Any[T, k, mode, c.dimξ, n.R, n.nW2 + 2,
                n.cur_primal.vars, n.cur_primal.cons, n.cur_primal.nnz,
                n.blk_primal.vars, n.blk_primal.cons, n.blk_primal.nnz,
                n.cur_dual.vars, n.cur_dual.cons, n.cur_dual.nnz, n.cur_dual.cone_nnz,
                n.str_dual.vars, n.str_dual.cons, n.str_dual.nnz, n.str_dual.cone_nnz,
                n.mask_dual_cone,
                round(n.cur_primal.nnz / n.blk_primal.nnz, digits = 2),
                round(n.cur_dual.nnz / n.str_dual.nnz, digits = 2),
                round(n.cur_dual.cone_nnz / n.str_dual.cone_nnz, digits = 2)]
            push!(rows, row)
            @printf("T=%-3d k=%d %-6s ξ=%-4d rows=%-5d | primal nnz %.3e -> %.3e (x%.1f) | dual nnz %.3e -> %.3e (x%.1f), cone %.3e -> %.3e (x%.1f) [%.1fs]\n",
                T, k, mode, c.dimξ, n.R, n.cur_primal.nnz, n.blk_primal.nnz, n.cur_primal.nnz / n.blk_primal.nnz,
                n.cur_dual.nnz, n.str_dual.nnz, n.cur_dual.nnz / n.str_dual.nnz,
                n.cur_dual.cone_nnz, n.str_dual.cone_nnz, n.cur_dual.cone_nnz / n.str_dual.cone_nnz, t)
        end
    end
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(row, ","))
        end
    end
    println("wrote $path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    what = isempty(ARGS) ? "all" : ARGS[1]
    what in ("validate", "all") && validate()
    what in ("check", "all") && check()
    what in ("sizes", "all") && sizes()
end
