# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The models of the stage-order tests: a five-stage inventory whose uncertainties are declared out
# of stage order (a law over stages 2 to 5, then scalars at stages 4, 3 and 2, then an unmarked one),
# with what the tests expect computed here from the declarations, independently of the package:
# the stage of every variable, the moments of every law, the support and the breakpoints; and the
# checks of a canonical form against them.

module StageOrderModels

using JuMP
using Test

import Distributions
import LinearAlgebra
import LinearDecisionRules:
    BreakPoints,
    CanonicalForm,
    ConfidenceMvNormal,
    InformationStructure,
    LDRModel,
    Moments,
    MvDiscreteNonParametric,
    Stage,
    Truncate,
    Uncertainty,
    ValidInequality,
    _cone_rows,
    _cone_violation,
    lift,
    observes,
    retraction,
    second_moment,
    set_stage,
    support_blocks,
    visible_columns
import MathOptInterface as MOI

# One declaration, with its moments computed here, and its atoms if it has finitely many.
struct Law
    variables::Vector{VariableRef}
    distribution::Distributions.Distribution
    atoms::Union{Nothing,Vector{Vector{Float64}}}
    probabilities::Vector{Float64}
    mean::Vector{Float64}
    covariance::Matrix{Float64}
end

function atom_moments(atoms, p)
    mean = sum(q * a for (a, q) in zip(atoms, p))
    covariance = sum(q * (a - mean) * (a - mean)' for (a, q) in zip(atoms, p))
    return mean, covariance
end

function discrete_law(variables, atoms, p, distribution)
    mean, covariance = atom_moments(atoms, p)
    return Law(variables, distribution, atoms, p, mean, covariance)
end

function scalar_law(x, d::Distributions.Uniform)
    a, b = d.a, d.b
    return Law([x], d, nothing, Float64[], [(a + b) / 2], fill((b - a)^2 / 12, 1, 1))
end

function scalar_law(x, d::Distributions.DiscreteNonParametric)
    values, p = Distributions.support(d), Distributions.probs(d)
    return discrete_law([x], [[v] for v in values], p, d)
end

# A discrete law on `values` if `discrete`, else the uniform law on their range.
function scalar_distribution(discrete::Bool, values, p)
    if discrete
        return Distributions.DiscreteNonParametric(values, p)
    end
    return Distributions.Uniform(minimum(values), maximum(values))
end

"""
    Staged

A model with what the tests expect of it:

 * `stages`: the stage of every variable, from the declarations; unmarked ones in the last stage.
 * `laws`: the declarations, with their moments.
 * `merges`: the groups of declarations (indices in `laws`) coupled by rows tagged `Truncate()`,
   with the predicate of the conditioning set, over a `Dict` of values.
 * `support`: predicates that the scenarios of the support satisfy, beyond the boxes of the laws.
 * `breakpoints`: the breakpoints set on each uncertainty.
"""
mutable struct Staged
    model::Model
    stages::Dict{VariableRef,Int}
    laws::Vector{Law}
    merges::Vector{Tuple{Vector{Int},Function}}
    support::Vector{Function}
    breakpoints::Dict{VariableRef,Vector{Float64}}
end

# A random walk over stages 2 to 5, so that it is correlated with linear conditional means.
function walk(variables)
    atoms, p = Vector{Float64}[], Float64[]
    for a in (8.0, 16.0), b in (-2.0, 4.0), c in (-3.0, 3.0), e in (-1.0, 5.0)
        push!(atoms, [a, a + b, a + b + c, a + b + c + e])
        push!(p, 1 / 16)
    end
    return discrete_law(variables, atoms, p, MvDiscreteNonParametric(atoms, p))
end

function product(variables)
    bounds = [(8.0, 16.0), (6.0, 20.0), (3.0, 23.0), (2.0, 28.0)]
    d = Distributions.product_distribution([Distributions.Uniform(a, b) for (a, b) in bounds])
    mean = [(a + b) / 2 for (a, b) in bounds]
    covariance = Matrix(LinearAlgebra.Diagonal([(b - a)^2 / 12 for (a, b) in bounds]))
    return Law(variables, d, nothing, Float64[], mean, covariance)
end

function confidence(variables)
    μ = [12.0, 13.0, 13.0, 15.0]
    Σ = [4.0 * 0.6^abs(i - j) for i in 1:4, j in 1:4]
    α = 0.9
    d = ConfidenceMvNormal(μ, Σ, α)
    # The covariance of the docstring of `ConfidenceMvNormal`.
    ρ² = Distributions.quantile(Distributions.Chisq(4), α)
    scaling = Distributions.cdf(Distributions.Chisq(6), ρ²) / α
    return Law(variables, d, nothing, Float64[], μ, scaling * Σ)
end

"""
    inventory(;
        law = :walk,
        discrete = false,
        valid = false,
        conic = false,
        truncate = :none,
        quadratic = false,
        optimizer,
    ) -> Staged

The five-stage inventory. Declared in this order: `w[t = 2:5]` with the multivariate `law` (`:walk`,
`:product` or `:confidence`) in `Stage(t)`; `v` in `Stage(4)`; `u3` in `Stage(3)`; `u2` in
`Stage(2)`; `z` unmarked (so in `Stage(5)`). The scalars `u3`, `u2` and `z` have discrete laws if
`discrete`. The decisions `buy[t]` and `stock[t]` are in `Stage(t)` for `t = 1:5`, `sell[t]` for
`t = 2:5`, and `spill` is unmarked; `sell[t]` is bounded by the demand of stage `t`, `w[t]` plus
`u2`, `u3`, `v` or `z`.

Rows of Ξ across stages (the boxes of the laws imply the valid ones, which couple support blocks):
`valid` adds `u2 + v ≤ 11` tagged `ValidInequality()` (stages 2 and 4); `conic` adds the disc
`‖(u2 - 4, v - 3)‖ ≤ 2.85` tagged `ValidInequality()`; `truncate` is `:none`,
`:scalars` (`u3 + z ≤ 4`, stages 3 and 5) or `:across` (`w[2] - z ≥ 7`, stages 2 and 5, which merges
the law of `w` with that of `z`, and `v ≥ 1.5`, which truncates `v` exactly). `quadratic` adds
quadratic terms to the objective.
"""
function inventory(;
    law::Symbol = :walk,
    discrete::Bool = false,
    valid::Bool = false,
    conic::Bool = false,
    truncate::Symbol = :none,
    quadratic::Bool = false,
    optimizer,
)
    model = LDRModel(optimizer)
    set_silent(model)
    laws = Law[]
    # The declarations, out of stage order.
    w_law = Dict(:walk => walk, :product => product, :confidence => confidence)[law]
    w_dist = w_law(VariableRef[]).distribution
    @variable(model, w[t = 2:5] in Uncertainty(w_dist), Stage(t))
    push!(laws, w_law(collect(w)))
    @variable(model, v in Uncertainty(Distributions.Uniform(1, 5)), Stage(4))
    push!(laws, scalar_law(v, Distributions.Uniform(1, 5)))
    u3_dist = scalar_distribution(discrete, [0.0, 1.0, 3.0, 4.0], [0.1, 0.4, 0.3, 0.2])
    @variable(model, u3 in Uncertainty(u3_dist), Stage(3))
    push!(laws, scalar_law(u3, u3_dist))
    u2_dist = scalar_distribution(discrete, [2.0, 4.0, 6.0], [0.25, 0.5, 0.25])
    @variable(model, u2 in Uncertainty(u2_dist), Stage(2))
    push!(laws, scalar_law(u2, u2_dist))
    z_dist = scalar_distribution(discrete, [0.0, 1.0, 2.0], [0.3, 0.3, 0.4])
    @variable(model, z in Uncertainty(z_dist))
    push!(laws, scalar_law(z, z_dist))
    @variable(model, buy[t = 1:5] >= 0, Stage(t))
    @variable(model, stock[t = 1:5] >= 0, Stage(t))
    @variable(model, sell[t = 2:5] >= 0, Stage(t))
    @variable(model, spill >= 0)
    extra = Dict(2 => u2, 3 => u3, 4 => v, 5 => z)
    @constraint(model, stock[1] == buy[1])
    @constraint(model, [t = 2:4], stock[t] == stock[t-1] + buy[t] - sell[t])
    @constraint(model, stock[5] + spill == stock[4] + buy[5] - sell[5])
    @constraint(model, demand[t = 2:5], sell[t] <= w[t] + extra[t])
    @constraint(model, [t = 1:5], buy[t] <= 12)
    @objective(
        model,
        Max,
        sum(10sell[t] for t in 2:5) - sum((1 + t) * buy[t] for t in 1:5) - 0.5sum(stock) - spill,
    )
    if quadratic
        # Every term of the objective: `x'Px`, `x'Cξ` across the columns of the law of `w`, and
        # `η'Qη + d'η` on the scalars of stages 2 and 4.
        quadratic_terms = sum(-0.05sell[t]^2 + 0.1w[t] * sell[t] for t in 2:5) + 0.01v * u2 + 0.1v
        set_objective_function(model, objective_function(model) + quadratic_terms)
    end
    stages = Dict{VariableRef,Int}(v => 4, u3 => 3, u2 => 2, z => 5, spill => 5)
    for t in 2:5
        stages[w[t]] = t
        stages[sell[t]] = t
    end
    for t in 1:5
        stages[buy[t]] = stages[stock[t]] = t
    end
    staged = Staged(model, stages, laws, [], Function[], Dict())
    if valid
        @constraint(model, u2 + v <= 11, ValidInequality())
        push!(staged.support, s -> s[u2] + s[v] <= 11 + 1e-9)
    end
    if conic
        @constraint(model, [2.85, u2 - 4, v - 3] in SecondOrderCone(), ValidInequality())
        push!(staged.support, s -> hypot(s[u2] - 4, s[v] - 3) <= 2.85 + 1e-9)
    end
    if truncate == :scalars
        @constraint(model, u3 + z <= 4, Truncate())
        push!(staged.merges, ([3, 5], s -> s[u3] + s[z] <= 4 + 1e-9))
        push!(staged.support, s -> s[u3] + s[z] <= 4 + 1e-9)
    elseif truncate == :across
        @constraint(model, w[2] - z >= 7, Truncate())
        push!(staged.merges, ([1, 5], s -> s[w[2]] - s[z] >= 7 - 1e-9))
        push!(staged.support, s -> s[w[2]] - s[z] >= 7 - 1e-9)
        @constraint(model, v >= 1.5, Truncate())
        # `v` is truncated exactly: a uniform law on [1.5, 5].
        staged.laws[2] = scalar_law(v, Distributions.Uniform(1.5, 5))
    end
    return staged
end

"""
    set_breakpoints!(s::Staged, x, points)

Set the breakpoints of `x` (a vector, or `nothing` to clear them) on the model and in `s`.
"""
function set_breakpoints!(s::Staged, x::VariableRef, points)
    set_attribute(x, BreakPoints(), points)
    if points === nothing
        delete!(s.breakpoints, x)
    else
        s.breakpoints[x] = sort(Vector{Float64}(points))
    end
    return
end

"""
    change_stage!(s::Staged, x, t)

`set_stage(x, t)` on the model, and in the expected stages of `s`.
"""
function change_stage!(s::Staged, x::VariableRef, t::Int)
    set_stage(x, t)
    s.stages[x] = t
    return
end

"""
    randomize!(s::Staged, rng)

Give every variable but the unmarked `z` and `spill` a random stage in `1:5`, so that these two
are in the last stage, and breakpoints to a random subset of the scalar uncertainties `v` and
`u2`, which no truncation conditions.
"""
function randomize!(s::Staged, rng)
    model = s.model
    unmarked = (model[:z], model[:spill])
    for x in sort!(collect(keys(s.stages)); by = x -> index(x).value)
        x in unmarked || change_stage!(s, x, rand(rng, 1:5))
    end
    T = max(2, maximum(t for (x, t) in s.stages if !(x in unmarked)))
    for x in unmarked
        s.stages[x] = T
    end
    for (x, points) in ((model[:v], [2.0, 3.5]), (model[:u2], [3.0]))
        set_breakpoints!(s, x, rand(rng, Bool) ? points : nothing)
    end
    return
end

"""
    delete_uncertainty!(s::Staged, x)

Delete the scalar uncertainty `x` (declared alone and in no row of Ξ) from the model and from `s`.
"""
function delete_uncertainty!(s::Staged, x::VariableRef)
    k = only(k for (k, law) in enumerate(s.laws) if x in law.variables)
    @assert isempty(s.merges) && isempty(s.support)
    delete(s.model, x)
    deleteat!(s.laws, k)
    delete!(s.stages, x)
    delete!(s.breakpoints, x)
    return
end

# The joint law of the declarations `group` conditioned on `predicate`, by enumerating the atoms of
# their product.
function conditioned(laws::Vector{Law}, predicate)
    variables = reduce(vcat, [law.variables for law in laws])
    atoms, p = Vector{Float64}[], Float64[]
    for combination in Iterators.product((eachindex(law.atoms) for law in laws)...)
        atom = reduce(vcat, [law.atoms[k] for (law, k) in zip(laws, combination)])
        if predicate(Dict(zip(variables, atom)))
            push!(atoms, atom)
            push!(p, prod(law.probabilities[k] for (law, k) in zip(laws, combination)))
        end
    end
    p ./= sum(p)
    mean, covariance = atom_moments(atoms, p)
    return variables, mean, covariance
end

"""
    direct_second_moment(s::Staged, order::Vector{VariableRef}) -> (M, known)

The second moment `E[ξ ξᵀ]` of `ξ = (1, η)` with `η` the uncertainties `order`, from the laws of
`s`, and whether each entry is known: those of a law conditioned by rejection sampling are not.
"""
function direct_second_moment(s::Staged, order::Vector{VariableRef})
    position = Dict(x => 1 + k for (k, x) in enumerate(order))
    m = 1 + length(order)
    μ, Σ = zeros(m), zeros(m, m)
    μ[1] = 1.0
    known = trues(m, m)
    merged = Set(k for (group, _) in s.merges for k in group)
    alone = s.laws[setdiff(eachindex(s.laws), merged)]
    blocks = [(law.variables, law.mean, law.covariance) for law in alone]
    for (group, predicate) in s.merges
        laws = s.laws[group]
        if all(law -> law.atoms !== nothing, laws)
            push!(blocks, conditioned(laws, predicate))
        else
            for law in laws, x in law.variables
                known[position[x], :] .= false
                known[:, position[x]] .= false
            end
        end
    end
    for (variables, mean, covariance) in blocks
        index = [position[x] for x in variables]
        μ[index] = mean
        Σ[index, index] = covariance
    end
    return μ * μ' + Σ, known
end

"""
    draw_scenario(rng, s::Staged) -> Dict{VariableRef,Float64}

A scenario in the support of the uncertainties of `s`: an atom of each discrete law, a draw of each
continuous one, by rejection on the rows of Ξ.
"""
function draw_scenario(rng, s::Staged)
    while true
        scenario = Dict{VariableRef,Float64}()
        for law in s.laws
            values = if law.atoms === nothing
                rand(rng, law.distribution)
            else
                law.atoms[rand(rng, 1:length(law.atoms))]
            end
            for (x, value) in zip(law.variables, values)
                scenario[x] = value
            end
        end
        all(predicate -> predicate(scenario), s.support) && return scenario
    end
end

"""
    pieces(points, value) -> Vector{Float64}

The coordinates of the `AxisPiecewise` lifting of `value` with the interior breakpoints `points`.
"""
function pieces(points::Vector{Float64}, value::Float64)
    n = length(points)
    out = zeros(n + 1)
    out[1] = min(value, points[1])
    for k in 2:n
        out[k] = clamp(value - points[k-1], 0.0, points[k] - points[k-1])
    end
    out[n+1] = max(value - points[n], 0.0)
    return out
end

# The checks, shared by the tests of the canonical form and of the JuMP models.

ref(s::Staged, v::MOI.VariableIndex) = VariableRef(s.model, v)

refs(s::Staged, vs) = [ref(s, v) for v in vs]

# The uncertainties of the columns of the original ξ: those of `form` if it is not lifted.
function original_uncertainties(form::CanonicalForm)
    return form.lifting === nothing ? form.uncertainties : form.lifting.uncertainties
end

"""
    by_hand(s::Staged, labels, scenario) -> Vector{Float64}

The lifted outcome of `scenario` in columns labelled as in a `RuleMatrix` (`:constant`, an
uncertainty, or `(η, k)` for the `k`-th piece of `η`), from the breakpoints of `s`.
"""
function by_hand(s::Staged, labels::Vector, scenario)
    return map(labels) do label
        if label == :constant
            return 1.0
        elseif label isa Tuple
            x, k = label
            return pieces(s.breakpoints[x], scenario[x])[k]
        end
        return scenario[label]
    end
end

# The labels of the columns of `form`, from its uncertainties and the breakpoints of `s`.
function labels(s::Staged, form::CanonicalForm)
    out = Any[:constant]
    seen = Dict{VariableRef,Int}()
    for x in refs(s, form.uncertainties)
        if haskey(s.breakpoints, x)
            seen[x] = get(seen, x, 0) + 1
            push!(out, (x, seen[x]))
        else
            push!(out, x)
        end
    end
    return out
end

"""
    check_columns(s::Staged, form::CanonicalForm)

The columns are sorted by stage, with the stages of the declarations, the laws are in the columns
of their variables, and every decision observes exactly the columns of its stage or earlier.
"""
function check_columns(s::Staged, form::CanonicalForm)
    m = length(form.uncertainties)
    stage(v) = s.stages[ref(s, v)]
    @test issorted(form.uncertainty_stages)
    @test form.uncertainty_stages == stage.(form.uncertainties)
    @test form.decision_stages == stage.(form.decisions)
    @test form.num_stages == max(2, maximum(values(s.stages)))
    @test Set(refs(s, form.uncertainties)) == Set(x for law in s.laws for x in law.variables)
    if form.lifting !== nothing
        @test issorted(form.lifting.uncertainty_stages)
        @test form.lifting.uncertainty_stages == stage.(form.lifting.uncertainties)
    end
    @test sort!(reduce(vcat, [law.columns for law in form.laws])) == 2:1+m
    for law in form.laws
        @test law.stages == form.uncertainty_stages[law.columns.-1]
    end
    # A declaration neither merged nor lifted is a law of the form, with its components in their
    # columns, in the order of the declaration.
    merged = Set(k for (group, _) in s.merges for k in group)
    column = Dict(x => 1 + j for (j, x) in enumerate(refs(s, form.uncertainties)))
    for (k, law) in enumerate(s.laws)
        (k in merged || any(x -> haskey(s.breakpoints, x), law.variables)) && continue
        columns = [column[x] for x in law.variables]
        @test count(l -> l.columns == columns, form.laws) == 1
    end
    info = InformationStructure(form)
    for (i, x) in enumerate(form.decisions)
        expected = [1; [1 + j for (j, v) in enumerate(form.uncertainties) if stage(v) <= stage(x)]]
        @test collect(visible_columns(info, i)) == expected
        @test [observes(info, i, j) for j in 1:1+m] == [j in expected for j in 1:1+m]
    end
    return
end

"""
    check_moments(s::Staged, form::CanonicalForm, mom::Moments)

The second moment of the columns of `form`, mapped back to the original uncertainties by the
retraction, is that of the laws in the order of the declarations, permuted. Laws conditioned by
rejection sampling are skipped.
"""
function check_moments(s::Staged, form::CanonicalForm, mom::Moments)
    M = second_moment(mom)
    if form.lifting !== nothing
        R = Matrix(retraction(form.lifting))
        M = R * M * R'
    end
    D, known = direct_second_moment(s, refs(s, original_uncertainties(form)))
    @test isapprox(M[known], D[known]; rtol = 1e-9)
    return
end

"""
    check_support(s::Staged, form::CanonicalForm, scenarios)

The support blocks partition the columns, and their rows hold on the `scenarios` of the support,
lifted.
"""
function check_support(s::Staged, form::CanonicalForm, scenarios)
    blocks = support_blocks(form)
    @test sort!(reduce(vcat, [b.columns for b in blocks])) == 2:1+length(form.uncertainties)
    columns = labels(s, form)
    violation = 0.0
    for scenario in scenarios, block in blocks
        v = block.W * by_hand(s, columns, scenario)[block.columns] - block.h
        for (cone, rows) in zip(block.cones, _cone_rows(block))
            violation = max(violation, _cone_violation(cone, v[rows]))
        end
    end
    @test violation <= 1e-9
    return
end

"""
    check_lifting(s::Staged, base::CanonicalForm, form::CanonicalForm, scenarios)

The lifting of `base` into `form`: `R ∘ L = id` on the `scenarios` in the order of the columns
of `base`, the lifted coordinates in the columns of their uncertainty, and the data of `base`
mapped by the retraction.
"""
function check_lifting(s::Staged, base::CanonicalForm, form::CanonicalForm, scenarios)
    map = form.lifting
    @test map.uncertainties == base.uncertainties
    @test map.uncertainty_stages == base.uncertainty_stages
    original = refs(s, base.uncertainties)
    Ξ = reduce(hcat, [[1.0; [scenario[x] for x in original]] for scenario in scenarios])
    Ξ̃ = lift(map, Ξ)
    columns = labels(s, form)
    @test Ξ̃ ≈ reduce(hcat, [by_hand(s, columns, scenario) for scenario in scenarios])
    @test retraction(map) * Ξ̃ ≈ Ξ
    @test form.B_u * Ξ̃ ≈ base.B_u * Ξ
    @test form.B_l * Ξ̃ ≈ base.B_l * Ξ
    @test form.B_e * Ξ̃ ≈ base.B_e * Ξ
    @test form.C * Ξ̃ ≈ base.C * Ξ
    @test form.d' * Ξ̃[2:end, :] ≈ base.d' * Ξ[2:end, :]
    rows = 1:length(base.h_valid)
    @test form.W_valid[rows, :] * Ξ̃[2:end, :] ≈ base.W_valid * Ξ[2:end, :]
    for (lifted, row) in zip(form.conic_rows, base.conic_rows)
        @test lifted.W * Ξ̃[2:end, :] ≈ row.W * Ξ[2:end, :]
    end
    for block in map.blocks
        x = base.uncertainties[only(block.inputs)-1]
        @test form.uncertainties[block.columns.-1] == fill(x, length(block.columns))
        @test haskey(s.breakpoints, ref(s, x))
        name = base.uncertainty_names[only(block.inputs)-1]
        @test form.uncertainty_names[block.columns.-1] ==
              ["$name[piece $k]" for k in eachindex(block.columns)]
    end
    for (j, k) in enumerate(map.kept)
        j > 1 && k > 0 && @test form.uncertainties[k-1] == base.uncertainties[j-1]
    end
    return
end

end # module StageOrderModels
