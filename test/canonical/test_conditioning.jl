# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The laws conditioned by the rows tagged `Truncate()`, in the canonical form.

module TestConditioning

using JuMP
using Test

import Distributions
import Distributions: Uniform
import LinearAlgebra
import LinearDecisionRules as LDR
import LinearDecisionRules:
    AxisPiecewise,
    ConditionedDistribution,
    ModelInfo,
    UncertaintyDeclaration,
    canonicalize,
    lift,
    moments,
    support_blocks
import Random

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$name" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function error_message(f)
    try
        f()
    catch err
        return sprint(showerror, err)
    end
    return ""
end

# A model with a first-stage decision `x`, and uncertainties declared by `(name, law)`.
function model_with(laws::Pair...)
    model = Model()
    @variable(model, x)
    decls = map(laws) do (name, law)
        n = LDR._dimension(law)
        v =
            n == 1 ? @variable(model, base_name = name) : @variable(model, [1:n], base_name = name)
        model[Symbol(name)] = v
        return UncertaintyDeclaration(law, n == 1 ? [index(v)] : index.(v))
    end
    return model, collect(decls)
end

function form_of(model, decls; truncate = [], valid = [], breakpoints = Dict())
    info = ModelInfo(;
        uncertainties = decls,
        decision_stages = Dict(index(model[:x]) => 1),
        truncate_rows = Set{MOI.ConstraintIndex}(index.(truncate)),
        valid_rows = Set{MOI.ConstraintIndex}(index.(valid)),
        breakpoints = Dict{MOI.VariableIndex,Union{Int,Vector{Float64}}}(breakpoints),
    )
    return canonicalize(backend(model), info)
end

# The old moment check: the mean of Uniform(80, 120) given 100 ≤ d ≤ 110 is 105.
function test_exact_truncation()
    model, decls = model_with("d" => Uniform(80, 120))
    d = model[:d]
    rows = [@constraint(model, d >= 100), @constraint(model, 2d <= 220)]
    form = form_of(model, decls; truncate = rows)
    @test only(form.laws).distribution == Uniform(100, 110)
    # The box of the truncated law replaces the rows.
    @test isempty(form.h_valid) && isempty(form.conditioning)
    block = only(support_blocks(form))
    @test block.roles == [:box, :box] && block.h == [100, -110]
    mom = moments(form)
    @test mom.mean == [1, 105]
    @test only(mom.blocks).covariance ≈ [100 / 12;;]
    @test only(mom.blocks).num_samples == 0
    # An interval row is two rows; a bound beyond the support does not truncate.
    model, decls = model_with("d" => Uniform(80, 120))
    row = @constraint(model, 70 <= model[:d] <= 110)
    @test only(form_of(model, decls; truncate = [row]).laws).distribution == Uniform(80, 110)
    return
end

function test_exact_truncation_normal()
    law = Distributions.truncated(Distributions.Normal(100, 10), 60, 140)
    model, decls = model_with("d" => law)
    row = @constraint(model, model[:d] >= 105)
    form = form_of(model, decls; truncate = [row])
    exact = Distributions.truncated(Distributions.Normal(100, 10), 105, 140)
    mom = moments(form)
    @test mom.mean[2] ≈ Distributions.mean(exact)
    @test only(mom.blocks).covariance[1] ≈ Distributions.var(exact)
    return
end

# Distributions has no moments for a truncated exponential law: they are by quadrature.
function test_exact_truncation_by_quadrature()
    law = Distributions.truncated(Distributions.Exponential(1), 0, 5)
    model, decls = model_with("d" => law)
    row = @constraint(model, 0.5 <= model[:d] <= 2)
    mom = moments(form_of(model, decls; truncate = [row]))
    a, b = 0.5, 2.0
    p = exp(-a) - exp(-b)
    mean = ((a + 1) * exp(-a) - (b + 1) * exp(-b)) / p
    second = ((a^2 + 2a + 2) * exp(-a) - (b^2 + 2b + 2) * exp(-b)) / p
    @test mom.mean[2] ≈ mean rtol = 1e-10
    @test only(mom.blocks).covariance[1] ≈ second - mean^2 rtol = 1e-8
    return
end

# A discrete law keeps its atoms in the interval, with the atoms on the boundary.
function test_exact_truncation_discrete()
    law = Distributions.DiscreteNonParametric([80.0, 90, 100, 110, 120], [0.1, 0.2, 0.4, 0.2, 0.1])
    model, decls = model_with("d" => law)
    rows = [@constraint(model, 0.1 * model[:d] <= 11), @constraint(model, model[:d] >= 90)]
    form = form_of(model, decls; truncate = rows)
    truncated = only(form.laws).distribution
    @test truncated isa Distributions.DiscreteNonParametric
    @test Distributions.support(truncated) == [90, 100, 110]
    @test Distributions.probs(truncated) ≈ [0.25, 0.5, 0.25]
    @test moments(form).mean ≈ [1, 100]
    # An integer law with a finite support, too.
    model, decls = model_with("d" => Distributions.DiscreteUniform(1, 10))
    row = @constraint(model, 2.5 <= model[:d] <= 4)
    truncated = only(form_of(model, decls; truncate = [row]).laws).distribution
    @test Distributions.support(truncated) == [3, 4]
    return
end

# A discrete law without a finite support is truncated by rejection sampling.
function test_infinite_discrete_by_rejection()
    model, decls = model_with("d" => Distributions.Poisson(3))
    row = @constraint(model, 1 <= model[:d] <= 5)
    form = form_of(model, decls; truncate = [row])
    d = only(form.laws).distribution
    @test d isa ConditionedDistribution && d.laws == [Distributions.Poisson(3)]
    @test form.conditioning == [true, true]
    p = [Distributions.pdf(Distributions.Poisson(3), k) for k in 1:5]
    mean = sum((1:5) .* p) / sum(p)
    sd = sqrt(sum(((1:5) .- mean) .^ 2 .* p) / sum(p))
    mom = moments(form)
    @test abs(mom.mean[2] - mean) < 5 * sd / sqrt(10_000)
    return
end

# A multivariate discrete law conditioned by a row on its components keeps its atoms that satisfy
# the row, with their probabilities renormalized: its moments are exact, and the row stays in Ξ.
function test_exact_conditioning_mv_discrete()
    scenarios = [[60.0, 60], [60, 80], [80, 60], [80, 80], [100, 60], [60, 100], [100, 100]]
    p = [0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2]
    law = LDR.MvDiscreteNonParametric(scenarios, p)
    model, decls = model_with("d" => law)
    d = model[:d]
    form = form_of(model, decls; truncate = [@constraint(model, d[1] + d[2] <= 160)])
    exact = only(form.laws).distribution
    @test exact isa LDR.MvDiscreteNonParametric
    @test exact.support == scenarios[1:6]
    @test exact.p ≈ p[1:6] ./ 0.8
    @test form.conditioning == [true]
    @test :truncate in only(support_blocks(form)).roles
    mean = sum(p[1:6] .* scenarios[1:6]) / 0.8
    covariance = sum(q * (x - mean) * (x - mean)' for (q, x) in zip(p[1:6], scenarios[1:6])) / 0.8
    mom = moments(form)
    @test mom.mean[2:3] ≈ mean
    @test only(mom.blocks).covariance ≈ covariance
    @test only(mom.blocks).num_samples == 0
    # Rejection sampling draws from the same law: its estimates agree within the Monte Carlo error.
    rejection = ConditionedDistribution([law], [-1.0 -1.0], [-160.0])
    n = 10_000
    draws = rand(Random.Xoshiro(1), rejection, n)
    sample_mean = vec(sum(draws; dims = 2)) ./ n
    standard_error = sqrt.(LinearAlgebra.diag(covariance) ./ n)
    @test all(abs.(sample_mean .- mean) .<= 4 .* standard_error)
    centered = draws .- sample_mean
    @test centered * centered' ./ n ≈ covariance rtol = 0.05
    return
end

# Coupled discrete laws with finitely many atoms are conditioned exactly too, as are declared
# conditioned laws of such laws; no atom in the set is an error that names the constraints.
function test_exact_conditioning_coupled_discrete()
    a = Distributions.DiscreteNonParametric([0.0, 1.0, 2.0], [0.2, 0.3, 0.5])
    b = Distributions.Categorical([0.5, 0.5])
    model, decls = model_with("a" => a, "b" => b)
    row = @constraint(model, model[:a] + model[:b] <= 2)
    form = form_of(model, decls; truncate = [row])
    exact = only(form.laws).distribution
    @test exact isa LDR.MvDiscreteNonParametric
    @test exact.support == [[0.0, 1.0], [1.0, 1.0], [0.0, 2.0]]
    @test exact.p ≈ [0.1, 0.15, 0.1] ./ 0.35
    @test only(moments(form).blocks).num_samples == 0
    declared = ConditionedDistribution([a, b], [-1.0 -1.0], [-2.0])
    model, decls = model_with("c" => declared)
    @test only(form_of(model, decls).laws).distribution == exact
    model, decls = model_with("a" => a, "b" => b)
    @constraint(model, impossible, model[:a] + model[:b] >= 5)
    message = error_message(() -> form_of(model, decls; truncate = [impossible]))
    @test occursin("The constraints impossible tagged `Truncate()`", message)
    @test occursin("uncertainties a, b on a set that contains none of its atoms", message)
    # Beyond a million atoms in the product, the law is estimated by rejection sampling.
    laws = ["e$i" => Distributions.DiscreteUniform(0, 100) for i in 1:3]
    model, decls = model_with(laws...)
    row = @constraint(model, model[:e1] + model[:e2] + model[:e3] <= 150)
    @test only(form_of(model, decls; truncate = [row]).laws).distribution isa
          ConditionedDistribution
    return
end

# Coupled declarations share one conditioned law, in the place of the first one.
function test_merging()
    pair = Distributions.product_distribution([Uniform(0, 1), Uniform(0, 2)])
    model, decls = model_with(
        "a" => Uniform(0, 1),
        "e" => Uniform(0, 4),
        "b" => pair,
        "c" => Uniform(0, 3),
        "f" => Uniform(0, 5),
    )
    a, b, c, e, f = model[:a], model[:b], model[:c], model[:e], model[:f]
    ab = @constraint(model, ab, a + b[1] <= 1.5)
    bc = @constraint(model, bc, 0.5 <= b[2] - c <= 1)
    ee = @constraint(model, ee, e <= 2)
    fc = @constraint(model, valid, f + c <= 7)
    form = form_of(model, decls; truncate = [ab, bc, ee], valid = [fc])
    @test length(form.laws) == 3
    merged, exact, alone = form.laws
    d = merged.distribution
    @test d isa ConditionedDistribution
    @test d.laws == [Uniform(0, 1), pair, Uniform(0, 3)]
    # The columns of a, b[1], b[2], c, in the order of the components.
    @test merged.columns == [2, 4, 5, 6]
    @test exact.distribution == Uniform(0, 2) && exact.columns == [3]
    @test alone.distribution == Uniform(0, 5)
    rows = sort(collect(zip(eachrow(d.W), d.h, d.names)); by = r -> (r[3], r[2]))
    @test [(collect(w), h, name) for (w, h, name) in rows] == [
        ([-1.0, -1.0, 0, 0], -1.5, "ab"),
        ([0, 0, -1.0, 1.0], -1.0, "bc"),
        ([0, 0, 1.0, -1.0], 0.5, "bc"),
    ]
    # The rows of e are dropped; the conditioning rows bound Ξ, with the valid row.
    @test count(form.conditioning) == 3 && length(form.conditioning) == 4
    blocks = support_blocks(form)
    # Ξ: (a, b[1]), (b[2], c, f) and e.
    @test length(blocks) == 3
    roles = reduce(vcat, block.roles for block in blocks)
    @test count(==(:truncate), roles) == 3 && count(==(:valid), roles) == 1
    mom = moments(form)
    @test [block.columns for block in mom.blocks] == [[2, 4, 5, 6], [3], [7]]
    @test [block.num_samples for block in mom.blocks] == [10_000, 0, 0]
    # The estimated mean is in the conditioning set.
    η = mom.mean[[2, 4, 5, 6]]
    @test all(d.W * η .>= d.h)
    return
end

# A declared conditioned law, coupled with another one, gives its laws and rows.
function test_merging_declared_conditioned_law()
    declared = ConditionedDistribution(
        [Uniform(0, 1), Uniform(0, 1)],
        [-1.0 -1.0],
        [-1.0];
        names = ["inner"],
    )
    model, decls = model_with("p" => declared, "q" => Uniform(0, 1))
    row = @constraint(model, outer, model[:p][2] + model[:q] <= 1)
    form = form_of(model, decls; truncate = [row])
    d = only(form.laws).distribution
    @test d.laws == [Uniform(0, 1), Uniform(0, 1), Uniform(0, 1)]
    @test d.W == [-1 -1 0; 0 -1 -1] && d.h == [-1, -1] && d.names == ["inner", "outer"]
    return
end

# A second-order cone row tagged `Truncate()` conditions the laws that it couples, by rejection,
# even a univariate law alone; it stays a row of Ξ.
function test_conic_truncation()
    model, decls = model_with("a" => Uniform(0, 2), "b" => Uniform(0, 2), "c" => Uniform(0, 2))
    a, b, c = model[:a], model[:b], model[:c]
    disc = @constraint(model, disc, [1, a - 1, b - 1] in SecondOrderCone())
    cut = @constraint(model, cut, b + c <= 2)
    form = form_of(model, decls; truncate = [disc, cut])
    d = only(form.laws).distribution
    @test d isa ConditionedDistribution && length(d) == 3
    @test d.cones == [MOI.Nonnegatives(1), MOI.SecondOrderCone(3)]
    @test d.W == [0 -1 -1; 0 0 0; 1 0 0; 0 1 0] && d.h == [-2, -1, 1, 1]
    @test d.names == ["cut", "disc", "disc", "disc"]
    @test only(form.conic_rows).conditioning
    block = only(support_blocks(form))
    @test block.cones == [MOI.Nonnegatives(7), MOI.SecondOrderCone(3)]
    @test block.roles == [fill(:box, 6); :truncate; fill(:truncate, 3)]
    mom = moments(form)
    # The disc is symmetric about (1, 1), and the cut keeps (a, b) in it.
    @test only(mom.blocks).num_samples == 10_000
    @test abs(mom.mean[2] - 1) < 0.02
    draws = rand(LDR.Random.Xoshiro(1), d, 1000)
    @test all((draws[1, :] .- 1) .^ 2 .+ (draws[2, :] .- 1) .^ 2 .<= 1 + 1e-9)
    @test all(draws[2, :] .+ draws[3, :] .<= 2 + 1e-9)
    # A univariate law in a cone alone is conditioned by rejection, not truncated.
    model, decls = model_with("e" => Uniform(0, 4))
    row = @constraint(model, [1, model[:e] - 1] in SecondOrderCone())
    form = form_of(model, decls; truncate = [row])
    @test only(form.laws).distribution isa ConditionedDistribution
    # A lifted form keeps the conic rows, over the lifted coordinates.
    model, decls = model_with("f" => Uniform(0, 2), "g" => Uniform(0, 2))
    row = @constraint(model, [3, model[:g] - 1] in SecondOrderCone())
    form = form_of(model, decls; valid = [row], breakpoints = Dict(index(model[:f]) => [1.0]))
    @test Matrix(only(form.conic_rows).W) == [0 0 0; 0 0 1]
    return
end

# A valid inequality holds at the mean of the truncated law, not of the original one.
function test_valid_row_at_truncated_mean()
    model, decls = model_with("d" => Uniform(80, 120))
    cut = @constraint(model, 101 <= model[:d] <= 110)
    valid = @constraint(model, model[:d] >= 101)
    form = form_of(model, decls; truncate = [cut], valid = [valid])
    @test only(form.laws).distribution == Uniform(101, 110)
    model, decls = model_with("d" => Uniform(80, 120))
    cut = @constraint(model, 101 <= model[:d] <= 110)
    valid = @constraint(model, wrong, model[:d] >= 108)
    msg = error_message(() -> form_of(model, decls; truncate = [cut], valid = [valid]))
    @test occursin("wrong is tagged `ValidInequality()`, but the mean", msg)
    return
end

# The breakpoints of a truncated law are in its support; a conditioned law cannot be lifted.
function test_lifting()
    model, decls = model_with("d" => Uniform(80, 120))
    cut = @constraint(model, 100 <= model[:d] <= 110)
    form = form_of(model, decls; truncate = [cut], breakpoints = Dict(index(model[:d]) => 1))
    @test only(form.lifting.blocks).lifting.breakpoints == [105]
    @test moments(form).mean ≈ [1, 103.75, 1.25]
    model, decls = model_with("d" => Uniform(80, 120), "e" => Uniform(0, 1))
    cut = @constraint(model, cut, model[:d] + model[:e] <= 110)
    msg = error_message(
        () ->
            form_of(model, decls; truncate = [cut], breakpoints = Dict(index(model[:d]) => 1)),
    )
    @test occursin(
        "The uncertainty d has a lifting, but its law is conditioned by the " *
        "constraints cut and estimated by rejection sampling",
        msg,
    )
    # A lifted form keeps the conditioning rows.
    model, decls = model_with("d" => Uniform(80, 120), "e" => Uniform(0, 1), "g" => Uniform(0, 1))
    cut = @constraint(model, cut, model[:e] + model[:g] <= 1)
    form = form_of(model, decls; truncate = [cut], breakpoints = Dict(index(model[:d]) => [100.0]))
    @test form.conditioning[1] && !any(form.conditioning[2:end])
    @test length(form.conditioning) == length(form.h_valid)
    return
end

end # module TestConditioning

TestConditioning.runtests()
