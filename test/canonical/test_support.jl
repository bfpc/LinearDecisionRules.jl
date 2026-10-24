# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestSupport

using JuMP
using Test

import Distributions
import LinearDecisionRules:
    ModelInfo,
    MvDiscreteNonParametric,
    SupportBlock,
    UncertaintyDeclaration,
    canonicalize,
    support_blocks

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

declare(dist, vars...) = UncertaintyDeclaration(dist, collect(index.(vars)))

# Two products with independent demands, and optionally `d[1] + d[2] <= 230` as a valid row.
function two_demands(; valid::Bool)
    model = Model()
    @variable(model, buy >= 0)
    @variable(model, sell[1:2] >= 0)
    @variable(model, d[1:2])
    @constraint(model, sum(sell) <= buy)
    @constraint(model, sell .<= d)
    @objective(model, Max, -10buy + 15sum(sell))
    rows = Set{MOI.ConstraintIndex}()
    if valid
        push!(rows, index(@constraint(model, total, d[1] + d[2] <= 230)))
    end
    info = ModelInfo(;
        uncertainties = [
            declare(Distributions.Uniform(80, 120), d[1]),
            declare(Distributions.Uniform(80, 120), d[2]),
        ],
        decision_stages = Dict(index(buy) => 1),
        valid_rows = rows,
    )
    return model, canonicalize(backend(model), info)
end

function test_independent_demands()
    _, form = two_demands(; valid = false)
    blocks = support_blocks(form)
    @test length(blocks) == 2
    for (b, block) in enumerate(blocks)
        @test block isa SupportBlock
        @test block.columns == [1 + b]
        @test Matrix(block.W) == [1; -1;;]
        @test block.h == [80, -120]
        @test block.cones == [MOI.Nonnegatives(2)]
        @test block.roles == [:box, :box]
        @test block.constraints == [nothing, nothing]
    end
    return
end

function test_coupled_demands()
    model, form = two_demands(; valid = true)
    blocks = support_blocks(form)
    @test length(blocks) == 1
    block = only(blocks)
    @test block.columns == [2, 3]
    @test Matrix(block.W) == [1 0; -1 0; 0 1; 0 -1; -1 -1]
    @test block.h == [80, -120, 80, -120, -230]
    @test block.cones == [MOI.Nonnegatives(5)]
    @test block.roles == [:box, :box, :box, :box, :valid]
    @test block.constraints == [nothing, nothing, nothing, nothing, index(model[:total])]
    return
end

function test_unbounded_and_multivariate()
    model = Model()
    @variable(model, x)
    @variable(model, a)
    @variable(model, b[1:2])
    @variable(model, c[1:2])
    @variable(model, e)
    @constraint(model, x >= a + sum(b) + sum(c) + e)
    @constraint(model, interval, -10 <= b[2] - e <= 10)
    mvn = Distributions.MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0])
    scenarios = MvDiscreteNonParametric([[1.0, 5.0], [3.0, 2.0]], [0.5, 0.5])
    info = ModelInfo(;
        uncertainties = [
            declare(Distributions.Normal(), a),
            declare(scenarios, b...),
            declare(mvn, c...),
            declare(Distributions.Exponential(), e),
        ],
        valid_rows = Set{MOI.ConstraintIndex}([index(interval)]),
    )
    form = canonicalize(backend(model), info)
    blocks = support_blocks(form)
    # Columns: a, b[1], b[2], c[1], c[2], e.
    @test [block.columns for block in blocks] == [[2], [3], [4, 7], [5], [6]]
    a_block, b1_block, coupled, c1_block, c2_block = blocks
    @test size(a_block.W) == (0, 1) && isempty(a_block.h)
    @test a_block.cones == [MOI.Nonnegatives(0)]
    @test Matrix(b1_block.W) == [1; -1;;] && b1_block.h == [1, -3]
    # b[2] in [2, 5], e >= 0, and -10 <= b[2] - e <= 10 (two rows).
    @test Matrix(coupled.W) == [1 0; -1 0; 0 1; 1 -1; -1 1]
    @test coupled.h == [2, -5, 0, -10, -10]
    @test coupled.roles == [:box, :box, :box, :valid, :valid]
    @test coupled.constraints[4] == coupled.constraints[5] == index(interval)
    # MvNormal has an infinite box, so its blocks have no rows.
    @test size(c1_block.W) == size(c2_block.W) == (0, 1)
    return
end

# A second-order cone row couples its coordinates into one block, after the linear rows.
function test_second_order_cone_row()
    model = Model()
    @variable(model, x)
    @variable(model, d[1:3])
    @constraint(model, x >= sum(d))
    disc = @constraint(model, [10, d[1] - 100, d[3] - 100] in SecondOrderCone())
    law = Distributions.Uniform(90, 110)
    info = ModelInfo(;
        uncertainties = [declare(law, d[1]), declare(law, d[2]), declare(law, d[3])],
        valid_rows = Set{MOI.ConstraintIndex}([index(disc)]),
    )
    form = canonicalize(backend(model), info)
    row = only(form.conic_rows)
    @test row.cone == MOI.SecondOrderCone(3) && !row.conditioning
    @test Matrix(row.W) == [0 0 0; 1 0 0; 0 0 1] && row.h == [-10, 100, 100]
    blocks = support_blocks(form)
    @test [block.columns for block in blocks] == [[2, 4], [3]]
    block = first(blocks)
    @test block.cones == [MOI.Nonnegatives(4), MOI.SecondOrderCone(3)]
    @test Matrix(block.W) == [1 0; -1 0; 0 1; 0 -1; 0 0; 1 0; 0 1]
    @test block.h == [90, -110, 90, -110, -10, 100, 100]
    @test block.roles == [fill(:box, 4); fill(:valid, 3)]
    @test block.constraints == [fill(nothing, 4); fill(index(disc), 3)]
    @test blocks[2].cones == [MOI.Nonnegatives(2)]
    return
end

# Parameters are read as constants in a second-order cone row; its errors name the constraint.
function test_second_order_cone_row_errors()
    model = Model()
    @variable(model, x)
    @variable(model, r in Parameter(5.0))
    @variable(model, d[1:2])
    @constraint(model, x >= sum(d))
    ok = @constraint(model, [r, d[1] - 1, d[2]] in SecondOrderCone())
    law = Distributions.Uniform(-1, 1)
    function form_with(rows...; truncate = false)
        info = ModelInfo(;
            uncertainties = [declare(law, d[1]), declare(law, d[2])],
            valid_rows = Set{MOI.ConstraintIndex}(truncate ? [] : index.(rows)),
            truncate_rows = Set{MOI.ConstraintIndex}(truncate ? index.(rows) : []),
        )
        return canonicalize(backend(model), info)
    end
    @test only(form_with(ok).conic_rows).h == [-5, 1, 0]
    far = @constraint(model, far, [1, d[1] - 3, d[2]] in SecondOrderCone())
    err = try
        form_with(ok, far)
    catch err
        err
    end
    @test occursin("far is tagged `ValidInequality()`, but the mean of the uncertainty", err.msg)
    delete.(model, [ok, far])
    constant = @constraint(model, constant, [r, 1] in SecondOrderCone())
    err = try
        form_with(constant; truncate = true)
    catch err
        err
    end
    @test occursin("constant is tagged `Truncate()`, but it has no uncertainty", err.msg)
    return
end

# A law that does not define `minimum` and `maximum` (the front end rejects it) has no box rows:
# Ξ stays a superset of its support.
function test_law_without_bounds()
    model = Model()
    @variable(model, x)
    @variable(model, w[1:3])
    @constraint(model, x >= sum(w))
    info = ModelInfo(; uncertainties = [declare(Distributions.Dirichlet([1.0, 1.0, 1.0]), w...)])
    blocks = support_blocks(canonicalize(backend(model), info))
    @test [block.columns for block in blocks] == [[2], [3], [4]]
    @test all(block -> isempty(block.h), blocks)
    return
end

end # module TestSupport

TestSupport.runtests()
