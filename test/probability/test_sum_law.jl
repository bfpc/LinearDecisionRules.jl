# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestSumLaw

using Test

import LinearDecisionRules

const LDR = LinearDecisionRules
const QuadGK = LDR.QuadGK

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

uniform(lo, width) = LDR._UniformTerm(lo, width)

# `E[(X - t)₊^q]` for `X` uniform on `[lo, lo + w]`.
function one_uniform(lo, w, q, t)
    t >= lo + w && return 0.0
    t >= lo && return (lo + w - t)^(q + 1) / ((q + 1) * w)
    m = lo + w / 2 - t
    return q == 1 ? m : w^2 / 12 + m^2
end

function test_one_uniform()
    law = LDR._SumLaw(1.0, [uniform(2.0, 4.0)])
    @test law.offset == 3.0 && law.width == 4.0
    for t in (-1.0, 3.0, 4.0, 5.5, 7.0, 9.0), q in (1, 2)
        @test LDR._stop_loss(law, q, t) ≈ one_uniform(3.0, 4.0, q, t) atol = 1e-14
    end
    return
end

# A discrete term merges equal values: the sum of two dice has 11 atoms.
function test_discrete_convolution()
    die = LDR._DiscreteTerm(collect(1.0:6.0), fill(1 / 6, 6))
    law = LDR._SumLaw(0.0, [die, die])
    @test law.atoms == collect(2.0:12.0)
    @test law.probabilities ≈ [min(k - 1, 13 - k) / 36 for k in 2:12]
    @test LDR._stop_loss(law, 1, 7.0) ≈ sum(max(k - 7, 0) * min(k - 1, 13 - k) for k in 2:12) / 36
    # `-0.0` and `0.0` are one atom.
    zero = LDR._DiscreteTerm([-0.0, 1.0], [0.5, 0.5])
    @test LDR._SumLaw(0.0, [zero, LDR._DiscreteTerm([0.0], [1.0])]).atoms == [0.0, 1.0]
    return
end

# The closed form for `d` terms is the expectation, over the `d`-th term, of the closed form for
# the first `d - 1`: a one-dimensional integral with the knots of the latter, by QuadGK.
function induction(terms, q, t)
    rest = LDR._SumLaw(0.0, terms[1:end-1])
    last = terms[end]
    knots = [t - k for k in LDR._stop_loss_knots(rest)]
    points = sort!([last.lower; [k for k in knots if last.lower < k < last.lower + last.width]])
    push!(points, last.lower + last.width)
    value, _ = QuadGK.quadgk(x -> LDR._stop_loss(rest, q, t - x), points...; rtol = 1e-14)
    return value / last.width
end

function test_closed_form_by_induction()
    terms = [uniform(0.5i, 0.3 + 0.7i) for i in 1:10]
    law = LDR._SumLaw(0.0, terms)
    @test length(law.knots) == 2^10 - count_equal_subset_sums(terms)
    lo, hi = law.offset, law.offset + law.width
    for t in range(lo - 1, hi + 1; length = 13), q in (1, 2)
        @test LDR._stop_loss(law, q, t) ≈ induction(terms, q, t) rtol = 1e-11 atol = 1e-12
    end
    return
end

# Subsets of the widths `0.3 + 0.7i` with equal sums share a knot.
function count_equal_subset_sums(terms)
    sums = Float64[]
    for S in 0:(2^length(terms)-1)
        push!(
            sums,
            sum((terms[i].width for i in eachindex(terms) if (S >> (i - 1)) & 1 == 1); init = 0.0),
        )
    end
    return length(sums) - length(unique(sums))
end

# Widths of very different sizes make the closed form cancel; it is then computed in BigFloat.
function test_disparate_widths()
    terms = [uniform(0.0, 1e-3), uniform(0.0, 1e3), uniform(5.0, 2.0)]
    law = LDR._SumLaw(0.0, terms)
    for t in (5.0004, 300.0, 600.0, 1006.9995), q in (1, 2)
        @test LDR._stop_loss(law, q, t) ≈ induction(terms, q, t) rtol = 1e-11
    end
    return
end

# A discrete term and uniform terms: a mixture of shifts of the uniform part.
function test_mixture()
    terms =
        [LDR._DiscreteTerm([0.0, 2.0, 5.0], [0.2, 0.5, 0.3]), uniform(0.0, 1.0), uniform(1.0, 3.0)]
    law = LDR._SumLaw(-1.0, terms)
    uniforms = LDR._SumLaw(-1.0, terms[2:3])
    for t in (-1.0, 0.5, 3.0, 6.5, 9.5), q in (1, 2)
        expected = sum(
            p * LDR._stop_loss(uniforms, q, t - z) for
            (z, p) in zip([0.0, 2.0, 5.0], [0.2, 0.5, 0.3])
        )
        @test LDR._stop_loss(law, q, t) ≈ expected atol = 1e-13
    end
    return
end

# The density of a sum of three uniform laws on [0, 1] (Irwin–Hall), and the integral of
# polynomials between knots.
function test_density_and_integral()
    law = LDR._SumLaw(0.0, [uniform(0.0, 1.0) for _ in 1:3])
    @test length(law.knots) == 4
    @test LDR._uniform_density(law, 0.5) ≈ 0.125
    @test LDR._uniform_density(law, 1.5) ≈ 0.75
    @test LDR._uniform_density(law, 2.5) ≈ 0.125
    @test LDR._uniform_density(law, 3.5) == 0.0
    density(x) = LDR._uniform_density(law, x)
    @test LDR._piecewise_integral(density, 0.0, 3.0, law.knots, 2) ≈ 1.0
    @test LDR._piecewise_integral(x -> x^2 * density(x), 0.0, 3.0, law.knots, 4) ≈ 1 / 4 + 9 / 4
    @test LDR._piecewise_integral(x -> x^5 - x, -1.0, 2.0, Float64[], 5) ≈ 63 / 6 - 3 / 2
    return
end

end # module TestSumLaw

TestSumLaw.runtests()
