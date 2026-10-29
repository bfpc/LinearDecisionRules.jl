# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

module TestUtils

using Test

import LinearDecisionRules as LDR

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

function test_union_find()
    u = LDR._UnionFind(7)
    LDR._join!(u, [6, 2])
    LDR._join!(u, Int[])
    LDR._join!(u, [4, 7])
    LDR._join!(u, [7, 2])
    # The groups are sorted, in the order of their smallest element.
    groups, group = LDR._groups(u)
    @test groups == [[1], [2, 4, 6, 7], [3], [5]]
    @test group == [1, 2, 3, 2, 4, 2, 2]
    return
end

function test_name_list()
    @test LDR._name_list(["a", "b"]) == "a, b"
    @test LDR._name_list(["a", "b", "c", "d"], 2) == "a, b and 2 more"
    return
end

function test_sparse_triplets()
    t = LDR._Triplets()
    LDR._push!(t, 1, 2, 3.0)
    LDR._push!(t, 2, 1, 0.0)  # zeros are not stored
    @test LDR._sparse(t, 2, 3) == [0.0 3.0 0.0; 0.0 0.0 0.0]
    @test LDR._nonzero_columns(LDR._sparse(t, 2, 3)) == [2]
    return
end

end # module TestUtils

TestUtils.runtests()
