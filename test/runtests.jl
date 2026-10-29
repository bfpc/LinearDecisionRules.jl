# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Every file named `test_*.jl` under `test/` is a module with a `runtests()` function that runs
# its own `test_*` functions. Set `LDR_TEST_FILES="a.jl,b.jl"` to run only some files, and
# `LDR_SKIP_TUTORIALS=true` or `LDR_SKIP_AQUA=true` to skip the two slowest ones.

using Test

const ONLY = filter(!isempty, split(get(ENV, "LDR_TEST_FILES", ""), ","))

@testset "LinearDecisionRules" begin
    for (root, _, files) in walkdir(@__DIR__), file in sort(files)
        if !startswith(file, "test_") || !endswith(file, ".jl") || !(isempty(ONLY) || file in ONLY)
            continue
        end
        @testset "$(relpath(joinpath(root, file), @__DIR__))" begin
            include(joinpath(root, file))
        end
    end
end
