# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Every file named `test_*.jl` under `test/` is a module with a `runtests()` function that runs
# its own `test_*` functions. Set `LDR_TEST_FILES="a.jl,moi/b.jl"` to run only some files, by name
# or by path relative to `test/` (two files are named `test_checks.jl`), and
# `LDR_SKIP_TUTORIALS=true` or `LDR_SKIP_AQUA=true` to skip the two slowest ones.

# The tests follow the layers of `src/`: `test/<layer>/test_<file>.jl` tests mainly
# `src/<layer>/<file>.jl`. The core layers (`canonical`, `lifting`, `probability`, `formulations`)
# build plain JuMP models with a hand-built `ModelInfo`; tests that need the tags of `LDRModel` or
# `optimize!` are in `moi/` or `jump/`. Shared problems and oracles are in `problems/` and
# `oracles/`, and shared helpers in `common.jl`, in files without the `test_` prefix.

using Test

const ONLY = filter(!isempty, split(get(ENV, "LDR_TEST_FILES", ""), ","))

@testset "LinearDecisionRules" begin
    for (root, _, files) in walkdir(@__DIR__), file in sort(files)
        path = replace(relpath(joinpath(root, file), @__DIR__), '\\' => '/')
        if !startswith(file, "test_") ||
           !endswith(file, ".jl") ||
           !(isempty(ONLY) || file in ONLY || path in ONLY)
            continue
        end
        @testset "$path" begin
            include(joinpath(root, file))
        end
    end
end
