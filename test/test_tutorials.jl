# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Runs every tutorial of `docs/src/tutorials`, each in its own module, as a script: the code must
# run and must not log a warning. The claims of their text are `@test`s on lines that end with
# `#src`, which Literate drops from the pages; with a seed, a claim is tested as an inequality.
# The numbers that matter are also tested in `test/jump`. Set `LDR_SKIP_TUTORIALS=true` to skip
# them.

module TestTutorials

using Test

const DIRECTORY = joinpath(@__DIR__, "..", "docs", "src", "tutorials")

function runtests()
    if get(ENV, "LDR_SKIP_TUTORIALS", "false") == "true"
        @info "Skipping the tutorials (LDR_SKIP_TUTORIALS=true)"
        return
    end
    for file in sort(filter(endswith(".jl"), readdir(DIRECTORY)))
        @testset "$file" begin
            test_tutorial(joinpath(DIRECTORY, file))
        end
    end
    return
end

# A tutorial prints tables, which the test log does not need.
function test_tutorial(file::String)
    sandbox = Module(:Tutorial)
    Core.eval(sandbox, :(include(path) = Base.include($sandbox, path)))
    @test_logs redirect_stdout(devnull) do
        return Base.include(sandbox, file)
    end
    return
end

end # module TestTutorials

TestTutorials.runtests()
