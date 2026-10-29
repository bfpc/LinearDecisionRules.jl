# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Runs the quality checks of Aqua.jl, which take about 30 seconds. Set `LDR_SKIP_AQUA=true` to skip
# them in local runs; CI runs them.

module TestAqua

using Test

import Aqua
import LinearDecisionRules

function runtests()
    if get(ENV, "LDR_SKIP_AQUA", "false") == "true"
        @info "Skipping the Aqua checks (LDR_SKIP_AQUA=true)"
        return
    end
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$name" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_aqua()
    Aqua.test_all(LinearDecisionRules)
    return
end

end # module TestAqua

TestAqua.runtests()
