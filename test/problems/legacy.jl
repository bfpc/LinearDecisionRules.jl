# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The golden values of version 0.3.1, in `test/reference/legacy_values.toml`, and the translation of
# their names: `x1` there is `x[1]` here, `buy` of a one-product model can be `buy[1]`, and the
# coefficient of a lifted uncertainty is the vector of its coefficients on its pieces. The core
# tests compare them with the canonical problems, and the JuMP tests with the models written with
# the tags.

module Legacy

import TOML

const VALUES = TOML.parsefile(joinpath(@__DIR__, "..", "reference", "legacy_values.toml"))

# The index in `names` of the legacy name `legacy`.
function name_index(names, legacy::String)
    for name in (legacy, replace(legacy, r"^([a-z_]+)(\d+)$" => s"\1[\2]"), "$legacy[1]")
        i = findfirst(==(name), names)
        i === nothing || return i
    end
    return error("No decision or uncertainty is named $legacy.")
end

"""
    rules(decision_names, uncertainty_names, rules) -> Vector{Pair{Int,Vector{Float64}}}

The legacy rules `rules` (such as `VALUES["newsvendor"]["rules"]["primal"]`), as the row on
`ξ = (1, η)` of each decision listed, by its index in `decision_names`. `uncertainty_names` are the
names of the columns `2:end` of ξ, with `name[piece k]` for the pieces of a lifted uncertainty.
"""
function rules(decision_names, uncertainty_names, rules)
    return map(collect(rules)) do (name, rule)
        x = zeros(1 + length(uncertainty_names))
        x[1] = rule["constant"]
        for (u, coefficient) in rule["coefficients"]
            if coefficient isa Vector
                for (k, c) in enumerate(coefficient)
                    x[1+name_index(uncertainty_names, "$u[piece $k]")] = c
                end
            else
                x[1+name_index(uncertainty_names, u)] = coefficient
            end
        end
        return name_index(decision_names, name) => x
    end
end

end # module Legacy
