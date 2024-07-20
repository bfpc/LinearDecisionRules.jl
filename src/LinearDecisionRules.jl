module LinearDecisionRules

using JuMP
import Distributions
import SparseArrays
using LinearAlgebra: tr
const MOI = JuMP.MOI

# TODO
# consider first stage variables?

include("jump.jl")
include("matrix_data.jl")
include("canonical.jl")
include("solve_primal.jl")
include("solve_dual.jl")
include("implement_rule.jl")

# # this is a simple lightweight extension
# # we do not extend JuMP.AbstractModel so error reporting is limited

# # to be sabed in model.ext
# Base.@kwdef mutable struct Cache
#     uncertainty::Vector{JuMP.VariableRef} = JuMP.VariableRef[]
# end

# function _init_cache(m::Model)
#     m.ext[:cache] = Cache()
#     return
# end

# function _get_cache(m::Model)
#     return m.ext[:cache]::Cache
# end

# function _optimize_hook(model)
#     # this is called after the model is optimized
#     # we can use it to update the cache
#     cache = _get_cache(model)
#     cache.uncertainty = JuMP.value.(cache.uncertainty)
#     return optimize!(model; ignore_optimize_hook = true)
# end

end # module LinearDecisionRules
