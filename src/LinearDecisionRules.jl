module LinearDecisionRules

using JuMP
import Distributions
import Random
import SparseArrays
using LinearAlgebra: tr
const MOI = JuMP.MOI

include("jump.jl")
include("matrix_data.jl")
include("canonical.jl")
include("solve_primal.jl")
include("solve_dual.jl")
include("implement_rule.jl")

# distributions
include("distributions/mv_discrete_non_parametric.jl")

end # module LinearDecisionRules
