module LinearDecisionRules

using JuMP
import Distributions
import Expectations
import Random
import SparseArrays
import LinearAlgebra
const MOI = JuMP.MOI

include("pwl_moments.jl")
include("jump.jl")
include("matrix_data.jl")
include("canonical.jl")
include("solve_primal.jl")
include("solve_dual.jl")
include("implement_rule.jl")

# distributions
include("distributions/mv_discrete_non_parametric.jl")

end # module LinearDecisionRules
