module LinearDecisionRules

using JuMP
import Distributions
import Random
import SparseArrays
import LinearAlgebra
const MOI = JuMP.MOI

include("jump.jl")
include("matrix_data.jl")
include("pwl.jl")
include("canonical.jl")
include("solve_primal.jl")
include("solve_dual.jl")
include("implement_rule.jl")

# distributions
include("distributions/mv_discrete_non_parametric.jl")
include("distributions/univariate_piece_wise.jl")

end # module LinearDecisionRules
