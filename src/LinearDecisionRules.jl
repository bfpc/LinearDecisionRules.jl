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
include("solve_sampled.jl")
include("implement_rule.jl")
include("recursion.jl")

# distributions
include("distributions/mv_discrete_non_parametric.jl")
include("distributions/univariate_piece_wise.jl")
include("distributions/confidence_mv_normal.jl")

end # module LinearDecisionRules
