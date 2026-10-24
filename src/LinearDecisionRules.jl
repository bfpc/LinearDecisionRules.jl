# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    LinearDecisionRules

Linear decision rules for multistage stochastic programs written in JuMP.

The user writes a plain `JuMP.Model` whose optimizer is [`LinearDecisionRules.Optimizer`](@ref);
at `optimize!`, the model is reformulated into primal and dual linear decision rule problems
(Kuhn, Wiesemann and Georghiou, 2011), which are solved by an inner optimizer.
"""
module LinearDecisionRules

import Distributions
import JuMP
import LinearAlgebra
import MathOptInterface as MOI
import Random
import SparseArrays

include("canonical/info.jl")
include("canonical/form.jl")
include("canonical/information.jl")
include("canonical/support.jl")
include("probability/moments.jl")
include("probability/distributions/mv_discrete_non_parametric.jl")
include("moi/sets.jl")
include("moi/attributes.jl")
include("jump/variables.jl")
include("jump/constraints.jl")

export BreakPoints,
    CanonicalForm,
    CovarianceBlock,
    FirstStage,
    InformationStructure,
    ModelInfo,
    Moments,
    MvDiscreteNonParametric,
    Stage,
    SupportBlock,
    Truncate,
    Uncertainty,
    UncertaintyBlock,
    UncertaintyDeclaration,
    ValidInequality,
    canonicalize,
    moments,
    observes,
    second_moment,
    set_stage,
    stage,
    support_blocks,
    unset_stage,
    visible_columns

end # module LinearDecisionRules
