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
include("formulations/formulation.jl")
include("formulations/primal.jl")
include("formulations/dual.jl")
include("formulations/sampled.jl")
include("moi/sets.jl")
include("moi/attributes.jl")
include("moi/model_info.jl")
include("moi/optimizer.jl")
include("moi/results.jl")
include("jump/variables.jl")
include("jump/constraints.jl")
include("jump/checks.jl")
include("jump/model.jl")
include("jump/results.jl")

export AbstractFormulation,
    BreakPoints,
    CanonicalForm,
    CovarianceBlock,
    DualLDR,
    FirstStage,
    FormulationAttribute,
    FormulationCapabilities,
    FormulationResult,
    Formulations,
    InformationStructure,
    InnerOptimizer,
    LDRModel,
    ModelInfo,
    Moments,
    MvDiscreteNonParametric,
    PrimalLDR,
    SampledLDR,
    Stage,
    SupportBlock,
    Truncate,
    Uncertainty,
    UncertaintyBlock,
    UncertaintyDeclaration,
    ValidInequality,
    canonicalize,
    capabilities,
    moments,
    observes,
    second_moment,
    set_stage,
    solve_formulation,
    stage,
    support_blocks,
    unset_stage,
    visible_columns

end # module LinearDecisionRules
