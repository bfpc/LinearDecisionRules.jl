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
import QuadGK
import Random
import SparseArrays

include("lifting/lifting.jl")
include("canonical/info.jl")
include("canonical/form.jl")
include("canonical/information.jl")
include("canonical/support.jl")
include("probability/moments.jl")
include("probability/distributions/mv_discrete_non_parametric.jl")
include("probability/distributions/lifted_distribution.jl")
include("lifting/lifted_form.jl")
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
include("jump/rules.jl")
include("jump/simulate.jl")

export AbstractFormulation,
    AbstractLifting,
    AxisPiecewise,
    BreakPoints,
    CanonicalForm,
    CovarianceBlock,
    DecisionRule,
    DualLDR,
    Evaluation,
    FirstStage,
    FormulationAttribute,
    FormulationCapabilities,
    FormulationResult,
    Formulations,
    InformationStructure,
    InnerOptimizer,
    LDRModel,
    LiftedDistribution,
    LiftedTerm,
    LiftingBlock,
    LiftingMap,
    ModelInfo,
    Moments,
    MvDiscreteNonParametric,
    PrimalLDR,
    RuleMatrix,
    SampledLDR,
    Simulation,
    Stage,
    SupportBlock,
    Truncate,
    Uncertainty,
    UncertaintyBlock,
    UncertaintyDeclaration,
    ValidInequality,
    canonicalize,
    capabilities,
    decision_rule,
    evaluate,
    expected_value,
    lift,
    moments,
    observes,
    retraction,
    rule_coefficient,
    rule_matrix,
    second_moment,
    set_stage,
    simulate,
    solve_formulation,
    stage,
    support_blocks,
    unset_stage,
    visible_columns

end # module LinearDecisionRules
