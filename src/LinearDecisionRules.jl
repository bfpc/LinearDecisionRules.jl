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

include("utils.jl")
include("probability/laws.jl")
include("lifting/lifting.jl")
include("lifting/axis_piecewise.jl")
include("lifting/combination_piecewise.jl")
include("probability/distributions/conditioned_distribution.jl")
include("probability/distributions/confidence_mv_normal.jl")
include("probability/distributions/mv_discrete_non_parametric.jl")
include("probability/distributions/lifted_distribution.jl")
include("probability/distributions/sum_law.jl")
include("probability/distributions/combination_law.jl")
include("canonical/declarations.jl")
include("canonical/types.jl")
include("canonical/names.jl")
include("canonical/read.jl")
include("canonical/information_structure.jl")
include("canonical/checks.jl")
include("canonical/support.jl")
include("canonical/conditioning.jl")
include("lifting/lifted_form.jl")
include("canonical/canonicalize.jl")
include("probability/rejection_sampling.jl")
include("probability/moments.jl")
include("formulations/interface.jl")
include("formulations/rows.jl")
include("formulations/primal.jl")
include("formulations/dual.jl")
include("formulations/sampled.jl")
include("formulations/result.jl")
include("formulations/inner.jl")
include("moi/sets.jl")
include("moi/attributes.jl")
include("moi/model_info.jl")
include("moi/checks.jl")
include("moi/inner_cache.jl")
include("moi/optimizer.jl")
include("moi/results.jl")
include("jump/variables.jl")
include("jump/constraints.jl")
include("jump/checks.jl")
include("jump/model.jl")
include("jump/results.jl")
include("jump/rules.jl")
include("jump/simulate.jl")
include("experimental.jl")

# The user API, by the page of the manual that documents it. The canonical form, the moments and
# `solve_formulation` are documented on the Internals page but not exported: their contract may
# change until the `Experimental` submodule makes them stable.

# Modelling.
export LDRModel,
    Stage,
    FirstStage,
    stage,
    set_stage,
    unset_stage,
    Uncertainty,
    ValidInequality,
    Truncate,
    BreakPoints,
    Formulations,
    InnerOptimizer,
    RejectionSamplingSeed,
    RejectionSamplingTimeLimit,
    RejectionSamplingMaxSamples

# Results and rules.
export AbstractFormulation,
    PrimalLDR,
    DualLDR,
    SampledLDR,
    FormulationAttribute,
    Certified,
    is_certified,
    inner_model,
    decision_rule,
    DecisionRule,
    LiftedTerm,
    expected_value,
    rule_coefficient,
    rule_matrix,
    RuleMatrix

# Simulation.
export evaluate, Evaluation, simulate, Simulation

# Distributions and liftings.
export ConditionedDistribution,
    ConfidenceMvNormal,
    MvDiscreteNonParametric,
    LiftedDistribution,
    support_constraints,
    AbstractLifting,
    AxisPiecewise,
    lift,
    retraction,
    LiftingMap,
    LiftingBlock

# `LDRModel(optimizer)` is `Model(() -> LinearDecisionRules.Optimizer(optimizer))`: the optimizer
# is public, but not exported, and so is the `Experimental` submodule. The keyword `public` needs
# Julia 1.11.
if VERSION >= v"1.11"
    eval(Expr(:public, :Optimizer, :Experimental))
end

end # module LinearDecisionRules
