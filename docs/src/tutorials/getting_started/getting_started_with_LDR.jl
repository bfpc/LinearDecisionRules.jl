# # Getting started with LinearDecisionRules

# This tutorial provides a quick introduction to modeling and solving stochastic
# optimization problems with LinearDecisionRules.jl.

# If you are new to JuMP, start by reading the
# [Getting started with JuMP](https://jump.dev/JuMP.jl/stable/tutorials/getting_started/getting_started_with_JuMP/)
# tutorial first.

# ## What is LinearDecisionRules.jl?

# LinearDecisionRules.jl is a Julia package for solving **two-stage stochastic
# optimization problems** using **linear decision rules** (LDRs).

# In a two-stage stochastic problem:
#  * **First-stage decisions** are made *before* the uncertainty is revealed
#  * **Second-stage decisions** are made *after* observing the uncertain data

# The key insight is that second-stage decisions can be modeled as **linear
# functions** of the uncertain parameters. This approximation makes the problem
# tractable while still capturing the adaptive nature of the decisions.

# ## What is a decision rule?

# A decision rule specifies how a decision variable depends on the realized
# uncertainty. For example, if `x` is a decision variable and `ξ` is an uncertain
# parameter, a linear decision rule takes the form:
# ```math
# x(ξ) = x_0 + x_1 ξ
# ```
# where `x_0` is the constant term and `x_1` is the coefficient that determines
# how `x` responds to changes in `ξ`.

# ## Installation

# LinearDecisionRules.jl is installed using the Julia package manager:
# ```julia
# import Pkg
# Pkg.add("LinearDecisionRules")
# ```

# You also need a solver. We recommend HiGHS:
# ```julia
# Pkg.add("HiGHS")
# ```

# ## An example: The Newsvendor problem

# Let's solve a classic stochastic optimization problem: the **Newsvendor problem**.

# A retailer must decide how many units of a product to buy before knowing
# the demand. The problem is:
#  * Buy cost: \$10 per unit
#  * Sell price: \$15 per unit
#  * Return value: \$8 per unit (for unsold items)
#  * Demand: uniformly distributed between 80 and 120 units

# Mathematically, we want to solve:
# ```math
# \begin{aligned}
# \max \quad & -10 \cdot \text{buy} + \mathbb{E}[15 \cdot \text{sell}(ξ) + 8 \cdot \text{return}(ξ)] \\
# \text{s.t.} \quad & \text{sell}(ξ) + \text{return}(ξ) \leq \text{buy} \\
# & \text{sell}(ξ) \leq \text{demand}(ξ) \\
# & \text{sell}(ξ), \text{return}(ξ), \text{buy} \geq 0
# \end{aligned}
# ```

# where `buy` is decided before demand is known, and `sell` and `return` adapt
# to the realized demand.

# Here's the complete code:

using JuMP
import LinearDecisionRules
import HiGHS
import Distributions

ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
set_silent(ldr)

@variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
@variable(ldr, sell >= 0)
@variable(ldr, ret >= 0)
@variable(ldr, demand in LinearDecisionRules.Uncertainty(
    distribution = Distributions.Uniform(80, 120),
))

@constraint(ldr, sell + ret <= buy)
@constraint(ldr, sell <= demand)

@objective(ldr, Max, -10 * buy + 8 * ret + 15 * sell)

optimize!(ldr)

# Let's check the solution:

solution_summary(ldr)

# ## Step-by-step walkthrough

# Let's break down each part of the code.

# ### Loading packages

# We need four packages:

using JuMP                      # Algebraic modeling language
import LinearDecisionRules      # Linear decision rules
import HiGHS                    # Solver
import Distributions            # For probability distributions

# ### Creating the model

# Create an `LDRModel` by passing a solver:

ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
set_silent(ldr)

# The `LDRModel` is similar to JuMP's `Model`, but it handles the reformulation
# needed for stochastic optimization with decision rules.

# ### First-stage variables

# Variables that must be decided *before* the uncertainty is revealed are
# marked with `FirstStage`:

@variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)

# First-stage variables have constant decision rules (they don't depend on ξ).

# ### Uncertainties

# Uncertain parameters are declared using `Uncertainty` with a distribution:

@variable(ldr, demand in LinearDecisionRules.Uncertainty(
    distribution = Distributions.Uniform(80, 120),
))

# The distribution specifies the support and probability distribution of the
# uncertainty. Here, demand is uniformly distributed between 80 and 120.

# ### Second-stage variables

# Variables declared *without* `FirstStage` are second-stage variables. Their
# decision rules can depend linearly on the uncertainty:

@variable(ldr, sell >= 0)
@variable(ldr, ret >= 0)

# ### Constraints

# Constraints are written just like in JuMP:

@constraint(ldr, sell + ret <= buy)
@constraint(ldr, sell <= demand)

# !!! note
#     Constraints are interpreted as holding *for all* scenarios. The package
#     reformulates these into deterministic constraints that guarantee
#     feasibility for the entire uncertainty set.

# ### Objective

# The objective is set using `@objective`:

@objective(ldr, Max, -10 * buy + 8 * ret + 15 * sell)

# !!! note
#     The objective is interpreted as an *expectation* over the uncertainty.
#     Terms involving second-stage variables are averaged over all scenarios.

# ### Solving the model

# Call `optimize!` to solve:

optimize!(ldr)

# ### Querying the solution

# Check if the model solved successfully:

termination_status(ldr)

# Query the objective value:

objective_value(ldr)

# The package also provides a **dual** (outer) bound. This is useful because
# linear decision rules provide an inner approximation, so the true optimal
# value lies between the primal and dual bounds:

objective_value(ldr; dual = true)

# ### Understanding decision rules

# The key feature of LinearDecisionRules.jl is extracting the decision rules.
# Use `get_decision` to see how each variable depends on the uncertainty.

# For first-stage variables, the decision rule is just a constant:

LinearDecisionRules.get_decision(ldr, buy)

# For second-stage variables, you can query both the constant term and the
# coefficient on the uncertainty:

sell_constant = LinearDecisionRules.get_decision(ldr, sell)

#-

sell_coefficient = LinearDecisionRules.get_decision(ldr, sell, demand)

# This means: `sell(ξ) = sell_constant + sell_coefficient * demand`.

# Similarly for `ret`:

ret_constant = LinearDecisionRules.get_decision(ldr, ret)

#-

ret_coefficient = LinearDecisionRules.get_decision(ldr, ret, demand)

# To evaluate the decision rule at a specific demand value, compute manually:

demand_value = 100  # Expected demand
sell_at_100 = sell_constant + sell_coefficient * demand_value

#-

demand_value = 80  # Minimum demand
sell_at_80 = sell_constant + sell_coefficient * demand_value

#-

demand_value = 120  # Maximum demand
sell_at_120 = sell_constant + sell_coefficient * demand_value

# ## Key concepts recap

# 1. **LDRModel**: The main model type, similar to JuMP's `Model`
# 2. **FirstStage**: Attribute for variables decided before uncertainty is revealed
# 3. **Uncertainty**: Set for declaring uncertain parameters with a distribution
# 4. **Constraints**: Interpreted as holding for all scenarios
# 5. **Objective**: Interpreted as an expectation
# 6. **get_decision**: Retrieve the decision rule coefficients

# ## What's next?

# Now that you understand the basics, you can:
#  * Learn about the [mathematical formulation](@ref math_formulation)
#  * Explore [piecewise linear extensions](@ref pwl_extensions) for better
#    approximations
