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

end # module LinearDecisionRules
