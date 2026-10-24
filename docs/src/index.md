# LinearDecisionRules.jl

Linear decision rules for multistage stochastic programs written in [JuMP](https://jump.dev).

The user writes a plain `JuMP.Model`. At `optimize!`, the model is reformulated into the primal and
dual linear decision rule problems of Kuhn, Wiesemann and Georghiou (2011): the primal gives a
feasible policy and its expected cost, the dual a bound on the optimal value.
