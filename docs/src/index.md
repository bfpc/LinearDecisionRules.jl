# LinearDecisionRules.jl

Linear decision rules for multistage stochastic programs written in [JuMP](https://jump.dev).

The user writes a plain `JuMP.Model`. At `optimize!`, the model is reformulated into the primal and
dual linear decision rule problems of Kuhn, Wiesemann and Georghiou (2011): the primal gives a
feasible policy and its expected cost, the dual a bound on the optimal value.

Start with [Getting started](getting_started.md), then the tutorials. The
[Developer documentation](@ref developer_overview) explains how the package works inside; it is
written for those who change the package, not for its users.
