# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# # [Ellipsoidal uncertainty with `ConfidenceMvNormal`](@id confidence_mvnormal_tutorial)

# ```@meta
# CurrentModule = LinearDecisionRules
# ```

# This tutorial introduces the [`ConfidenceMvNormal`](@ref) distribution, which models uncertainty
# as a multivariate normal law conditioned on an ellipsoid with a chosen fraction of its
# probability mass. It is especially useful when you want to control how *conservative* the
# uncertainty set is. We also see the two ways to give the ellipsoid to the formulations, as a
# second-order cone or as boxes, and how `support = :auto` chooses between them.

# ## Setup

using JuMP
using HiGHS
using Clarabel
using Distributions
using LinearDecisionRules
using Random

# ## What is `ConfidenceMvNormal`?

# Suppose that the demand follows a multivariate normal law, ``d \sim \mathcal{N}(\mu, \Sigma)``.
# In practice we may not want to plan for the entire (unbounded) support.
# `ConfidenceMvNormal(μ, Σ, α)` restricts the uncertainty to the **ellipsoid** centered at
# ``\mu`` that contains the fraction ``\alpha`` of the probability mass,
# ```math
# E_{\alpha} = \{\, x :  (x - \mu)^\top \Sigma^{-1} (x - \mu) \leq \rho^2(\alpha) \,\},
# ```
# where ``\rho^2(\alpha) = \text{quantile}(\chi^2_d, \alpha)``, and its law is the normal law
# *conditioned* on ``E_\alpha``. The distribution gives the formulations:
# * the **mean**, equal to ``\mu`` (the ellipsoid is symmetric);
# * the **covariance**, a scalar multiple of ``\Sigma``, smaller than ``\Sigma`` because the tails
#   are cut;
# * a finite **box**, `minimum` and `maximum`, which is the box tangent to the ellipsoid along the
#   coordinates;
# * its **support** beyond the box, as rows or cones, with [`support_constraints`](@ref).

# ## Inspecting the distribution

μ = [100.0, 80.0]
Σ = [100.0 40.0; 40.0 64.0]

law = ConfidenceMvNormal(μ, Σ, 0.95)
(ρ = law.ρ, mean = mean(law), cov = cov(law), lower = minimum(law), upper = maximum(law))

# As ``\alpha`` grows, the ellipsoid expands, so that the covariance approaches ``\Sigma`` and the
# box widens:

println("α      ρ        cov scaling   box of the first demand")
for α in [0.50, 0.80, 0.95, 0.99]
    d = ConfidenceMvNormal(μ, Σ, α)
    box = round.([minimum(d)[1], maximum(d)[1]]; digits = 2)
    println(
        rpad(α, 7),
        rpad(round(d.ρ; digits = 3), 9),
        rpad(round(cov(d)[1, 1] / Σ[1, 1]; digits = 3), 14),
        box,
    )
end

# ## Ellipsoid or boxes

# The formulations are not given the ellipsoid in the same way. The `support` keyword chooses it:

# * `support = :ellipsoid`: the ellipsoid itself, as **one second-order cone** row. The inner
#   problems of the primal and the dual formulations are second-order cone programs (SOCP), which
#   need a conic solver, such as Clarabel.
# * `support = :boxes`: the **box along the principal axes** of ``\Sigma`` (the rectangle that
#   contains the ellipse, with its sides parallel to its axes), in addition to the box of the law.
#   These are linear rows, so that the inner problems stay linear programs (LP), solved by HiGHS,
#   for instance. The boxes are an outer approximation of the ellipsoid: the primal rule is still
#   feasible on the ellipsoid, and the dual bound is valid, but both are looser.
# * `support = :auto` (the default): the optimizer chooses, as the next sections show.

# The cones that the formulations see are in `support_constraints`: one second-order cone for the
# ellipsoid, and four linear inequalities for the boxes (two sides for each of the two axes).

ellipsoid = ConfidenceMvNormal(μ, Σ, 0.95; support = :ellipsoid)
boxes = ConfidenceMvNormal(μ, Σ, 0.95; support = :boxes)
only(support_constraints(ellipsoid).cones), only(support_constraints(boxes).cones)

# The areas of the sets, in this two-dimensional example, give an idea of the loss of the
# approximation. The ellipse has the area ``\pi \rho^2 \sqrt{\det \Sigma}``; the box along the
# principal axes has the area ``4 \rho^2 \sqrt{\det \Sigma}``, which is ``4/\pi \approx 1.27``
# times larger; and the box of the law, along the coordinates, has the area
# ``4 \rho^2 \sqrt{\Sigma_{11} \Sigma_{22}}``, which is larger when the demands are correlated.

detΣ = Σ[1, 1] * Σ[2, 2] - Σ[1, 2]^2
areas = (
    ellipse = π * law.ρ^2 * sqrt(detΣ),
    principal_box = 4 * law.ρ^2 * sqrt(detΣ),
    coordinate_box = prod(maximum(law) .- minimum(law)),
)

# ## An inventory problem with correlated demands

# Two products share a stock that is bought before their correlated demands are known. Selling
# more of one product leaves less stock for the other, so the **joint** behavior of the demands
# matters, and the shape of their support matters with it.

function shared_stock(optimizer, α; support = :auto, covariance = Σ)
    model = LDRModel(optimizer)
    set_silent(model)
    law = ConfidenceMvNormal(μ, covariance, α; support)
    @variable(model, buy >= 0, FirstStage)
    @variable(model, sell[1:2] >= 0)
    @variable(model, d[1:2] in Uncertainty(law))
    @constraint(model, sum(sell) <= buy)
    @constraint(model, sell .<= d)
    @objective(model, Max, -10buy + 15sell[1] + 12sell[2])
    optimize!(model)
    return model
end

# We solve it for several confidence levels with the boxes (LP, with HiGHS) and with the
# ellipsoid (SOCP, with Clarabel), and print the primal value and the dual bound of each.

println("         boxes (HiGHS)         ellipsoid (Clarabel)")
println("α      primal     dual       primal     dual")
for α in [0.50, 0.80, 0.90, 0.95, 0.99]
    a = shared_stock(HiGHS.Optimizer, α; support = :boxes)
    b = shared_stock(Clarabel.Optimizer, α; support = :ellipsoid)
    v =
        round.(
            [objective_value(a), objective_bound(a), objective_value(b), objective_bound(b)];
            digits = 2,
        )
    println(rpad(α, 7), rpad(v[1], 11), rpad(v[2], 11), rpad(v[3], 11), v[4])
end

# The ellipsoid is a smaller set than the boxes, so the primal value is **higher** (the rule has
# fewer scenarios to be feasible for) and the dual bound is **lower**, hence tighter. For both,
# the primal value decreases with ``\alpha``: a larger set is more conservative, and the planner
# gives up some profit for it, even if the dual bound does not change much.

# !!! note
#     A higher ``\alpha`` means a wider ellipsoid, where the demand can be further from its mean,
#     and the primal rule is feasible for all of them. A smaller ``\alpha`` is a tighter set,
#     and a more optimistic policy that is protected only against the demands in this smaller
#     set.

# ### The effect of the shape

# When the demands are uncorrelated, the principal axes are the coordinates, and `:boxes` is
# exactly the box of the law: it does not cut the corners of the box, where the ellipse is far
# from. The difference between the formulations is then the largest:

diagonal = [100.0 0.0; 0.0 64.0]
for (name, optimizer, support) in
    [("boxes", HiGHS.Optimizer, :boxes), ("ellipsoid", Clarabel.Optimizer, :ellipsoid)]
    m = shared_stock(optimizer, 0.95; support, covariance = diagonal)
    println(
        rpad(name, 10),
        round(objective_value(m); digits = 2),
        "   ",
        round(objective_bound(m); digits = 2),
    )
end

# ## The `:auto` heuristic

# By default, `support = :auto` lets the optimizer choose, by the rule of
# [`ConfidenceMvNormal`](@ref):
# * HiGHS does not solve second-order cone programs, so `:auto` gives the boxes and the same
#   values as `:boxes`;
# * Clarabel does, so `:auto` gives the ellipsoid;
# * with an integer decision, `:auto` gives the boxes even for a conic solver, since a
#   mixed-integer second-order cone program is much harder than a mixed-integer linear one.
# The dimension of the law is not part of the heuristic: the ellipsoid is one cone of dimension
# ``d + 1``, while the boxes add ``2d`` rows.

for (name, optimizer) in [("HiGHS", HiGHS.Optimizer), ("Clarabel", Clarabel.Optimizer)]
    auto = shared_stock(optimizer, 0.95; support = :auto)
    explicit = shared_stock(optimizer, 0.95; support = name == "HiGHS" ? :boxes : :ellipsoid)
    println(
        rpad(name, 10),
        round(objective_value(auto); digits = 4),
        "   ",
        round(objective_value(explicit); digits = 4),
    )
end

# Asking for `:ellipsoid` with a solver that does not support the cone is an error of the inner
# optimizer, and the message says that the constraint is not supported:

try
    shared_stock(HiGHS.Optimizer, 0.95; support = :ellipsoid)
catch err
    ## The message has the constraint type, then the explanation.
    println(split(sprint(showerror, err), "\n\n")[2])
end

# !!! tip
#     Use `:boxes` when the inner optimizer is an LP solver, or when the decisions are integer,
#     and `:ellipsoid` (or `:auto`) with a conic solver. With `:auto`, the same model changes
#     its formulation, and its values, with the solver, so state the support if you want
#     to compare results across solvers.

# ## Simulating the policy

# [`simulate`](@ref) draws from the law (by rejection, in the standardized space, so that all the
# draws are in the ellipsoid) and applies the rule. The primal rule of the ellipsoid has no
# violation, and the mean of the objective estimates its value:

model = shared_stock(Clarabel.Optimizer, 0.95; support = :ellipsoid)
(objective_value(model), simulate(model, 10_000; rng = Xoshiro(1)).objective_mean)

# ## What's next?

# This is the last tutorial. The API pages, from [Modelling](@ref), document every function.
