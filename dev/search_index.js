var documenterSearchIndex = {"docs":
[{"location":"math/#Mathematical-formulations","page":"Mathematical formulations","title":"Mathematical formulations","text":"","category":"section"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"We follow the scheme from Primal and dual linear decision rules in stochastic and robust optimization, by Kuhn, Wiesemann and Georghiou.","category":"page"},{"location":"math/#Derivation-of-primal-LDR-problem","page":"Mathematical formulations","title":"Derivation of primal LDR problem","text":"","category":"section"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"We start from","category":"page"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"beginarrayrl\nmin   E c(ξ)^top x(ξ) + x(ξ)^top Q x(ξ) + r  05ex\ntextst  A_e x(ξ) = b_e(ξ) \n A_u x(ξ)  b_u(ξ) \n A_l x(ξ)  b_l(ξ) \n x(ξ)  x_u \n x(ξ)  x_l \n x_i(ξ) text is non-anticipative for  i  I \n  ξ  Ξ\nendarray","category":"page"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"where Ξ  ℝ^m is a polytope described by","category":"page"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"beginalign*\nΞ  =  ξ = (1 η)  ℝ^m mid W_u η  h_u W_l η  h_l lb  η  ub  \n =  ξ  ℝ^m mid W ξ  h \nendalign*","category":"page"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"Recall that the linear span of Ξ must be all of ℝ^m.","category":"page"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"We introduce positive slack variables for the inequality constraints, so that the problem can be written as","category":"page"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"beginarrayrl\nmin   E c(ξ)^top x(ξ) + x(ξ)^top Q x(ξ) + r  05ex\ntextst  A_e x(ξ) = b_e(ξ) \n A_u x(ξ) + s_u(ξ) = b_u(ξ) \n A_l x(ξ) - s_l(ξ) = b_l(ξ) \n x(ξ) + s_xu(ξ) = x_u \n x(ξ) - s_xl(ξ) = x_l \n s_u(ξ)  0 \n s_l(ξ)  0 \n s_xu(ξ)  0 \n s_xl(ξ)  0 \n  ξ  Ξ\nendarray","category":"page"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"Assuming a linear decision rule x(ξ) = X ξ, etc, and that scenario-dependent data c(ξ), b_e(ξ), b_u(ξ), b_l(ξ) can also be transformed to linear forms, we write the problem as","category":"page"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"beginarrayrl\nmin   E ξ^top C^top X ξ + ξ^top X^top Q X ξ + r  05ex\ntextst  A_e X ξ = B_e ξ \n A_u X ξ + S_u ξ = B_u ξ \n A_l X ξ - S_l ξ = B_l ξ \n X ξ + S_xu ξ = X_u ξ \n X ξ - S_xl ξ = X_l ξ \n S_u ξ  0 \n S_l ξ  0 \n S_xu ξ  0 \n S_xl ξ  0 \n  ξ  Ξ\nendarray","category":"page"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"where X_u = x_u 0 and X_l = x_l 0, since the first component of ξ is equal to 1 by definition.","category":"page"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"Since the linear span of Ξ is all of ℝ^m, the equalities must hold for all ξ, so they can be replaced by equalities between the corresponding matrices. Moreover, applying the trace trick to the objective function, we obtain","category":"page"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"beginarrayrl\nmin   texttrBig( (C^top X + X^top Q X) Eξ ξ^top Big) + r 05ex\ntextst  A_e X = B_e \n A_u X + S_u = B_u \n A_l X - S_l = B_l \n X + S_xu = X_u \n X - S_xl = X_l \n S_u ξ  0 \n S_l ξ  0 \n S_xu ξ  0 \n S_xl ξ  0 \n  ξ  Ξ\nendarray","category":"page"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"The non-negativity constraints are dealt with by duality. If S ξ  0 for all ξ in Ξ =  ξ  ℝ^m mid W ξ  h , then there exists a matrix Λ  0 such that S = Λ W and Λ h  0. Therefore, we introduce the matrices Λ_Su, Λ_Sl, Λ_Sxu, Λ_Sxl and obtain the equivalent problem","category":"page"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"beginarrayrl\nmin   texttrBig( (C^top X + X^top Q X) Eξ ξ^top Big) + r 05ex\ntextst  A_e X = B_e \n A_u X + S_u = B_u \n A_l X - S_l = B_l \n X + S_xu = X_u \n X - S_xl = X_l \n S_u = Λ_Su W \n S_l = Λ_Sl W \n S_xu = Λ_Sxu W \n S_xl = Λ_Sxl W \n Λ_Su h  0 \n Λ_Sl h  0 \n Λ_Sxu h  0 \n Λ_Sxl h  0 \n Λ_Su  0 \n Λ_Sl  0 \n Λ_Sxu  0 \n Λ_Sxl  0\nendarray","category":"page"},{"location":"math/#Derivation-of-dual-LDR-problem","page":"Mathematical formulations","title":"Derivation of dual LDR problem","text":"","category":"section"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"Imposing a linear decision rule on the constraint multipliers corresponds to a relaxation of the primal problem, where constraints are taken as expectation. For example, the equality constraints become:","category":"page"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"E (A_e x(ξ) - b_e(ξ)) ξ^top  = 0","category":"page"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"If the second-moment matrix M = Eξ ξ^top is invertible, we are justified to search for x(ξ) in the form X ξ, since for every x(ξ) there is an X such that E x(ξ) ξ^top  = X M.","category":"page"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"The slack variables s_cdot(xi) remain scenario-wise positive, and we demand that E s_cdot(ξ) ξ^top  = S_cdot M for the reformulation of the inequality constraints. Again using duality, this leads to the constraints (W - h e_0^top) M S_cdot^top  0, where e_0 is the first canonical vector.","category":"page"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"Then, the reformulation of the dual LDR problem is","category":"page"},{"location":"math/","page":"Mathematical formulations","title":"Mathematical formulations","text":"beginarrayrl\nmin   texttrBig( (C^top X + X^top Q X) Eξ ξ^top Big) + r 05ex\ntextst  A_e X = B_e \n A_u X + S_u = B_u \n A_l X - S_l = B_l \n X + S_xu = X_u \n X - S_xl = X_l \n (W - h e_0^top) M S_u^top  0 \n (W - h e_0^top) M S_l^top  0 \n (W - h e_0^top) M S_xu^top  0 \n (W - h e_0^top) M S_xl^top  0\nendarray","category":"page"},{"location":"#LinearDecisionRules.jl-Documentation","page":"LinearDecisionRules.jl Documentation","title":"LinearDecisionRules.jl Documentation","text":"","category":"section"},{"location":"","page":"LinearDecisionRules.jl Documentation","title":"LinearDecisionRules.jl Documentation","text":"CurrentModule = LinearDecisionRules","category":"page"},{"location":"","page":"LinearDecisionRules.jl Documentation","title":"LinearDecisionRules.jl Documentation","text":"The LinearDecisionRules.jl package provides a simple JuMP abstraction to represent decision rules in (stochastic) optimization problems as linear functions of random variables.","category":"page"},{"location":"","page":"LinearDecisionRules.jl Documentation","title":"LinearDecisionRules.jl Documentation","text":"The problems the package deals with are of the form","category":"page"},{"location":"","page":"LinearDecisionRules.jl Documentation","title":"LinearDecisionRules.jl Documentation","text":"beginarrayrl\nmin   E c(ξ)^top x(ξ) + x(ξ)^top Q x(ξ) + r  05ex\ntextst  A_e x(ξ) = b_e(ξ) \n A_u x(ξ)  b_u(ξ) \n A_l x(ξ)  b_l(ξ) \n x(ξ)  x_u \n x(ξ)  x_l \n x_i(ξ) text is non-anticipative for  i  I \n  ξ  Ξ\nendarray","category":"page"},{"location":"","page":"LinearDecisionRules.jl Documentation","title":"LinearDecisionRules.jl Documentation","text":"where Ξ  ℝ^m is a polytope described by","category":"page"},{"location":"","page":"LinearDecisionRules.jl Documentation","title":"LinearDecisionRules.jl Documentation","text":"beginalign*\nΞ  =  ξ = (1 η)  ℝ^m mid W_u η  h_u W_l η  h_l lb  η  ub  \n =  ξ  ℝ^m mid W ξ  h \nendalign*","category":"page"},{"location":"","page":"LinearDecisionRules.jl Documentation","title":"LinearDecisionRules.jl Documentation","text":"Variable η cannot appear in equality constraints, since the linear span of Ξ must be all of ℝ^m.","category":"page"},{"location":"","page":"LinearDecisionRules.jl Documentation","title":"LinearDecisionRules.jl Documentation","text":"Non-anticipative variables are not allowed to depend on the random variable ξ. This is enforced by fixing their decision rules to have coefficient equal to zero, except for the constant term.","category":"page"},{"location":"#Example","page":"LinearDecisionRules.jl Documentation","title":"Example","text":"","category":"section"},{"location":"","page":"LinearDecisionRules.jl Documentation","title":"LinearDecisionRules.jl Documentation","text":"Consider the following classical \"Newsvendor\" problem:","category":"page"},{"location":"","page":"LinearDecisionRules.jl Documentation","title":"LinearDecisionRules.jl Documentation","text":"A retailer must decide how many units of a product to buy (at a cost of $10).\nThe demand is uniformly distributed between 80 and 120 units, and unavailable at buying time; units are sold for $12.\nLeftover units can be returned (for $8).","category":"page"},{"location":"","page":"LinearDecisionRules.jl Documentation","title":"LinearDecisionRules.jl Documentation","text":"This leads to the following optimization problem:","category":"page"},{"location":"","page":"LinearDecisionRules.jl Documentation","title":"LinearDecisionRules.jl Documentation","text":"beginarrayrl\nmax   - 10 cdot textbuy + E  8 cdot textreturn + 12 cdot textsell 05ex\ntextst  textsell(ξ) + textreturn(ξ)  textbuy \n textsell(ξ)  textdemand(ξ) \n 0  textsell(ξ) textreturn(ξ) textbuy\nendarray","category":"page"},{"location":"","page":"LinearDecisionRules.jl Documentation","title":"LinearDecisionRules.jl Documentation","text":"where we indicate that buy is a first-stage decision, and sell and return are second-stage decisions, depending on the scenario xi that fixes value of the random variable demand.","category":"page"},{"location":"","page":"LinearDecisionRules.jl Documentation","title":"LinearDecisionRules.jl Documentation","text":"using JuMP\nusing LinearDecisionRules\nusing HiGHS\nusing Distributions\n\nbuy_cost = 10\nreturn_value = 8\nsell_value = 12\n\ndemand_max = 120\ndemand_min = 80\n\nldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)\nset_silent(ldr)\n\n@variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)\n@variable(ldr, sell >= 0)\n@variable(ldr, ret >= 0)\n@variable(ldr, demand, LinearDecisionRules.Uncertainty,\n    distribution = Uniform(demand_min, demand_max)\n)\n\n@constraint(ldr, sell + ret <= buy)\n@constraint(ldr, sell <= demand)\n\n@objective(ldr, Max,\n    - buy_cost * buy\n    + return_value * ret\n    + sell_value * sell\n)\n\noptimize!(ldr)\n\n@show objective_value(ldr)\n@show LinearDecisionRules.get_decision(ldr, buy)\n@show objective_value(ldr, dual = true)\n@show LinearDecisionRules.get_decision(ldr, buy, dual = true)","category":"page"},{"location":"","page":"LinearDecisionRules.jl Documentation","title":"LinearDecisionRules.jl Documentation","text":"get_decision","category":"page"},{"location":"#LinearDecisionRules.get_decision","page":"LinearDecisionRules.jl Documentation","title":"LinearDecisionRules.get_decision","text":"get_decision(m, x, η; dual = false)\n\nCoefficient of η in the LDR of x\n\n\n\n\n\nget_decision(m, x; dual = false)\n\nConstant term in the LDR of x\n\n\n\n\n\n","category":"function"}]
}