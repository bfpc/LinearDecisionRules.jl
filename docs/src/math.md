# Mathematical formulations

We follow the scheme from [Primal and dual linear decision rules in stochastic and robust optimization, by Kuhn, Wiesemann and Georghiou](https://link.springer.com/article/10.1007/s10107-009-0331-4).

## Derivation of primal LDR problem

We start from
```math
\begin{array}{rl}
\min \ & E[ c(ξ)^\top x(ξ) + x(ξ)^\top Q x(ξ) + r ] \\[0.5ex]
\text{s.t.} & A_e x(ξ) = b_e(ξ) \\
& A_u x(ξ) ≤ b_u(ξ) \\
& A_l x(ξ) ≥ b_l(ξ) \\
& x(ξ) ≤ x_u \\
& x(ξ) ≥ x_l \\
& x_i(ξ) \text{ is non-anticipative, for } i ∈ I \\
& ∀ ξ ∈ Ξ
\end{array}
```
where $Ξ ⊂ ℝ^m$ is a polytope described by
```math
\begin{align*}
Ξ & = \{\, ξ = (1, η) ∈ ℝ^m \mid W_u η ≤ h_u, W_l η ≥ h_l, lb ≤ η ≤ ub \,\} \\
& = \{\, ξ ∈ ℝ^m \mid W ξ ≥ h \,\}.
\end{align*}
```
Recall that the linear span of $Ξ$ must be all of $ℝ^m$.

We introduce positive slack variables for the inequality constraints, so that the problem can be written as
```math
\begin{array}{rl}
\min \ & E[ c(ξ)^\top x(ξ) + x(ξ)^\top Q x(ξ) + r ] \\[0.5ex]
\text{s.t.} & A_e x(ξ) = b_e(ξ) \\
& A_u x(ξ) + s_u(ξ) = b_u(ξ) \\
& A_l x(ξ) - s_l(ξ) = b_l(ξ) \\
& x(ξ) + s_{x,u}(ξ) = x_u \\
& x(ξ) - s_{x,l}(ξ) = x_l \\
& s_u(ξ) ≥ 0 \\
& s_l(ξ) ≥ 0 \\
& s_{x,u}(ξ) ≥ 0 \\
& s_{x,l}(ξ) ≥ 0 \\
& ∀ ξ ∈ Ξ
\end{array}
```

Assuming a linear decision rule $x(ξ) = X ξ$, etc, and that scenario-dependent data $c(ξ)$, $b_u(ξ)$, $b_l(ξ)$ can also be transformed to a linear form, we write the problem as
```math
\begin{array}{rl}
\min \ & E[ ξ^\top C^\top X ξ + ξ^\top X^\top Q X ξ + r ] \\[0.5ex]
\text{s.t.} & A_e X ξ = B_e ξ \\
& A_u X ξ + S_u ξ = B_u ξ \\
& A_l X ξ - S_l ξ = B_l ξ \\
& X ξ + S_{x,u} ξ = X_u ξ \\
& X ξ - S_{x,l} ξ = X_l ξ \\
& S_u ξ ≥ 0 \\
& S_l ξ ≥ 0 \\
& S_{x,u} ξ ≥ 0 \\
& S_{x,l} ξ ≥ 0 \\
& ∀ ξ ∈ Ξ
\end{array}
```
where $X_u = [x_u; 0]$ and $X_l = [x_l; 0]$, since the first component of $ξ$ is equal to $1$ by definition.

Since the linear span of $Ξ$ is all of $ℝ^m$, the equalities must hold for all $ξ$, so they can be replaced by equalities between the corresponding matrices.
Moreover, applying the trace trick to the objective function, we obtain
```math
\begin{array}{rl}
\min \ & \text{tr}\Big( (C^\top X + X^\top Q X) E[ξ ξ^\top] \Big) + r \\[0.5ex]
\text{s.t.} & A_e X = B_e \\
& A_u X + S_u = B_u \\
& A_l X - S_l = B_l \\
& X + S_{x,u} = X_u \\
& X - S_{x,l} = X_l \\
& S_u ξ ≥ 0 \\
& S_l ξ ≥ 0 \\
& S_{x,u} ξ ≥ 0 \\
& S_{x,l} ξ ≥ 0 \\
& ∀ ξ ∈ Ξ
\end{array}
```

The non-negativity constraints are dealt with by duality.
If $S ξ ≥ 0$ for all $ξ$ in $Ξ = \{\, ξ ∈ ℝ^m \mid W ξ ≥ h \,\}$, then there exists a matrix $Λ$ such that $S = Λ W$, $Λ ≥ 0$ and $Λ h ≥ 0$.
Therefore, we introduce the matrices $Λ_{S,u}$, $Λ_{S,l}$, $Λ_{S,x,u}$, $Λ_{S,x,l}$ and obtain the equivalent problem
```math
\begin{array}{rl}
\min \ & \text{tr}\Big( (C^\top X + X^\top Q X) E[ξ ξ^\top] \Big) + r \\[0.5ex]
\text{s.t.} & A_e X = B_e \\
& A_u X + S_u = B_u \\
& A_l X - S_l = B_l \\
& X + S_{x,u} = X_u \\
& X - S_{x,l} = X_l \\
& S_u = Λ_{S,u} W \\
& S_l = Λ_{S,l} W \\
& S_{x,u} = Λ_{S,x,u} W \\
& S_{x,l} = Λ_{S,x,l} W \\
& Λ_{S,u} h ≥ 0 \\
& Λ_{S,l} h ≥ 0 \\
& Λ_{S,x,u} h ≥ 0 \\
& Λ_{S,x,l} h ≥ 0 \\
& Λ_{S,u} ≥ 0 \\
& Λ_{S,l} ≥ 0 \\
& Λ_{S,x,u} ≥ 0 \\
& Λ_{S,x,l} ≥ 0
\end{array}
```

## Derivation of dual LDR problem

Imposing a linear decision rule on the constraint multipliers corresponds to a relaxation of the primal problem, where constraints are taken as expectation.
For example, the equality constraints become:
```math
E[ (A_e x(ξ) - b_e(ξ)) ξ^\top ] = 0.
```

If the second-moment matrix $M = E[ξ ξ^\top]$ is invertible, we are justified to search for $x(ξ)$ in the form $X ξ$, since for every $x(ξ)$ there is an $X$ such that $E[ x(ξ) ξ^\top ] = X M$.
Then, the reformulation of the dual LDR problem is
```math
\begin{array}{rl}
\min \ & \text{tr}\Big( (C^\top X + X^\top Q X) E[ξ ξ^\top] \Big) + r \\[0.5ex]
\text{s.t.} & A_e X = B_e \\
& A_u X + S_u = B_u \\
& A_l X - S_l = B_l \\
& X + S_{x,u} = X_u \\
& X - S_{x,l} = X_l \\
& (W - h e_0^\top) M S_u^\top ≥ 0 \\
& (W - h e_0^\top) M S_l^\top ≥ 0 \\
& (W - h e_0^\top) M S_{x,u}^\top ≥ 0 \\
& (W - h e_0^\top) M S_{x,l}^\top ≥ 0
\end{array}
```
where $e_0$ is the first canonical vector.
