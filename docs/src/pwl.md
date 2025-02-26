# Piecewise linear lifts

If we generalize the linear decision rule to a piecewise linear decision rule, we can get a better approximation of the optimal decision rule.

This implies some changes in the structure of the problem.
For simplicity, we will assume that we have a single coordinate of the uncertainty $η$.

## Breakpoints and modification of the uncertainty polyhedron $Ξ$

We will assume that the uncertainty $η$ is a scalar random variable, and that the uncertainty polyhedron $\Xi$ is a segment ${1} \times [η_{\min}, η_{\max}]$.
We assume the breakpoints are given by $η_0 = η_{\min} < η_1 < \dots < η_k = η_{\max}$.
The lengths of the segments are $Δ_i = η_i - η_{i-1}$.

The random variable $\eta$ will be the sum of its components along the segments, i.e.,
```math
\eta = \eta_{\min} + \tilde{\eta}_1 + \dots + \tilde{\eta}_k = [\eta_{\min} \ 1 \cdots \ 1] \cdot \tilde{\eta} = \eta_{\min} + e^\top \tilde{\eta},
```
where $\tilde{\eta} = [\tilde{\eta}_1; \dots; \tilde{\eta}_k]$ is the lifted vector corresponding to $\eta$ and $e = [1; \dots; 1]$.

So a previous constraint of the form $W_u \eta \leq h_u$ becomes $W_u (\eta_{\min} + e^\top \tilde{\eta}) \leq h_u$, which defines a matrix $\tilde{W}_u = W_u e^\top$ and modifies the right-hand side vector to $\tilde{h}_u = h_u - W_u \eta_{\min}$.
In this example, $\eta$ is a 1-dimensional random variable, so $W_u$ is a column vector.
If we were lifting several variables, this expansion of the coefficient matrix would have to be performed column by column.

Furthermore, we observe that the lifted vector $\tilde{\eta}$ describes a piecewise linear path, whose vertices are
$(0, 0, \ldots, 0)$, $(\Delta_1, 0, \ldots, 0)$, $(\Delta_1, \Delta_2, \ldots, 0)$, $\ldots$, $(\Delta_1, \Delta_2, \ldots, \Delta_k)$.
The convex hull of these vertices is given by the inequalities
```math
\begin{align*}
0 \leq \frac{\tilde{\eta}_k}{Δ_k} \leq \dots \leq \frac{\tilde{\eta}_2}{Δ_2} \leq \frac{\tilde{\eta}_1}{Δ_1} \leq 1,
\end{align*}
```
which are to be added to the constraints defining the lifted uncertainty polyhedron $\tilde{\Xi}$.

## Modification of decisions and constraints

We must rewrite the decisions constraints in terms of the lifted variable $\tilde{\xi} = [1; \tilde{\eta}]$.
The PWL decision rule $x(\xi) = X \xi$ will be replaced by $x(\xi) = \tilde{X} \tilde{\xi}$, which poses no additionnal difficulty.

The data terms ($b_e$, ...) are already exact linear functions of $\xi$, so their transformation is similar to the one for the polyhedron constraints.
For example, right-hand side $b_e(ξ) = B_e \xi$ becomes:
```math
\begin{align*}
b_e(ξ) = B_e \xi & = [B_{e,0} \ B_{e,\eta}] [1; \eta] \\
& = B_{e,0} + B_{e,\eta}\eta \\
& = B_{e,0} + B_{e,\eta}(\eta_{\min} + e^\top \tilde{\eta}) \\
& = [(B_{e,0} + B_{e,\eta}\eta_{\min}) \ \ B_{e,\eta} e^\top] [1; \tilde{\eta}]
\end{align*}
```
and therefore the equality constraint $A_e x(ξ) = b_e(ξ)$ becomes
```math
\begin{align*}
A_e \tilde{X} = [(B_{e,0} + B_{e,\eta}\eta_{\min}) \ \ B_{e,\eta} e^\top].
\end{align*}
```

Again, this expansion of the coefficient matrix would have to be performed column by column if we were lifting several variables.

## Second-moment matrix

We need to rewrite the second-moment matrix $E[ξ ξ^\top]$, which is now $E[\tilde{\xi} \tilde{\xi}^\top]$.
We must take into account that the coordinates of $\tilde{\eta}$ are not independent.

### Uniform variable with breakpoints

If $\eta$ is uniformly distributed on the segment $[η_{\min}, η_{\max}]$, and its lifted vector is $\tilde{\eta} = [\tilde{\eta}_1; \dots; \tilde{\eta}_k]$, then the $(i,j)$ entry in the second-moment matrix is given by
```math
m_{i,j} = E[\tilde{\eta}_i \tilde{\eta}_j]
= \int_0^\Delta \min(\Delta_i, \max(0, x - \eta_{i-1})) \min(\Delta_j, \max(0, x - \eta_{j-1})) \, dx.
```

For $i < j$, these integrals can be split in three parts:
- for $x < \eta_{j-1}$, the integrand is zero;
- for $\eta_{j-1} \leq x < \eta_j$, the integrand is $\Delta_i (x - \eta_{j-1})$, so we have $\int_{\eta_{j-1}}^{\eta_j} \Delta_i (x - \eta_{j-1}) \ \rho(x) dx = \Delta_i E_{\text{truncated }\eta}[x - \eta_{j-1}] \cdot P[\eta_{j-1} < \eta < \eta_j]$;
- for $\eta_j \leq x < \eta_{\max}$, the integrand is $\Delta_i \Delta_j$, and its contribution is $\Delta_i \Delta_j \cdot P[\eta > \eta_{j}]$.

When $i = j$, the integrand for $x \in [\eta_{i-1}, \eta_i]$ is $(x - \eta_{i-1})^2$, so we have: $\int_{\eta_{j-1}}^{\eta_j} (x - \eta_{i-1})^2 \ \rho(x) dx = E_{\text{truncated }\eta}[(x - \eta_{j-1})^2] \cdot P[\eta_{j-1} < \eta < \eta_j]$

The other two are still given by the same formula as above.

### Packages

We rely on the `truncated(dist, a, b)` mechanism of [`Distributions.jl`](https://juliastats.org/Distributions.jl/stable/truncate) to generate the truncations, and on the `expectation` numerical integration from the [`Expectations`](https://quantecon.github.io/Expectations.jl/dev/) package for integration.
Specific distributions (`Uniform` and `Normal`) have mean and variance directly calculated by the `Distributions.jl` package.

The integrals are performed with [`FastGaussQuadrature.jl`](https://juliaapproximation.github.io/FastGaussQuadrature.jl/stable/), which implements several Gauss quadrature rules, which are dispatched by the `Expectations` package with a reasonable setting.
In principle, we could override the use of truncation + expectation by directly using the Gauss quadrature rules for numerical integration.
