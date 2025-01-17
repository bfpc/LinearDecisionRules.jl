# Calculation for Confidence MvNormal

Suppose we have a normal distribution with mean $\mu$ and covariance $\Sigma$.
The `ConfidenceMvNormal` with parameters `μ`, `Σ` and `α` represents the truncated distribution on an ellipsoid centered at $\mu$, with $0 \leq \alpha \leq 1$ probability mass.

## Standardization

We start by calculating the Cholesky factorization of $\Sigma = LL^\top$.
Then, we standardize the distribution by defining $z = L^{-1}(x - \mu)$.

## Ellipsoid radius

We then need to find $\rho$ such that the mass of $L \cdot B(0, \rho) + \mu$ is equal to $\alpha$.
This is equivalent to finding $\rho$ such that the mass of $B(0, \rho)$ is equal to $\alpha$ under the standard normal $d$-dimensional distribution.
That is, we need to find $\rho$ such that

```math
\alpha 
= \sqrt{(2\pi)}^{-d} \int_{B(0, \rho)}  e^{-\frac{1}{2} \|z\|^2} dz
= \sqrt{(2\pi)}^{-d} \cdot \text{Vol}(S^{d-1}) \int_0^\rho t^{d-1} e^{-t^2/2} dt.
```

Integrating the probability density over the entire space gives
```math
1 = (2\pi)^{-d/2} \int_{\mathbb{R}^d} e^{-\frac{1}{2} \|z\|^2} dz
= (2\pi)^{-d/2} \cdot \text{Vol}(S^{d-1}) \int_0^\infty t^{d-1} e^{-t^2/2} dt,
```
so that $\alpha = \frac{\int_0^\rho t^{d-1} e^{-t^2/2} dt}{\int_0^\infty t^{d-1} e^{-t^2/2} dt}$.

## Covariance matrix

Finally, we must calculate the covariance matrix of the truncated distribution.
It is, by definition:
```math
\frac{1}{\alpha \sqrt{(2\pi)^d \det \Sigma}}\int_{L \cdot B(0, \rho) + \mu} (x - \mu) (x - \mu)^\top e^{-\frac{1}{2} (x - \mu)^\top \Sigma^{-1} (x - \mu)} dx.
```

The standardization yields:
```math
\frac{1}{\alpha \sqrt{(2\pi)^d}}\int_{B(0, \rho)} L z z^\top L^\top e^{-\frac{1}{2} \|z\|^2} dz.
```

And a polar change of variables $z = t \cdot \omega$ leads to:
```math
\frac{1}{\alpha \sqrt{(2\pi)^d}}\int_0^\rho \int_{S^{d-1}} L t^2 \omega \omega^\top L^\top e^{-\frac{1}{2} t^2} t^{d-1} \, d\omega dt
= \frac{\int_0^\rho t^{d+1} e^{-t^2/2}\, dt}{\alpha \sqrt{2\pi}^d} L M_d L^\top.
```
This depends on a matrix $M_d = \int_{S^{d-1}} \omega \omega^\top d\omega$.
It is easy to show that $M_d$ must be a multiple of the identity matrix: an analogous argument shows that the covariance of the standard multivariate normal is $\frac{1}{\sqrt{2\pi}^d} \int_0^\infty t^2 t^{d-1} M_d e^{-t^2/2} dt$.
So $M_d = \sqrt{2\pi}^d \big( \int_0^\infty t^{d+1} e^{-t^2/2}\, dt \big)^{-1} I$.

Putting this all together, we obtain $\displaystyle C_{d,\alpha} = \frac{1}{\alpha} \frac{\int_0^\rho t^{d+1} e^{-t^2/2}\, dt}{\int_0^\infty t^{d+1} e^{-t^2/2}\, dt} L L^\top$.

## Numerical evaluation

To find the radius $\rho$ and the scaling of the covariance matrix, we need a change of variables and the gamma functions
```math
\begin{align*}
\text{``complete''} && \Gamma(a) & = \int_0^\infty u^{a-1} e^{-u} \, du \\
\text{``lower incomplete''} && \gamma(a, x) & = \int_0^x u^{a-1} e^{-u} \, du \\
\text{``upper incomplete''} && \Gamma(a, x) & = \int_x^\infty u^{a-1} e^{-u} \, du .
\end{align*}
```

### Finding $\rho$

We had
$\displaystyle \alpha = \frac{\int_0^\rho t^{d-1} e^{-t^2/2} \, dt}{\int_0^\infty t^{d-1} e^{-t^2/2} \, dt}$.

With $u = t^2/2$, we get $du = t \, dt$ and the integrals become
```math
\begin{align*}
\alpha \cdot \int_0^\infty t^{d-1} e^{-t^2/2} \, dt
& = \int_0^\rho t^{d-1} e^{-t^2/2} \, dt \\
\alpha \cdot \int_0^\infty \sqrt{2u}^{d-2} e^{-u} \, du
& = \int_0^{\rho^2/2} \sqrt{2u}^{d-2} e^{-u} \, du \\
\alpha \cdot \Gamma(d/2) & = \gamma(d/2, \rho^2/2).
\end{align*}
```

So $\rho^2 = 2 \gamma^{-1}(d/2, \alpha \cdot \Gamma(d/2))$.

If $\alpha$ is near $1$, it might be more stable to evaluate $\Gamma^{-1}(d/2, (1 - \alpha) \cdot \Gamma(d/2))$.

### Finding the scaling

We need to evaluate the ratio of integrals in
$\displaystyle C_{d,\alpha} = \frac{1}{\alpha} \frac{\int_0^\rho t^{d+1} e^{-t^2/2}\, dt}{\int_0^\infty t^{d+1} e^{-t^2/2}\, dt} L L^\top$.

The same change of variables leads to
```math
\begin{align*}
\int_0^\rho t^{d+1} e^{-t^2/2}\, dt
& = \int_0^{\rho^2/2} (2u)^{d/2} e^{-u} \, du \\
& = 2^{d/2} \gamma(d/2 + 1, \rho^2/2) \\
\int_0^\infty t^{d+1} e^{-t^2/2}\, dt
& = 2^{d/2} \gamma(d/2 + 1, \infty) = 2^{d/2} \Gamma(d/2 + 1) \end{align*}
```

As $\alpha \cdot \Gamma(d/2) = \gamma(d/2, \rho^2/2)$, using integration by parts, we have an alternative expression for the scaling factor:
```math
\frac{\gamma(d/2+1, \rho^2/2)}{\gamma(d/2, \rho^2/2)} \frac{\Gamma(d/2)}{\Gamma(d/2+1)} = \frac{d/2 - (\rho^2/2)^{d/2}e^{-\rho^2/2}}{d/2}.
```
