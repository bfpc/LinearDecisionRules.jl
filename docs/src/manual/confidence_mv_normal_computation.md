# ConfidenceMvNormal: math vs implementation

The [mathematical derivation](ConfidenceNormal.md) expresses the ellipsoid radius and
covariance scaling in terms of the **incomplete gamma function**.
The implementation uses **chi-squared CDF/quantile functions** from
`Distributions.jl`.
This page shows the two formulations are identical.

## The linking identity

The CDF of the chi-squared distribution with $k$ degrees of freedom evaluated
at $x \geq 0$ is

```math
F_{\chi^2(k)}(x) = \frac{\gamma(k/2,\; x/2)}{\Gamma(k/2)},
```

where $\gamma(a, x) = \int_0^x u^{a-1} e^{-u}\,du$ is the lower incomplete
gamma function.  All equivalences below follow from this identity.

---

## Ellipsoid radius $\rho$

**Derivation** (from `ConfidenceNormal.md`)

Require the ball $B(0,\rho)$ to contain fraction $\alpha$ of the standard
$d$-dimensional normal mass.  A polar change of variables gives:

```math
\alpha = \frac{\int_0^\rho t^{d-1} e^{-t^2/2}\,dt}{\int_0^\infty t^{d-1} e^{-t^2/2}\,dt}.
```

Substituting $u = t^2/2$ turns both integrals into incomplete gammas:

```math
\alpha = \frac{\gamma(d/2,\; \rho^2/2)}{\Gamma(d/2)},
```

so $\rho$ is the solution of $\gamma(d/2, \rho^2/2) = \alpha\,\Gamma(d/2)$.

**Implementation**

```julia
ρ² = Distributions.quantile(Distributions.Chisq(d), α)
```

**Equivalence**

`quantile(Chisq(d), α)` returns the value $x$ satisfying
$F_{\chi^2(d)}(x) = \alpha$, i.e.

```math
\frac{\gamma(d/2,\; x/2)}{\Gamma(d/2)} = \alpha
\quad\Longleftrightarrow\quad
\gamma(d/2,\; x/2) = \alpha\,\Gamma(d/2).
```

Setting $x = \rho^2$ recovers the derivation exactly.

---

## Covariance scaling

**Derivation** (from `ConfidenceNormal.md`)

The covariance of the truncated distribution is $C_{d,\alpha} = s \cdot \Sigma$
where

```math
s = \frac{1}{\alpha} \cdot \frac{\int_0^\rho t^{d+1} e^{-t^2/2}\,dt}
                                  {\int_0^\infty t^{d+1} e^{-t^2/2}\,dt}.
```

The same substitution $u = t^2/2$ gives:

```math
\int_0^\rho t^{d+1} e^{-t^2/2}\,dt = 2^{d/2}\,\gamma\!\left(\tfrac{d}{2}+1,\; \tfrac{\rho^2}{2}\right),
\qquad
\int_0^\infty t^{d+1} e^{-t^2/2}\,dt = 2^{d/2}\,\Gamma\!\left(\tfrac{d}{2}+1\right),
```

so

```math
s = \frac{1}{\alpha} \cdot \frac{\gamma(d/2+1,\; \rho^2/2)}{\Gamma(d/2+1)}.
```

**Implementation**

```julia
scaling = Distributions.cdf(Distributions.Chisq(d + 2), ρ²) / α
```

**Equivalence**

Applying the linking identity with $k = d+2$:

```math
F_{\chi^2(d+2)}(\rho^2) = \frac{\gamma((d+2)/2,\; \rho^2/2)}{\Gamma((d+2)/2)}
= \frac{\gamma(d/2+1,\; \rho^2/2)}{\Gamma(d/2+1)}.
```

Dividing by $\alpha$ gives exactly $s$.

---

## Summary

| Quantity | Math document | Code |
|----------|--------------|------|
| Radius $\rho$ | solve $\gamma(d/2, \rho^2/2) = \alpha\,\Gamma(d/2)$ | `quantile(Chisq(d), α)` |
| Cov. scaling $s$ | $\gamma(d/2+1, \rho^2/2)\;/\;(\alpha\,\Gamma(d/2+1))$ | `cdf(Chisq(d+2), ρ²) / α` |

Both columns use the relation $F_{\chi^2(k)}(x) = \gamma(k/2, x/2) / \Gamma(k/2)$
to pass between incomplete-gamma integrals and chi-squared distribution functions.
