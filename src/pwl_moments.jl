using Distributions: cdf, ccdf, truncated
using Expectations: expectation

# TODO: If dist is TruncatedNormal or Uniform, then we don't need to use
# expectation, since Distributions support mean and var for those;, but
# at least this is a uniform interface
function second_moment_matrix(dist, breakpoints::Vector)
    d = length(breakpoints)
    Deltas = diff(breakpoints)

    M = zeros(d, d)

    # The first entry is just 1
    M[1, 1] = 1

    # The first row and column are the expectations
    for i in 1:d-1
        td = truncated(dist, breakpoints[i], breakpoints[i+1])
        E = expectation(td)
        prob = cdf(dist, breakpoints[i+1]) - cdf(dist, breakpoints[i])
        I1 = E(x -> x - breakpoints[i]) * prob

        I2 = Deltas[i] * ccdf(dist, breakpoints[i+1])
        M[1, i+1] = I1 + I2
        M[i+1, 1] = M[1, i+1]
    end

    for i in 1:d-1
        for j in i:d-1
            # We split the integral in 3 terms
            # The first term is from -Inf to the second breakpoint and is zero
            # The second term depends on whether we're on the diagonal or not
            if i == j
                td = truncated(dist, breakpoints[i], breakpoints[i+1])
                E = expectation(td)
                prob = cdf(dist, breakpoints[i+1]) - cdf(dist, breakpoints[i])
                I1 = E(x -> (x - breakpoints[i])^2) * prob
            else
                td = truncated(dist, breakpoints[j], breakpoints[j+1])
                E = expectation(td)
                prob = cdf(dist, breakpoints[j+1]) - cdf(dist, breakpoints[j])
                I1 = Deltas[i] * E(x -> (x - breakpoints[j])) * prob
            end

            # The third term is from the last breakpoint to Inf
            I2 = Deltas[i] * Deltas[j] * ccdf(dist, breakpoints[j+1])

            M[i+1, j+1] = I1 + I2
            M[j+1, i+1] = M[1+i, 1+j]
        end
    end

    return M
end

