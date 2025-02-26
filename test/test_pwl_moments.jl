using Distributions
import LinearDecisionRules

function sum_yyt(dist, breakpoints, nsamples)
    d = length(breakpoints)
    M = zeros(d, d)
    for _ in 1:nsamples
        x = rand(dist)
        y = lift(x, breakpoints)
        M .+= y * y'
    end
    return M
end

function second_moment_matrix(dist, breakpoints, nsamples::Int)
    d = length(breakpoints)
    M = zeros(d, d)
    for _ in 1:nsamples
        x = rand(dist)
        y = lift(x, breakpoints)
        M .+= y * y'
    end
    return M / nsamples
end

function second_moment_matrix_pmap(dist, breakpoints, nsamples::Int)
    nw = nworkers()
    thread_sums = pmap(_ -> sum_yyt(dist, breakpoints, nsamples), 1:nw+1)
    return sum(thread_sums) / nsamples / (nw+1)
end

function lift(x, breakpoints)
    d = length(breakpoints)
    Deltas = diff(breakpoints)
    y = zeros(d)
    y[1] = 1
    for i in 1:d-1
        y[i+1] = min(Deltas[i], max(x - breakpoints[i], 0))
    end
    return y
end

println("Symmetric normal, with evenly spaced breakpoints")
n = truncated(Normal(5, 0.5), 3, 7)
breakpoints = 3 .+ [0, 1, 2, 3, 4]
N_samples = 100_000_000

M = LinearDecisionRules.second_moment_matrix(n, breakpoints)
M_sample = second_moment_matrix(n, breakpoints, N_samples)
scaled_err = sqrt(N_samples) * (M_sample .- M)
@show scaled_err


println("Symmetric normal, with unevenly spaced breakpoints")
n = truncated(Normal(5, 0.5), 3, 7)
breakpoints = 3 .+ [0, 1, 2.5, 3.3, 4]
N_samples = 100_000_000

M = LinearDecisionRules.second_moment_matrix(n, breakpoints)
M_sample = second_moment_matrix(n, breakpoints, N_samples)
scaled_err = sqrt(N_samples) * (M_sample .- M)
@show scaled_err


println("Non-symmetric normal, with strange breakpoints")
n = truncated(Normal(5, 0.5), 3, 8)
breakpoints = 3 .+ [0, 1, 3.2, 51]
N_samples = 100_000_000

M = LinearDecisionRules.second_moment_matrix(n, breakpoints)
M_sample = second_moment_matrix(n, breakpoints, N_samples)

err = sqrt(N_samples) * (M_sample .- M)
@show err
