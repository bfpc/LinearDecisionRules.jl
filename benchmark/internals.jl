# The internals of the package that the benchmarks read: the cache of inner problems of the
# optimizer (`src/moi/inner_cache.jl`), and the choice of the rescaling of the pieces of liftings
# (`src/formulations/rescaling.jl`). They are not public and may change without notice; this file
# is the only place in `benchmark/` that reads them, so that a change breaks it alone.

import JuMP
import LinearDecisionRules as LDR

_cache(model) = JuMP.unsafe_backend(model).inner_cache

# The number of inner problems that `model` has built, not reused, since it was created.
num_builds(model) = _cache(model).num_builds

# Drop the inner problems of `model`, so that the next `optimize!` builds them all again. The count
# of `num_builds` is kept.
function empty_cache!(model)
    cache = _cache(model)
    empty!(cache.problems)
    cache.fingerprint = nothing
    return
end

# What the inner problems of the last `optimize!` of `model` depend on, except the values of the
# parameters: the canonical form, the moments and the settings.
fingerprint(model) = _cache(model).fingerprint

# Whether the inner problems built for the fingerprint `a` can be reused for `b`.
same_problem(a, b) = LDR._same_problem(a, b)

# How the next `optimize!` rescales the pieces of the liftings of breakpoints, with
# `RescaleUncertainties()`: `:per_piece` (the default), `:before_lifting` or `:hybrid` (see
# `_LIFTED_RESCALING`). It holds for every model until it is set again.
set_lifted_rescaling!(mode::Symbol) = (LDR._LIFTED_RESCALING[] = mode; nothing)
