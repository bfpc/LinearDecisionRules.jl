# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# The inner problem of a formulation: `solve_formulation`, the build and the solve within a time limit, and the parameters as decisions.

"""
    solve_formulation(
        form::CanonicalForm,
        f::AbstractFormulation,
        mom::Moments,
        optimizer_factory;
        silent::Bool = true,
        time_limit::Union{Nothing,Real} = nothing,
        raw_attributes = Dict{String,Any}(),
        names::Bool = false,
    ) -> FormulationResult

Build the inner problem of the formulation `f` for `form` and the moments `mom`, solve it with
`optimizer_factory` and return a [`FormulationResult`](@ref).

`time_limit` is in seconds, for building and solving the inner problem: the inner optimizer gets
what remains after the build, and with no time left (`time_limit = 0`) the inner problem is not
built and the result has status `TIME_LIMIT`. `raw_attributes` is an iterable of
`name => value` pairs passed to `JuMP.set_attribute` on the inner problem, followed by the
`attributes` of `f` (see [`DualLDR`](@ref)). [`SampledLDR`](@ref) ignores `mom` and uses the
empirical moments of its outcomes.

With `names = true`, the variables of the inner problem are named, so that `write_to_file` is
readable: `X[i,j]` for the rule of decision `i` on column `j` of ξ, and by inequality `r`,
`λ[r,b]` (the multipliers of the linear rows of support block `b`) and `λ[r,b,p]` (those of its
`p`-th piece, a second-order cone), `w[r,j]` (the auxiliary of the closed form on a box, for column
`j`), or `u[r]` and `S[r,j]` (the slack rule). Names slow the build
down, so they are off by default.

## Example

```julia
import LinearDecisionRules as LDR
form = LDR.canonicalize(backend(model), LDR.model_info(backend(model)))
result = LDR.solve_formulation(form, PrimalLDR(), LDR.moments(form), HiGHS.Optimizer)
result.objective_value
```
"""
function solve_formulation(
    form::CanonicalForm,
    f::AbstractFormulation,
    mom::Moments,
    optimizer_factory;
    silent::Bool = true,
    time_limit::Union{Nothing,Real} = nothing,
    raw_attributes = Dict{String,Any}(),
    names::Bool = false,
)
    return _solve!(f, form, mom, time_ns(), time_limit) do
        return _build_inner(form, f, mom, optimizer_factory; silent, raw_attributes, names)
    end
end

# `time_ns` rather than `time`, whose resolution can be 16 ms on Windows.
_seconds_since(start::UInt64) = (time_ns() - start) / 1e9

# The inner problem of a formulation, which the optimizer keeps to solve it again when only the
# values of the parameters change (see `ParameterTerms`).
mutable struct _InnerProblem
    model::JuMP.Model
    # The rule variables of the decisions of the form (see `_rule_variables!`).
    X::Vector{Vector{JuMP.VariableRef}}
    # The variable of `model` fixed to each parameter of `form.parameters`, if it has one.
    parameters::Vector{Union{Nothing,JuMP.VariableRef}}
    # Whether a time limit was set on `model`, to be removed by a solve without one.
    time_limited::Bool
end

function _build_inner(
    form::CanonicalForm,
    f::AbstractFormulation,
    mom::Moments,
    optimizer_factory;
    silent::Bool = true,
    raw_attributes = Dict{String,Any}(),
    names::Bool = false,
)
    model = JuMP.Model(optimizer_factory)
    JuMP.set_string_names_on_creation(model, names)
    if silent
        JuMP.set_silent(model)
    end
    for (name, value) in raw_attributes
        JuMP.set_attribute(model, name, value)
    end
    for (name, value) in _formulation_attributes(f)
        JuMP.set_attribute(model, name, value)
    end
    extended, used = _with_parameter_decisions(form)
    X = _build!(model, f, extended, mom)
    n = length(form.decisions)
    parameters = Vector{Union{Nothing,JuMP.VariableRef}}(nothing, length(form.parameters.values))
    for (q, k) in enumerate(used)
        parameters[k] = only(X[n+q])
    end
    return _InnerProblem(model, X[1:n], parameters, false)
end

# Fix the parameters of `inner` to their values in `form` and solve it. A parameter whose value
# does not change is not set again, so that the solver keeps its solution when nothing changes.
function _solve_inner!(
    inner::_InnerProblem,
    f::AbstractFormulation,
    form::CanonicalForm,
    mom::Moments;
    time_limit::Union{Nothing,Real} = nothing,
)
    model = inner.model
    for (p, value) in zip(inner.parameters, form.parameters.values)
        if p !== nothing && JuMP.fix_value(p) != value
            JuMP.fix(p, value)
        end
    end
    if time_limit !== nothing || inner.time_limited
        JuMP.set_time_limit_sec(model, time_limit)
        inner.time_limited = time_limit !== nothing
    end
    solve_start = time_ns()
    JuMP.optimize!(model)
    return _formulation_result(f, form, mom, model, inner.X, _seconds_since(solve_start))
end

# Solve the inner problem of `f`, built by `build()` unless `inner` is given, within `time_limit`
# seconds from `start` (`nothing` for no limit). The limit covers the build too, which can take
# longer than the solve: past the limit, the problem is not built and the result has status
# `TIME_LIMIT`, and the inner optimizer gets what remains after the build.
function _solve!(
    build::Function,
    f::AbstractFormulation,
    form::CanonicalForm,
    mom::Moments,
    start::UInt64,
    time_limit::Union{Nothing,Real};
    inner::Union{Nothing,_InnerProblem} = nothing,
)
    if time_limit !== nothing && _seconds_since(start) >= time_limit
        return _time_limit_result(f, form, mom)
    end
    if inner === nothing
        inner = build()
    end
    remaining = time_limit === nothing ? nothing : max(0.0, time_limit - _seconds_since(start))
    return _solve_inner!(inner, f, form, mom; time_limit = remaining)
end

# The form that the formulations build when constants depend on parameters (see `ParameterTerms`):
# each such parameter becomes a decision of stage 0, which observes only the constant column and
# is a fixed variable of the inner problem (see `_rule_variables!`). Its column in `A` is `-Bp`, so
# that `A x - Bp p (=, ≤, ≥) b + (B ξ without its constant)`, and its cost in the constant column
# of `C` is `fp`; `B[:, 1]` and `f` are those without the parameters. Returns the form and the
# columns of `ParameterTerms` of the parameters added, in order.
function _with_parameter_decisions(form::CanonicalForm)
    terms = form.parameters
    _has_parameter_terms(terms) || return form, Int[]
    in_use(Bp) = [!isempty(SparseArrays.nzrange(Bp, k)) for k in axes(Bp, 2)]
    in_rows = in_use(terms.Bp_e) .| in_use(terms.Bp_u) .| in_use(terms.Bp_l)
    used = findall(in_rows .| (terms.fp .!= 0))
    q = length(used)
    extend(A, Bp) = hcat(A, -Bp[:, used])
    cost = SparseArrays.sparse(1:q, ones(Int, q), terms.fp[used], q, size(form.C, 2))
    names = ["p[$k]" for k in used]
    extended = _replace(
        form;
        decisions = vcat(form.decisions, terms.parameters[used]),
        decision_names = vcat(form.decision_names, names),
        decision_stages = vcat(form.decision_stages, zeros(Int, q)),
        integer = vcat(form.integer, falses(q)),
        binary = vcat(form.binary, falses(q)),
        lower = vcat(form.lower, fill(-Inf, q)),
        upper = vcat(form.upper, fill(Inf, q)),
        A_e = extend(form.A_e, terms.Bp_e),
        B_e = _with_constant_column(form.B_e, terms.b_e),
        A_u = extend(form.A_u, terms.Bp_u),
        B_u = _with_constant_column(form.B_u, terms.b_u),
        A_l = extend(form.A_l, terms.Bp_l),
        B_l = _with_constant_column(form.B_l, terms.b_l),
        P = SparseArrays.blockdiag(form.P, SparseArrays.spzeros(q, q)),
        C = vcat(form.C, cost),
        f = terms.f,
    )
    return extended, used
end

# `B` with its first column replaced by `b`.
function _with_constant_column(B::_SparseMatrix, b::Vector{Float64})
    I, J, V = SparseArrays.findnz(B[:, 2:end])
    rows = findall(!iszero, b)
    return SparseArrays.sparse(
        [rows; I],
        [ones(Int, length(rows)); J .+ 1],
        [b[rows]; V],
        size(B)...,
    )
end
