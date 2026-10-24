# Solve one MPS file with Ipopt through JuMP (`read_from_file`), in this (fresh) process.
#   julia --project=benchmark/compare benchmark/compare/solve_ipopt.jl file.mps [name=value ...]
# Options are Ipopt options (print_level = 0 and max_cpu_time = 120 by default). The last line is
#   WALL read=<s> run=<s> status=<termination status> obj=<value> iter=<n>
# `run` is the wall time of `optimize!` (copy of the model into Ipopt included, as HiGHS's clock);
# `read` is the time of `read_from_file`, excluded.
using JuMP
import Ipopt

# MOI's MPS reader fails on non-ASCII names (the 0.4 primal has `λ[...]`): read an ASCII copy.
function ascii_copy(path)
    text = read(path, String)
    isascii(text) && return path
    text = replace(text, "λ" => "lam")
    text = replace(text, r"[^\x00-\x7f]" => c -> "_u" * string(Int(first(c))) * "_")
    copy = tempname() * ".mps"
    write(copy, text)
    return copy
end

function main(args)
    path = ascii_copy(args[1])
    t_read = @elapsed model = read_from_file(path)
    path == args[1] || rm(path; force = true)
    set_optimizer(model, Ipopt.Optimizer)
    set_attribute(model, "print_level", 0)
    set_attribute(model, "max_cpu_time", 120.0)
    for a in args[2:end]
        k, v = split(a, "="; limit = 2)
        x = tryparse(Int, v)
        x = x === nothing ? tryparse(Float64, v) : x
        set_attribute(model, String(k), x === nothing ? String(v) : x)
    end
    t_run = @elapsed optimize!(model)
    iter = try
        MOI.get(model, MOI.BarrierIterations())
    catch
        -1
    end
    obj = has_values(model) ? objective_value(model) : NaN
    println("WALL read=$t_read run=$t_run status=$(termination_status(model)) obj=$obj iter=$iter")
end

main(ARGS)
