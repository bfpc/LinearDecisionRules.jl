# Uniform measurements of the three spike designs (audit).
#   julia --project=spike spike/audit/metrics.jl            # runs the three designs, each in a fresh process
#   julia --project=spike spike/audit/metrics.jl A|B|C      # one design
# Measures: (a) code lines, (b) methods added to functions owned by other modules (reflection), (c) new types,
# (d) list of foreign methods with file:line (input to the by-inspection forwarding count), (e) Aqua piracies and
# ambiguities on a throw-away package made from the unmodified design file (same trick for all three).
using Test

const SPIKE = dirname(@__DIR__)
const DESIGNS = Dict(
    "A" => (name = "LDRWrapper", file = joinpath(SPIKE, "wrapper", "LDRWrapper.jl")),
    "B" => (name = "LDROptimizer", file = joinpath(SPIKE, "optimizer", "LDROptimizer.jl")),
    "C" => (name = "LDRHook", file = joinpath(SPIKE, "hook", "LDRHook.jl")),
)

# ------------------------------------------------------------------ (a) lines
function count_lines(file)
    total = blank = comment = doc = code = 0
    lines = readlines(file)
    in_doc = false
    for (i, l) in enumerate(lines)
        s = strip(l)
        total += 1
        if in_doc
            doc += 1
            occursin("\"\"\"", s) && (in_doc = false)
        elseif startswith(s, "\"\"\"")
            doc += 1
            count("\"\"\"", s) == 1 && (in_doc = true)
        elseif isempty(s)
            blank += 1
        elseif startswith(s, "#")
            comment += 1
        elseif occursin(r"^\"[^\"]*\"$", s)          # one-line docstring "..." on its own line
            doc += 1
        else
            code += 1
        end
    end
    return (; total, blank, comment, doc, code)
end

# ------------------------------------------------------------------ (b), (c) reflection
function all_modules()
    seen = Set{Module}()
    stack = Module[Main, Base, Core]
    append!(stack, collect(values(Base.loaded_modules)))
    while !isempty(stack)
        M = pop!(stack)
        M in seen && continue
        push!(seen, M)
        for n in names(M; all = true)
            isdefined(M, n) || continue
            v = try getfield(M, n) catch; continue end
            v isa Module && v !== M && !(v in seen) && push!(stack, v)
        end
    end
    return seen
end

within(m::Module, D::Module) = m === D || (parentmodule(m) !== m && within(parentmodule(m), D))

function foreign_methods(D::Module)
    objs = IdDict{Any,Nothing}()
    for M in all_modules()
        M === D && continue          # nested modules (B and C embed their own SpikeCore copy) count as foreign owners
        for n in names(M; all = true)
            isdefined(M, n) || continue
            v = try getfield(M, n) catch; continue end
            (v isa Function || (v isa Type && Base.unwrap_unionall(v) isa DataType)) || continue
            objs[v] = nothing
        end
    end
    objs[Core.kwcall] = nothing
    out = Tuple{String,String,Method}[]           # (owner module, function name, method)
    for f in keys(objs)
        owner = f isa Type ? parentmodule(Base.unwrap_unionall(f)) : parentmodule(f)
        owner === D && continue
        ms = try methods(f) catch; continue end
        for m in ms
            m.module === D || continue
            push!(out, (string(owner), string(f isa Type ? nameof(Base.unwrap_unionall(f)) : nameof(f)), m))
        end
    end
    return unique(p -> p[3], out)
end

function new_types(D::Module)
    concrete, abstract_, aliases = String[], String[], String[]
    for n in names(D; all = true)
        startswith(string(n), "#") && continue
        isdefined(D, n) || continue
        v = getfield(D, n)
        v isa Type || continue
        u = Base.unwrap_unionall(v)
        if u isa DataType && parentmodule(u) === D && nameof(u) === n
            push!(isabstracttype(u) ? abstract_ : concrete, string(n))
        else
            push!(aliases, string(n))
        end
    end
    return (; concrete, abstract_, aliases)
end

# ------------------------------------------------------------------ (e) Aqua on a throw-away package
function aqua_package(name, file)
    root = mktempdir()
    dir = joinpath(root, name)
    mkpath(joinpath(dir, "src"))
    uuid = Base.UUID(hash(name) % UInt128 | UInt128(1) << 64)   # deterministic, distinct per design
    open(joinpath(dir, "Project.toml"), "w") do io
        println(io, "name = \"$name\"\nuuid = \"$uuid\"\n[deps]")
        println(io, "Distributions = \"31c24e10-a181-5473-b8eb-7969acd0382f\"")
        println(io, "JuMP = \"4076af6c-e467-56ae-b986-b466b2749572\"")
        println(io, "LinearAlgebra = \"37e2e46d-f89d-539d-b4ee-838fcccc9c8e\"")
        println(io, "MathOptInterface = \"b8f27783-ece8-5eb3-8dc8-9495eed66fee\"")
    end
    # the design file is included unmodified; core/ is reached through its own relative include
    write(joinpath(dir, "src", "$name.jl"), "Base.include(Base.__toplevel__, $(repr(file)))\n")
    push!(LOAD_PATH, root)
    return Base.require(Base.PkgId(uuid, name))
end

function run_one(key)
    d = DESIGNS[key]
    println("=================== design $key: $(d.name) ===================")
    L = count_lines(d.file)
    println("LINES total=$(L.total) blank=$(L.blank) comment=$(L.comment) docstring=$(L.doc) code=$(L.code)")
    # Aqua first, on the package copy (before Main.SpikeCore exists, which design A would otherwise pick up)
    P = aqua_package(d.name, d.file)
    for (label, f) in (("test_piracies", () -> Main.Aqua.test_piracies(P)),
                       ("test_ambiguities", () -> Main.Aqua.test_ambiguities(P)))
        ok = try
            @testset "$label" begin f() end
            "PASS"
        catch e
            "FAIL (" * first(replace(sprint(showerror, e), "\n" => " "), 200) * ")"
        end
        println("AQUA $label: $ok")
    end
    # load the design exactly as its checklist does (A needs Main.SpikeCore first)
    key == "A" && Base.include(Main, joinpath(SPIKE, "core", "SpikeCore.jl"))
    Base.include(Main, d.file)
    D = Base.invokelatest(getfield, Main, Symbol(d.name))
    @eval Main using JuMP, HiGHS, Distributions   # make sure every module a user would load is present
    Base.invokelatest(analyze, D)
end

function analyze(D::Module)
    fm = foreign_methods(D)
    kw = filter(p -> p[2] == "kwcall", fm)
    fm = filter(p -> p[2] != "kwcall", fm)
    byowner = Dict{String,Int}()
    for (o, _, _) in fm
        byowner[o] = get(byowner, o, 0) + 1
    end
    nfun = length(unique((o, f) for (o, f, _) in fm))
    println("FOREIGN_METHODS total=$(length(fm)) functions=$nfun  (+ $(length(kw)) Core.kwcall keyword-sorter methods, not counted)")
    for (o, k) in sort!(collect(byowner); by = p -> -p[2])
        println("  OWNER $o => $k")
    end
    own = 0
    for n in names(D; all = true)
        isdefined(D, n) || continue
        f = getfield(D, n)
        (f isa Function && parentmodule(f) === D) || continue
        own += count(m -> m.module === D, methods(f))
    end
    println("OWN_FUNCTION_METHODS $own")
    T = new_types(D)
    println("TYPES concrete=$(length(T.concrete)) abstract=$(length(T.abstract_)) aliases_or_unions=$(length(T.aliases))")
    println("  concrete: ", join(T.concrete, ", "))
    println("  abstract: ", join(T.abstract_, ", "))
    println("  aliases/unions/foreign: ", join(T.aliases, ", "))
    println("FOREIGN METHOD LIST (owner.function  file:line)")
    for (o, f, m) in sort(fm; by = p -> (p[1], p[2], m_line(p[3])))
        println("  $o.$f  L$(m.line)  $(m.sig)")
    end
    # ambiguities, Test.detect_ambiguities on the included module (pairs involving a design method)
    mods = Module[D, Base, Core]
    for M in all_modules()
        nameof(M) in (:JuMP, :MathOptInterface, :Distributions) && parentmodule(M) === M && push!(mods, M)
    end
    amb = filter(p -> p[1].module === D || p[2].module === D, Test.detect_ambiguities(mods...; recursive = true))
    println("DETECT_AMBIGUITIES pairs involving the design: $(length(amb))")
    foreach(p -> println("  AMB ", p[1], "\n      ", p[2]), amb)
end
m_line(m) = m.line

if isempty(ARGS)
    for key in ("A", "B", "C")
        run(`$(Base.julia_cmd()) --project=$(SPIKE) $(@__FILE__) $key`)
    end
else
    @eval Main using Aqua
    run_one(ARGS[1])
end
