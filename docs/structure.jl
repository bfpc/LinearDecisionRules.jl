# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Extracts the structure of `src/` from the source itself: every top-level definition (file, line,
# exported, documented), the names it references, the dependencies between files and between
# layers, the references to files included later (which only late binding makes work), and the
# cycles. The role of each file comes from `docs/structure_notes.toml`.
#
# Run with `julia docs/structure.jl`; it needs only the standard library. It writes
# `docs/src/developer/structure_current.md`, a page of the manual. With `--json path`, it writes the
# data of the structure, and with `--explorer path`, an interactive page built from
# `docs/structure_explorer.html`.
#
# To plan a reorganization, describe it in a TOML file (moves of definitions between files,
# renames, new definitions, a new include order; see `proposed_structure`), and pass it with
# `--proposal path`. The script then runs the same analysis on the proposed structure: `--proposed
# path` writes its page, and the JSON data and the explorer include it.

module DeveloperStructure

import TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
const SRC = joinpath(ROOT, "src")
const MODULE_FILE = "src/LinearDecisionRules.jl"
const REPO = "https://github.com/bfpc/LinearDecisionRules.jl/blob/main"

"""
A top-level definition: a method, a struct, an abstract type or a constant. `refs` are the names it
uses, with `:signature` for the uses needed when the file is loaded (argument and field types,
supertypes, constant values) and `:body` for the uses inside a function body.
"""
mutable struct Def
    name::String
    kind::Symbol
    file::String
    line::Int
    signature::String
    documented::Bool
    refs::Vector{Tuple{String,Symbol}}
    # `file:line` of the current definition, kept when the proposal moves it.
    origin::String
end

function Def(name, kind, file, line, signature, documented, refs)
    return Def(name, kind, file, line, signature, documented, refs, "$file:$line")
end

# --- Reading the source ---------------------------------------------------------------------------

_rel(path) = replace(relpath(path, ROOT), '\\' => '/')

"""The files of the module in include order, and the exported names."""
function module_files()
    ex = Meta.parseall(read(joinpath(ROOT, MODULE_FILE), String))
    files, exports = String[MODULE_FILE], Set{String}()
    walk(e) =
        if e isa Expr
            if e.head === :call && e.args[1] === :include
                push!(files, "src/" * e.args[2])
            elseif e.head === :export
                union!(exports, string.(e.args))
            else
                foreach(walk, e.args)
            end
        end
    walk(ex)
    return files, exports
end

_name(s::Symbol) = string(s)
_name(q::QuoteNode) = string(q.value)
function _name(e::Expr)
    e.head === :. && return _name(e.args[1]) * "." * _name(e.args[2])
    e.head === :curly && return _name(e.args[1])
    e.head === :(::) && return "(::" * string(e.args[end]) * ")"
    return string(e)
end
_name(x) = string(x)

"""The called expression of a signature, without `where` and a return type."""
function _call(sig)
    while sig isa Expr &&
              sig.head in (:where, :(::)) &&
              !(sig.head === :(::) && length(sig.args) == 1)
        sig = sig.args[1]
    end
    return sig
end

function _arg_string(a)
    a isa Symbol && return string(a)
    a isa Expr || return string(a)
    a.head === :(::) && return "::" * string(a.args[end])
    a.head === :kw && return _arg_string(a.args[1])
    a.head === :parameters && return ""
    a.head === :... && return _arg_string(a.args[1]) * "..."
    return string(a)
end

function _signature(call)
    args = [_arg_string(a) for a in call.args[2:end] if !(a isa Expr && a.head === :parameters)]
    has_kw = any(a -> a isa Expr && a.head === :parameters, call.args[2:end])
    return "(" * join(args, ", ") * (has_kw ? "; …" : "") * ")"
end

function _type_name(e)
    e isa Symbol && return string(e)
    e.head === :curly && return _type_name(e.args[1])
    e.head === :<: && return _type_name(e.args[1])
    return string(e)
end

# Locals of a function: its arguments, the left-hand sides of assignments, loop and generator
# variables, the arguments of closures and `do` blocks, and inner functions. A use of a local in a
# position other than a call is not a reference to a definition of the package.
function _locals!(locals, e)
    e isa Expr || return
    if e.head === :(=) || e.head === :kw
        _lhs!(locals, e.args[1])
    elseif e.head === :-> || e.head === :function
        _lhs!(locals, e.args[1])
    elseif e.head === :local
        foreach(a -> _lhs!(locals, a), e.args)
    end
    foreach(a -> _locals!(locals, a), e.args)
    return
end

function _lhs!(locals, e)
    if e isa Symbol
        push!(locals, e)
    elseif e isa Expr
        if e.head in (:tuple, :parameters)
            foreach(a -> _lhs!(locals, a), e.args)
        elseif e.head === :(::) && length(e.args) == 2
            _lhs!(locals, e.args[1])
        elseif e.head === :kw || e.head === :...
            _lhs!(locals, e.args[1])
        elseif e.head === :call  # an inner function, `g(x) = …`
            _lhs!(locals, e.args[1])
            foreach(a -> _lhs!(locals, a), e.args[2:end])
        elseif e.head === :where
            _lhs!(locals, e.args[1])
        end
    end
    return
end

const _QUALIFIERS = (:LinearDecisionRules,)

function _refs!(out, e, locals, context)
    if e isa Symbol
        e in locals || push!(out, (string(e), context))
    elseif e isa Expr
        h = e.head
        if h === :call ||
           (h === :. && length(e.args) == 2 && e.args[2] isa Expr && e.args[2].head === :tuple)
            f = e.args[1]
            if f isa Symbol
                push!(out, (string(f), context))
            elseif f isa Expr && f.head === :. && f.args[1] in _QUALIFIERS
                push!(out, (_name(f.args[2]), context))
            else
                _refs!(out, f, locals, context)
            end
            rest = h === :call ? e.args[2:end] : e.args[2].args
            foreach(a -> _refs!(out, a, locals, context), rest)
        elseif h === :.
            _refs!(out, e.args[1], locals, context)  # `x.field`: the field is not a reference
        elseif h === :kw
            _refs!(out, e.args[2], locals, context)
        elseif h === :(=)
            e.args[1] isa Symbol || _refs!(out, e.args[1], locals, context)
            _refs!(out, e.args[2], locals, context)
        elseif h === :macrocall
            foreach(a -> _refs!(out, a, locals, context), e.args[2:end])
        elseif h === :quote
            return
        else
            foreach(a -> _refs!(out, a, locals, context), e.args)
        end
    end
    return
end

# The line of a definition: the first line at or after `from` that starts it.
function _find_line(lines, from, name, kind)
    short = last(split(name, '.'))
    patterns = if kind === :struct
        [Regex("^(Base\\.@kwdef )?(mutable )?struct " * short * "\\b")]
    elseif kind === :abstract
        [Regex("^abstract type " * short * "\\b")]
    elseif kind === :const
        [Regex("^const " * short * "\\b")]
    else
        q = replace(name, r"([.()\[\]{}*+?^$|:!=\\])" => s"\\\1")
        # A callable struct, `(rule::DecisionRule)(…)`, is named `(::DecisionRule)`.
        startswith(name, "(") ? [Regex("^(function )?\\([a-z_]*" * q[3:end])] :
        [Regex("^(function )?" * q * "[({ ]")]
    end
    # A docstring can be long: search up to its end.
    for i in max(from, 1):min(from + 150, length(lines)), p in patterns
        occursin(p, lines[i]) && return i
    end
    return from
end

function read_file(file)
    text = read(joinpath(ROOT, file), String)
    lines = split(text, '\n')
    defs = Def[]
    toplevel = Tuple{String,Symbol}[]
    line = 1
    function visit(e, documented)
        if e isa LineNumberNode
            line = e.line
        elseif e isa Expr
            h = e.head
            if h === :macrocall && e.args[1] isa GlobalRef && e.args[1].name === Symbol("@doc")
                e.args[2] isa LineNumberNode && (line = e.args[2].line)
                visit(e.args[4], true)
            elseif h === :macrocall && _name(e.args[1]) in ("Base.@kwdef", "@kwdef")
                visit(e.args[end], documented)
            elseif h === :module
                foreach(a -> visit(a, false), e.args[3].args)
            elseif h === :toplevel || h === :block
                foreach(a -> visit(a, false), e.args)
            elseif h === :function ||
                   (h === :(=) && _call(e.args[1]) isa Expr && _call(e.args[1]).head === :call)
                sig = e.args[1]
                call = _call(sig)
                name = call isa Symbol ? string(call) : _name(call.args[1])
                locals = Set{Symbol}()
                signature = ""
                refs = Tuple{String,Symbol}[]
                if call isa Expr
                    foreach(a -> _lhs!(locals, a), call.args[2:end])
                    signature = _signature(call)
                    # The types of the arguments are needed at load time.
                    sigrefs = Tuple{String,Symbol}[]
                    for a in call.args[2:end], t in _types_of(a)
                        _refs!(sigrefs, t, Set{Symbol}(), :signature)
                    end
                    sig isa Expr &&
                        sig.head === :where &&
                        for t in sig.args[2:end]
                            _refs!(sigrefs, t, Set{Symbol}(), :signature)
                        end
                    append!(refs, sigrefs)
                end
                if length(e.args) >= 2
                    _locals!(locals, e.args[2])
                    _refs!(refs, e.args[2], locals, :body)
                end
                kind = :function
                push!(
                    defs,
                    Def(
                        name,
                        kind,
                        file,
                        _find_line(lines, line, name, kind),
                        signature,
                        documented,
                        refs,
                    ),
                )
            elseif h === :struct
                name = _type_name(e.args[2])
                refs = Tuple{String,Symbol}[]
                _refs!(
                    refs,
                    e.args[2] isa Expr && e.args[2].head === :<: ? e.args[2].args[2] : nothing,
                    Set{Symbol}(),
                    :signature,
                )
                for f in e.args[3].args
                    if f isa Expr && f.head === :(::)
                        _refs!(refs, f.args[end], Set{Symbol}(), :signature)
                    elseif f isa Expr && f.head in (:function, :(=))
                        locals = Set{Symbol}()
                        _locals!(locals, f)
                        _refs!(refs, f.args[end], locals, :body)
                    end
                end
                push!(
                    defs,
                    Def(
                        name,
                        :struct,
                        file,
                        _find_line(lines, line, name, :struct),
                        "",
                        documented,
                        refs,
                    ),
                )
            elseif h === :abstract
                name = _type_name(e.args[1])
                refs = Tuple{String,Symbol}[]
                e.args[1] isa Expr &&
                    e.args[1].head === :<: &&
                    _refs!(refs, e.args[1].args[2], Set{Symbol}(), :signature)
                push!(
                    defs,
                    Def(
                        name,
                        :abstract,
                        file,
                        _find_line(lines, line, name, :abstract),
                        "",
                        documented,
                        refs,
                    ),
                )
            elseif h === :const && e.args[1] isa Expr && e.args[1].head === :(=)
                name = _name(e.args[1].args[1])
                refs = Tuple{String,Symbol}[]
                _refs!(refs, e.args[1].args[2], Set{Symbol}(), :signature)
                push!(
                    defs,
                    Def(
                        name,
                        :const,
                        file,
                        _find_line(lines, line, name, :const),
                        "",
                        documented,
                        refs,
                    ),
                )
            elseif h in (:import, :using, :export)
                return
            else
                _refs!(toplevel, e, Set{Symbol}(), :signature)
            end
        end
        return
    end
    visit(Meta.parseall(text; filename = file), false)
    return defs, toplevel
end

function _types_of(a)
    a isa Expr || return Any[]
    a.head === :(::) && return Any[a.args[end]]
    a.head in (:kw, :...) && return _types_of(a.args[1])
    a.head === :parameters && return reduce(vcat, (_types_of(p) for p in a.args); init = Any[])
    return Any[]
end

# --- A structure: files, definitions, dependencies ------------------------------------------------

struct PackageStructure
    label::String
    files::Vector{String}                 # in include order
    defs::Vector{Def}
    toplevel::Dict{String,Vector{Tuple{String,Symbol}}}
    exports::Set{String}
    notes::Dict{String,String}            # file => one line on its role
end

function layer(file)
    file == MODULE_FILE && return "module"
    file == "src/utils.jl" && return "utils"
    file == "src/experimental.jl" && return "experimental"
    return split(file, '/')[2]
end

function owners(s::PackageStructure)
    order = Dict(f => i for (i, f) in enumerate(s.files))
    owner = Dict{String,Def}()
    for d in sort(s.defs; by = d -> (order[d.file], d.line))
        occursin('.', d.name) ||
            startswith(d.name, "(") ||
            haskey(owner, d.name) ||
            (owner[d.name] = d)
    end
    return owner
end

"""
The references between files: for each pair `(from, to)`, the names that `from` uses and `to`
owns (the file of their first definition in include order), with the kind of use.
"""
function file_edges(s::PackageStructure)
    owner = owners(s)
    edges = Dict{Tuple{String,String},Dict{String,Symbol}}()
    function add!(from, name, context)
        haskey(owner, name) || return
        to = owner[name].file
        to == from && return
        names = get!(edges, (from, to), Dict{String,Symbol}())
        return names[name] = get(names, name, :body) === :signature ? :signature : context
    end
    for d in s.defs
        foreach(((n, c),) -> add!(d.file, n, c), d.refs)
        # A method of a function owned by another file extends it.
        haskey(owner, d.name) && owner[d.name].file != d.file && add!(d.file, d.name, :signature)
    end
    for (f, refs) in s.toplevel
        foreach(((n, c),) -> add!(f, n, c), refs)
    end
    return edges
end

"""Strongly connected components with more than one file (Tarjan)."""
function cycles(files, edges)
    succ = Dict(f => String[] for f in files)
    for (from, to) in keys(edges)
        push!(succ[from], to)
    end
    index, low, onstack, stack, out =
        Dict{String,Int}(), Dict{String,Int}(), Set{String}(), String[], Vector{String}[]
    counter = 0
    function visit(v)
        counter += 1
        index[v] = low[v] = counter
        push!(stack, v)
        push!(onstack, v)
        for w in succ[v]
            if !haskey(index, w)
                visit(w)
                low[v] = min(low[v], low[w])
            elseif w in onstack
                low[v] = min(low[v], index[w])
            end
        end
        if low[v] == index[v]
            comp = String[]
            while true
                w = pop!(stack)
                delete!(onstack, w)
                push!(comp, w)
                w == v && break
            end
            length(comp) > 1 && push!(out, sort(comp; by = f -> findfirst(==(f), files)))
        end
    end
    foreach(f -> haskey(index, f) || visit(f), files)
    return out
end

function current_structure(notes = Dict{String,String}())
    files, exports = module_files()
    defs, toplevel = Def[], Dict{String,Vector{Tuple{String,Symbol}}}()
    for f in files
        d, t = read_file(f)
        append!(defs, d)
        toplevel[f] = t
    end
    return PackageStructure("current", files, defs, toplevel, exports, notes)
end

# --- The proposed structure -------------------------------------------------------------------------

# The lines of `from` from the first definition named `a` to the last one named `b`, so that a move
# survives edits of `src/` that do not reorder the definitions. `name#k` is the `k`-th definition of
# `name` in the file, for a name with several methods.
function _range(defs, origin, from, a, b, rn)
    function lines(entry)
        name, k = occursin('#', entry) ? split(entry, '#') : (entry, nothing)
        found = sort([origin[d][2] for d in defs if origin[d][1] == from && d.name == rn(name)])
        isempty(found) && error("`$entry`, in a range of a move from $from, names no definition")
        return k === nothing ? found : found[parse(Int, k):parse(Int, k)]
    end
    return minimum(lines(a)):maximum(lines(b))
end

"""
The current structure with the moves, renames, additions, include order and exports of `toml`. A
move takes the definitions of a file to another file: all of them, those from a definition to
another (`range`, by name), or those with some names.
"""
function proposed_structure(current::PackageStructure, toml)
    rename = Dict{String,String}(get(toml, "rename", Dict()))
    rn(n) = get(rename, n, n)
    defs = [
        Def(
            rn(d.name),
            d.kind,
            d.file,
            d.line,
            d.signature,
            d.documented,
            [(rn(n), c) for (n, c) in d.refs],
        ) for d in current.defs
    ]
    origin = Dict(d => (c.file, c.line) for (d, c) in zip(defs, current.defs))
    for m in get(toml, "move", [])
        from, to = m["from"], m["to"]
        lines = haskey(m, "range") ? _range(defs, origin, from, m["range"]..., rn) : nothing
        names = haskey(m, "names") ? Set(rn.(m["names"])) : nothing
        moved = 0
        for d in defs
            f, l = origin[d]
            f == from || continue
            lines === nothing || l in lines || continue
            names === nothing || d.name in names || continue
            d.file = to
            moved += 1
        end
        moved == 0 && @warn "The move from $from to $to matches no definition" m
    end
    # Definitions that a refactoring absorbs into a new one: `name`, or `name@file` (current file).
    for r in get(toml, "remove", String[])
        name, file = occursin('@', r) ? split(r, '@'; limit = 2) : (r, nothing)
        n = length(defs)
        filter!(d -> !(d.name == rn(name) && (file === nothing || origin[d][1] == file)), defs)
        n == length(defs) && @warn "The removal of `$r` matches no definition"
    end
    toplevel = Dict(f => [(rn(n), c) for (n, c) in r] for (f, r) in current.toplevel)
    for a in get(toml, "add", [])
        refs = [(n, Symbol(get(a, "context", "body"))) for n in get(a, "uses", String[])]
        d = Def(
            a["name"],
            Symbol(get(a, "kind", "function")),
            a["file"],
            0,
            get(a, "signature", ""),
            get(a, "documented", false),
            refs,
        )
        push!(defs, d)
        # `name` or `name@file`, for a name with methods in several files, such as `MOI.get`.
        for user in get(a, "used_by", String[])
            name, file = occursin('@', user) ? split(user, '@'; limit = 2) : (user, nothing)
            users = [u for u in defs if u.name == name && (file === nothing || u.file == file)]
            isempty(users) &&
                @warn "`$user`, a user of the added `$(a["name"])`, matches no definition"
            foreach(u -> push!(u.refs, (a["name"], :body)), users)
        end
    end
    files = toml["include_order"]
    unknown = setdiff(unique(d.file for d in defs), files)
    leftover = ["$(d.name)@$(d.file):$(origin[d][2])" for d in defs if d.file in unknown]
    isempty(unknown) || error("Definitions in files not in include_order: $(join(leftover, ", "))")
    exports = Set(rn.(collect(current.exports)))
    setdiff!(exports, get(toml, "unexport", String[]))
    union!(exports, get(toml, "export", String[]))
    notes = Dict{String,String}(get(toml, "notes", Dict()))
    return PackageStructure("proposed", files, defs, toplevel, exports, notes)
end

# --- Output -----------------------------------------------------------------------------------------

_lines(s::PackageStructure, f) = s.label == "current" ? countlines(joinpath(ROOT, f)) : nothing

# The line, not a link: a link per definition would make the page too large for Documenter.
function _link(s::PackageStructure, d::Def)
    s.label == "current" || return "`$(d.name)`"
    return "`$(d.name)` (L$(d.line))"
end

function _file_link(s::PackageStructure, f)
    isfile(joinpath(ROOT, f)) && s.label == "current" && return "[`$f`]($(REPO)/$f)"
    return "`$f`"
end

_short(f) = replace(f, r"^src/" => "", r"\.jl$" => "")
_id(f) = replace(_short(f), r"[^A-Za-z0-9]" => "_")

const LAYERS = [
    "module",
    "utils",
    "lifting",
    "canonical",
    "probability",
    "formulations",
    "moi",
    "jump",
    "experimental",
]

function layer_graph(s::PackageStructure, edges)
    order = Dict(f => i for (i, f) in enumerate(s.files))
    counts = Dict{Tuple{String,String},Tuple{Int,Int}}()
    for ((from, to), names) in edges
        lf, lt = layer(from), layer(to)
        (lf == lt || lf == "module") && continue
        n, b = get(counts, (lf, lt), (0, 0))
        counts[(lf, lt)] = (n + length(names), b + (order[to] > order[from] ? length(names) : 0))
    end
    present = [l for l in LAYERS if any(f -> layer(f) == l, s.files) && l != "module"]
    io = IOBuffer()
    println(io, "```mermaid")
    println(io, "flowchart TD")
    for l in present
        n = count(f -> layer(f) == l, s.files)
        label = l in ("utils", "experimental") ? "$l.jl" : "$l/ ($n file$(n == 1 ? "" : "s"))"
        println(io, "    $l[\"$label\"]")
    end
    k = 0
    style = String[]
    for ((lf, lt), (n, b)) in sort(
        collect(counts);
        by = x -> (findfirst(==(x[1][1]), LAYERS), findfirst(==(x[1][2]), LAYERS)),
    )
        label = b > 0 ? "$n names, $b to later files" : "$n names"
        println(io, "    $lf -->|\"$label\"| $lt")
        b > 0 && push!(style, "    linkStyle $k stroke:#c0392b,stroke-width:2px")
        k += 1
    end
    foreach(x -> println(io, x), style)
    println(io, "```")
    return String(take!(io))
end

function write_markdown(path, s::PackageStructure)
    edges = file_edges(s)
    owner = owners(s)
    order = Dict(f => i for (i, f) in enumerate(s.files))
    cyc = cycles(s.files, edges)
    backward = sort(
        [(from, to, names) for ((from, to), names) in edges if order[to] > order[from]];
        by = x -> (order[x[1]], order[x[2]]),
    )
    loadfail =
        [(from, to, [n for (n, c) in names if c === :signature]) for (from, to, names) in backward]
    filter!(x -> !isempty(x[3]), loadfail)
    io = IOBuffer()
    title =
        s.label == "current" ? "Structure of `src/`: current (generated)" :
        "Structure of `src/`: proposed (generated)"
    println(io, "# ", title, "\n")
    println(
        io,
        "Generated by `docs/structure.jl` from ",
        s.label == "current" ? "the source" : "the source and a proposal file",
        "; do not edit by hand.",
    )
    println(
        io,
        s.label == "current" ?
        "See [the developer documentation](@ref developer_overview) for how to read it.\n" :
        "It reads as the page of the current structure.\n",
    )
    nfun = length(unique(d.name for d in s.defs if d.kind === :function))
    println(
        io,
        "$(length(s.files)) files, $(length(s.defs)) top-level definitions ($nfun functions, ",
        count(d -> d.kind in (:struct, :abstract), s.defs),
        " types, ",
        count(d -> d.kind === :const, s.defs),
        " constants), ",
        "$(length(s.exports)) exported names.\n",
    )
    println(io, "## Layers\n")
    println(
        io,
        "An arrow `a → b` means that files of `a` use names defined in `b`; red arrows include uses of files that are included later (late binding).\n",
    )
    println(io, layer_graph(s, edges))
    println(io, "## Include order\n")
    for (i, f) in enumerate(s.files)
        note = get(s.notes, f, "")
        println(io, "$i. ", _file_link(s, f), isempty(note) ? "" : ": " * note)
    end
    println(io)
    println(io, "## Uses of files included later\n")
    if isempty(backward)
        println(io, "None: every file uses only names defined in files included before it.\n")
    else
        println(
            io,
            "These uses work because Julia resolves names in function bodies when the function first runs. They read as a layer calling the one above it.\n",
        )
        println(io, "| File | uses, from a later file | names |")
        println(io, "|---|---|---|")
        for (from, to, names) in backward
            println(
                io,
                "| `",
                _short(from),
                "` | `",
                _short(to),
                "` | ",
                join(("`$n`" for n in sort(collect(keys(names)))), ", "),
                " |",
            )
        end
        println(io)
    end
    if !isempty(loadfail)
        println(
            io,
            "!!! warning \"Would not load\"\n    These uses are in signatures, fields or constants, so the include order must put the defining file first:\n",
        )
        for (from, to, names) in loadfail
            println(
                io,
                "    - `",
                _short(from),
                "` needs `",
                _short(to),
                "` for ",
                join(("`$n`" for n in names), ", "),
            )
        end
        println(io)
    end
    println(io, "## Cycles between files\n")
    if isempty(cyc)
        println(io, "None.\n")
    else
        println(io, "Groups of files that use each other, directly or through other files:\n")
        for c in cyc
            println(io, "- ", join(("`" * _short(f) * "`" for f in c), ", "))
        end
        println(io)
    end
    println(io, "## Files\n")
    println(
        io,
        "For each file: what it defines, which files it uses (and for what), and which files use it.",
        " Exported names are in **bold**; names with a docstring are marked †.\n",
    )
    for f in s.files
        ds = sort([d for d in s.defs if d.file == f]; by = d -> d.line)
        println(io, "### `", _short(f), "`\n")
        note = get(s.notes, f, "")
        isempty(note) || println(io, note, "\n")
        lines = _lines(s, f)
        info = String[]
        lines === nothing || push!(info, "$lines lines")
        isempty(info) || println(io, _file_link(s, f), ": ", join(info, ", "), ".\n")
        if !isempty(ds)
            println(io, "Defines:")
            groups = Dict{String,Vector{Def}}()
            names = String[]
            for d in ds
                haskey(groups, d.name) || push!(names, d.name)
                push!(get!(groups, d.name, Def[]), d)
            end
            items = String[]
            for n in names
                g = groups[n]
                label = n in s.exports ? "**" * _link(s, g[1]) * "**" : _link(s, g[1])
                any(d -> d.documented, g) && (label *= "†")
                length(g) > 1 && (label *= " ($(length(g)) methods)")
                ext =
                    haskey(owner, n) && owner[n].file != f ?
                    " (extends `" * _short(owner[n].file) * "`)" : ""
                push!(items, label * ext)
            end
            println(io, join(items, ", "), ".\n")
        end
        uses =
            sort([(to, names) for ((from, to), names) in edges if from == f]; by = x -> order[x[1]])
        if !isempty(uses)
            println(io, "Uses:\n")
            for (to, names) in uses
                later = order[to] > order[f] ? " (included later)" : ""
                println(
                    io,
                    "- `",
                    _short(to),
                    "`",
                    later,
                    ": ",
                    join(("`$n`" for n in sort(collect(keys(names)))), ", "),
                )
            end
            println(io)
        end
        users =
            sort([(from, names) for ((from, to), names) in edges if to == f]; by = x -> order[x[1]])
        if !isempty(users)
            println(
                io,
                "Used by: ",
                join(("`" * _short(u) * "` (" * string(length(n)) * ")" for (u, n) in users), ", "),
                ".\n",
            )
        end
    end
    write(path, String(take!(io)))
    return (; backward = length(backward), loadfail = length(loadfail), cycles = cyc)
end

# A small JSON writer, to avoid a dependency.
_json(io, x::AbstractString) = print(io, '"', escape_string(x), '"')
_json(io, x::Symbol) = _json(io, string(x))
_json(io, x::Union{Integer,AbstractFloat,Bool}) = print(io, x)
_json(io, ::Nothing) = print(io, "null")
function _json(io, x::AbstractDict)
    print(io, '{')
    for (i, (k, v)) in enumerate(x)
        i > 1 && print(io, ',')
        _json(io, string(k))
        print(io, ':')
        _json(io, v)
    end
    return print(io, '}')
end
function _json(io, x::Union{AbstractVector,Tuple,AbstractSet})
    print(io, '[')
    for (i, v) in enumerate(x)
        i > 1 && print(io, ',')
        _json(io, v)
    end
    return print(io, ']')
end

function json_data(s::PackageStructure)
    edges = file_edges(s)
    owner = owners(s)
    order = Dict(f => i for (i, f) in enumerate(s.files))
    known = Set(keys(owner))
    defs = [
        Dict(
            "name" => d.name,
            "kind" => d.kind,
            "file" => d.file,
            "line" => d.line,
            "signature" => d.signature,
            "documented" => d.documented,
            "exported" => d.name in s.exports,
            "uses" => unique(n for (n, _) in d.refs if n in known && n != d.name),
            "origin" => d.line == 0 ? "" : d.origin,
        ) for d in s.defs
    ]
    return Dict(
        "label" => s.label,
        "files" => [
            Dict(
                "path" => f,
                "layer" => layer(f),
                "order" => order[f],
                "note" => get(s.notes, f, ""),
                "lines" => _lines(s, f),
            ) for f in s.files
        ],
        "defs" => defs,
        "edges" => [
            Dict(
                "from" => from,
                "to" => to,
                "later" => order[to] > order[from],
                "names" => Dict(n => string(c) for (n, c) in names),
            ) for ((from, to), names) in edges
        ],
        "cycles" => cycles(s.files, edges),
        "exports" => sort(collect(s.exports)),
    )
end

function main(args)
    # The current structure is always written, to the manual; a proposed one only on request.
    notes = TOML.parsefile(joinpath(@__DIR__, "structure_notes.toml"))
    j = findfirst(==("--proposal"), args)
    toml = j === nothing ? nothing : TOML.parsefile(args[j+1])
    structures = [current_structure(Dict{String,String}(notes))]
    toml === nothing || push!(structures, proposed_structure(structures[1], toml))
    i = findfirst(==("--proposed"), args)
    if i !== nothing && toml === nothing
        error("`--proposed` needs a proposal file, given with `--proposal`")
    end
    for s in structures
        file = if s.label == "current"
            joinpath(@__DIR__, "src", "developer", "structure_current.md")
        elseif i !== nothing
            args[i+1]
        else
            continue
        end
        r = write_markdown(file, s)
        println(
            rpad(s.label, 9),
            ": ",
            length(s.files),
            " files, ",
            length(s.defs),
            " definitions, ",
            r.backward,
            " backward file pairs, ",
            r.loadfail,
            " that would not load, ",
            length(r.cycles),
            " cycles",
        )
    end
    io = IOBuffer()
    _json(io, Dict(s.label => json_data(s) for s in structures))
    data = String(take!(io))
    i = findfirst(==("--json"), args)
    if i !== nothing
        write(args[i+1], data)
    end
    # The explorer page: the template with the data, safe inside a `<script>` element.
    i = findfirst(==("--explorer"), args)
    if i !== nothing
        template = read(joinpath(@__DIR__, "structure_explorer.html"), String)
        commit = readchomp(`git -C $ROOT rev-parse --short HEAD`)
        dirty = !isempty(readchomp(`git -C $ROOT status --porcelain -- src`))
        page = replace(
            template,
            "__DATA__" => replace(data, "</" => "<\\/"),
            "__COMMIT__" => commit * (dirty ? " (with changes to src/)" : ""),
        )
        write(args[i+1], page)
    end
    return
end

end # module DeveloperStructure

if abspath(PROGRAM_FILE) == @__FILE__
    DeveloperStructure.main(ARGS)
end
