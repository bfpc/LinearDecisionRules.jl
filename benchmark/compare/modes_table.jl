# Post-process ../results/compare_modes.csv (rewrites it with the columns `ref_objective` and
# `rel_err`, the relative error of the objective against the HiGHS optimum of the same file: the
# first OPTIMAL row among dsimplex, psimplex, ipm, ipx_nox) and print the Markdown table of the
# README (one row per file, one column per mode: seconds, with the status or the error when not a
# plain optimal solve).
#   julia benchmark/compare/modes_table.jl [table] [table=mode,mode,...]
const RESULTS = joinpath(@__DIR__, "..", "results", "compare_modes.csv")
const REF_ORDER = ["dsimplex", "psimplex", "ipm", "ipx_nox"]
const MODE_ORDER = [
    "dsimplex", "psimplex", "ipm", "ipm_nox", "ipx_nox", "pdlp", "hipdlp", "ipopt", "ipopt_tol6", "ipopt_tol10", "ipopt_br0",
]

function read_rows()
    lines = readlines(RESULTS)
    header = split(lines[1], ",")
    keep = findall(!in(("ref_objective", "rel_err")), header)
    rows = [Dict(String(header[j]) => String(split(l, ",")[j]) for j in keep) for l in lines[2:end]]
    return String.(header[keep]), rows
end

stemof(r) = "$(r["problem"])_T$(r["T"])_$(r["package"])_$(r["formulation"])"

function main(args)
    header, rows = read_rows()
    refs = Dict{String,Float64}()
    for mode in REF_ORDER, r in rows
        r["mode"] == mode && r["status"] == "OPTIMAL" && !haskey(refs, stemof(r)) &&
            (refs[stemof(r)] = parse(Float64, r["objective"]))
    end
    # no HiGHS optimum for this file (time limit everywhere): the other package's, same instance and
    # formulation family (the packages agree on the objective, see README)
    family(r) = "$(r["problem"])_T$(r["T"])_" * (startswith(r["formulation"], "primal") ? "primal" : "dual")
    famref = Dict{String,Float64}()
    for r in rows
        haskey(refs, stemof(r)) && !haskey(famref, family(r)) && (famref[family(r)] = refs[stemof(r)])
    end
    for r in rows
        ref = get(refs, stemof(r), get(famref, family(r), NaN))
        obj = r["objective"] == "" ? NaN : parse(Float64, r["objective"])
        r["ref_objective"] = isnan(ref) ? "" : string(ref)
        r["rel_err"] = isnan(ref) || isnan(obj) ? "" : string(abs(obj - ref) / max(abs(ref), 1e-12))
    end
    open(RESULTS, "w") do io
        cols = [header; "ref_objective"; "rel_err"]
        println(io, join(cols, ","))
        for r in rows
            println(io, join((r[c] for c in cols), ","))
        end
    end
    for a in args
        a == "table" && print_table(rows, ["dsimplex", "psimplex", "ipm", "ipm_nox", "ipx_nox", "ipopt"])
        startswith(a, "table=") && print_table(rows, String.(split(a[7:end], ",")))
    end
end

function fmt_time(t)
    t < 10 ? string(round(t; sigdigits = 2)) : string(round(t; digits = t < 100 ? 1 : 0))
end

function cell(r)
    s = r["status"]
    s == "SKIPPED" && return "skip"
    s in ("TIME_LIMIT", "KILLED") && return ">120 (TL)"
    t = fmt_time(parse(Float64, r["run_s"]))
    e = r["rel_err"] == "" ? NaN : parse(Float64, r["rel_err"])
    flags = String[]
    s in ("OPTIMAL", "LOCALLY_SOLVED") || push!(flags, s == "UNKNOWN" ? "unk" : lowercase(s))
    isnan(e) || e > 1e-6 && push!(flags, "err " * string(round(e; sigdigits = 1)))
    return isempty(flags) ? t : "$t ($(join(flags, "; ")))"
end

function print_table(rows, modes)
    present = [m for m in MODE_ORDER if m in modes && any(r["mode"] == m for r in rows)]
    rows = [r for r in rows if r["mode"] in present]
    println("| Instance | Formulation | Package | ", join(present, " | "), " |")
    println("|---|---|---|", join(fill("--:", length(present)), "|"), "|")
    stems = unique(stemof(r) for r in rows)
    lookup = Dict((stemof(r), r["mode"]) => r for r in rows)
    for s in stems
        r1 = first(r for r in rows if stemof(r) == s)
        println(
            "| $(r1["problem"]) T=$(r1["T"]) | $(r1["formulation"]) | $(r1["package"]) | ",
            join((haskey(lookup, (s, m)) ? cell(lookup[(s, m)]) : "" for m in present), " | "), " |",
        )
    end
end

main(ARGS)
