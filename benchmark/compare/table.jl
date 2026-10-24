# Markdown tables from ../results/compare.csv.
#   julia benchmark/compare/table.jl [default|settings]
const CSV = joinpath(@__DIR__, "..", "results", "compare.csv")

function read_rows()
    lines = readlines(CSV)
    header = split(lines[1], ",")
    return [Dict(zip(header, split(l, ","))) for l in lines[2:end]]
end

f(x; d = 2) = x == "" ? "" : string(round(parse(Float64, x); digits = d))
k(x) = replace(string(parse(Int, x)), r"(?<=\d)(?=(\d{3})+$)" => ",")

function main(kind)
    rows = read_rows()
    if kind == "default"
        rows = filter(r -> r["setting"] == "default", rows)
        println("| Instance | Formulation | Package | Rows x cols | After presolve | Iterations | Presolve s | Total s | Solve s |")
        println("|---|---|---|---|---|--:|--:|--:|--:|")
        order(r) = (r["problem"], parse(Int, r["T"]), r["formulation"] == "primal" || startswith(r["formulation"], "primal") ? 0 : 1, r["package"] == "new" ? 0 : 1, r["formulation"])
        for r in sort(rows; by = order)
            pre = r["presolve_s"]
            solve = pre == "" ? "" : f(string(parse(Float64, r["run_s"]) - parse(Float64, pre)))
            println("| $(r["problem"]) T=$(r["T"]) | $(r["formulation"]) | $(r["package"]) | $(k(r["rows"])) x $(k(r["cols"])) | ",
                "$(k(r["pre_rows"])) x $(k(r["pre_cols"])) ($(k(r["pre_nonzeros"])) nz) | ",
                "$(r["simplex_iter"] != "0" ? k(r["simplex_iter"]) : "IPM " * r["ipm_iter"]) | $(f(pre)) | $(f(r["run_s"])) | $solve |")
        end
    else
        rows = filter(r -> r["setting"] != "default" && r["setting"] != "presolve_only", rows)
        println("| Instance | Formulation | Package | Setting | Iterations | Total s | Status | Objective |")
        println("|---|---|---|---|--:|--:|--:|--:|")
        for r in sort(rows; by = r -> (r["problem"], parse(Int, r["T"]), r["formulation"], r["package"], r["setting"]))
            its = r["simplex_iter"] != "0" ? r["simplex_iter"] : "IPM " * r["ipm_iter"]
            println("| $(r["problem"]) T=$(r["T"]) | $(r["formulation"]) | $(r["package"]) | $(r["setting"]) | $its | $(f(r["run_s"])) | $(r["status"]) | $(f(r["objective"]; d = 6)) |")
        end
    end
end

main(isempty(ARGS) ? "default" : ARGS[1])
