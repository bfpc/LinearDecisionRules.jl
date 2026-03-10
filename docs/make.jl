using Documenter
using LinearDecisionRules
using Literate

# ==============================================================================
#  Process Literate.jl tutorials
# ==============================================================================

const _TUTORIAL_DIR = joinpath(@__DIR__, "src", "tutorials")

"""
    _include_sandbox(filename)

Include the `filename` in a temporary module that acts as a sandbox.
"""
function _include_sandbox(filename)
    mod = @eval module $(gensym()) end
    return Base.include(mod, filename)
end

function _literate_directory(dir)
    for filename in readdir(dir; join = true)
        if !endswith(filename, ".jl")
            continue
        end
        # Test the file in a sandbox
        @info "Testing $filename"
        _include_sandbox(filename)
        # Generate markdown
        Literate.markdown(
            filename,
            dir;
            documenter = true,
            credit = false,
        )
    end
    return nothing
end

# Process tutorial files in the root tutorials directory
_literate_directory(_TUTORIAL_DIR)

# Process all tutorial subdirectories
for (root, dirs, files) in walkdir(_TUTORIAL_DIR)
    for dir in dirs
        _literate_directory(joinpath(root, dir))
    end
end

# ==============================================================================
#  Documentation structure
# ==============================================================================

const _PAGES = [
    "Home" => "index.md",
    "Tutorials" => [
        "Getting started" => [
            "tutorials/getting_started/introduction.md",
            "tutorials/getting_started/getting_started_with_LDR.md",
        ],
        "tutorials/piecewise_linear.md",
        "tutorials/distributions.md",
        "tutorials/advanced_distributions.md",
    ],
    "Manual" => [
        "manual/math.md",
        "manual/pwl.md",
        "manual/ConfidenceNormal.md",
    ],
    "API Reference" => "api.md",
]

# ==============================================================================
#  Build documentation
# ==============================================================================

makedocs(;
    sitename = "LinearDecisionRules.jl",
    authors = "Bernardo Freitas Paulo da Costa, Joaquim Dias Garcia, and contributors",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        mathengine = Documenter.MathJax2(),
        collapselevel = 1,
    ),
    modules = [LinearDecisionRules],
    pages = _PAGES,
    repo = "https://github.com/bfpc/LinearDecisionRules.jl/blob/{commit}{path}#{line}",
    checkdocs = :none,
)

deploydocs(;
    repo = "github.com/bfpc/LinearDecisionRules.jl.git",
    push_preview = true,
)
