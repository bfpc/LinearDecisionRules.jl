# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Build locally with `julia --project=docs docs/make.jl`; the pages are written to docs/build.

import Documenter
import LinearDecisionRules

Documenter.makedocs(;
    sitename = "LinearDecisionRules.jl",
    modules = [LinearDecisionRules],
    repo = Documenter.Remotes.GitHub("bfpc", "LinearDecisionRules.jl"),
    format = Documenter.HTML(; edit_link = "main", prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "API" => [
            "Modelling" => "modelling.md",
            "Results and rules" => "results.md",
            "Simulation" => "simulation.md",
            "Distributions and liftings" => "distributions.md",
            "Internals" => "internals.md",
        ],
    ],
    checkdocs = :exports,
)

# Deploy only from CI.
if get(ENV, "CI", nothing) == "true"
    Documenter.deploydocs(;
        repo = "github.com/bfpc/LinearDecisionRules.jl.git",
        push_preview = true,
    )
end
