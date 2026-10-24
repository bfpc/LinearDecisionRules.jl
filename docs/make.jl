# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

import Documenter
import LinearDecisionRules

Documenter.makedocs(;
    sitename = "LinearDecisionRules.jl",
    modules = [LinearDecisionRules],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = ["Home" => "index.md", "API" => "api.md"],
    checkdocs = :exports,
)

Documenter.deploydocs(; repo = "github.com/bfpc/LinearDecisionRules.jl.git", push_preview = true)
