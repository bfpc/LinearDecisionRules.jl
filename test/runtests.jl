# How to run only specific test subfiles:
#   LDR_TEST_FILES="distributions.jl" julia --project=test -e 'include("test/runtests.jl")'
#   LDR_TEST_FILES="test_core.jl,distributions.jl" julia --project=test -e 'include("test/runtests.jl")'
#
# Notes:
# - File names are resolved relative to test/.
# - If LDR_TEST_FILES is not set, this file runs all tests

using Test

# Argument / subfile logic

ALL_FILES = ["test_core.jl", "distributions.jl", "sampled_mode.jl"]

function _parse_selected_test_files()
    raw = get(ENV, "LDR_TEST_FILES", "")
    isempty(strip(raw)) && return String[]
    files = [strip(f) for f in split(raw, ',') if !isempty(strip(f))]
    return files
end

function _run_test_files(files::Vector{String})
    for file in files
        path = joinpath(@__DIR__, file)
        if !isfile(path)
            error("Test file not found: $(file)")
        end
        @testset "$(file)" begin
            include(path)
        end
    end
    return
end

function dispatch()
    selected_files = _parse_selected_test_files()
    if isempty(selected_files)
        _run_test_files(ALL_FILES)
    else
        _run_test_files(selected_files)
    end
    return
end

dispatch()
