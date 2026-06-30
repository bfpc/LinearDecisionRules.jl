module TutorialsTests

include("common.jl")

function test_tutorials()
    # Automatically find and run all tutorial .jl files
    tutorials_dir = joinpath(dirname(@__DIR__), "docs", "src", "tutorials")
    for (root, dirs, files) in walkdir(tutorials_dir)
        for file in files
            if endswith(file, ".jl")
                tutorial_path = joinpath(root, file)
                @testset "$file" begin
                    # Include in a sandbox module to avoid polluting the namespace
                    mod = @eval module $(gensym("TutorialTest")) end
                    Base.include(mod, tutorial_path)
                end
            end
        end
    end
    return nothing
end

end # module

TutorialsTests.runtests()
