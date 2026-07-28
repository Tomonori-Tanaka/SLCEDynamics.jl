using SLCEDynamics
using SLCE     # the SLCE fitting core, for the executed `@example` model builds
using SLCEMonteCarlo  # the tiled Hamiltonian / observable layer the examples run on
using Documenter
using Documenter: Remotes

DocMeta.setdocmeta!(SLCEDynamics, :DocTestSetup, :(using SLCEDynamics);
                    recursive = true)

makedocs(;
    sitename = "SLCEDynamics.jl",
    modules = [SLCEDynamics],
    repo = Remotes.GitHub("Tomonori-Tanaka", "SLCEDynamics.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        mathengine = Documenter.MathJax3(),
        canonical = "https://tomonori-tanaka.github.io/SLCEDynamics.jl/dev",
        edit_link = "main",
        footer = "Built with [Documenter.jl](https://documenter.juliadocs.org).",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Guide" => [
            "guide/dynamics.md",
            "guide/thermal.md",
            "guide/quantum_thermostat.md",
            "guide/structure_factor.md",
            "guide/checkpointing.md",
            "guide/gpu.md",
        ],
        "API reference" => "api.md",
    ],
    checkdocs = :exports,
    doctest = false,
)

# Publishes to https://tomonori-tanaka.github.io/SLCEDynamics.jl/ from the `documentation build`
# CI job (which needs `permissions: contents: write`). Outside CI this is a no-op, so a
# local `julia --project=docs docs/make.jl` still just builds into `docs/build/`.
deploydocs(;
    repo = "github.com/Tomonori-Tanaka/SLCEDynamics.jl",
    devbranch = "main",
    push_preview = false,
)
