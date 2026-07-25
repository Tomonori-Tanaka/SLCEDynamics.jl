using SLCEDynamics
using SLCE     # the SCE fitting core, for the executed `@example` model builds
using SLCEMonteCarlo  # the tiled Hamiltonian / observable layer the examples run on
using Documenter

DocMeta.setdocmeta!(SLCEDynamics, :DocTestSetup, :(using SLCEDynamics);
                    recursive = true)

makedocs(;
    sitename = "SLCEDynamics.jl",
    modules = [SLCEDynamics],
    # The SLCE/SLCEMonteCarlo dependencies are path-devs without a resolvable
    # remote in this build, so per-line source/edit links stay disabled; the navbar
    # links to the repository (private: github.com/Tomonori-Tanaka/SLCEDynamics.jl).
    remotes = nothing,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        mathengine = Documenter.MathJax3(),
        edit_link = nothing,
        repolink = "https://github.com/Tomonori-Tanaka/SLCEDynamics.jl",
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
