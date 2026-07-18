# Documentation build for SCESpinDynamics.jl.
#
#   make build        build the HTML site into build/
#   make serve        build, then serve it locally with live reload (LiveServer.jl)
#   make open         open the built site in a browser
#   make clean        remove the build/ output
#
# The first run resolves docs/Project.toml (Documenter + LiveServer + SCESpinDynamics +
# SCEFitting + SCEMonteCarlo, all by path). The siblings must sit at
# ../../SCEFitting.jl and ../../SCEMonteCarlo.jl.

.PHONY: build serve open clean help

# -t 4: some executed examples pass ntasks > 1 to the drivers (results are
# bit-identical regardless of the thread count — that is the point they make).
build:
	julia -t 4 --project=. -e 'using Pkg; Pkg.instantiate(); include("make.jl")'

serve: build
	julia --project=. -e 'using LiveServer; serve(dir = "build")'

open:
	open build/index.html

clean:
	rm -rf build/

help:
	@echo "build  - build the HTML documentation into build/"
	@echo "serve  - build and serve locally (live reload)"
	@echo "open   - open the built site in a browser"
	@echo "clean  - remove build/"
