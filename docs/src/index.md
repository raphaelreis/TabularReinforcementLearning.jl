A reinforcement learning library for tabular environments.


# What is reinforcement learning?

- [Wikipedia](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [New Sutton & Barto book](http://incompleteideas.net/sutton/book/the-book-2nd.html)

# Installation

julia v0.6:

	Pkg.clone("https://github.com/jbrea/TabularReinforcementLearning.jl")

julia v0.5:

	Pkg.clone("https://github.com/jbrea/TabularReinforcementLearning.jl")
	Pkg.checkout("TabularReinforcementLearning", "v0.5")


If you want to use the utilities [`compare`](@ref) and 
[`plotcomparison`](@ref) make sure that `DataFrames` and `PyPlot` is installed.

	Pkg.add("DataFrames") 
	Pkg.add("PyPlot")

