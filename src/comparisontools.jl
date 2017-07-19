function compare end
@doc """
This function is loaded with `loadcompare()` (requires DataFrames).

	compare(N, 
		    environment, 
			metric::AbstractEvaluationMetrics, 
			stoppingcriterion::StoppingCriterion, 
			agent1generator::Function, 
			agent2generator::Function, ...)

Returns a DataFrame of `N` runs of all the agents on the `environment`. 

The DataFrame has the columns :learner (a string identifying the agent), :value
(the result of `getvalue(metric)` and :seed (the random seed used for this run).
This macro requires and loads the module DataFrames (can be installed with
`Pkg.add("DataFrames")`. If `environment` is an environment generator function
a new environment is generated `N` times. Otherwise the same environment is
reset `N` times.

# Examples
	result = compare(10, () -> MDP(), MeanReward(), ConstantNumberSteps(100), 
					 () -> Agent(QLearning(λ = 0.)), 
					 () -> Agent(QLearning(λ = .8)))

This can also be written as:

	metric = MeanReward()
	stopcrit = ConstantNumberSteps(100)
	pol = VeryOptimisticEpsilonGreedyPolicy(.1)
	getnewmdp() = MDP()
	getnewQ1() = Agent(QLearning(λ = 0.), policy = pol)
	getnewQ2() = Agent(QLearning(λ = .8), policy = pol)
	result = compare(10, getnewmdp, metric, stopcrit, getnewQ1, getnewQ2)

""" compare

function plotcomparison end
@doc """
This function is loaded with `loadplotcomparison()` (requires PyPlot, DataFrames).

	plotcomparison(results; labels = Dict(), colors = [], thin = .1, thick = 2, smoothingwindow = 0)

Plots results obtained with [`compare`](@ref).

The dictionary `labels` can be used to rename the legend entries, e.g. `labels =
Dict("QLearning_1" => "QLearning λ = .8")`. The data is smoothed by a moving
average of size `smoothingwindow` (default: no smoothing).
""" plotcomparison

loadcompare() = include("$(dirname(@__FILE__))/compare.jl")
loadplotcomparison() = include("$(dirname(@__FILE__))/plotcomparison.jl")
function loadcomparisontools()
	loadcompare()
	loadplotcomparison()
end
export loadcompare, loadplotcomparison, loadcomparisontools, compare, plotcomparison

