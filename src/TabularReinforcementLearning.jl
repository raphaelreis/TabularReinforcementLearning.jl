__precompile__()

module TabularReinforcementLearning

using DataStructures
include("abstracttypes.jl")
include("helper.jl")
include("forced.jl")
include("buffers.jl")
include("traces.jl")
include("epsilongreedypolicies.jl")
include("softmaxpolicy.jl")
include("nsteplearner.jl")
include("tdlearning.jl")
include("prioritizedsweeping.jl")
include("policygradientlearning.jl")
include("mdp.jl")
include("mdplearner.jl")
include("montecarlo.jl")
include("metrics.jl")
include("stoppingcriterion.jl")
include("callbacks.jl")
include("preprocessor.jl")
# include("knet.jl")
include("flux.jl")
include("deepactorcritic.jl")
include("dqn.jl")
include("learn.jl")
include("comparisontools.jl")
    

end # module
