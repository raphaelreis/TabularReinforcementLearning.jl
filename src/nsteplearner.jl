abstract AbstractMultistepLearner <: AbstractReinforcementLearner
"""
	type NstepLearner <: AbstractReinforcementLearner
		nsteps::Int64
		learner::AbstractReinforcementLearner
"""
type NstepLearner <: AbstractMultistepLearner
	nsteps::Int64
	learner::AbstractReinforcementLearner
end
"""
	NstepLearner(; nsteps = 10, learner = Sarsa, kwargs...) = 
		NstepLearner(nsteps, learner(; kwargs...))
"""
NstepLearner(; nsteps = 10, learner = Sarsa, kwargs...) = 
	NstepLearner(nsteps, learner(; λ = 0, kwargs...))
export NstepLearner

"""
	type EpisodicLearner <: AbstractMultistepLearner
		learner::AbstractReinforcementLearner
"""
type EpisodicLearner <: AbstractMultistepLearner
	learner::AbstractReinforcementLearner
end
export EpisodicLearner
