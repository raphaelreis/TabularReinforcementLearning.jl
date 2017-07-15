abstract AbstractMultistepLearner <: AbstractReinforcementLearner
"""
	type NstepLearner <: AbstractReinforcementLearner
		nsteps::Int64
		learner::Union{AbstractTDLearner, AbstractPolicyGradient}
"""
type NstepLearner <: AbstractMultistepLearner
	nsteps::Int64
	learner::Union{AbstractTDLearner, AbstractPolicyGradient}
end
"""
	NstepLearner(; nsteps = 10, learner = Sarsa, kwargs...) = 
		NstepLearner(nsteps, learner(; kwargs...))
"""
NstepLearner(; nsteps = 10, learner = Sarsa, kwargs...) = 
	NstepLearner(nsteps, learner(; kwargs...))
export NstepLearner

"""
	type EpisodicLearner <: AbstractMultistepLearner
		learner::Union{AbstractTDLearner, AbstractPolicyGradient}
"""
type EpisodicLearner <: AbstractMultistepLearner
	learner::Union{AbstractTDLearner, AbstractPolicyGradient}
end
