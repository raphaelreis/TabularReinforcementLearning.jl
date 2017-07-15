abstract type AbstractMultistepLearner <: AbstractReinforcementLearner end
"""
	struct NstepLearner <: AbstractReinforcementLearner
		nsteps::Int64
		learner::Union{AbstractTDLearner, AbstractPolicyGradient}
"""
struct NstepLearner <: AbstractMultistepLearner
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
	struct EpisodicLearner <: AbstractMultistepLearner
		learner::Union{AbstractTDLearner, AbstractPolicyGradient}
"""
struct EpisodicLearner <: AbstractMultistepLearner
	learner::Union{AbstractTDLearner, AbstractPolicyGradient}
end
