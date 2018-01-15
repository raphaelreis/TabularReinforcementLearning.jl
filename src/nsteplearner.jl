abstract type AbstractMultistepLearner <: AbstractReinforcementLearner end
"""
    struct NstepLearner <: AbstractReinforcementLearner
        nsteps::Int64
        learner::AbstractReinforcementLearner
"""
struct NstepLearner <: AbstractMultistepLearner
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
    struct EpisodicLearner <: AbstractMultistepLearner
        learner::AbstractReinforcementLearner
"""
struct EpisodicLearner <: AbstractMultistepLearner
    learner::AbstractReinforcementLearner
end
export EpisodicLearner
