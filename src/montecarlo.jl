"""
    mutable struct MonteCarlo <: AbstractReinforcementLearner
        Nsa::Array{Int64, 2}
        γ::Float64
        Q::Array{Float64, 2}

Estimate Q values by averaging over returns.
"""
mutable struct MonteCarlo{Tbuff} <: AbstractReinforcementLearner
    @common_learner_fields
    Nsa::Array{Int64, 2}
    Q::Array{Float64, 2}
end
"""
    MonteCarlo(; ns = 10, na = 4, γ = .9)
"""
MonteCarlo(; ns = 10, na = 4, γ = .9, initvalue = Inf64, discretestates = false,
             statetype = discretestates ? Int64 : Array{Float64, 1},
             buffer = EpisodeBuffer(statetype = statetype)) = 
    MonteCarlo(γ, buffer, zeros(Int64, na, ns), zeros(na, ns) + Inf64)
export MonteCarlo

function update!(learner::MonteCarlo)
    rewards = learner.buffer.rewards
    states = learner.buffer.states
    actions = learner.buffer.actions
    if learner.Q[actions[end-1], states[end-1]] == Inf64
        learner.Q[actions[end-1], states[end-1]] = 0.
    end
    if learner.buffer.done[end]
        G = 0.
        for t in length(rewards):-1:1
            G = learner.γ * G + rewards[t]
            n = learner.Nsa[actions[t], states[t]] += 1
            learner.Q[actions[t], states[t]] *= (1 - 1/n)
            learner.Q[actions[t], states[t]] += 1/n * G
        end
    end
end
