"""
	type MonteCarlo <: AbstractReinforcementLearner
		Nsa::Array{Int64, 2}
		γ::Float64
		Q::Array{Float64, 2}

Estimate Q values by averaging over returns.
"""
type MonteCarlo <: AbstractReinforcementLearner
	Nsa::Array{Int64, 2}
	γ::Float64
	Q::Array{Float64, 2}
end
"""
	MonteCarlo(; ns = 10, na = 4, γ = .9)
"""
MonteCarlo(; ns = 10, na = 4, γ = .9, initvalue = Inf64) = 
	EpisodicLearner(MonteCarlo(zeros(na, ns), γ, zeros(na, ns) + Inf64))
export MonteCarlo

function update!(::EpisodicLearner, learner::MonteCarlo, rewards, states,
				actions, isterminal)
	if learner.Q[actions[end-1], states[end-1]] == Inf64
		learner.Q[actions[end-1], states[end-1]] = 0.
	end
	if isterminal
		G = rewards[end]
		for t in length(rewards)-1:-1:1
			G = learner.γ * G + rewards[t]
			n = learner.Nsa[actions[t], states[t]] += 1
			learner.Q[actions[t], states[t]] *= (1 - 1/n)
			learner.Q[actions[t], states[t]] += 1/n * G

		end
	end
end
