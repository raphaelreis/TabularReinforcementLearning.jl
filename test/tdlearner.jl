episode = [(1, 2, 0.), (3, 1, 1.), (1, 1, 0.), (2, 2, 1.), (3, 2, 0.)]
episode_sarsnan = [(episode[i]..., episode[i+1][1:2]...) 
                            for i in 1:length(episode) - 1]
γ = .9
λ = .8
α = .1
results = Dict()
results[QLearning, NoTraces] = [0. Inf64 1.; 0. 1. + γ 0.]
for tdkind in (QLearning)#, Sarsa, ExpectedSarsa)
    for tracekind in (NoTraces)#, AccumulatingTraces, ReplacingTraces)
        learner = tdkind(ns = 3, na = 2, γ = γ, λ = λ, α = α, tracekind = tracekind)
        for sarsnan in episode_sarsnan
            update!(learner, sarsnan...)
        end
        @assert learner.params == 
                results[tdkind, tracekind] "$tdkind $tracekind $(learner.params)"
    end
end


