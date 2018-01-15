episode = [(0., 1, 2), (1., 3, 1), (0., 1, 2), (1., 2, 2), (0., 3, 2)]
episode_rsasnan = [(episode[i]..., episode[i+1][2:3]..., false) 
                                for i in 1:length(episode) - 1]
γ = .9
λ = .8
α = .1
γλ = γ*λ
results = Dict()
results[QLearning, NoTraces] = [Inf64 Inf64 1.; 0. 1. + γ Inf64]
results[Sarsa, NoTraces] = [Inf64 Inf64 1.; 0. 1. Inf64]
δ2 = 1.
δ3 = - α * γλ
δ4 = 1. + γ * (δ2 + α * δ3 * γλ)
δ4Sarsa = 1.
tmp = [Inf64 Inf64 0.; 0. δ4 Inf64]
tmp[2, 1] += α * (δ2 * γλ + δ3 * (1 + γλ^2) + δ4 * (γλ + γλ^3))
tmp[1, 3] += δ2 + α * (δ3 * γλ + δ4 * γλ^2)
results[QLearning, AccumulatingTraces] = deepcopy(tmp)
tmp[2, 1] -=  α * (δ3 * γλ^2 + δ4 * γλ^3)
results[QLearning, ReplacingTraces] = deepcopy(tmp)
tmp[2, 2] = δ4Sarsa
tmp[2, 1] += α * (δ4Sarsa - δ4) * γλ
tmp[1, 3] += α * (δ4Sarsa - δ4) * γλ^2
results[Sarsa, ReplacingTraces] = deepcopy(tmp)
tmp[2, 1] += α * (δ3 * γλ^2 + δ4Sarsa * γλ^3)
results[Sarsa, AccumulatingTraces] = deepcopy(tmp)
for tdkind in [QLearning, Sarsa] #, ExpectedSarsa]
    for tracekind in [NoTraces, AccumulatingTraces, ReplacingTraces]
        learner = tdkind(ns = 3, na = 2, γ = γ, λ = λ, α = α, tracekind = tracekind)
        for rsasnan in episode_rsasnan
            update!(learner, rsasnan...)
        end
        @assert(norm((learner.params - 
                      results[tdkind, tracekind])[[2,4,5]]) < 1e-15 &&
            learner.params[[1,3,6]] == results[tdkind, tracekind][[1,3,6]],
            "$tdkind $tracekind $(learner.params) $(results[tdkind, tracekind])")
    end
end


