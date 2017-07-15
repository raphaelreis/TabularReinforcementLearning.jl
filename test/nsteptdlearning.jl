na = 4; depth = 4; α = .1; nepisodes = 10^3
mdp = treeMDP(na, depth, init = "deterministic")
learner = NstepLearner(depth + 1, Sarsa(α = α, λ = 0., na = na, ns = mdp.ns))
metric = RecordAll()
params = ConstantNumberSteps(nepisodes*(depth + 1))
learn!(learner, VeryOptimisticEpsilonGreedyPolicy(.1), NoCallback(), mdp, metric, params)
learner.learner.params
amaxs1 = indmax(learner.learner.params[:, 1])
amaxs1episodes = find(x -> x == (1, amaxs1), collect(zip(metric.s, metric.a)))
rewards = learner.learner.γ^depth * metric.r[amaxs1episodes + depth]
@test learner.learner.params[amaxs1, 1] ≈ (1 - α)^(length(rewards) - 1) * rewards[1] + 
							α * dot(rewards[2:end], 
									(1 - α).^collect(length(rewards) - 2:-1:0)) atol = 1e-2

