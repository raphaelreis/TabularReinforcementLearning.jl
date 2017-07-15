getgreedystates = TabularReinforcementLearning.getgreedystates
for (v, rO, rVO, r, rP) in (([-9., 12., Inf64], [2, 3], [3], [3], [2]),
							([-9., -12.], [1], [1], [1], [1]),
							([Inf64, Inf64], [1, 2], [1, 2], [1, 2], [1, 2]))
	@assert getgreedystates(OptimisticEpsilonGreedyPolicy(0.), v) == rO
	@assert getgreedystates(VeryOptimisticEpsilonGreedyPolicy(0.), v) == rVO
	@assert getgreedystates(PesimisticEpsilonGreedyPolicy(0.), v) == rP
end

for (v, rO, rVO, r, rP) in (([-9., 12., Inf64], [0, .5, .5], [0, 0., 1.], 
												[0., 0., 1.], [0., 1., 0.]),
							([-100, Inf64, Inf64], [1/3, 1/3, 1/3], 
												   [0., 0.5, 0.5], 
												   [0., 0.5, 0.5], 
												   [1., 0., 0.]))
	@assert getactionprobabilities(OptimisticEpsilonGreedyPolicy(0.), v) == rO
	@assert getactionprobabilities(VeryOptimisticEpsilonGreedyPolicy(0.), v) == rVO
	@assert getactionprobabilities(PesimisticEpsilonGreedyPolicy(0.), v) == rP
end



