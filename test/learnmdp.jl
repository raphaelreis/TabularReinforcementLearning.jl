srand(123)
mdp = MDP(ns = 5, na = 3); 
γ = .5
mdpl = MDPLearner(mdp, γ); policy_iteration!(mdpl)
x = RLSetup(Agent(QLearning(ns = 5, na = 3, γ = γ, λ = 0., α = 1e-3)), 
	   mdp,
	   MeanReward(),
	   ConstantNumberSteps(10^6))
learn!(x)
@test mdpl.values ≈ getvalues(x.agent.learner) atol=0.3
