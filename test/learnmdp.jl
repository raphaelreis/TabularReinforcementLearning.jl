srand(123)
mdp = MDP(ns = 5, na = 3); 
γ = .5
mdpl = MDPLearner(mdp, γ); policy_iteration!(mdpl)
x = RLSetup(Agent(QLearning(ns = 5, na = 3, γ = γ, λ = 0., α = 1e-3, 
                            discretestates = true)), 
            mdp,
            MeanReward(),
            ConstantNumberSteps(10^6))
learn!(x)
@test mdpl.values ≈ getvalues(x.agent.learner) atol=0.3

include(joinpath(Pkg.dir("TabularReinforcementLearning"),
                 "environments", "discretestates", "randommdp.jl"))
mdp = DetTreeMDP()
mdpl = MDPLearner(mdp, .9); policy_iteration!(mdpl)
x = RLSetup(Agent(mdpl, policy = EpsilonGreedyPolicy(0)), mdp, MeanReward(),
            ConstantNumberEpisodes(2))
run!(x)
@test 5 * getvalue(x.metric) ≈ maximum(mdp.reward[find(mdp.reward)])
