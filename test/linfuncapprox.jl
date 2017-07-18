mutable struct OneHotMDP
	mdp::MDP
end
import TabularReinforcementLearning.interact!,
TabularReinforcementLearning.reset!, TabularReinforcementLearning.getstate

toonehot(s, ns) = Float64[i == s for i in 1:ns]
function interact!(a, env::OneHotMDP)
	s, r, isterminal = interact!(a, env.mdp)
	toonehot(s, env.mdp.ns), r, isterminal
end
getstate(env::OneHotMDP) = (toonehot(env.mdp.state, env.mdp.ns),
							env.mdp.isterminal[env.mdp.state] == 1)
reset!(env::OneHotMDP) = reset(env.mdp)
x = RLSetup(Agent(QLearning(initvalue = 0., λ = 0)), OneHotMDP(MDP()),
			RecordAll(),
			ConstantNumberSteps(100))
srand(124)
s0 = x.environment.mdp.state
learn!(x)
srand(124)
x.environment.mdp.state = s0
y = RLSetup(Agent(QLearning(initvalue = 0., λ = 0)), x.environment.mdp,
			RecordAll(),
			ConstantNumberSteps(100))
learn!(y)
@test x.agent.learner.params ≈ y.agent.learner.params 
