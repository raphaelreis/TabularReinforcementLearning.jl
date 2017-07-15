using TabularReinforcementLearning

struct RewardHitsThreshold <: TabularReinforcementLearning.StoppingCriterion
	θ::Float64
end
import TabularReinforcementLearning.isbreak!
function isbreak!(p::RewardHitsThreshold, r, s, a, isterminal)
	r > p.θ
end

x = RLSetup(Agent(QLearning()), MDP(), RecordAll(), RewardHitsThreshold(.5))
learn!(x)
