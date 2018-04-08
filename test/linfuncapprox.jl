import TabularReinforcementLearning.preprocessstate
struct OneHotPreprocessor 
    ns::Int64
end
preprocessstate(p::OneHotPreprocessor, s) = Float64[i == s for i in 1:p.ns]
for λ in [0, .8]
    mdp = MDP()
    x = RLSetup(Agent(QLearning(λ = λ, tracekind = AccumulatingTraces),
                      # with ReplacingTraces the results will be different
                      # because of different replacingAccumulatingTraces 
                      preprocessor = OneHotPreprocessor(mdp.ns)), mdp,
                RecordAll(),
                ConstantNumberSteps(100))
    srand(124)
    s0 = mdp.state
    learn!(x)
    srand(124)
    mdp.state = s0
    y = RLSetup(Agent(QLearning(initvalue = 0., λ = λ, discretestates = true,
                                tracekind = AccumulatingTraces)), 
                mdp,
                RecordAll(),
                ConstantNumberSteps(100))
    learn!(y)
    @test x.agent.learner.params ≈ y.agent.learner.params 
end

for learner in [PolicyGradientBackward, EpisodicReinforce,
                ActorCriticPolicyGradient]
    mdp = MDP()
    x = RLSetup(Agent(learner(),
                      preprocessor = OneHotPreprocessor(mdp.ns)), mdp,
                RecordAll(),
                ConstantNumberSteps(100))
    srand(124)
    s0 = mdp.state
    learn!(x)
    srand(124)
    mdp.state = s0
    y = RLSetup(Agent(learner(discretestates = true, initvalue = 0.)), 
                mdp,
                RecordAll(),
                ConstantNumberSteps(100))
    learn!(y)
    @test x.agent.learner.params ≈ y.agent.learner.params 
end
