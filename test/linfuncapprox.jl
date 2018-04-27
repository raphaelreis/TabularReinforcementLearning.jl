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

using Flux
ns = 10; na = 4;
env = MDP(; ns = ns, na = na, init = "deterministic")
policy = ForcedPolicy(rand(1:na, 200))
learner = DQN(Linear(ns, na), replaysize = 2, updatetargetevery = 1, 
              updateevery = 1, startlearningat = 1, opt = x -> Flux.SGD(x, .05), 
              minibatchsize = 1)
agent = Agent(learner, preprocessor = OneHotPreprocessor(ns), policy = policy)
x = RLSetup(agent, env, EvaluationPerT(10^3, MeanReward()),
            ConstantNumberSteps(90))
x2 = RLSetup(Agent(QLearning(λ = 0, γ = .99, initvalue = 0., 
                             discretestates = true, α = .1), policy = policy),
             env, EvaluationPerT(10^3, MeanReward()), ConstantNumberSteps(90))
srand(445)
reset!(env)
learn!(x)
srand(445)
reset!(env)
x2.agent.policy.t = 1
learn!(x2)
@test x.agent.learner.policynet.W ≈ x2.agent.learner.params
