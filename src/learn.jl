"""
    mutable struct Agent
        learner::AbstractReinforcementLearner
        policy::AbstractPolicy
        callback::AbstractCallback
        preprocessor
"""
struct Agent{TLearner, TPolicy, TCallback, TPreprocessor}
    learner::TLearner
    policy::TPolicy
    callback::TCallback
    preprocessor::TPreprocessor
end
export Agent
"""
    Agent(learner; policy = EpsilonGreedyPolicy(.1),  callback = NoCallback())
"""
Agent(learner; policy = EpsilonGreedyPolicy(.1), callback = NoCallback(), 
      preprocessor = NoPreprocessor()) = 
      Agent(learner, policy, callback, preprocessor)
"""
    Agent(learner::AbstractPolicyGradient; policy = SoftmaxPolicy1(), callback = NoCallback())
"""
Agent(learner::Union{AbstractPolicyGradient, A2C}; policy = SoftmaxPolicy1(), 
      preprocessor = NoPreprocessor(), callback = NoCallback()) = 
    Agent(learner, policy, callback, preprocessor)
"""
    mutable struct RLSetup
        agent::Agent
        environment
        metric::AbstractEvaluationMetrics
        stoppingcriterion::StoppingCriterion
"""
mutable struct RLSetup{TMetric, TStopping}
    agent::Agent
    environment
    metric::TMetric
    stoppingcriterion::TStopping
end
export RLSetup

@inline getvalue(params, state::Int) = params[:, state]
@inline getvalue(params::Vector, state::Int) = params[state]
@inline getvalue(params, action::Int, state::Int) = params[action, state]
@inline getvalue(params, state::Vector) = params * state
@inline getvalue(params::Vector, state::Vector) = dot(params, state)
@inline getvalue(params, action::Int, state::Vector) = dot(params[action, :], state)

selectaction(agent::Agent, state) = selectaction(agent.learner, agent.policy, state) 
@inline function selectaction(learner::Union{TDLearner, AbstractPolicyGradient}, 
                              policy::AbstractPolicy,
                              state)
    selectaction(policy, getvalue(learner.params, state))
end
@inline function selectaction(learner::Union{SmallBackups, MonteCarlo}, 
                              policy::AbstractPolicy,
                              state)
    selectaction(policy, getvalue(learner.Q, state))
end
@inline function selectaction(learner::MDPLearner, 
                              policy::AbstractEpsilonGreedyPolicy, state)
    if rand() < policy.ϵ
        rand(1:learner.mdp.na)
    else
        learner.policy[state]
    end
end

"""
    learn!(x::RLSetup)

"""
learn!(x::RLSetup) = learn!(x.agent, x.environment, x.metric, x.stoppingcriterion)
"""
    run!(x::RLSetup)

"""
run!(x::RLSetup) = run!(x.agent, x.environment, x.metric, x.stoppingcriterion)

"""
    learn!(agent::Agent, environment, metric, stoppingcriterion)
"""
@inline learn!(agent::Agent, env, metric, stop) =
        learn!(agent.learner, agent.policy,  
               agent.callback, agent.preprocessor, env, metric, stop)
"""
    run!(agent::Agent, environment, metric, stoppingcriterion)
"""
@inline run!(agent::Agent, env, metric, stop) =
        run!(agent.learner, agent.policy,              
             agent.callback, agent.preprocessor, env, metric, stop)

"""
    learn!(learner, policy, callback, environment, metric, stoppingcriterion)
"""
@inline learn!(learner, policy, callback, preprocessor, env, metric, stop) = 
    run!(learner, policy, callback, preprocessor, env, metric, stop, true)
"""
    run!(learner, policy, callback, environment, metric, stoppingcriterion)
"""
@inline function run!(learner, policy, callback, preprocessor,
                      env, metric, stop, withlearning = false)
    s = preprocessstate(preprocessor, getstate(env)[1])
    a = selectaction(learner, policy, s)
    pushstateaction!(learner.buffer, s, a)
    while true
        s, r, done = preprocess(preprocessor, interact!(a, env)...)
        pushreturn!(learner.buffer, r, done)
        if done; s = preprocessstate(preprocessor, reset!(env)) end
        a = selectaction(learner, policy, s)
        pushstateaction!(learner.buffer, s, a)
        if withlearning; update!(learner); end
        evaluate!(metric, r, done, learner.buffer)
        callback!(callback, learner, policy)
        if isbreak!(stop, done, learner.buffer); break; end
    end
end
export run!, learn!
