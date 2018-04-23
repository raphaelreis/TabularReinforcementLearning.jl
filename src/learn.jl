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
Agent(learner::Union{AbstractPolicyGradient,DeepActorCritic}; policy = SoftmaxPolicy1(), 
      preprocessor = NoPreprocessor(), callback = NoCallback()) = 
    Agent(learner, policy, callback, preprocessor)
"""
    mutable struct RLSetup
        agent::Agent
        environment
        metric::AbstractEvaluationMetrics
        stoppingcriterion::StoppingCriterion
"""
struct ParallelAgents{TLearner, TPolicy, TCallback, TPreprocessor} 
    learner::Array{TLearner, 1}
    policy::TPolicy
    callback::TCallback
    preprocessor::TPreprocessor
end
export ParallelAgents
function ParallelAgents(learners::AbstractArray; policy = SoftmaxPolicy1(),
                        preprocessor = NoPreprocessor(), callback = NoCallback())
    ParallelAgents(learners, policy, callback, preprocessor)
end
function ParallelAgents(learner, n; policy = SoftmaxPolicy1(),
                        preprocessor = NoPreprocessor(), callback = NoCallback())
    learners = [deepcopy(learner) for _ in 1:n]
    w = params(learners[1])
    if typeof(w[1]) == Array{Any, 1}
        w = map(x -> SharedArray.(x), w)
    else
        w = SharedArray.(w)
    end
    for learner in learners; setparams!(learner, w); end # share parameters
    ParallelAgents(learners, policy, callback, preprocessor)
end

mutable struct RLSetup{TMetric, TStopping}
    agent
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
learn!(x::RLSetup) = run!(x, true)
"""
    run!(x::RLSetup)

"""
run!(x::RLSetup, withlearning = false) = 
    run!(x.agent, x.environment, x.metric, x.stoppingcriterion, withlearning)

"""
    learn!(agent::Agent, environment, metric, stoppingcriterion)
"""
learn!(agent, env, metric, stop) = run!(agent, env, metric, stop, true)
"""
    run!(agent::Agent, environment, metric, stoppingcriterion)
"""
run!(agent, env, metric, stop, withlearning = false) =
        run!(agent.learner, agent.policy, agent.callback, agent.preprocessor, 
             env, metric, stop, withlearning)

"""
    learn!(learner, policy, callback, environment, metric, stoppingcriterion)
"""
@inline function step!(learner, policy, env, preprocessor)
    s, r, done = preprocess(preprocessor, 
                            interact!(learner.buffer.actions[end], env)...)
    pushreturn!(learner.buffer, r, done)
    if done; s = preprocessstate(preprocessor, reset!(env)) end
    a = selectaction(learner, policy, s)
    pushstateaction!(learner.buffer, s, a)
    r, done
end
@inline function firststateaction!(learner, policy, preprocessor, env)
    if isempty(learner.buffer.actions)
        s = preprocessstate(preprocessor, getstate(env)[1])
        a = selectaction(learner, policy, s)
        pushstateaction!(learner.buffer, s, a)
    end
end

"""
    run!(learner, policy, callback, environment, metric, stoppingcriterion)
"""
@inline function run!(learner, policy, callback, preprocessor,
                      env, metric, stop, withlearning = false)
    firststateaction!(learner, policy, preprocessor, env)
    while true
        r, done = step!(learner, policy, env, preprocessor)
        if withlearning; update!(learner); end
        evaluate!(metric, r, done, learner.buffer)
        callback!(callback, learner, policy)
        if isbreak!(stop, done, learner.buffer); break; end
    end
end
@inline function run!(learners::AbstractArray, policy, callback, preprocessor,
                      envs::AbstractArray, metric, stop, withlearning = false)
    @sync @parallel for i in 1:length(learners)
        run!(learners[i], policy, callback, preprocessor, envs[i], 
             metric, stop, withlearning)
    end
end
export run!, learn!
