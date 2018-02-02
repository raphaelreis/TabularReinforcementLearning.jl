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
Agent(learner; policy = EpsilonGreedyPolicy(.1), 
      callback = NoCallback(), preprocessor = NoPreprocessor()) = 
      Agent(learner, policy, callback, preprocessor)
"""
    Agent(learner::AbstractPolicyGradient; policy = SoftmaxPolicy1(), callback = NoCallback())
"""
Agent(learner::AbstractPolicyGradient;
      policy = SoftmaxPolicy1(), preprocessor = NoPreprocessor(),
      callback = NoCallback()) = Agent(learner, policy, callback, preprocessor)
"""
    Agent(learner::NstepLearner; policy = EpsilonGreedyPolicy(.1), callback = NoCallback())

Replaces `policy` with SoftmaxPolicy1 for baselearner of type
AbstractPolicyGradient.
"""
function Agent(learner::AbstractMultistepLearner; 
               policy = EpsilonGreedyPolicy(.1), callback = NoCallback())
    if typeof(learner.learner) <: AbstractPolicyGradient
        Agent(learner, SoftmaxPolicy1(), callback)
    else
        Agent(learner, policy, callback)
    end
end
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

getvalue(params, state::Int64) = params[:, state]
getvalue(params, action::Int64, state::Int64) = params[action, state]
getvalue(params, state::Array{Float64, 1}) = params * state
getvalue(params, action::Int64, state::Array{Float64, 1}) = dot(params[action, :], state)

act(agent::Agent, state) = act(agent.learner, agent.policy, state) 
function act(learner::Union{AbstractTDLearner, AbstractPolicyGradient}, 
             policy::AbstractPolicy,
             state)
    act(policy, getvalue(learner.params, state))
end
function act(learner::Union{SmallBackups, MonteCarlo}, 
             policy::AbstractPolicy,
             state)
    act(policy, getvalue(learner.Q, state))
end
function act(learner::MDPLearner, policy::AbstractEpsilonGreedyPolicy, state)
    if rand() < policy.ϵ
        rand(1:learner.mdp.na)
    else
        learner.policy[state]
    end
end

"""
    learn!(x::RLSetup)

"""
learn!(x::RLSetup) = learn!(x.agent.learner, x.agent.policy, x.agent.callback,
                            x.agent.preprocessor, x.environment, x.metric, 
                            x.stoppingcriterion)
"""
    run!(x::RLSetup)

"""
run!(x::RLSetup) = learn!(x.agent.learner, x.agent.policy, x.agent.callback,
                          x.agent.preprocessor, x.environment, x.metric, 
                          x.stoppingcriterion)

"""
    learn!(agent::Agent, environment, metric, stoppingcriterion)
"""
learn!(agent::Agent, env, metric, stop) =
        learn!(agent.learner, agent.policy, agent.callback, agent.preprocessor,
               env, metric, stop)
"""
    run!(agent::Agent, environment, metric, stoppingcriterion)
"""
run!(agent::Agent, env, metric, stop) =
        run!(agent.learner, agent.policy, agent.callback, agent.preprocessor,
             env, metric, stop)

"""
    learn!(learner, policy, callback, environment, metric, stoppingcriterion)
"""
learn!(learner, policy, callback, preprocessor, env, metric, stop) = 
    run!(learner, policy, callback, preprocessor, env, metric, stop, true)
"""
    run!(learner, policy, callback, environment, metric, stoppingcriterion)
"""
function run!(learner, policy, callback, preprocessor,
              env, metric, stop, withlearning = false)
    s0, iss0terminal = getstate(env)
    s0p = preprocess(preprocessor, s0)
    a0 = act(learner, policy, s0p)
    while true
        s1, r, iss1terminal = interact!(a0, env)
        if iss1terminal 
            s1 = reset!(env) 
            r = 0
        end
        s1p = preprocess(preprocessor, s1)
        a1 = act(learner, policy, s1p)
        if withlearning; update!(learner, r, s0p, a0, s1p, a1, iss0terminal); end
        evaluate!(metric, r, a0, s0p, iss0terminal)
        callback!(callback, learner, policy, r, a0, s0p, iss0terminal)
        if isbreak!(stop, r, a0, s0p, iss0terminal); break; end
        s0p = deepcopy(s1p)
        a0 = a1
        iss0terminal = iss1terminal
    end
end
export run!, learn!

# TODO: make nstep nicer.
# also: above the order is update!, evaluate!, callback!, isbreak!
#       here it is evaluate!, callback!, isbreak!, update!
function step!(learner, policy, callback, env,
               rewards, states, actions, iss0terminal, 
               nsteps, metric, stop)
    s0p = states[end]
    iss1terminal = iss0terminal
    for i in 1:nsteps
        s1, r, iss1terminal = interact!(actions[end], env)
        if iss1terminal 
            s1 = reset!(env)
            r = 0
        end
        s1p = preprocess(preprocessor, s1)
        a1 = act(learner, policy, s1p)
        push!(actions, a1)
        push!(rewards, r)
        push!(states, s1p)
        evaluate!(metric, r, actions[end-1], s0p, iss0terminal)
        callback!(callback, learner, policy, r, actions[end-1], s0p, iss0terminal)
        if isbreak!(stop, r, actions[end-1], s0p, iss0terminal)
            if iss0terminal
                return 4
            else
                return 2
            end
        end
        if iss0terminal; return 3; end
        iss0terminal = iss1terminal
        s0p = deepcopy(s1)
    end
    return iss1terminal
end
function run!(learner::AbstractMultistepLearner, policy, callback,
                env, metric, stop, withlearning = false)
    s0, ret = getstate(env)
    s0p = preprocess(preprocessor, s0)
    a0 = act(learner.learner, policy, s0p)
    actions = Int64[a0]
    rewards = Float64[]
    states = Int64[s0p]
    while true
        if ret < 2 || length(actions) == 1
            ret = step!(learner.learner, policy, callback, env,
                        rewards, states, actions, ret == 1, 
                        1, metric, stop)
        end
        if withlearning
            update!(learner, learner.learner, rewards, states, actions, ret > 2)
        end
        if typeof(learner) == NstepLearner && 
             (length(rewards) == learner.nsteps || 
                (ret > 1 && length(actions) > 1))
            shift!(actions); shift!(states); shift!(rewards)
        end
        if typeof(learner) == EpisodicLearner && ret > 2
            empty!(rewards)
            deleteat!(states, 1:length(states) - 1) 
            deleteat!(actions, 1:length(actions) - 1)
        end
        if (ret == 2 || ret == 4) && 
            (length(actions) == 1 || typeof(learner) == EpisodicLearner)
            break
        end
    end
end

