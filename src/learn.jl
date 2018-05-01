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
struct AsyncParallelAgents{TLearner, TPolicy, TCallback, TPreprocessor} 
    learner::Array{TLearner, 1}
    policy::TPolicy
    callback::TCallback
    preprocessor::TPreprocessor
end
export AsyncParallelAgents
function AsyncParallelAgents(learners::AbstractArray; policy = SoftmaxPolicy1(),
                        preprocessor = NoPreprocessor(), callback = NoCallback())
    AsyncParallelAgents(learners, policy, callback, preprocessor)
end
function AsyncParallelAgents(learner, n; policy = SoftmaxPolicy1(),
                        preprocessor = NoPreprocessor(), callback = NoCallback())
    learners = [deepcopy(learner) for _ in 1:n]
    w = params(learners[1])
    if typeof(w[1]) == Array{Any, 1}
        w = map(x -> SharedArray.(x), w)
    else
        w = SharedArray.(w)
    end
    for learner in learners; setparams!(learner, w); end # share parameters
    AsyncParallelAgents(learners, policy, callback, preprocessor)
end

function copyandreplace(obj, fieldname, newval)
    fields = []
    for field in typeof(obj).name.names
        if field == fieldname
            push!(fields, newval)
        else
            push!(fields, getfield(obj, field))
        end
    end
    typeof(obj).name.wrapper(fields...)
end
mutable struct RLSetup{TM, TS}
    agent
    environment
    metric::TM
    stoppingcriterion::TS
    RLSetup{TM, TS}(a, e, m, s) where {TM, TS} = new(a, e, m, s)
end
export RLSetup    
function RLSetup(agent, env, metric::TM, stop::TS) where {TM, TS} # most elegant way?
    if typeof(agent.learner) <: DeepActorCritic
        if agent.learner.nenvs > 1 && length(env) != agent.learner.nenvs
            error("Learner expects $(agent.learner.nenvs) environments; $(length(env)) given.")
        end
    end
    if typeof(env) <: AbstractArray
        s = getstate(env[1])[1]
    else
        s = getstate(env)[1]
    end
    s = preprocessstate(agent.preprocessor, s)
    Tstate = typeof(s)
    b = agent.learner.buffer
    if Tstate != typeof(b).parameters[1] || typeof(b) <: ArrayStateBuffer
        info("Detected type of state after preprocessing: $Tstate.")
        buffertype = typeof(b)
        fields = []
        for field in buffertype.name.names
            if field == :states
                if typeof(b.states) <: CircularBuffer
                    push!(fields, CircularBuffer{Tstate}(b.states.capacity))
                elseif typeof(b.states) <: ArrayCircularBuffer
                    push!(fields, ArrayCircularBuffer(Tstate.name.wrapper,
                                                      Tstate.parameters[1],
                                                      size(s),
                                                      b.states.capacity))
                else
                    push!(fields, Array{Tstate, 1})
                end
            else
                push!(fields, getfield(b, field))
            end
        end
        newbuffer = buffertype.name.wrapper(fields...)
        agent = copyandreplace(agent, :learner, 
                               copyandreplace(agent.learner, :buffer, 
                                              newbuffer))
    end
    RLSetup{typeof(metric),typeof(stop)}(agent, env, metric, stop)
end


@inline getvalue(params, state::Int) = params[:, state]
@inline getvalue(params::Vector, state::Int) = params[state]
@inline getvalue(params, action::Int, state::Int) = params[action, state]
@inline getvalue(params, state::AbstractArray) = params * state
@inline getvalue(params::Vector, state::Vector) = dot(params, state)
@inline getvalue(params, action::Int, state::AbstractArray) = 
    dot(view(params, action, :), state)

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
@inline function step!(learner, policy, env, preprocessor, offset = 0)
    s, r, done = preprocess(preprocessor, 
                            interact!(learner.buffer.actions[end - offset], env)...)
    pushreturn!(learner.buffer, r, done)
    if done; s = preprocessstate(preprocessor, reset!(env)) end
    pushstate!(learner.buffer, s)
    a = selectaction(learner, policy, s)
    pushaction!(learner.buffer, a)
    r, done
end
@inline function firststateaction!(learner, policy, preprocessor, env, 
                                   force = false)
    if isempty(learner.buffer.actions) || force
        s = preprocessstate(preprocessor, getstate(env)[1])
        pushstate!(learner.buffer, s)
        a = selectaction(learner, policy, s)
        pushaction!(learner.buffer, a)
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
        callback!(callback, learner, policy, metric, stop)
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
@inline function run!(learner, policy, callback, preprocessor,
                      envs::AbstractArray, metric, stop, withlearning = false)
    for env in envs; firststateaction!(learner, policy, preprocessor, env, true) end
    while true
        for env in envs
            r, done = step!(learner, policy, env, preprocessor, length(envs) - 1)
            evaluate!(metric, r, done, learner.buffer)
            callback!(callback, learner, policy, metric, stop)
        end
        if withlearning; update!(learner); end
        if isbreak!(stop, done, learner.buffer); break; end
    end
end

export run!, learn!
