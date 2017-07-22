"""
	mutable struct Agent
		learner::AbstractReinforcementLearner
		policy::AbstractPolicy
		callback::AbstractCallback
"""
mutable struct Agent
	learner::AbstractReinforcementLearner
	policy::AbstractPolicy
	callback::AbstractCallback
end
export Agent
"""
	Agent(learner; policy = EpsilonGreedyPolicy(.1),  callback = NoCallback())
"""
Agent(learner; policy = EpsilonGreedyPolicy(.1), callback = NoCallback()) = 
	Agent(learner, policy, callback)
"""
	Agent(learner::AbstractPolicyGradient; policy = SoftmaxPolicy1(), callback = NoCallback())
"""
Agent(learner::AbstractPolicyGradient;
	  policy = SoftmaxPolicy1(),
	  callback = NoCallback()) = Agent(learner, policy, callback)
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
mutable struct RLSetup
	agent::Agent
	environment
	metric::AbstractEvaluationMetrics
	stoppingcriterion::StoppingCriterion
end
export RLSetup

act(agent::Agent, state) = act(agent.learner, agent.policy, state) 
function act(learner::Union{AbstractTDLearner, AbstractPolicyGradient}, 
			 policy::AbstractPolicy,
			 state)
	act(policy, learner.params[:, state])
end
function act(learner::SmallBackups, 
			 policy::AbstractPolicy,
			 state)
	act(policy, learner.Q[:, state])
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
					   x.environment, x.metric, x.stoppingcriterion)
"""
	run!(x::RLSetup)

"""
run!(x::RLSetup) = learn!(x.agent.learner, x.agent.policy, x.agent.callback,
					   x.environment, x.metric, x.stoppingcriterion)

"""
	learn!(agent::Agent, environment, metric, stoppingcriterion)
"""
learn!(agent::Agent, env, metric, stop) =
		learn!(agent.learner, agent.policy, agent.callback, env, metric, stop)
"""
	run!(agent::Agent, environment, metric, stoppingcriterion)
"""
run!(agent::Agent, env, metric, stop) =
		run!(agent.learner, agent.policy, agent.callback, env, metric, stop)

"""
	learn!(learner, policy, callback, environment, metric, stoppingcriterion)
"""
learn!(learner, policy, callback, env, metric, stop) = 
	run!(learner, policy, callback, env, metric, stop, true)
"""
	run!(learner, policy, callback, environment, metric, stoppingcriterion)
"""
function run!(learner, policy, callback, 
			  env, metric, stop, withlearning = false)
	s0, iss0terminal = getstate(env)
	a0 = act(learner, policy, s0)
	while true
		s1, r, iss1terminal = interact!(a0, env)
		a1 = act(learner, policy, s1)
		if withlearning; update!(learner, r, s0, a0, s1, a1, iss0terminal); end
		evaluate!(metric, r, a0, s0, iss0terminal)
		callback!(callback, learner, policy, r, a0, s0, iss0terminal)
		if isbreak!(stop, r, a0, s0, iss0terminal); break; end
		s0 = s1
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
	s0 = states[end]
	iss1terminal = iss0terminal
	for i in 1:nsteps
		s1, r, iss1terminal = interact!(actions[end], env)
		a1 = act(learner, policy, s1)
		push!(actions, a1)
		push!(rewards, r)
		push!(states, s1)
		evaluate!(metric, r, actions[end-1], s0, iss0terminal)
		callback!(callback, learner, policy, r, actions[end-1], s0, iss0terminal)
		if isbreak!(stop, r, actions[end-1], s0, iss0terminal)
			if iss0terminal
				return 4
			else
				return 2
			end
		end
		if iss0terminal; return 3; end
		iss0terminal = iss1terminal
		s0 = s1
	end
	return iss1terminal
end
function run!(learner::AbstractMultistepLearner, policy, callback,
				env, metric, stop, withlearning = false)
	s0, ret = getstate(env)
	a0 = act(learner.learner, policy, s0)
	actions = Int64[a0]
	rewards = Float64[]
	states = Int64[s0]
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

