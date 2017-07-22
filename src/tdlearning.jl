for kind in (:QLearning, :Sarsa)
	@eval begin
		type $kind <: AbstractTDLearner
			α::Float64
			γ::Float64
			unseenvalue::Float64
			params::Array{Float64, 2}
			traces::AbstractTraces
		end; 
		export $kind
		function $kind(; ns = 10, na = 4, α = .1, γ = .9, λ = .8, 
						   tracekind = ReplacingTraces, initvalue = Inf64,
						   unseenvalue = 0.)
				$kind(α, γ, unseenvalue, zeros(na, ns) .+ initvalue,
					  λ == 0. || tracekind == NoTraces ? NoTraces() : 
						tracekind(ns, na, λ, γ))
		  end
	  end
end
# TODO: how can one generate the doc nicer?
@doc """
	type QLearning <: AbstractTDLearner
		α::Float64
		γ::Float64
		unseenvalue::Float64
		params::Array{Float64, 2}
		traces::AbstractTraces

QLearner with learning rate `α`, discount factor `γ`, Q-values `params` and
eligibility `traces`.

The Q-values are updated "off-policy" according to
``Q(a, s) ← α δ e(a, s)``
where ``δ = r + γ \\max_{a'} Q(a', s') - Q(a, s)`` with next state ``s'`` and
``e(a, s)`` is the eligibility trace (see [`NoTraces`](@ref), 
[`ReplacingTraces`](@ref) and [`AccumulatingTraces`](@ref)).
""" QLearning
@doc """
	QLearning(; ns = 10, na = 4, α = .1, γ = .9, λ = .8, 
				tracekind = ReplacingTraces, initvalue = Inf64, unseenvalue = 0.)

See also  [Initial values, novel actions and unseen values](@ref initunseen).
""" QLearning()
@doc """
	type Sarsa <: AbstractTDLearner
		α::Float64
		γ::Float64
		unseenvalue::Float64
		params::Array{Float64, 2}
		traces::AbstractTraces

Sarsa Learner with learning rate `α`, discount factor `γ`, Q-values `params` and
eligibility `traces`.

The Q-values are updated "on-policy" according to
``Q(a, s) ← α δ e(a, s)``
where ``δ = r + γ Q(a', s') - Q(a, s)`` with next state ``s'``, next action
``a'`` and ``e(a, s)`` is the eligibility trace (see [`NoTraces`](@ref), 
[`ReplacingTraces`](@ref) and [`AccumulatingTraces`](@ref)).
""" Sarsa
@doc """
	Sarsa(; ns = 10, na = 4, α = .1, γ = .9, λ = .8, 
			tracekind = ReplacingTraces, initvalue = Inf64, unseenvalue = 0.)

See also  [Initial values, novel actions and unseen values](@ref initunseen).
""" Sarsa()

"""
	type ExpectedSarsa <: AbstractTDLearner
		α::Float64
		γ::Float64
		unseenvalue::Float64
		params::Array{Float64, 2}
		traces::AbstractTraces
		policy::AbstractPolicy

Expected Sarsa Learner with learning rate `α`, discount factor `γ`, 
Q-values `params` and eligibility `traces`.

The Q-values are updated according to
``Q(a, s) ← α δ e(a, s)``
where ``δ = r + γ \\sum_{a'} \\pi(a', s') Q(a', s') - Q(a, s)`` 
with next state ``s'``, probability ``\\pi(a', s')`` of choosing action ``a'`` in
next state ``s'`` and ``e(a, s)`` is the eligibility trace (see [`NoTraces`](@ref), 
[`ReplacingTraces`](@ref) and [`AccumulatingTraces`](@ref)).
"""
type ExpectedSarsa <: AbstractTDLearner
	α::Float64
	γ::Float64
	unseenvalue::Float64
	params::Array{Float64, 2}
	traces::AbstractTraces
	policy::AbstractPolicy
end
export ExpectedSarsa
"""
	ExpectedSarsa(; ns = 10, na = 4, α = .1, γ = .9, λ = .8, 
					tracekind = ReplacingTraces, initvalue = Inf64,
					unseenvalue = 0.,
					policy = VeryOptimisticEpsilonGreedyPolicy(.1))

See also  [Initial values, novel actions and unseen values](@ref initunseen).
"""
function ExpectedSarsa(; ns = 10, na = 4, α = .1, γ = .9, λ = .8, 
						 tracekind = ReplacingTraces, initvalue = Inf64,
						 unseenvalue = 0, 
						 policy = VeryOptimisticEpsilonGreedyPolicy(.1))
	ExpectedSarsa(α, γ, unseenvalue, zeros(na, ns) .+ initvalue,
					  λ == 0. || tracekind == NoTraces ? NoTraces() : 
					  tracekind(ns, na, λ, γ), policy)
end


function futurediscountedcheckinf(γ, value, learner)
	if value == Inf64
#		TODO: unseenvalue could be updated through a callback	
		γ * learner.unseenvalue
	else
		γ * value
	end
end

function futurediscountedvalue(learner::QLearning, γ, nextstate, nextaction)
	futurediscountedcheckinf(γ, maximumbelowInf(learner.params[:, nextstate]),
							 learner)
end
function futurediscountedvalue(learner::Sarsa, γ, nextstate, nextaction)
	futurediscountedcheckinf(γ, learner.params[nextaction, nextstate], learner)
end
function futurediscountedvalue(learner::ExpectedSarsa, γ, nextstate, nextaction)
	actionprobabilites = getactionprobabilities(learner.policy,
												learner.params[:, nextstate])
	m = 0.
	for (a, w) in enumerate(actionprobabilites)
		if w != 0.
			m += w * futurediscountedcheckinf(γ, learner.params[a, nextstate],
											  learner)
		 end
	end
	m
end

function gettderror(learner, r, s, a, nexts, nexta, isterminal)
	r + 
	(isterminal ? 0 : futurediscountedvalue(learner, learner.γ, nexts, nexta)) -
	(learner.params[a, s] == Inf64 ? 0. : learner.params[a, s])
end
function getnsteptderror(rewards::Array{Float64, 1}, γ, value1, valueend, 
						 isterminal)
	gammaeff = 1.
	advantage = -value1
	for r in rewards
		advantage += gammaeff * r
		gammaeff *= γ
	end
	if !isterminal
		advantage += gammaeff * valueend
	end
	advantage
end
function getnsteptderror(learner, rewards, states, actions, isterminal)
	getnsteptderror(rewards, 
					learner.γ, 
					(learner.params[actions[1], states[1]] == Inf64 ?
					 0 : learner.params[actions[1], states[1]]),
					futurediscountedvalue(learner, 1., states[end], actions[end]),
					isterminal)
end

function update!(::NstepLearner, learner::AbstractTDLearner, 
				 rewards, states, actions, isterminal)
# 	if isterminal || length(rewards) == learner.nsteps
		δ = getnsteptderror(learner, rewards, states, actions, isterminal)
		update!(learner, states[1], actions[1], δ, 
				isterminal && length(states) == 2)
# 	end
end
function update!(learner::AbstractTDLearner, r, s, a, nexts, nexta, isterminal) 
	δ = gettderror(learner, r, s, a, nexts, nexta, isterminal)
	update!(learner, s, a, δ, isterminal)
end
function update!(learner::AbstractTDLearner, s::Int64, a::Int64, δ, isterminal)
	updatetraceandparams!(learner.traces, learner, s, a, δ, isterminal)
end

export update!

function updateterminalstate!(learner, s, δ)
	for a in 1:size(learner.params, 1)
		updateparam!(learner, s, a, learner.α, δ)
	end
end
function updateparam!(learner, s, a, α, Δ)
	if learner.params[a, s] == Inf64
		learner.params[a, s] = Δ
	else
		learner.params[a, s] += α * Δ
	end
end

function updatetraceandparams!(trace::NoTraces, learner, s, a, δ, isterminal)
	if isterminal
		updateterminalstate!(learner, s, δ)
	else
		updateparam!(learner, s, a, learner.α, δ)
	end
end

function updatetraceandparams!(trace, learner, state, action, δ, isterminal)
	increasetrace!(learner.traces, state, action, isterminal)
	updatetraceandparams!(learner.traces, learner.params, learner.α * δ)
	if learner.params[action, state] == Inf64
		learner.params[action, state] = δ
	end
end

 
function getvalues(learner::AbstractTDLearner)
	[maximum(learner.params[:, i]) for i in 1:size(learner.params, 2)]
end
export getvalues
