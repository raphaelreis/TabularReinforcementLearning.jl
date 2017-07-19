# learner
abstract AbstractReinforcementLearner
abstract AbstractTDLearner <: AbstractReinforcementLearner
abstract AbstractPolicyGradient <: AbstractReinforcementLearner
function update! end
@doc """
	update!(learner::TabularReinforcementLearning.AbstractReinforcementLearner, 
			r, s0, a0, s1, a1, iss0terminal)

Update `learner` after observing state `s0`, performing action `a0`, receiving
reward `r`, observing next state `s1` and performing next action `a1`. The
boolean `iss0terminal` is `true` if `s0` is a terminal state.


	update!(learner::Union{NstepLearner, EpisodicLearner}, 
			baselearner::TabularReinforcementLearning.AbstractReinforcementLearner, 
			rewards, states, actions, isterminal)

Update `baselearner` with arrays of maximally `n+1` `states`, `n+1` `actions`,
`n` rewards, if `learner` is [`NstepLearner`](@ref). If `learner` is
[`EpisodicLearner`](@ref) the arrays grow until the end of an episode.
The boolean `isterminal` is `true` if `states[end-1]` is a terminal state.
""" update!
export update!

# policy
abstract AbstractPolicy
abstract AbstractEpsilonGreedyPolicy <: AbstractPolicy
abstract AbstractSoftmaxPolicy <: AbstractPolicy
function act end
@doc """
	act(learner::TabularReinforcementLearning.AbstractReinforcementLearner,
		policy::TabularReinforcementLearning.AbstractPolicy,
		state)

Returns an action for a `learner`, using `policy` in `state`.
""" act(learner, policy, state)
@doc """
	act(policy::TabularReinforcementLearning.AbstractPolicy, values)

Returns an action given an array of `values` (one value for each possible action) 
using `policy`.
""" act(policy, values)
function getactionprobabilities end
@doc """
	getactionprobabilities(policy::TabularReinforcementLearning.AbstractPolicy, values)

Returns a array of action probabilities for a given array of `values` 
(one value for each possible action) and `policy`.
"""	getactionprobabilities
export act, getactionprobabilities

# environment
function interact! end
@doc """
	interact!(action, environment)

Updates the `environment` and returns the triple `state`, `reward`, `isterminal`, 
where `state` is the new state of the environment (an integer), `reward` is the
reward obtained for the performed `action` and `isterminal` is `true` if the 
`state` is terminal.
""" interact!
function getstate end
@doc """
	getstate(environment)

Returns the tuple `state`, `isterminal`. See also [`interact!(action, environment)`](@ref).
""" getstate
function reset! end
@doc """
	reset!(environment)

Resets the `environment` to a possible initial state.
""" reset!
export interact!, getstate, reset!

# metrics
abstract AbstractEvaluationMetrics
abstract SimpleEvaluationMetric <: AbstractEvaluationMetrics
function evaluate! end
@doc """
	evaluate!(metric::TabularReinforcementLearning.AbstractEvaluationMetrics, 
			  reward, action, state, isterminal)

Updates the `metric` based on the experienced (`reward`, `action`, `state`)
triplet and the boolean `isterminal` that is `true` if `state` is terminal.

""" evaluate!
function getvalue end
@doc """
	getvalue(metric)

Returns the value of a metric.
""" getvalue
export evaluate!, getvalue 

# stopping
abstract StoppingCriterion
function isbreak! end
@doc """
	isbreak!(criterion::TabularReinforcementLearning.StoppingCriterion, r, a, s, isterminal)

Return `true` if `criterion` is matched.
See [`ConstantNumberSteps`](@ref) and [`ConstantNumberEpisodes`](@ref) for
builtin criterions and [example](https://github.com/jbrea/TabularReinforcementLearning.jl/blob/master/examples/definestopcriterion.jl) for how
to define new criterions.
""" isbreak!
export isbreak!

# Callbacks
abstract AbstractCallback
function callback! end
@doc """
	callback!(callback::AbstractCallback, learner, policy, r, a, s, isterminal)

Can be used to manipulate the learner or the policy during learning, e.g. to
change the learning rate or the exploration rate.
""" callback!
export callback!

# misc

abstract AbstractBiasCorrector

abstract AbstractTraces


