"""
    mutable struct PolicyGradientBackward <: AbstractPolicyGradient
        α::Float64
        γ::Float64
        params::Array{Float64, 2}
        traces::AccumulatingTraces
        biascorrector::AbstractBiasCorrector
        
Policy gradient learning in the backward view.

The parameters are updated according to
``
params[a, s] += α * r_{eff} * e[a, s]
``
where ``r_{eff} =  r`` for [`NoBiasCorrector`](@ref), ``r_{eff} =  r - rmean``
for [`RewardLowpassFilterBiasCorrector`](@ref) and e[a, s] is the eligibility
trace.
""" 
mutable struct PolicyGradientBackward <: AbstractPolicyGradient
    α::Float64
    γ::Float64
    params::Array{Float64, 2}
    traces::AccumulatingTraces
    biascorrector::AbstractBiasCorrector
end
export PolicyGradientBackward
"""
    PolicyGradientBackward(; ns = 10, na = 4, α = .1, γ = .9, 
                   tracekind = AccumulatingTraces, initvalue = Inf64,
                   biascorrector = NoBiasCorrector())
"""
function PolicyGradientBackward(; ns = 10, na = 4, α = .1, γ = .9, 
                                  tracekind = AccumulatingTraces,
                                  initvalue = Inf64,
                                  biascorrector = NoBiasCorrector())
        PolicyGradientBackward(α, γ, zeros(na, ns) + initvalue,
                               tracekind(ns, na, 1., γ),
                               biascorrector)
end

"""
    mutable struct PolicyGradientForward <: AbstractPolicyGradient
        α::Float64
        γ::Float64
        params::Array{Float64, 2}
        biascorrector::AbstractBiasCorrector
"""
mutable struct PolicyGradientForward <: AbstractPolicyGradient
    α::Float64
    γ::Float64
    params::Array{Float64, 2}
    biascorrector::AbstractBiasCorrector
end
export PolicyGradientForward
function PolicyGradientForward(; ns = 10, na = 4, α = .1, γ = .9,
                                 initvalue = Inf64,
                                 biascorrector = NoBiasCorrector())
    PolicyGradientForward(α, γ, zeros(na, ns) + initvalue, biascorrector)
end
"""
    EpisodicReinforce(; kwargs...) = EpisodicLearner(PolicyGradientForward(; kwargs...))
"""
EpisodicReinforce(; kwargs...) =
    EpisodicLearner(PolicyGradientForward(; kwargs...))
export EpisodicReinforce
"""
    ActorCriticPolicyGradient(; nsteps = 1, γ = .9, ns = 10, na = 4, 
                                α = .1, αcritic = .1, initvalue = Inf64)
"""
ActorCriticPolicyGradient(; nsteps = 1, γ = .9, ns = 10, na = 4, 
                            α = .1, αcritic = .1, initvalue = Inf64) =
    NstepLearner(nsteps, 
                 PolicyGradientForward(; biascorrector = Critic(ns = ns, 
                                                                α = αcritic),
                                              ns = ns, na = na, γ = γ, 
                                              α = α, initvalue = initvalue))
export ActorCriticPolicyGradient

function update!(learner, r, s, a)
    δ = correct(learner.biascorrector, r, s)
    updatetraceandparams!(learner.traces, learner.params, learner.α * δ)
    if learner.params[a, s] == Inf64
        learner.params[a, s] = learner.α * δ * learner.traces.trace[a, s] /
                                    learner.traces.γλ 
                                    # because updatetraceandparams updates 
    end
end

"""
    struct NoBiasCorrector <: AbstractBiasCorrector
"""
struct NoBiasCorrector <: AbstractBiasCorrector end
export NoBiasCorrector
correct(::NoBiasCorrector, r, s, G = r) = G

"""
    mutable struct RewardLowpassFilterBiasCorrector <: AbstractBiasCorrector
    γ::Float64
    rmean::Float64

Filters the reward with factor γ and uses effective reward (r - rmean) to update
the parameters.
"""
mutable struct RewardLowpassFilterBiasCorrector <: AbstractBiasCorrector
    γ::Float64
    rmean::Float64
end
export RewardLowpassFilterBiasCorrector
RewardLowpassFilterBiasCorrector(γ) = RewardLowpassFilterBiasCorrector(γ, 0.)
function correct(corrector::RewardLowpassFilterBiasCorrector, r, s, G = r)
    corrector.rmean *= corrector.γ
    corrector.rmean += (1 - corrector.γ) * r
    G - corrector.rmean
end

function update!(learner::PolicyGradientBackward, r, s, a, nexts, nexta, isterminal)
    gradlogpolicy!(getactionprobabilities(learner, s), s, a, learner.traces.trace)
    update!(learner, r, s, a)
    if isterminal; resettraces!(learner.traces); end
end

"""
    mutable struct Critic <: AbstractBiasCorrector
        α::Float64
        V::Array{Float64, 1}
"""
mutable struct Critic <: AbstractBiasCorrector
    α::Float64
    V::Array{Float64, 1}
end
export Critic
"""
    Critic(; α = .1, ns = 10, initvalue = 0.)
"""
Critic(; α = .1, ns = 10, initvalue = 0.) = Critic(α, zeros(ns) + initvalue)
function correct(corrector::Critic, r, s, G = r)
    δ = G - corrector.V[s]
    corrector.V[s] += corrector.α * δ
    δ
end
function correct(corrector::Critic, rewards, γ, s, lasts, isterminal)
    δ = getnsteptderror(rewards, γ, corrector.V[s], corrector.V[lasts], isterminal)
    corrector.V[s] += corrector.α * δ
    δ
end

function update!(::EpisodicLearner, learner, rewards, states, actions, isterminal)
    if isterminal
        G = rewards[end]
        gammaeff = learner.γ^length(rewards)
        tmp = deepcopy(learner.params)
        for t in length(rewards)-1:-1:1
            G = learner.γ * G + rewards[t]
            δ = correct(learner.biascorrector, rewards[t], states[t], G)
            gammaeff *= 1/learner.γ
            probs = getactionprobabilities(learner, states[t])
            gradlogpolicy!(probs, states[t], actions[t], tmp,
                           learner.α * gammaeff * δ)
        end
        copy!(learner.params, tmp)
    else
        if learner.params[actions[end], states[end]] == Inf64
            learner.params[actions[end], states[end]] = 0.
        end
    end
end

# Note: Actor-Critic (episodic) on p 344 of Sutton & Barto 2017 draft optimizes
# for V[s1] and therefore discounts all other values (see I in algo). I don't do
# this here.
function update!(::NstepLearner, learner, rewards, states, actions, isterminal)
    δ = correct(learner.biascorrector, rewards, learner.γ, states[1], states[end],
                isterminal)
    if learner.params[actions[end], states[end]] == Inf64
        learner.params[actions[end], states[end]] = 0.
    end
    gradlogpolicy!(getactionprobabilities(learner, states[1]),
                   states[1], actions[1], learner.params, learner.α * δ)
end

getactionprobabilities(learner::AbstractPolicyGradient, state) =
    getactionprobabilities(SoftmaxPolicy1(), learner.params[:, state])

function gradlogpolicy!(probs, state, action, output, factor = 1.)
    output[action, state] += factor
    na, ns = size(output)
    BLAS.axpy!(-factor, probs, 1:na,
               output, (state - 1) * na + 1 : state * na)
end
