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
mutable struct PolicyGradientBackward{Tbuff} <: AbstractPolicyGradient
    @common_learner_fields
    α::Float64
    params::Array{Float64, 2}
    traces::AccumulatingTraces
    biascorrector::AbstractBiasCorrector
    initvalue::Float64
end
export PolicyGradientBackward
"""
    PolicyGradientBackward(; ns = 10, na = 4, α = .1, γ = .9, 
                   tracekind = AccumulatingTraces, initvalue = Inf64,
                   biascorrector = NoBiasCorrector())
"""
function PolicyGradientBackward(; ns = 10, na = 4, α = .1, γ = .9, 
                                  tracekind = AccumulatingTraces,
                                  discretestates = false,
                                  initvalue = discretestates ? Inf64 : 0., 
                                  statetype = discretestates ? Int64 : Array{Float64, 1},
                                  buffer = Buffer(statetype = statetype),
                                  biascorrector = NoBiasCorrector())
        PolicyGradientBackward(γ, buffer, α, zeros(na, ns) + initvalue,
                               tracekind(ns, na, 1., γ),
                               biascorrector, Float64(initvalue))
end

"""
    mutable struct PolicyGradientForward <: AbstractPolicyGradient
        α::Float64
        γ::Float64
        params::Array{Float64, 2}
        biascorrector::AbstractBiasCorrector
"""
mutable struct PolicyGradientForward{Tbuff} <: AbstractPolicyGradient
    @common_learner_fields
    α::Float64
    params::Array{Float64, 2}
    biascorrector::AbstractBiasCorrector
    initvalue::Float64
end
export PolicyGradientForward
function PolicyGradientForward(; ns = 10, na = 4, α = .1, γ = .9,
                         discretestates = false,
                         initvalue = discretestates ? Inf64 : 0., 
                         statetype = discretestates ? Int64 : Array{Float64, 1},
                         buffertype = Buffer,
                         buffer = buffertype(statetype = statetype),
                         biascorrector = NoBiasCorrector())
    PolicyGradientForward(Float64(γ), buffer, Float64(α), zeros(na, ns) + initvalue, 
                          biascorrector, Float64(initvalue))
end
"""
    EpisodicReinforce(; kwargs...) = EpisodicLearner(PolicyGradientForward(; kwargs...))
"""
EpisodicReinforce(; kwargs...) =
    PolicyGradientForward(; buffertype = EpisodeBuffer, kwargs...)
export EpisodicReinforce
"""
    ActorCriticPolicyGradient(; nsteps = 1, γ = .9, ns = 10, na = 4, 
                                α = .1, αcritic = .1, initvalue = Inf64)
"""
ActorCriticPolicyGradient(; nsteps = 1, γ = .9, ns = 10,
                            αcritic = .1, kargs...) =
        PolicyGradientForward(; biascorrector = Critic(γ = γ, ns = ns, α = αcritic),
                        ns = ns, γ = γ, 
                        buffertype = (;x...) -> Buffer(; capacity = nsteps + 1, x...),
                        kargs...)
export ActorCriticPolicyGradient


# bias correctors

"""
    struct NoBiasCorrector <: AbstractBiasCorrector
"""
struct NoBiasCorrector <: AbstractBiasCorrector end
export NoBiasCorrector
correct(::NoBiasCorrector, buffer, t = 1, G = buffer.rewards[t]) = G

"""
    mutable struct RewardLowpassFilterBiasCorrector <: AbstractBiasCorrector
    λ::Float64
    rmean::Float64

Filters the reward with factor λ and uses effective reward (r - rmean) to update
the parameters.
"""
mutable struct RewardLowpassFilterBiasCorrector <: AbstractBiasCorrector
    λ::Float64
    rmean::Float64
end
export RewardLowpassFilterBiasCorrector
RewardLowpassFilterBiasCorrector(λ) = RewardLowpassFilterBiasCorrector(λ, 0.)
function correct(corrector::RewardLowpassFilterBiasCorrector, buffer, 
                 t = 1, G = buffer.rewards[t])
    corrector.rmean *= corrector.λ
    corrector.rmean += (1 - corrector.λ) * buffer.rewards[t]
    G - corrector.rmean
end


"""
    mutable struct Critic <: AbstractBiasCorrector
        α::Float64
        V::Array{Float64, 1}
"""
mutable struct Critic <: AbstractBiasCorrector
    α::Float64
    γ::Float64
    V::Array{Float64, 1}
end
export Critic
"""
    Critic(; γ = .9, α = .1, ns = 10, initvalue = 0.)
"""
Critic(; γ = .9, α = .1, ns = 10, initvalue = 0.) = Critic(α, γ, zeros(ns) + initvalue)
function correct(corrector::Critic, buffer, t = 1, G = buffer.rewards[t])
    s = buffer.states[t]
    δ = tderror(buffer.rewards, buffer.done, corrector.γ,
                getvalue(corrector.V, s), 
                getvalue(corrector.V, buffer.states[end]))
    if typeof(s) <: Int
        corrector.V[s] += corrector.α * δ
    else
        corrector.V .+= corrector.α * δ * s
    end
    δ
end


# update helper 

getactionprobabilities(learner::AbstractPolicyGradient, s) =
    getactionprobabilities(SoftmaxPolicy1(), getvalue(learner.params, s))

function gradlogpolicy!(probs, state::Int, action, output, factor = 1.)
    na, ns = size(output)
    output[action, state] += factor
    BLAS.axpy!(-factor, probs, 1:na, output, (state - 1) * na + 1 : state * na)
end

function gradlogpolicy!(probs, state::Vector, action, output, factor = 1.)
    na, ns = size(output)
    output[action, :] += factor * state
    BLAS.ger!(-factor, probs, state, output)
end

function update!(learner, r, s, a)
    δ = correct(learner.biascorrector, learner.buffer)
    updatetraceandparams!(learner.traces, learner.params, learner.α * δ)
    if learner.initvalue == Inf && learner.params[a, s] == Inf
        learner.params[a, s] = learner.α * δ * learner.traces.trace[a, s] /
                                    learner.traces.γλ 
                                    # because updatetraceandparams updates 
    end
end

# update

function update!(learner::PolicyGradientBackward)
    s = learner.buffer.states[1]; a = learner.buffer.actions[1];
    gradlogpolicy!(getactionprobabilities(learner, s), s, a, learner.traces.trace)
    update!(learner, learner.buffer.rewards[1], s, a)
    if learner.buffer.done[1]; resettraces!(learner.traces); end
end


function update!(learner::PolicyGradientForward{<:EpisodeBuffer})
    if learner.buffer.done[end]
        rewards = learner.buffer.rewards
        states = learner.buffer.states
        actions = learner.buffer.actions
        G = rewards[end]
        gammaeff = learner.γ^length(rewards)
        tmp = deepcopy(learner.params)
        for t in length(rewards)-1:-1:1
            G = learner.γ * G + rewards[t]
            δ = correct(learner.biascorrector, learner.buffer, t, G)
            gammaeff *= 1/learner.γ
            probs = getactionprobabilities(learner, states[t])
            gradlogpolicy!(probs, states[t], actions[t], tmp,
                           learner.α * gammaeff * δ)
        end
        copy!(learner.params, tmp)
    else
        if learner.initvalue == Inf && learner.params[actions[end], states[end]] == Inf
            learner.params[actions[end], states[end]] = 0.
        end
    end
end

# Note: Actor-Critic (episodic) on p 344 of Sutton & Barto 2017 draft optimizes
# for V[s1] and therefore discounts all other values (see I in algo). I don't do
# this here.
function update!(learner::PolicyGradientForward{<:Buffer})
    !isfull(learner.buffer) && return
    rewards = learner.buffer.rewards
    states = learner.buffer.states
    actions = learner.buffer.actions
    δ = correct(learner.biascorrector, learner.buffer)
    if learner.initvalue == Inf && learner.params[actions[end], states[end]] == Inf
        learner.params[actions[end], states[end]] = 0.
    end
    gradlogpolicy!(getactionprobabilities(learner, states[1]),
                   states[1], actions[1], learner.params, learner.α * δ)
end


