mutable struct TDLearner{T, Tbuff}
    @common_learner_fields
    α::Float64
    unseenvalue::Float64
    params::Array{Float64, 2}
    traces::T
    initvalue::Float64
    endvaluepolicy
end

function TDLearner(; ns = 10, na = 4, α = .1, γ = .9, λ = .8, 
                     nsteps = 1, 
                     discretestates = false,
                     initvalue = discretestates ? Inf64 : 0.,
                     statetype = discretestates ? Int64 : Array{Float64, 1},
                     buffer = nsteps == Inf ?
                        EpisodeBuffer(statetype = statetype) :
                        Buffer(capacity = nsteps + 1, statetype = statetype),
                     tracekind = ReplacingTraces, 
                     unseenvalue = initvalue == Inf64 ? 0. : initvalue, 
                     endvaluepolicy = :Sarsa)
    TDLearner(γ, buffer, Float64(α), Float64(unseenvalue), zeros(na, ns) .+ initvalue,
               λ == 0. || tracekind == NoTraces ? NoTraces() : 
               tracekind(ns, na, λ, γ), Float64(initvalue), endvaluepolicy)
end
Sarsa(; kargs...) = TDLearner(; kargs...)
QLearning(; kargs...) = TDLearner(; endvaluepolicy = :QLearning, kargs...)
ExpectedSarsa(; kargs...) = TDLearner(; endvaluepolicy = VeryOptimisticEpsilonGreedyPolicy(.1), kargs...)
export Sarsa, QLearning, ExpectedSarsa

# td error

@inline getvaluecheckinf(learner, a, s) = checkinf(learner, getvalue(learner.params, a, s))
@inline getvaluecheckinf(learner, a, s::AbstractArray) = getvalue(learner.params, a, s)
@inline checkinf(learner, value) = (value == Inf64 ? learner.unseenvalue : value)

@inline function futurevalue(learner)
    a = learner.buffer.actions[end]
    s = learner.buffer.states[end]
    if learner.endvaluepolicy == :QLearning
        checkinf(learner, maximumbelowInf(getvalue(learner.params, s)))
    elseif learner.endvaluepolicy == :Sarsa
        getvaluecheckinf(learner, a, s)
    else
        actionprobabilites = getactionprobabilities(learner.endvaluepolicy,
                                                    getvalue(learner.params, s))
        m = 0.
        for (a, w) in enumerate(actionprobabilites)
            if w != 0.
                m += w * getvaluecheckinf(learner, a, s)
            end
        end
        m
    end
end

@inline function discountedrewards(rewards, done, γ)
    gammaeff = 1.
    discr = 0.
    for (r, done) in zip(rewards, done)
        discr += gammaeff * r
        done && return [discr; 0.]
        gammaeff *= γ
    end
    [discr; gammaeff]
end
@inline function tderror(rewards, done, γ, startvalue, endvalue)
    discr, gammaeff = discountedrewards(rewards, done, γ)
    discr + gammaeff * endvalue - startvalue
end

function tderror(learner)
    tderror(learner.buffer.rewards, learner.buffer.done, learner.γ,
            getvaluecheckinf(learner, learner.buffer.actions[1], learner.buffer.states[1]),
            futurevalue(learner))
end

# update params

@inline function updateparam!(learner, s, a, δ)
    if learner.params[a, s] == Inf64
        learner.params[a, s] = learner.unseenvalue + δ
    else
        learner.params[a, s] += learner.α * δ
    end
end
@inline function updateparam!(learner, s::Vector, a, δ)
    na, ns = size(learner.params)
    BLAS.axpy!(learner.α * δ, s, 1:ns, learner.params, a:na:na * (ns - 1) + a)
end
@inline function updateparam!(learner, s::SparseVector, a, δ)
    @simd for i in 1:length(s.nzind)
        learner.params[a, s.nzind[i]] += learner.α * δ * s.nzval[i]
    end
end

@inline updatetraceandparams!(learner::TDLearner{NoTraces, <:Any}, s, a, δ) =
    updateparam!(learner, s, a, δ)
@inline function updatetraceandparams!(learner, s, a, δ)
    increasetrace!(learner.traces, s, a)
    updatetraceandparams!(learner.traces, learner.params, learner.α * δ)
    if learner.initvalue == Inf && learner.params[a, s] == Inf
        learner.params[a, s] = learner.unseenvalue + δ
    end
    if learner.buffer.done[1]; resettraces!(learner.traces); end
end

# update

function update!(learner::TDLearner)
    !isfull(learner.buffer) && return
    updatetraceandparams!(learner, 
                          learner.buffer.states[1], 
                          learner.buffer.actions[1],
                          tderror(learner))
end
 
function getvalues(learner::TDLearner)
    [maximum(learner.params[:, i]) for i in 1:size(learner.params, 2)]
end
export getvalues

