"""
    struct NoCallback <: AbstractCallback end
"""
struct NoCallback <: AbstractCallback end
export NoCallback
callback!(::NoCallback, learner, policy, metric, stop) = Void

"""
    struct ListofCallbacks <: AbstractCallback 
        callbacks::Array{AbstractCallback, 1}

Loops over all `callbacks`.
"""
struct ListofCallbacks <: AbstractCallback
    callbacks::Array{AbstractCallback, 1}
end
export ListofCallbacks
function callback!(c::ListofCallbacks, learner, policy, metric, stop)
    for callback in c.callbacks
        callback!(callback, learner, policy, metric, stop)
    end
end

"""
    mutable struct ReduceEpsilonPerEpisode <: AbstractCallback
        ϵ0::Float64
        counter::Int64

Reduces ϵ of an [`EpsilonGreedyPolicy`](@ref) after each episode.

In episode n, ϵ = ϵ0/n
"""
mutable struct ReduceEpsilonPerEpisode <: AbstractCallback
    ϵ0::Float64
    counter::Int64
end
"""
    ReduceEpsilonPerEpisode()

Initialize callback.
"""
ReduceEpsilonPerEpisode() = ReduceEpsilonPerEpisode(0., 1)
function callback!(c::ReduceEpsilonPerEpisode, learner, 
                   policy::AbstractEpsilonGreedyPolicy, metric, stop)
    if learner.buffer.done[end]
        if c.counter == 1
            c.ϵ0 = policy.ϵ
        end
        c.counter += 1
        policy.ϵ = c.ϵ0 / c.counter
    end
end
export ReduceEpsilonPerEpisode

"""
    mutable struct ReduceEpsilonPerT <: AbstractCallback
        ϵ0::Float64
        T::Int64
        n::Int64
        counter::Int64

Reduces ϵ of an [`EpsilonGreedyPolicy`](@ref) after every `T` steps.

After n * T steps, ϵ = ϵ0/n
"""
mutable struct ReduceEpsilonPerT <: AbstractCallback
    ϵ0::Float64
    T::Int64
    n::Int64
    counter::Int64
end
"""
    ReduceEpsilonPerT()

Initialize callback.
"""
ReduceEpsilonPerT(T) = ReduceEpsilonPerT(0., T, 1, 1)
function callback!(c::ReduceEpsilonPerT, learner, 
                   policy::AbstractEpsilonGreedyPolicy, metric, stop)
    c.counter += 1
    if c.counter == c.T
        c.counter == 1
        if c.n == 1
            c.ϵ0 = policy.ϵ
        end
        c.n += 1
        policy.ϵ = c.ϵ0 / c.n
    end
end
export ReduceEpsilonPerT

mutable struct LinearDecreaseEpsilon <: AbstractCallback
    start::Int64
    stop::Int64
    initval::Float64
    finalval::Float64
    t::Int64
    step::Float64
end
export LinearDecreaseEpsilon
function LinearDecreaseEpsilon(start, stop, initval, finalval)
    step = (finalval - initval)/(stop - start)
    LinearDecreaseEpsilon(start, stop, initval, finalval, 0, step)
end
function callback!(c::LinearDecreaseEpsilon, learner, policy, metric, stop)
    c.t += 1
    if c.t == 1 policy.ϵ = c.initval
    elseif c.t >= c.start && c.t < c.stop
        policy.ϵ += c.step
    else
        policy.ϵ = c.finalvalue
    end
end

struct Progress <: AbstractCallback end
export Progress
function callback!(c::Progress, learner, policy, metric, stop::ConstantNumberSteps)
    if stop.counter % div(stop.T, 10) == 0
        value = getvalue(metric)
        lastvaluestring = length(value) > 1 ? "; last value: $(value[end])" : ""
        info("$(now()) $(stop.counter) of $(stop.T) steps$lastvaluestring.")
    end
end
