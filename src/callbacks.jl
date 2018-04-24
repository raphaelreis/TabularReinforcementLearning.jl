"""
    struct NoCallback <: AbstractCallback end
"""
struct NoCallback <: AbstractCallback end
export NoCallback
callback!(::NoCallback, learner, policy) = Void

"""
    struct ListofCallbacks <: AbstractCallback 
        callbacks::Array{AbstractCallback, 1}

Loops over all `callbacks`.
"""
struct ListofCallbacks <: AbstractCallback
    callbacks::Array{AbstractCallback, 1}
end
export ListofCallbacks
function callback!(c::ListofCallbacks, learner, policy)
    for callback in c.callbacks
        callback!(callback, learner, policy)
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
                   policy::AbstractEpsilonGreedyPolicy)
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
                   policy::AbstractEpsilonGreedyPolicy)
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
function callback!(c::LinearDecreaseEpsilon, learner, policy)
    c.t += 1
    if c.t == 1 policy.ϵ = c.initval end
    if c.t >= c.start && c.t < c.stop
        policy.ϵ -= c.step
    end
end


