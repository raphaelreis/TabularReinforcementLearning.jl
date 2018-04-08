"""
    mutable struct MeanReward <: TabularReinforcementLearning.SimpleEvaluationMetric
        meanreward::Float64
        counter::Int64

Computes iteratively the mean reward.
"""
mutable struct MeanReward <: SimpleEvaluationMetric
    meanreward::Float64
    counter::Int64
end
"""
    MeanReward()

Initializes `counter` and `meanreward` to 0.
"""
MeanReward() = MeanReward(0., 0)
function evaluate!(p::MeanReward, r, done, buffer)
    p.counter += 1
    p.meanreward += 1/p.counter * (r - p.meanreward)
end
function reset!(p::MeanReward)
    p.counter = 0
    p.meanreward = 0.
end
getvalue(p::MeanReward) = p.meanreward
export MeanReward, getvalue

"""
    mutable struct TotalReward <: TabularReinforcementLearning.SimpleEvaluationMetric
        reward::Float64

Accumulates all rewards.
"""
mutable struct TotalReward <: SimpleEvaluationMetric
    reward::Float64
end
"""
    TotalReward()

Initializes `reward` to 0.
"""
TotalReward() = TotalReward(0.)
function evaluate!(p::TotalReward, r, done, buffer)
    p.reward += r
end
function reset!(p::TotalReward)
    p.reward = 0.
end
getvalue(p::TotalReward) = p.reward
export TotalReward

"""
    mutable struct TimeSteps <: SimpleEvaluationMetric
        counter::Int64

Counts the number of timesteps the simulation is running.
"""
mutable struct TimeSteps <: SimpleEvaluationMetric
    counter::Int64
end
"""
    TimeSteps()

Initializes `counter` to 0.
"""
TimeSteps() = TimeSteps(0)
function evaluate!(p::TimeSteps, r, done, buffer)
    p.counter += 1
end
function reset!(p::TimeSteps)
    p.counter = 0
end
getvalue(p::TimeSteps) = p.counter
export TimeSteps

"""
    EvaluationPerEpisode <: AbstractEvaluationMetrics
        values::Array{Float64, 1}
        metric::SimpleEvaluationMetric

Stores the value of the simple `metric` for each episode in `values`.
"""
struct EvaluationPerEpisode <: AbstractEvaluationMetrics
    values::Array{Float64, 1}
    metric::SimpleEvaluationMetric
end
"""
    EvaluationPerEpisode(metric = MeanReward())

Initializes with empty `values` array and simple `metric` (default
[`MeanReward`](@ref)).
Other options are [`TimeSteps`](@ref) (to measure the lengths of episodes) or
[`TotalReward`](@ref).
"""
EvaluationPerEpisode(metric = MeanReward()) = EvaluationPerEpisode(Float64[],
                                                                   metric)
function evaluate!(p::EvaluationPerEpisode, r, done, buffer)
    evaluate!(p.metric, r, done, buffer)
    if done
        push!(p.values, getvalue(p.metric))
        reset!(p.metric)
    end
end
function reset!(p::EvaluationPerEpisode)
    reset!(p.metric)
    empty!(p.values)
end
getvalue(p::EvaluationPerEpisode) = deepcopy(p.values)
export EvaluationPerEpisode

"""
    EvaluationPerT <: AbstractEvaluationMetrics
        T::Int64
        counter::Int64
        values::Array{Float64, 1}
        metric::SimpleEvaluationMetric

Stores the value of the simple `metric` after every `T` steps in `values`.
"""
mutable struct EvaluationPerT <: AbstractEvaluationMetrics
    T::Int64
    counter::Int64
    values::Array{Float64, 1}
    metric::SimpleEvaluationMetric
end
"""
    EvaluationPerT(T, metric = MeanReward())

Initializes with `T`, `counter` = 0, empty `values` array and simple `metric`
(default [`MeanReward`](@ref)).  Another option is [`TotalReward`](@ref).
"""
EvaluationPerT(T, metric = MeanReward()) = EvaluationPerT(T, 0, Float64[],
                                                          metric)
function evaluate!(p::EvaluationPerT, r, done, buffer)
    evaluate!(p.metric, r, done, buffer)
    p.counter += 1
    if p.counter == p.T
        push!(p.values, getvalue(p.metric))
        reset!(p.metric)
        p.counter = 0
    end
end
function reset!(p::EvaluationPerT)
    reset!(p.metric)
    p.counter = 0
    empty!(p.values)
end
getvalue(p::EvaluationPerT) = deepcopy(p.values)
export EvaluationPerT

"""
    struct RecordAll <: AbstractEvaluationMetrics
        r::Array{Float64, 1}
        a::Array{Int64, 1}
        s::Array{Int64, 1}
        done::Array{Bool, 1}

Records everything.
"""
struct RecordAll <: AbstractEvaluationMetrics
    r::Array{Float64, 1}
    a::Array{Int64, 1}
    s::Array{Any, 1}
    done::Array{Bool, 1}
end
"""
    RecordAll()

Initializes with empty arrays.
"""
RecordAll() = RecordAll(Float64[], Int64[], [], Bool[])
function evaluate!(p::RecordAll, r, done, buffer)
    push!(p.r, r)
    push!(p.a, buffer.actions[1])
    push!(p.s, buffer.states[1])
    push!(p.done, done)
end
function reset!(p::RecordAll)
    empty!(p.r); empty!(p.a); empty!(p.s); empty!(p.done)
end
getvalue(p::RecordAll) = deepcopy(p)
export RecordAll

"""
    struct AllRewards <: AbstractEvaluationMetrics
        rewards::Array{Float64, 1}
    
Records all rewards.
"""
struct AllRewards <: AbstractEvaluationMetrics
    rewards::Array{Float64, 1}
end
"""
    AllRewards()

Initializes with empty array.
"""
AllRewards() = AllRewards(Float64[])
function evaluate!(p::AllRewards, r, done, buffer)
    push!(p.rewards, r)
end
function reset!(p::AllRewards)
    empty!(p.rewards)
end
getvalue(p::AllRewards) = deepcopy(p.rewards)
export AllRewards
