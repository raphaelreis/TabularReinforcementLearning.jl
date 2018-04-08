"""
    mutable struct ConstantNumberSteps <: StoppingCriterion
        T::Int64
        counter::Int64

Stops learning when the agent has taken 'T' actions.
"""
mutable struct ConstantNumberSteps <: StoppingCriterion
    T::Int64
    counter::Int64
end
"""
    ConstantNumberSteps(T) = ConstantNumberSteps(T, 0)
"""
ConstantNumberSteps(T) = ConstantNumberSteps(T, 0)
function isbreak!(criterion::ConstantNumberSteps, done, buffer)
    criterion.counter += 1
    if criterion.counter == criterion.T
        criterion.counter = 0
        return true
    end
    false
end
export ConstantNumberSteps

"""
    mutable struct ConstantNumberEpisodes <: StoppingCriterion
        N::Int64
        counter::Int64

Stops learning when the agent has finished 'N' episodes.
"""

mutable struct ConstantNumberEpisodes <: StoppingCriterion
    N::Int64
    counter::Int64
end
"""
        ConstantNumbeEpisodes(N) = ConstantNumberEpisodes(N, 0)
"""
ConstantNumberEpisodes(N) = ConstantNumberEpisodes(N, 0)
function isbreak!(criterion::ConstantNumberEpisodes, done, buffer)
    if done
        criterion.counter += 1
        if criterion.counter == criterion.N
            criterion.counter = 0
            return true
        end
    end
    false
end
export ConstantNumberEpisodes

