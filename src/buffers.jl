struct Buffer{Ts, Ta}
    states::CircularBuffer{Ts}
    actions::CircularBuffer{Ta}
    rewards::CircularBuffer{Float64}
    done::CircularBuffer{Bool}
end
export Buffer
function Buffer(; statetype = Int64, actiontype = Int64, capacity = 2)
    Buffer(CircularBuffer{statetype}(capacity),
           CircularBuffer{actiontype}(capacity),
           CircularBuffer{Float64}(capacity-1),
           CircularBuffer{Bool}(capacity-1))
end
function pushstateaction!(b, s, a)
    push!(b.states, deepcopy(s))
    push!(b.actions, deepcopy(a))
end
function pushreturn!(b, r, done)
    push!(b.rewards, r)
    push!(b.done, done)
end

struct EpisodeBuffer{Ts, Ta}
    states::Array{Ts, 1}
    actions::Array{Ta, 1}
    rewards::Array{Float64, 1}
    done::Array{Bool, 1}
end
EpisodeBuffer(; statetype = Int64, actiontype = Int64) = 
    EpisodeBuffer(statetype[], actiontype[], Float64[], Bool[])
function pushreturn!(b::EpisodeBuffer, r, done)
    if length(b.done) > 0 && b.done[end]
        s = b.states[end]; a = b.actions[end]
        empty!(b.states); empty!(b.actions); empty!(b.rewards); empty!(b.done)
        push!(b.states, s)
        push!(b.actions, a)
    end
    push!(b.rewards, r)
    push!(b.done, done)
end

struct AdvantageBuffer{Tg, Ta}
    pg::CircularBuffer{Tg}
    actions::CircularBuffer{Ta}
    values::CircularBuffer{Float64}
    rewards::CircularBuffer{Float64}
    done::CircularBuffer{Bool}
end
export AdvantageBuffer
function AdvantageBuffer(; gradienttype = Array{Any, 1}, actiontype = Int64, 
                           nsteps = 1, capacity = nsteps + 1, γ = .9)
    AdvantageBuffer(CircularBuffer{gradienttype}(capacity),
                    CircularBuffer{actiontype}(capacity),
                    CircularBuffer{Float64}(capacity),
                    CircularBuffer{Float64}(capacity - 1),
                    CircularBuffer{Bool}(capacity - 1))
end
pushstateaction!(b::AdvantageBuffer, s, a) = push!(b.actions, a)

import DataStructures.isfull
isfull(b::Union{Buffer, AdvantageBuffer}) = isfull(b.rewards)
