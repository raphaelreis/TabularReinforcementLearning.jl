mutable struct ForcedPolicy
    t::Int64
    actions::Array{Int64}
end
export ForcedPolicy
ForcedPolicy(actions) = ForcedPolicy(1, actions)
function selectaction(p::ForcedPolicy, ::)
    if p.t > length(p.actions)
        p.t = 1
    else
        p.t += 1
    end
    p.actions[p.t]
end
mutable struct ForcedEpisode{Ts}
    t::Int64
    states::Ts
    dones::Array{Bool, 1}
    rewards::Array{Float64, 1}
end
export ForcedEpisode
ForcedEpisode(states, dones, rewards) = ForcedEpisode(1, states, dones, rewards)
function interact!(a, env::ForcedEpisode)
    env.t += 1
    env.states[env.t], env.rewards[env.t], env.dones[env.t]
end
function reset!(env::ForcedEpisode)
    env.t = 1
    env.states[1]
end
getstate(env::ForcedEpisode) = (env.states[env.t], env.dones[env.t])
