using PyCall, TabularReinforcementLearning
@pyimport gym
@pyimport roboschool

import TabularReinforcementLearning.interact!,
TabularReinforcementLearning.getstate,
TabularReinforcementLearning.reset!

function getspace(space)
    if pyisinstance(space, gym.spaces[:box][:Box])
        TabularReinforcementLearning.Box(space[:low], space[:high])
    elseif pyisinstance(space, gym.spaces[:discrete][:Discrete])
        1:space[:n]
    else
        error("Don't know how to convert $(pytypeof(space)).")
    end
end
struct GymEnv{TObject, TObsSpace, TActionSpace}
    pyobj::TObject
    observation_space::TObsSpace
    action_space::TActionSpace
end
function GymEnv(name::String)
    pyenv = gym.make(name)
    obsspace = getspace(pyenv[:observation_space])
    actspace = getspace(pyenv[:action_space])
    env = GymEnv(pyenv, obsspace, actspace)
    reset!(env)
    env
end

interact!(action, env::GymEnv) = env.pyobj[:step](action)
interact!(action::Int64, env::GymEnv) = env.pyobj[:step](action - 1)
reset!(env::GymEnv) = env.pyobj[:reset]()
getstate(env::GymEnv) = (env.pyobj[:env][:state], false) # doesn't work for all envs

gym.envs[:registry][:all]()

env = GymEnv("CartPole-v0")
agent = Agent(Sarsa(na = 2, ns = 160, initvalue = 0, α = .01), 
              preprocessor = StateAggregator(env.observation_space, 
                                             40 * ones(4)))
metric = EvaluationPerEpisode(TotalReward())
x = RLSetup(agent, env, metric, ConstantNumberEpisodes(10^4))
@time learn!(x)
using PlotlyJS
plot(metric.values)

for _ in 1:500
    env.pyobj[:render]()
    run!(agent, env, AllRewards(), ConstantNumberSteps(1))
end
env.pyobj[:close]()

env = GymEnv("RoboschoolHumanoidFlagrun-v1")
for _ in 1:1000
    env.pyobj[:render]()
    s, r, d = interact!(rand(17), env)
    if d break end
end

# env = GymEnv("RoboschoolHumanoidFlagrun-v1")
# a = rand(17)
# using BenchmarkTools
# @benchmark interact!($a, $env)
# 
# env = GymEnv("RoboschoolHumanoidFlagrun-v1")
# i = 0
# res = @time for _ in 1:100
#     for _ in 1:1000
#         i += 1
#         s, r, d = interact!(rand(17), env)
#         if d break end
#     end
#     reset!(env)
# end

