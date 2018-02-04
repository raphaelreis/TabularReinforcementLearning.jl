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
mutable struct GymEnvState
    done::Bool
end
struct GymEnv{TObject, TObsSpace, TActionSpace}
    pyobj::TObject
    observation_space::TObsSpace
    action_space::TActionSpace
    state::GymEnvState
end
function GymEnv(name::String)
    pyenv = gym.make(name)
    obsspace = getspace(pyenv[:observation_space])
    actspace = getspace(pyenv[:action_space])
    env = GymEnv(pyenv, obsspace, actspace, GymEnvState(false))
    reset!(env)
    env
end

function interactgym!(action, env)
    if env.state.done 
        s = reset!(env)
        r = 0
        d = false
    else
        s, r, d = env.pyobj[:step](action)
    end
    env.state.done = d
    s, r, d
end
interact!(action, env::GymEnv) = interactgym!(action, env)
interact!(action::Int64, env::GymEnv) = interactgym!(action - 1, env)
reset!(env::GymEnv) = env.pyobj[:reset]()
getstate(env::GymEnv) = (env.pyobj[:env][:state], false) # doesn't work for all envs


# List all envs

gym.envs[:registry][:all]()



# CartPole example

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



# RoboSchool example

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



# MountainCar
struct MountainCarPreprocessor{TFuncApprox}
    funcapprox::TFuncApprox
end
import TabularReinforcementLearning.preprocess
function preprocess(p::MountainCarPreprocessor, s, r, done)
    preprocess(p.funcapprox, s, r, s[1] >= .5) # could do reward shaping here
end
env = GymEnv("MountainCar-v0")
mcpreprocessor = MountainCarPreprocessor(StateAggregator(env.observation_space, 
                                         100 * ones(2)))
agent = Agent(Sarsa(na = 2, ns = 200, initvalue = 0, 
                    α = .5, γ = 1., λ = 0.),
              policy = VeryOptimisticEpsilonGreedyPolicy(.2),
              preprocessor = mcpreprocessor)
metric = EvaluationPerEpisode(TimeSteps())
x = RLSetup(agent, env, metric, ConstantNumberSteps(2 * 10^6))
@time learn!(x) # takes ~10 min on my machine
using PlotlyJS
plot(metric.values)

# s1 = map(s -> findfirst(s[1:100])/100*1.8 - 1.2, metric.s)
