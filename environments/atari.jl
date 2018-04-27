using ArcadeLearningEnvironment, Images
import DataStructures: CircularBuffer
import TabularReinforcementLearning
import Flux
const T = TabularReinforcementLearning
import T.interact!, T.getstate, T.reset!, T.preprocessstate, T.selectaction,
T.callback!

struct AtariEnv
    ale::Ptr{Void}
    screen::Array{UInt8, 1}
end
function AtariEnv(name)
    if isfile(name)
        ale = ALE_new()
        loadROM(ale, name)
    else
        error("ROM $name could not be found.")
    end
    screen = Array{UInt8}(210*160*3)
    AtariEnv(ale, screen)
end
function interact!(a, env::AtariEnv)
    r = act(env.ale, Int32(a - 1))
    getScreenRGB(env.ale, env.screen)
    env.screen, r, game_over(env.ale)
end
function getstate(env::AtariEnv)
    getScreenRGB(env.ale, env.screen)
    env.screen, game_over(env.ale)
end
reset!(env::AtariEnv) = reset_game(env.ale)

struct AtariPreprocessor end
function preprocessstate(::AtariPreprocessor, s)
    colorimg = colorview(RGB, reshape(normedview(s), 3, 160, 210))
    Flux.gpu(reshape(Float16.(imresize(Gray.(colorimg), 84, 84)), 84, 84, 1))
end
function preprocessstate(p::AtariPreprocessor, ::Void)
    Flux.gpu(zeros(Float16, 84, 84, 1))
end

mutable struct RepeatActionPolicy{T}
    i::Int64
    k::Int64
    a::Int64
    policy::T
end # TODO: how to reset at end of episode?
RepeatActionPolicy(; k = 4, a = 1, policy = EpsilonGreedyPolicy(1.)) = 
    RepeatActionPolicy(0, k, a, policy)
function selectaction(learner::T.DQN, 
                      p::RepeatActionPolicy, state)
    p.i += 1
    if p.i == p.k
        p.i = 0
        p.a = selectaction(learner, p.policy, state)
    else
        p.a
    end
end
callback!(c::T.LinearDecreaseEpsilon, learner, p::RepeatActionPolicy, args...) = 
    callback!(c, learner, p.policy, args...)
