using ArcadeLearningEnvironment, Images
import DataStructures:CircularBuffer
import TabularReinforcementLearning.interact!,
TabularReinforcementLearning.getstate,
TabularReinforcementLearning.reset!,
TabularReinforcementLearning.preprocessstate,
TabularReinforcementLearning.selectaction

struct AtariEnv
    ale::Ptr{Void}
    screen::Array{UInt8, 1}
end
function AtariEnv(name)
    ale = ALE_new()
    try
        loadROM(ale, name)
    catch err
        println(err)
        error("ROM $name could not be loaded.")
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

struct AtariPreprocessor
    buffer::CircularBuffer{Array{Float64, 4}}
end
function AtariPreprocessor(; n = 4, statetype = Array{Float64, 4})
    p = AtariPreprocessor(CircularBuffer{statetype}(n))
    for i in 1:n
        push!(p.buffer, zeros(84, 84, 1, 1))
    end
    p
end
function preprocessstate(p::AtariPreprocessor, s)
    colorimg = colorview(RGB, reshape(normedview(s), 3, 160, 210))
    push!(p.buffer, reshape(Float64.(imresize(Gray.(colorimg), 84, 84)), 
                            84, 84, 1, 1))
    cat(3, p.buffer...)
end
function preprocessstate(p::AtariPreprocessor, ::Void)
    push!(p.buffer, zeros(84, 84, 1, 1))
    cat(3, p.buffer...)
end

mutable struct RepeatActionPolicy{T}
    i::Int64
    k::Int64
    a::Int64
    policy::T
end # TODO: how to reset at end of episode?
RepeatActionPolicy(; k = 4, a = 1, policy = SoftmaxPolicy1()) = 
    RepeatActionPolicy(0, k, a, policy)
function selectaction(learner::TabularReinforcementLearning.DQN, 
                      p::RepeatActionPolicy, state)
    p.i += 1
    if p.i == p.k
        p.i = 0
        p.a = selectaction(learner, p.policy, state)
    else
        p.a
    end
end
