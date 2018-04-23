include(joinpath(Pkg.dir("TabularReinforcementLearning"),
                 "environments", "atari.jl"))

using TabularReinforcementLearning, Flux
env = AtariEnv("/home/j/.julia/v0.6/AtariAlgos/deps/rom_files/breakout.bin")
na = length(getMinimalActionSet(env.ale))
learner = DQN(Flux.Chain(Flux.Conv((8, 8), 4 => 32, relu, stride = (4, 4)), 
                         Flux.Conv((4, 4), 32 => 64, relu, stride = (2, 2)),
                         Flux.Conv((3, 3), 64 => 64, relu),
                         x -> reshape(x, :, size(x, 4)),
                         Flux.Dense(3136, 512, relu), Linear(512, na)),
              updatetargetevery = 500, replaysize = 10^6);
x = RLSetup(Agent(learner, policy = RepeatActionPolicy(), preprocessor =
                  AtariPreprocessor()), 
            env,
            EvaluationPerEpisode(TotalReward()),
            ConstantNumberSteps(2000));
@time learn!(x)
