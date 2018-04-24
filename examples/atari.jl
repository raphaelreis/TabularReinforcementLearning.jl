include(joinpath(Pkg.dir("TabularReinforcementLearning"),
                 "environments", "atari.jl"))

using TabularReinforcementLearning, Flux
# using CuArrays
env = AtariEnv("examples/atarirom_files/breakout.bin")
na = length(getMinimalActionSet(env.ale))
learner = DQN(Flux.Chain(Flux.Conv((8, 8), 4 => 32, relu, stride = (4, 4)), 
                         Flux.Conv((4, 4), 32 => 64, relu, stride = (2, 2)),
                         Flux.Conv((3, 3), 64 => 64, relu),
                         x -> reshape(x, :, size(x, 4)),
                         Flux.Dense(3136, 512, relu), Linear(512, na)),
              updatetargetevery = 500, replaysize = 10^6,
              startlearningat = 50000);
x = RLSetup(Agent(learner, policy = RepeatActionPolicy(), 
                  preprocessor = AtariPreprocessor(),
                  callback = LinearDecreaseEpsilon(5 * 10^4, 10^6, 1, .1)), 
            env,
            EvaluationPerEpisode(TotalReward()),
            ConstantNumberSteps(200000));
@time learn!(x)
