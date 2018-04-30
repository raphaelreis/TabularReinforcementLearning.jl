include(joinpath(Pkg.dir("TabularReinforcementLearning"),
                 "environments", "atari.jl"))

using TabularReinforcementLearning, Flux
# using CuArrays
env = AtariEnv("examples/atarirom_files/breakout.bin")
na = length(getMinimalActionSet(env.ale))
learner = DQN(Chain(Conv((8, 8), 4 => 32, relu, stride = (4, 4)), 
                         Conv((4, 4), 32 => 64, relu, stride = (2, 2)),
                         Conv((3, 3), 64 => 64, relu),
                         x -> reshape(x, :, size(x, 4)),
                         Dense(3136, 512, relu), Linear(512, na)),
              updatetargetevery = 10^4, replaysize = 35*10^4, nmarkov = 4,
              startlearningat = 50000, minibatchsize = 14);
x = RLSetup(Agent(learner, policy = EpsilonGreedyPolicy(1.), 
                  preprocessor = AtariPreprocessor(),
                  callback = ListofCallbacks([Progress(100),
                             LinearDecreaseEpsilon(5 * 10^4, 10^6, 1, .1)])), 
            env,
            EvaluationPerEpisode(TotalReward()),
            ConstantNumberSteps(60000));
@time learn!(x)

env = AtariEnv("examples/atarirom_files/breakout.bin", colorspace = "Raw")
na = length(getMinimalActionSet(env.ale))
learner2 = Sarsa(na = na, ns = 20652352, λ = .9, γ = .99, α = .005, 
                 tracekind = ReplacingTraces)
x2 = RLSetup(Agent(learner2, policy = EpsilonGreedyPolicy(.01),
             preprocessor = AtariBPROST(),
             callback = Progress(100)),
             env,
             EvaluationPerEpisode(TotalReward()),
             ConstantNumberSteps(60000));
@time learn!(x2)
