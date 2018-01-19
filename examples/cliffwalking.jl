include("mdpexamples.jl")
loadcomparisontools()
params = ((:ns, 48), (:na, 4), (:γ, 1.), (:λ, .0))
ql() = Agent(QLearning(; α = .2, params...))
sarsa() = Agent(Sarsa(; α = .2, params...))
esarsa() = Agent(ExpectedSarsa(; α = .2, params...))
nstep() = Agent(NstepLearner(; nsteps = 10, learner = Sarsa, 
                           α = .02, params...))
pg() = Agent(EpisodicReinforce(; α = 2e-7, params[1:3]...))
pgbc() = Agent(EpisodicReinforce(; α = 2e-7, 
                                 biascorrector = Critic(ns = 48, α = 1e-7),
                                 params[1:3]...))
pgb() = Agent(PolicyGradientBackward(; α = 2e-7, biascorrector =
                                     RewardLowpassFilterBiasCorrector(.99, 0.), 
                                     params[1:3]...))
pgac() = Agent(ActorCriticPolicyGradient(; α = 1e-1, αcritic = .1, nsteps = 1,
                                 params[1:3]...))
results = compare(200, getcliffwalkingmdp(), EvaluationPerEpisode(TotalReward()),
                   ConstantNumberEpisodes(500), ql, sarsa, esarsa,
                   nstep, pgac);
plotcomparison(results, smoothingwindow = 20, thin = .01,
               labels = Dict("NstepLearner_1" => "ActorCriticPolicyGradient",
                             "EpisodicLearner" => "EpisodicReinforce",
                             "EpisodicLearner_1" => "bias corrected Reinforce"))
plt[:yscale]("symlog")
#plt[:ylim]([-100,-10])
results2 = @compare(10, getcliffwalkingmdp(), 
                    EvaluationPerT(10^3, TotalReward()),
                    ConstantNumberSteps(10^6), pgac(), pg(), pgbc(), pgb());
figure()
plotcomparison(results2, smoothingwindow = 20, thin = .01,
               labels = Dict("NstepLearner" => "ActorCriticPolicyGradient",
                             "EpisodicLearner" => "EpisodicReinforce",
                             "EpisodicLearner_1" => "bias corrected Reinforce"))
plt[:yscale]("symlog")


# 
# x = RLSetup(pgb(), getcliffwalkingmdp(), EvaluationPerEpisode(TotalReward()),
#           ConstantNumberSteps(50000))
# learn!(x)
