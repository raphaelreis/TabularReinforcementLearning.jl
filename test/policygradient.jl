na = 4; depth = 4; α = .1; γ = .9; nepisodes = 1
mdp = treeMDP(na, depth, init = "deterministic")
learner = PolicyGradientBackward(na = na, ns = mdp.ns, α = α, γ = γ)
metric = RecordAll()
params = ConstantNumberSteps(nepisodes*(depth + 1))
x = RLSetup(Agent(learner), mdp, metric, params)
learn!(x)
tmp = zeros(na, mdp.ns) + Inf64
i = depth
for (a, s) in zip(metric.a, metric.s)
    tmp[a, s] = α * (1 - 1/na) * metric.r[end] * γ^i
    i -= 1
end
@test learner.params ≈ tmp

# learner = EpisodicReinforce(na = na, ns = mdp.ns, α = α, γ = γ)
# metric = RecordAll()
# nepisodes = 10^4
# params = ConstantNumberSteps(nepisodes*(depth + 1))
# x = RLSetup(Agent(learner), mdp, metric, params)
# learn!(x)

