include("mdpexamples.jl")
loadcomparisontools()
m, mdp, goal, mapping = getmazemdp(returnall = true, nwalls = 200)
params = ((:na, 4), (:ns, 1600), (:γ, .99), (:initvalue, Inf64))
ql() = Agent(QLearning(; α = .05, λ = .9, params...))
sb() = Agent(SmallBackups(; maxcount = 10, params...))
pgac() = Agent(ActorCriticPolicyGradient(; α = 1e-1, αcritic = .1, nsteps = 1,
                                         params...))
result = @compare(10, getmazemdp(nwalls = 200, offset = 1), EvaluationPerT(10^4),
                  ConstantNumberSteps(10^6),
                  ql(), sb(), pgac())
plotcomparison(result)

