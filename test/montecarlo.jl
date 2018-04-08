T = TabularReinforcementLearning
γ = .9
learner = MonteCarlo(ns = 4, na = 1, γ = γ, discretestates = true)
T.pushstateaction!(learner.buffer, 1, 1)
for i in 2:4
    T.pushreturn!(learner.buffer, iseven(i), i == 4)
    T.pushstateaction!(learner.buffer, i, 1)
    update!(learner)
end
@test learner.Q == [1 + γ^2 γ 1 Inf64]

