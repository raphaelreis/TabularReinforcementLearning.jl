## Simple usage

1. Define an [`Agent`](@ref).
2. Choose an [environment](@ref mdp).
3. Choose a [metric](@ref metrics).
4. Choose a [stopping criterion](@ref stop).
5. (Optionally) define an [`RLSetup`](@ref).
6. Learn with [`learn!`](@ref).
7. Look at results with [`getvalue`](@ref).

Example

```julia
agent = Agent(QLearning())
env = MDP()
metric = TotalReward()
stop = ConstantNumberSteps(100)
x = RLSetup(agent, env, metric, stop)
learn!(x)
getvalue(metric)
```

## Advanced Usage

1. Define an [`Agent`](@ref) by choosing one of the [learners](@ref learners), one of the
   [policies](@ref policies) and one of the [callbacks](@ref callbacks) (e.g. to have an
   exploration schedule).
2. Choose an [environment](@ref mdp) or define the [interaction with a custom
   environment](@ref api_environments).
3. ( - 7.) as above.
8. (Optionally) compare with optimal solution.

Example

```julia
learner = QLearning(na = 5, ns = 500, λ = .8, γ = .95,
                    tracekind = ReplacingTraces, initvalue = 10.)
policy = EpsilonGreedyPolicy(.2)
callback = ReduceEpsilonPerT(10^4)
agent = Agent(learner, policy, callback)
env = MDP(na = 5, ns = 500, init = "deterministic")
metric = EvaluationPerT(10^4)
stop = ConstantNumberSteps(10^6)
x = RLSetup(agent, env, metric, stop)
@time learn!(x)
res = getvalue(metric)
mdpl = MDPLearner(env, .95)
policy_iteration!(mdpl)
reset!(env)
x2 = RLSetup(Agent(mdpl, EpsilonGreedyPolicy(.2), ReduceEpsilonPerT(10^4)), 
             env, EvaluationPerT(10^4), ConstantNumberSteps(10^6))
run!(x2)
res2 = getvalue(x2.metric)
```

## Comparisons

See section [`Comparison`](@ref comparison).
## Examples

See [examples](https://github.com/jbrea/TabularReinforcementLearning.jl/tree/master/examples).
