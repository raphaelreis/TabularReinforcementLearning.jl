var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Introduction",
    "title": "Introduction",
    "category": "page",
    "text": "A reinforcement learning library for tabular environments."
},

{
    "location": "#What-is-reinforcement-learning?-1",
    "page": "Introduction",
    "title": "What is reinforcement learning?",
    "category": "section",
    "text": "Wikipedia\nNew Sutton & Barto book"
},

{
    "location": "#Installation-1",
    "page": "Introduction",
    "title": "Installation",
    "category": "section",
    "text": "julia v0.6:Pkg.clone(\"https://github.com/jbrea/TabularReinforcementLearning.jl\")julia v0.5:Pkg.clone(\"https://github.com/jbrea/TabularReinforcementLearning.jl\")\nPkg.checkout(\"TabularReinforcementLearning\", \"v0.5\")If you want to use the utilities compare and  plotcomparison make sure that DataFrames and PyPlot is installed.Pkg.add(\"DataFrames\") \nPkg.add(\"PyPlot\")"
},

{
    "location": "usage/#",
    "page": "Usage",
    "title": "Usage",
    "category": "page",
    "text": ""
},

{
    "location": "usage/#Simple-usage-1",
    "page": "Usage",
    "title": "Simple usage",
    "category": "section",
    "text": "Define an Agent.\nChoose an environment.\nChoose a metric.\nChoose a stopping criterion.\n(Optionally) define an RLSetup.\nLearn with learn!.\nLook at results with getvalue.Exampleagent = Agent(QLearning())\nenv = MDP()\nmetric = TotalReward()\nstop = ConstantNumberSteps(100)\nx = RLSetup(agent, env, metric, stop)\nlearn!(x)\ngetvalue(metric)"
},

{
    "location": "usage/#Advanced-Usage-1",
    "page": "Usage",
    "title": "Advanced Usage",
    "category": "section",
    "text": "Define an Agent by choosing one of the learners, one of the policies and one of the callbacks (e.g. to have an exploration schedule).\nChoose an environment or define the interaction with a custom environment.\n( - 7.) as above.\n(Optionally) compare with optimal solution.Examplelearner = QLearning(na = 5, ns = 500, λ = .8, γ = .95,\n					tracekind = ReplacingTraces, initvalue = 10.)\npolicy = EpsilonGreedyPolicy(.2)\ncallback = ReduceEpsilonPerT(10^4)\nagent = Agent(learner, policy, callback)\nenv = MDP(na = 5, ns = 500, init = \"deterministic\")\nmetric = EvaluationPerT(10^4)\nstop = ConstantNumberSteps(10^6)\nx = RLSetup(agent, env, metric, stop)\n@time learn!(x)\nres = getvalue(metric)\nmdpl = MDPLearner(env, .95)\npolicy_iteration!(mdpl)\nreset!(env)\nx2 = RLSetup(Agent(mdpl, EpsilonGreedyPolicy(.2), ReduceEpsilonPerT(10^4)), \n			 env, EvaluationPerT(10^4), ConstantNumberSteps(10^6))\nrun!(x2)\nres2 = getvalue(x2.metric)"
},

{
    "location": "usage/#Comparisons-1",
    "page": "Usage",
    "title": "Comparisons",
    "category": "section",
    "text": "See section Comparison."
},

{
    "location": "usage/#Examples-1",
    "page": "Usage",
    "title": "Examples",
    "category": "section",
    "text": "See examples."
},

{
    "location": "comparison/#",
    "page": "Comparison",
    "title": "Comparison",
    "category": "page",
    "text": ""
},

{
    "location": "comparison/#TabularReinforcementLearning.compare",
    "page": "Comparison",
    "title": "TabularReinforcementLearning.compare",
    "category": "Function",
    "text": "This function is loaded with loadcompare() (requires DataFrames).\n\ncompare(N, \n	    environment, \n		metric::AbstractEvaluationMetrics, \n		stoppingcriterion::StoppingCriterion, \n		agent1generator::Function, \n		agent2generator::Function, ...)\n\nReturns a DataFrame of N runs of all the agents on the environment. \n\nThe DataFrame has the columns :learner (a string identifying the agent), :value (the result of getvalue(metric) and :seed (the random seed used for this run). This macro requires and loads the module DataFrames (can be installed with Pkg.add(\"DataFrames\"). If environment is an environment generator function a new environment is generated N times. Otherwise the same environment is reset N times.\n\nExamples\n\nresult = compare(10, () -> MDP(), MeanReward(), ConstantNumberSteps(100), \n				 () -> Agent(QLearning(λ = 0.)), \n				 () -> Agent(QLearning(λ = .8)))\n\nThis can also be written as:\n\nmetric = MeanReward()\nstopcrit = ConstantNumberSteps(100)\npol = VeryOptimisticEpsilonGreedyPolicy(.1)\ngetnewmdp() = MDP()\ngetnewQ1() = Agent(QLearning(λ = 0.), policy = pol)\ngetnewQ2() = Agent(QLearning(λ = .8), policy = pol)\nresult = compare(10, getnewmdp, metric, stopcrit, getnewQ1, getnewQ2)\n\n\n\n"
},

{
    "location": "comparison/#TabularReinforcementLearning.plotcomparison",
    "page": "Comparison",
    "title": "TabularReinforcementLearning.plotcomparison",
    "category": "Function",
    "text": "This function is loaded with loadplotcomparison() (requires PyPlot, DataFrames).\n\nplotcomparison(results; labels = Dict(), colors = [], thin = .1, thick = 2, smoothingwindow = 0)\n\nPlots results obtained with compare.\n\nThe dictionary labels can be used to rename the legend entries, e.g. labels = Dict(\"QLearning_1\" => \"QLearning λ = .8\"). The data is smoothed by a moving average of size smoothingwindow (default: no smoothing).\n\n\n\n"
},

{
    "location": "comparison/#comparison-1",
    "page": "Comparison",
    "title": "Comparison Tools",
    "category": "section",
    "text": "Since the comparison tools depend on the packages DataFrames and PyPlot they are not loaded automatically. To use them call loadcomparisontools() or individually loadcompare() and loadplotcomparison().Modules = [TabularReinforcementLearning]\nPages   = [\"comparisontools.jl\"]"
},

{
    "location": "learning/#TabularReinforcementLearning.Agent",
    "page": "Learning",
    "title": "TabularReinforcementLearning.Agent",
    "category": "Type",
    "text": "mutable struct Agent\n	learner::AbstractReinforcementLearner\n	policy::AbstractPolicy\n	callback::AbstractCallback\n\n\n\n"
},

{
    "location": "learning/#TabularReinforcementLearning.Agent-Tuple{Any}",
    "page": "Learning",
    "title": "TabularReinforcementLearning.Agent",
    "category": "Method",
    "text": "Agent(learner; policy = EpsilonGreedyPolicy(.1),  callback = NoCallback())\n\n\n\n"
},

{
    "location": "learning/#TabularReinforcementLearning.Agent-Tuple{TabularReinforcementLearning.AbstractMultistepLearner}",
    "page": "Learning",
    "title": "TabularReinforcementLearning.Agent",
    "category": "Method",
    "text": "Agent(learner::NstepLearner; policy = EpsilonGreedyPolicy(.1), callback = NoCallback())\n\nReplaces policy with SoftmaxPolicy1 for baselearner of type AbstractPolicyGradient.\n\n\n\n"
},

{
    "location": "learning/#TabularReinforcementLearning.Agent-Tuple{TabularReinforcementLearning.AbstractPolicyGradient}",
    "page": "Learning",
    "title": "TabularReinforcementLearning.Agent",
    "category": "Method",
    "text": "Agent(learner::AbstractPolicyGradient; policy = SoftmaxPolicy1(), callback = NoCallback())\n\n\n\n"
},

{
    "location": "learning/#TabularReinforcementLearning.RLSetup",
    "page": "Learning",
    "title": "TabularReinforcementLearning.RLSetup",
    "category": "Type",
    "text": "mutable struct RLSetup\n	agent::Agent\n	environment\n	metric::AbstractEvaluationMetrics\n	stoppingcriterion::StoppingCriterion\n\n\n\n"
},

{
    "location": "learning/#TabularReinforcementLearning.learn!-NTuple{6,Any}",
    "page": "Learning",
    "title": "TabularReinforcementLearning.learn!",
    "category": "Method",
    "text": "learn!(learner, policy, callback, environment, metric, stoppingcriterion)\n\n\n\n"
},

{
    "location": "learning/#TabularReinforcementLearning.learn!-Tuple{TabularReinforcementLearning.Agent,Any,Any,Any}",
    "page": "Learning",
    "title": "TabularReinforcementLearning.learn!",
    "category": "Method",
    "text": "learn!(agent::Agent, environment, metric, stoppingcriterion)\n\n\n\n"
},

{
    "location": "learning/#TabularReinforcementLearning.learn!-Tuple{TabularReinforcementLearning.RLSetup}",
    "page": "Learning",
    "title": "TabularReinforcementLearning.learn!",
    "category": "Method",
    "text": "learn!(x::RLSetup)\n\n\n\n"
},

{
    "location": "learning/#TabularReinforcementLearning.run!",
    "page": "Learning",
    "title": "TabularReinforcementLearning.run!",
    "category": "Function",
    "text": "run!(learner, policy, callback, environment, metric, stoppingcriterion)\n\n\n\n"
},

{
    "location": "learning/#TabularReinforcementLearning.run!-Tuple{TabularReinforcementLearning.Agent,Any,Any,Any}",
    "page": "Learning",
    "title": "TabularReinforcementLearning.run!",
    "category": "Method",
    "text": "run!(agent::Agent, environment, metric, stoppingcriterion)\n\n\n\n"
},

{
    "location": "learning/#TabularReinforcementLearning.run!-Tuple{TabularReinforcementLearning.RLSetup}",
    "page": "Learning",
    "title": "TabularReinforcementLearning.run!",
    "category": "Method",
    "text": "run!(x::RLSetup)\n\n\n\n"
},

{
    "location": "learning/#",
    "page": "Learning",
    "title": "Learning",
    "category": "page",
    "text": "Modules = [TabularReinforcementLearning]\nPages   = [\"learn.jl\"]"
},

{
    "location": "learners/#",
    "page": "Learners",
    "title": "Learners",
    "category": "page",
    "text": ""
},

{
    "location": "learners/#learners-1",
    "page": "Learners",
    "title": "Learners",
    "category": "section",
    "text": ""
},

{
    "location": "learners/#TabularReinforcementLearning.ExpectedSarsa",
    "page": "Learners",
    "title": "TabularReinforcementLearning.ExpectedSarsa",
    "category": "Type",
    "text": "mutable struct ExpectedSarsa <: AbstractTDLearner\n	α::Float64\n	γ::Float64\n	unseenvalue::Float64\n	params::Array{Float64, 2}\n	traces::AbstractTraces\n	policy::AbstractPolicy\n\nExpected Sarsa Learner with learning rate α, discount factor γ,  Q-values params and eligibility traces.\n\nThe Q-values are updated according to Q(a s)    e(a s) where  = r +  sum_a pi(a s) Q(a s) - Q(a s)  with next state s, probability pi(a s) of choosing action a in next state s and e(a s) is the eligibility trace (see NoTraces,  ReplacingTraces and AccumulatingTraces).\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.ExpectedSarsa-Tuple{}",
    "page": "Learners",
    "title": "TabularReinforcementLearning.ExpectedSarsa",
    "category": "Method",
    "text": "ExpectedSarsa(; ns = 10, na = 4, α = .1, γ = .9, λ = .8, \n				tracekind = ReplacingTraces, initvalue = Inf64,\n				unseenvalue = 0.,\n				policy = VeryOptimisticEpsilonGreedyPolicy(.1))\n\nSee also  Initial values, novel actions and unseen values.\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.QLearning",
    "page": "Learners",
    "title": "TabularReinforcementLearning.QLearning",
    "category": "Type",
    "text": "mutable struct QLearning <: AbstractTDLearner\n	α::Float64\n	γ::Float64\n	unseenvalue::Float64\n	params::Array{Float64, 2}\n	traces::AbstractTraces\n\nQLearner with learning rate α, discount factor γ, Q-values params and eligibility traces.\n\nThe Q-values are updated \"off-policy\" according to Q(a s)    e(a s) where  = r +  max_a Q(a s) - Q(a s) with next state s and e(a s) is the eligibility trace (see NoTraces,  ReplacingTraces and AccumulatingTraces).\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.QLearning-Tuple{}",
    "page": "Learners",
    "title": "TabularReinforcementLearning.QLearning",
    "category": "Method",
    "text": "QLearning(; ns = 10, na = 4, α = .1, γ = .9, λ = .8, \n			tracekind = ReplacingTraces, initvalue = Inf64, unseenvalue = 0.)\n\nSee also  Initial values, novel actions and unseen values.\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.Sarsa",
    "page": "Learners",
    "title": "TabularReinforcementLearning.Sarsa",
    "category": "Type",
    "text": "mutable struct Sarsa <: AbstractTDLearner\n	α::Float64\n	γ::Float64\n	unseenvalue::Float64\n	params::Array{Float64, 2}\n	traces::AbstractTraces\n\nSarsa Learner with learning rate α, discount factor γ, Q-values params and eligibility traces.\n\nThe Q-values are updated \"on-policy\" according to Q(a s)    e(a s) where  = r +  Q(a s) - Q(a s) with next state s, next action a and e(a s) is the eligibility trace (see NoTraces,  ReplacingTraces and AccumulatingTraces).\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.Sarsa-Tuple{}",
    "page": "Learners",
    "title": "TabularReinforcementLearning.Sarsa",
    "category": "Method",
    "text": "Sarsa(; ns = 10, na = 4, α = .1, γ = .9, λ = .8, \n		tracekind = ReplacingTraces, initvalue = Inf64, unseenvalue = 0.)\n\nSee also  Initial values, novel actions and unseen values.\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.AccumulatingTraces",
    "page": "Learners",
    "title": "TabularReinforcementLearning.AccumulatingTraces",
    "category": "Type",
    "text": "struct AccumulatingTraces <: AbstractTraces\n	λ::Float64\n	γλ::Float64\n	trace::Array{Float64, 2}\n	minimaltracevalue::Float64\n\nDecaying traces with factor γλ. \n\nTraces are updated according to	e(a s)   1 + e(a s) for the current action-state pair and e(a s)    e(a s) for all other pairs unless e(a s)  minimaltracevalue where the trace is set to 0  (for computational efficiency).\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.AccumulatingTraces-Tuple{}",
    "page": "Learners",
    "title": "TabularReinforcementLearning.AccumulatingTraces",
    "category": "Method",
    "text": "AccumulatingTraces(ns, na, λ::Float64, γ::Float64; minimaltracevalue = 1e-12)\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.NoTraces",
    "page": "Learners",
    "title": "TabularReinforcementLearning.NoTraces",
    "category": "Type",
    "text": "struct NoTraces <: AbstractTraces\n\nNo eligibility traces, i.e. e(a s) = 1 for current action a and state s and zero otherwise.\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.ReplacingTraces",
    "page": "Learners",
    "title": "TabularReinforcementLearning.ReplacingTraces",
    "category": "Type",
    "text": "struct ReplacingTraces <: AbstractTraces\n	λ::Float64\n	γλ::Float64\n	trace::Array{Float64, 2}\n	minimaltracevalue::Float64\n\nDecaying traces with factor γλ. \n\nTraces are updated according to	e(a s)   1 for the current action-state pair and e(a s)    e(a s) for all other pairs unless e(a s)  minimaltracevalue where the trace is set to 0  (for computational efficiency).\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.ReplacingTraces-Tuple{}",
    "page": "Learners",
    "title": "TabularReinforcementLearning.ReplacingTraces",
    "category": "Method",
    "text": "ReplacingTraces(ns, na, λ::Float64, γ::Float64; minimaltracevalue = 1e-12)\n\n\n\n"
},

{
    "location": "learners/#TD-Learner-1",
    "page": "Learners",
    "title": "TD Learner",
    "category": "section",
    "text": "Modules = [TabularReinforcementLearning]\nPages   = [\"tdlearning.jl\", \"traces.jl\"]"
},

{
    "location": "learners/#initunseen-1",
    "page": "Learners",
    "title": "Initial values, novel actions and unseen values",
    "category": "section",
    "text": "For td-error dependent methods, the exploration-exploitation trade-off depends on the initvalue and the unseenvalue.  To distinguish actions that were never choosen before, i.e. novel actions, the default initial Q-value (field param) is initvalue = Inf64. In a state with novel actions, the policy determines how to deal with novel actions. To compute the td-error the unseenvalue is used for states with novel actions.  One way to achieve agressively exploratory behavior is to assure that unseenvalue (or initvalue) is larger than the largest possible Q-value."
},

{
    "location": "learners/#TabularReinforcementLearning.Critic",
    "page": "Learners",
    "title": "TabularReinforcementLearning.Critic",
    "category": "Type",
    "text": "mutable struct Critic <: AbstractBiasCorrector\n	α::Float64\n	V::Array{Float64, 1}\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.Critic-Tuple{}",
    "page": "Learners",
    "title": "TabularReinforcementLearning.Critic",
    "category": "Method",
    "text": "Critic(; α = .1, ns = 10, initvalue = 0.)\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.NoBiasCorrector",
    "page": "Learners",
    "title": "TabularReinforcementLearning.NoBiasCorrector",
    "category": "Type",
    "text": "struct NoBiasCorrector <: AbstractBiasCorrector\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.PolicyGradientBackward",
    "page": "Learners",
    "title": "TabularReinforcementLearning.PolicyGradientBackward",
    "category": "Type",
    "text": "mutable struct PolicyGradientBackward <: AbstractPolicyGradient\n	α::Float64\n	γ::Float64\n	params::Array{Float64, 2}\n	traces::AccumulatingTraces\n	biascorrector::AbstractBiasCorrector\n\nPolicy gradient learning in the backward view.\n\nThe parameters are updated according to paramsa s +=  * r_eff * ea s where r_eff =  r for NoBiasCorrector, r_eff =  r - rmean for RewardLowpassFilterBiasCorrector and e[a, s] is the eligibility trace.\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.PolicyGradientBackward-Tuple{}",
    "page": "Learners",
    "title": "TabularReinforcementLearning.PolicyGradientBackward",
    "category": "Method",
    "text": "PolicyGradientBackward(; ns = 10, na = 4, α = .1, γ = .9, \n			   tracekind = AccumulatingTraces, initvalue = Inf64,\n			   biascorrector = NoBiasCorrector())\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.PolicyGradientForward",
    "page": "Learners",
    "title": "TabularReinforcementLearning.PolicyGradientForward",
    "category": "Type",
    "text": "mutable struct PolicyGradientForward <: AbstractPolicyGradient\n	α::Float64\n	γ::Float64\n	params::Array{Float64, 2}\n	biascorrector::AbstractBiasCorrector\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.RewardLowpassFilterBiasCorrector",
    "page": "Learners",
    "title": "TabularReinforcementLearning.RewardLowpassFilterBiasCorrector",
    "category": "Type",
    "text": "mutable struct RewardLowpassFilterBiasCorrector <: AbstractBiasCorrector\nγ::Float64\nrmean::Float64\n\nFilters the reward with factor γ and uses effective reward (r - rmean) to update the parameters.\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.ActorCriticPolicyGradient-Tuple{}",
    "page": "Learners",
    "title": "TabularReinforcementLearning.ActorCriticPolicyGradient",
    "category": "Method",
    "text": "ActorCriticPolicyGradient(; nsteps = 1, γ = .9, ns = 10, na = 4, \n					        α = .1, αcritic = .1, initvalue = Inf64)\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.EpisodicReinforce-Tuple{}",
    "page": "Learners",
    "title": "TabularReinforcementLearning.EpisodicReinforce",
    "category": "Method",
    "text": "EpisodicReinforce(; kwargs...) = EpisodicLearner(PolicyGradientForward(; kwargs...))\n\n\n\n"
},

{
    "location": "learners/#Policy-Gradient-Learner-1",
    "page": "Learners",
    "title": "Policy Gradient Learner",
    "category": "section",
    "text": "Modules = [TabularReinforcementLearning]\nPages   = [\"policygradientlearning.jl\"]"
},

{
    "location": "learners/#TabularReinforcementLearning.EpisodicLearner",
    "page": "Learners",
    "title": "TabularReinforcementLearning.EpisodicLearner",
    "category": "Type",
    "text": "struct EpisodicLearner <: AbstractMultistepLearner\n	learner::AbstractReinforcementLearner\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.NstepLearner",
    "page": "Learners",
    "title": "TabularReinforcementLearning.NstepLearner",
    "category": "Type",
    "text": "struct NstepLearner <: AbstractReinforcementLearner\n	nsteps::Int64\n	learner::AbstractReinforcementLearner\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.NstepLearner-Tuple{}",
    "page": "Learners",
    "title": "TabularReinforcementLearning.NstepLearner",
    "category": "Method",
    "text": "NstepLearner(; nsteps = 10, learner = Sarsa, kwargs...) = \n	NstepLearner(nsteps, learner(; kwargs...))\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.MonteCarlo",
    "page": "Learners",
    "title": "TabularReinforcementLearning.MonteCarlo",
    "category": "Type",
    "text": "mutable struct MonteCarlo <: AbstractReinforcementLearner\n	Nsa::Array{Int64, 2}\n	γ::Float64\n	Q::Array{Float64, 2}\n\nEstimate Q values by averaging over returns.\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.MonteCarlo-Tuple{}",
    "page": "Learners",
    "title": "TabularReinforcementLearning.MonteCarlo",
    "category": "Method",
    "text": "MonteCarlo(; ns = 10, na = 4, γ = .9)\n\n\n\n"
},

{
    "location": "learners/#N-step-Learner-1",
    "page": "Learners",
    "title": "N-step Learner",
    "category": "section",
    "text": "Modules = [TabularReinforcementLearning]\nPages   = [\"nsteplearner.jl\", \"montecarlo.jl\"]"
},

{
    "location": "learners/#TabularReinforcementLearning.SmallBackups",
    "page": "Learners",
    "title": "TabularReinforcementLearning.SmallBackups",
    "category": "Type",
    "text": "mutable struct SmallBackups <: AbstractReinforcementLearner\n	γ::Float64\n	maxcount::UInt64\n	minpriority::Float64\n	counter::Int64\n	Q::Array{Float64, 2}\n	Qprev::Array{Float64, 2}\n	V::Array{Float64, 1}\n	Nsa::Array{Int64, 2}\n	Ns1a0s0::Array{Dict{Tuple{Int64, Int64}, Int64}, 1}\n	queue::PriorityQueue\n\nSee Harm Van Seijen, Rich Sutton ; Proceedings of the 30th International Conference on Machine Learning, PMLR 28(3):361-369, 2013.\n\nmaxcount defines the maximal number of backups per action, minpriority is the smallest priority still added to the queue.\n\n\n\n"
},

{
    "location": "learners/#TabularReinforcementLearning.SmallBackups-Tuple{}",
    "page": "Learners",
    "title": "TabularReinforcementLearning.SmallBackups",
    "category": "Method",
    "text": "SmallBackups(; ns = 10, na = 4, γ = .9, initvalue = Inf64, maxcount = 3,  				   minpriority = 1e-8)\n\n\n\n"
},

{
    "location": "learners/#Model-Based-Learner-1",
    "page": "Learners",
    "title": "Model Based Learner",
    "category": "section",
    "text": "Modules = [TabularReinforcementLearning]\nPages   = [\"prioritizedsweeping.jl\"]"
},

{
    "location": "policies/#",
    "page": "Policies",
    "title": "Policies",
    "category": "page",
    "text": ""
},

{
    "location": "policies/#policies-1",
    "page": "Policies",
    "title": "Policies",
    "category": "section",
    "text": ""
},

{
    "location": "policies/#TabularReinforcementLearning.EpsilonGreedyPolicy",
    "page": "Policies",
    "title": "TabularReinforcementLearning.EpsilonGreedyPolicy",
    "category": "Type",
    "text": "mutable struct EpsilonGreedyPolicy <: AbstractEpsilonGreedyPolicy\n	ϵ::Float64\n\nChooses the action with the highest value with probability 1 - ϵ and selects an action uniformly random with probability ϵ. For states with actions that where never performed before, the behavior of the VeryOptimisticEpsilonGreedyPolicy is followed.\n\n\n\n"
},

{
    "location": "policies/#TabularReinforcementLearning.OptimisticEpsilonGreedyPolicy",
    "page": "Policies",
    "title": "TabularReinforcementLearning.OptimisticEpsilonGreedyPolicy",
    "category": "Type",
    "text": "mutable struct OptimisticEpsilonGreedyPolicy <: AbstractEpsilonGreedyPolicy\n	ϵ::Float64\n\nEpsilonGreedyPolicy that samples uniformly from the actions with the highest Q-value and novel actions in each state where actions are available that where never chosen before. \n\n\n\n"
},

{
    "location": "policies/#TabularReinforcementLearning.PesimisticEpsilonGreedyPolicy",
    "page": "Policies",
    "title": "TabularReinforcementLearning.PesimisticEpsilonGreedyPolicy",
    "category": "Type",
    "text": "mutable struct PesimisticEpsilonGreedyPolicy <: AbstractEpsilonGreedyPolicy\n	ϵ::Float64\n\nEpsilonGreedyPolicy that does not handle novel actions differently.\n\n\n\n"
},

{
    "location": "policies/#TabularReinforcementLearning.VeryOptimisticEpsilonGreedyPolicy",
    "page": "Policies",
    "title": "TabularReinforcementLearning.VeryOptimisticEpsilonGreedyPolicy",
    "category": "Type",
    "text": "mutable struct VeryOptimisticEpsilonGreedyPolicy <: AbstractEpsilonGreedyPolicy\n	ϵ::Float64\n\nEpsilonGreedyPolicy that samples uniformly from novel actions in each state where actions are available that where never chosen before. See also  Initial values, novel actions and unseen values.\n\n\n\n"
},

{
    "location": "policies/#Epsilon-Greedy-Policies-1",
    "page": "Policies",
    "title": "Epsilon Greedy Policies",
    "category": "section",
    "text": "Modules = [TabularReinforcementLearning]\nPages   = [\"epsilongreedypolicies.jl\"]"
},

{
    "location": "policies/#TabularReinforcementLearning.SoftmaxPolicy",
    "page": "Policies",
    "title": "TabularReinforcementLearning.SoftmaxPolicy",
    "category": "Type",
    "text": "mutable struct SoftmaxPolicy <: AbstractSoftmaxPolicy\n	β::Float64\n\nChoose action a with probability\n\nfrace^beta x_asum_a e^beta x_a\n\nwhere x is a vector of values for each action. In states with actions that were never chosen before, a uniform random novel action is returned.\n\nSoftmaxPolicy(; β = 1.)\n\nReturns a SoftmaxPolicy with default β = 1.\n\n\n\n"
},

{
    "location": "policies/#Softmax-Policies-1",
    "page": "Policies",
    "title": "Softmax Policies",
    "category": "section",
    "text": "Modules = [TabularReinforcementLearning]\nPages   = [\"softmaxpolicy.jl\"]"
},

{
    "location": "mdp/#",
    "page": "Environments",
    "title": "Environments",
    "category": "page",
    "text": ""
},

{
    "location": "mdp/#mdp-1",
    "page": "Environments",
    "title": "Environments",
    "category": "section",
    "text": "To use other environment, please have a look at the API"
},

{
    "location": "mdp/#TabularReinforcementLearning.MDP",
    "page": "Environments",
    "title": "TabularReinforcementLearning.MDP",
    "category": "Type",
    "text": "mutable struct MDP \n	ns::Int64\n	na::Int64\n	state::Int64\n	trans_probs::Array{AbstractArray, 2}\n	reward::Array{Float64, 2}\n	initialstates::Array{Int64, 1}\n	isterminal::Array{Int64, 1}\n\nA Markov Decision Process with ns states, na actions, current state, naxns - array of transition probabilites trans_props which consists for every (action, state) pair of a (potentially sparse) array that sums to 1 (see getprobvecrandom, getprobvecuniform, getprobvecdeterministic for helpers to constract the transition probabilities) naxns - array of reward, array of initial states initialstates, and ns - array of 0/1 indicating if a state is terminal.\n\n\n\n"
},

{
    "location": "mdp/#TabularReinforcementLearning.MDP-Tuple{Any,Any}",
    "page": "Environments",
    "title": "TabularReinforcementLearning.MDP",
    "category": "Method",
    "text": "MDP(ns, na; init = \"random\")\nMDP(; ns = 10, na = 4, init = \"random\")\n\nReturn MDP with init in (\"random\", \"uniform\", \"deterministic\"), where the keyword init determines how to construct the transition probabilites (see also  getprobvecrandom, getprobvecuniform, getprobvecdeterministic).\n\n\n\n"
},

{
    "location": "mdp/#TabularReinforcementLearning.run!-Tuple{TabularReinforcementLearning.MDP,Array{Int64,1}}",
    "page": "Environments",
    "title": "TabularReinforcementLearning.run!",
    "category": "Method",
    "text": "run!(mdp::MDP, policy::Array{Int64, 1}) = run!(mdp, policy[mdp.state])\n\n\n\n"
},

{
    "location": "mdp/#TabularReinforcementLearning.run!-Tuple{TabularReinforcementLearning.MDP,Int64}",
    "page": "Environments",
    "title": "TabularReinforcementLearning.run!",
    "category": "Method",
    "text": "run!(mdp::MDP, action::Int64)\n\nTransition to a new state given action. Returns the new state.\n\n\n\n"
},

{
    "location": "mdp/#TabularReinforcementLearning.setterminalstates!-Tuple{Any,Any}",
    "page": "Environments",
    "title": "TabularReinforcementLearning.setterminalstates!",
    "category": "Method",
    "text": "setterminalstates!(mdp, range)\n\nSets mdp.isterminal[range] .= 1, empties the table of transition probabilities for terminal states and sets the reward for all actions in the terminal state to the same value.\n\n\n\n"
},

{
    "location": "mdp/#TabularReinforcementLearning.treeMDP-Tuple{Any,Any}",
    "page": "Environments",
    "title": "TabularReinforcementLearning.treeMDP",
    "category": "Method",
    "text": "treeMDP(na, depth; init = \"random\", branchingfactor = 3)\n\nReturns a tree structured MDP with na actions and depth of the tree. If init is random, the branchingfactor determines how many possible states a (action, state) pair has. If init = \"deterministic\" the branchingfactor = na.\n\n\n\n"
},

{
    "location": "mdp/#TabularReinforcementLearning.getprobvecdeterministic",
    "page": "Environments",
    "title": "TabularReinforcementLearning.getprobvecdeterministic",
    "category": "Function",
    "text": "getprobvecdeterministic(n, min = 1, max = n)\n\nReturns a SparseVector of length n where one element in min:max has  value 1.\n\n\n\n"
},

{
    "location": "mdp/#TabularReinforcementLearning.getprobvecrandom-Tuple{Any,Any,Any}",
    "page": "Environments",
    "title": "TabularReinforcementLearning.getprobvecrandom",
    "category": "Method",
    "text": "getprobvecrandom(n, min, max)\n\nReturns an array of length n that sums to 1 where all elements outside of min:max are zero.\n\n\n\n"
},

{
    "location": "mdp/#TabularReinforcementLearning.getprobvecrandom-Tuple{Any}",
    "page": "Environments",
    "title": "TabularReinforcementLearning.getprobvecrandom",
    "category": "Method",
    "text": "getprobvecrandom(n)\n\nReturns an array of length n that sums to 1. More precisely, the array is a sample of a Dirichlet distribution with n categories and lpha_1 = cdots =lpha_n = 1.\n\n\n\n"
},

{
    "location": "mdp/#TabularReinforcementLearning.getprobvecuniform-Tuple{Any}",
    "page": "Environments",
    "title": "TabularReinforcementLearning.getprobvecuniform",
    "category": "Method",
    "text": "getprobvecuniform(n)  = fill(1/n, n)\n\n\n\n"
},

{
    "location": "mdp/#Markov-Decision-Processes-(MDPs)-1",
    "page": "Environments",
    "title": "Markov Decision Processes (MDPs)",
    "category": "section",
    "text": "Modules = [TabularReinforcementLearning]\nPages   = [\"mdp.jl\"]"
},

{
    "location": "mdp/#TabularReinforcementLearning.MDPLearner",
    "page": "Environments",
    "title": "TabularReinforcementLearning.MDPLearner",
    "category": "Type",
    "text": "struct MDPLearner\n	gamma::Float64\n	policy::Array{Int64, 1}\n	values::Array{Float64, 1}\n	mdp::MDP\n\nUsed to solve mdp with discount factor gamma.\n\n\n\n"
},

{
    "location": "mdp/#TabularReinforcementLearning.policy_iteration!-Tuple{TabularReinforcementLearning.MDPLearner}",
    "page": "Environments",
    "title": "TabularReinforcementLearning.policy_iteration!",
    "category": "Method",
    "text": "policy_iteration!(mdplearner::MDPLearner)\n\nSolve MDP with policy iteration using MDPLearner.\n\n\n\n"
},

{
    "location": "mdp/#Solving-MDPs-1",
    "page": "Environments",
    "title": "Solving MDPs",
    "category": "section",
    "text": "Modules = [TabularReinforcementLearning]\nPages   = [\"mdplearner.jl\"]"
},

{
    "location": "metrics/#",
    "page": "Evaluation Metrics",
    "title": "Evaluation Metrics",
    "category": "page",
    "text": ""
},

{
    "location": "metrics/#TabularReinforcementLearning.AllRewards",
    "page": "Evaluation Metrics",
    "title": "TabularReinforcementLearning.AllRewards",
    "category": "Type",
    "text": "struct AllRewards <: AbstractEvaluationMetrics\n	rewards::Array{Float64, 1}\n\nRecords all rewards.\n\n\n\n"
},

{
    "location": "metrics/#TabularReinforcementLearning.AllRewards-Tuple{}",
    "page": "Evaluation Metrics",
    "title": "TabularReinforcementLearning.AllRewards",
    "category": "Method",
    "text": "AllRewards()\n\nInitializes with empty array.\n\n\n\n"
},

{
    "location": "metrics/#TabularReinforcementLearning.EvaluationPerEpisode",
    "page": "Evaluation Metrics",
    "title": "TabularReinforcementLearning.EvaluationPerEpisode",
    "category": "Type",
    "text": "EvaluationPerEpisode <: AbstractEvaluationMetrics\n	values::Array{Float64, 1}\n	metric::SimpleEvaluationMetric\n\nStores the value of the simple metric for each episode in values.\n\n\n\n"
},

{
    "location": "metrics/#TabularReinforcementLearning.EvaluationPerEpisode",
    "page": "Evaluation Metrics",
    "title": "TabularReinforcementLearning.EvaluationPerEpisode",
    "category": "Type",
    "text": "EvaluationPerEpisode(metric = MeanReward())\n\nInitializes with empty values array and simple metric (default MeanReward). Other options are TimeSteps (to measure the lengths of episodes) or TotalReward.\n\n\n\n"
},

{
    "location": "metrics/#TabularReinforcementLearning.EvaluationPerT",
    "page": "Evaluation Metrics",
    "title": "TabularReinforcementLearning.EvaluationPerT",
    "category": "Type",
    "text": "EvaluationPerT <: AbstractEvaluationMetrics\n	T::Int64\n	counter::Int64\n	values::Array{Float64, 1}\n	metric::SimpleEvaluationMetric\n\nStores the value of the simple metric after every T steps in values.\n\n\n\n"
},

{
    "location": "metrics/#TabularReinforcementLearning.EvaluationPerT",
    "page": "Evaluation Metrics",
    "title": "TabularReinforcementLearning.EvaluationPerT",
    "category": "Type",
    "text": "EvaluationPerT(T, metric = MeanReward())\n\nInitializes with T, counter = 0, empty values array and simple metric (default MeanReward).  Another option is TotalReward.\n\n\n\n"
},

{
    "location": "metrics/#TabularReinforcementLearning.MeanReward",
    "page": "Evaluation Metrics",
    "title": "TabularReinforcementLearning.MeanReward",
    "category": "Type",
    "text": "mutable struct MeanReward <: TabularReinforcementLearning.SimpleEvaluationMetric\n	meanreward::Float64\n	counter::Int64\n\nComputes iteratively the mean reward.\n\n\n\n"
},

{
    "location": "metrics/#TabularReinforcementLearning.MeanReward-Tuple{}",
    "page": "Evaluation Metrics",
    "title": "TabularReinforcementLearning.MeanReward",
    "category": "Method",
    "text": "MeanReward()\n\nInitializes counter and meanreward to 0.\n\n\n\n"
},

{
    "location": "metrics/#TabularReinforcementLearning.RecordAll",
    "page": "Evaluation Metrics",
    "title": "TabularReinforcementLearning.RecordAll",
    "category": "Type",
    "text": "struct RecordAll <: AbstractEvaluationMetrics\n	r::Array{Float64, 1}\n	a::Array{Int64, 1}\n	s::Array{Int64, 1}\n	isterminal::Array{Bool, 1}\n\nRecords everything.\n\n\n\n"
},

{
    "location": "metrics/#TabularReinforcementLearning.RecordAll-Tuple{}",
    "page": "Evaluation Metrics",
    "title": "TabularReinforcementLearning.RecordAll",
    "category": "Method",
    "text": "RecordAll()\n\nInitializes with empty arrays.\n\n\n\n"
},

{
    "location": "metrics/#TabularReinforcementLearning.TimeSteps",
    "page": "Evaluation Metrics",
    "title": "TabularReinforcementLearning.TimeSteps",
    "category": "Type",
    "text": "mutable struct TimeSteps <: SimpleEvaluationMetric\n	counter::Int64\n\nCounts the number of timesteps the simulation is running.\n\n\n\n"
},

{
    "location": "metrics/#TabularReinforcementLearning.TimeSteps-Tuple{}",
    "page": "Evaluation Metrics",
    "title": "TabularReinforcementLearning.TimeSteps",
    "category": "Method",
    "text": "TimeSteps()\n\nInitializes counter to 0.\n\n\n\n"
},

{
    "location": "metrics/#TabularReinforcementLearning.TotalReward",
    "page": "Evaluation Metrics",
    "title": "TabularReinforcementLearning.TotalReward",
    "category": "Type",
    "text": "mutable struct TotalReward <: TabularReinforcementLearning.SimpleEvaluationMetric\n	reward::Float64\n\nAccumulates all rewards.\n\n\n\n"
},

{
    "location": "metrics/#TabularReinforcementLearning.TotalReward-Tuple{}",
    "page": "Evaluation Metrics",
    "title": "TabularReinforcementLearning.TotalReward",
    "category": "Method",
    "text": "TotalReward()\n\nInitializes reward to 0.\n\n\n\n"
},

{
    "location": "metrics/#metrics-1",
    "page": "Evaluation Metrics",
    "title": "Evaluation Metrics",
    "category": "section",
    "text": "Modules = [TabularReinforcementLearning]\nPages   = [\"metrics.jl\"]"
},

{
    "location": "stop/#",
    "page": "Stopping Criteria",
    "title": "Stopping Criteria",
    "category": "page",
    "text": ""
},

{
    "location": "stop/#TabularReinforcementLearning.ConstantNumberEpisodes",
    "page": "Stopping Criteria",
    "title": "TabularReinforcementLearning.ConstantNumberEpisodes",
    "category": "Type",
    "text": "mutable struct ConstantNumberEpisodes <: StoppingCriterion\n	N::Int64\n	counter::Int64\n\nStops learning when the agent has finished 'N' episodes.\n\n\n\n"
},

{
    "location": "stop/#TabularReinforcementLearning.ConstantNumberEpisodes-Tuple{Any}",
    "page": "Stopping Criteria",
    "title": "TabularReinforcementLearning.ConstantNumberEpisodes",
    "category": "Method",
    "text": "	ConstantNumbeEpisodes(N) = ConstantNumberEpisodes(N, 0)\n\n\n\n"
},

{
    "location": "stop/#TabularReinforcementLearning.ConstantNumberSteps",
    "page": "Stopping Criteria",
    "title": "TabularReinforcementLearning.ConstantNumberSteps",
    "category": "Type",
    "text": "mutable struct ConstantNumberSteps <: StoppingCriterion\n	T::Int64\n	counter::Int64\n\nStops learning when the agent has taken 'T' actions.\n\n\n\n"
},

{
    "location": "stop/#TabularReinforcementLearning.ConstantNumberSteps-Tuple{Any}",
    "page": "Stopping Criteria",
    "title": "TabularReinforcementLearning.ConstantNumberSteps",
    "category": "Method",
    "text": "ConstantNumberSteps(T) = ConstantNumberSteps(T, 0)\n\n\n\n"
},

{
    "location": "stop/#stop-1",
    "page": "Stopping Criteria",
    "title": "Stopping Criteria",
    "category": "section",
    "text": "Modules = [TabularReinforcementLearning]\nPages   = [\"stoppingcriterion.jl\"]"
},

{
    "location": "callbacks/#",
    "page": "Callbacks",
    "title": "Callbacks",
    "category": "page",
    "text": ""
},

{
    "location": "callbacks/#TabularReinforcementLearning.ListofCallbacks",
    "page": "Callbacks",
    "title": "TabularReinforcementLearning.ListofCallbacks",
    "category": "Type",
    "text": "struct ListofCallbacks <: AbstractCallback \n	callbacks::Array{AbstractCallback, 1}\n\nLoops over all callbacks.\n\n\n\n"
},

{
    "location": "callbacks/#TabularReinforcementLearning.NoCallback",
    "page": "Callbacks",
    "title": "TabularReinforcementLearning.NoCallback",
    "category": "Type",
    "text": "struct NoCallback <: AbstractCallback end\n\n\n\n"
},

{
    "location": "callbacks/#TabularReinforcementLearning.ReduceEpsilonPerEpisode",
    "page": "Callbacks",
    "title": "TabularReinforcementLearning.ReduceEpsilonPerEpisode",
    "category": "Type",
    "text": "mutable struct ReduceEpsilonPerEpisode <: AbstractCallback\n	ϵ0::Float64\n	counter::Int64\n\nReduces ϵ of an EpsilonGreedyPolicy after each episode.\n\nIn episode n, ϵ = ϵ0/n\n\n\n\n"
},

{
    "location": "callbacks/#TabularReinforcementLearning.ReduceEpsilonPerEpisode-Tuple{}",
    "page": "Callbacks",
    "title": "TabularReinforcementLearning.ReduceEpsilonPerEpisode",
    "category": "Method",
    "text": "ReduceEpsilonPerEpisode()\n\nInitialize callback.\n\n\n\n"
},

{
    "location": "callbacks/#TabularReinforcementLearning.ReduceEpsilonPerT",
    "page": "Callbacks",
    "title": "TabularReinforcementLearning.ReduceEpsilonPerT",
    "category": "Type",
    "text": "mutable struct ReduceEpsilonPerT <: AbstractCallback\n	ϵ0::Float64\n	T::Int64\n	n::Int64\n	counter::Int64\n\nReduces ϵ of an EpsilonGreedyPolicy after every T steps.\n\nAfter n * T steps, ϵ = ϵ0/n\n\n\n\n"
},

{
    "location": "callbacks/#TabularReinforcementLearning.ReduceEpsilonPerT-Tuple{Any}",
    "page": "Callbacks",
    "title": "TabularReinforcementLearning.ReduceEpsilonPerT",
    "category": "Method",
    "text": "ReduceEpsilonPerT()\n\nInitialize callback.\n\n\n\n"
},

{
    "location": "callbacks/#callbacks-1",
    "page": "Callbacks",
    "title": "Callbacks",
    "category": "section",
    "text": "Modules = [TabularReinforcementLearning]\nPages   = [\"callbacks.jl\"]"
},

{
    "location": "api/#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": ""
},

{
    "location": "api/#API-1",
    "page": "API",
    "title": "API",
    "category": "section",
    "text": "New learners, policies, callbacks, environments, evaluation metrics or stopping criteria need to implement the following functions."
},

{
    "location": "api/#TabularReinforcementLearning.update!",
    "page": "API",
    "title": "TabularReinforcementLearning.update!",
    "category": "Function",
    "text": "update!(learner::TabularReinforcementLearning.AbstractReinforcementLearner, \n		r, s0, a0, s1, a1, iss0terminal)\n\nUpdate learner after observing state s0, performing action a0, receiving reward r, observing next state s1 and performing next action a1. The boolean iss0terminal is true if s0 is a terminal state.\n\nupdate!(learner::Union{NstepLearner, EpisodicLearner}, \n		baselearner::TabularReinforcementLearning.AbstractReinforcementLearner, \n		rewards, states, actions, isterminal)\n\nUpdate baselearner with arrays of maximally n+1 states, n+1 actions, n rewards, if learner is NstepLearner. If learner is EpisodicLearner the arrays grow until the end of an episode. The boolean isterminal is true if states[end-1] is a terminal state.\n\n\n\n"
},

{
    "location": "api/#TabularReinforcementLearning.act-Tuple{Any,Any,Any}",
    "page": "API",
    "title": "TabularReinforcementLearning.act",
    "category": "Method",
    "text": "act(learner::TabularReinforcementLearning.AbstractReinforcementLearner,\n	policy::TabularReinforcementLearning.AbstractPolicy,\n	state)\n\nReturns an action for a learner, using policy in state.\n\n\n\n"
},

{
    "location": "api/#Learners-1",
    "page": "API",
    "title": "Learners",
    "category": "section",
    "text": "Learners that require only a (state, action, reward) triple and possibly the next state and action should implement the first definition. If the learner is also to be used with a NstepLearner one also needs to implement the second  definition.update!act(learner, policy, state)"
},

{
    "location": "api/#TabularReinforcementLearning.act-Tuple{Any,Any}",
    "page": "API",
    "title": "TabularReinforcementLearning.act",
    "category": "Method",
    "text": "act(policy::TabularReinforcementLearning.AbstractPolicy, values)\n\nReturns an action given an array of values (one value for each possible action)  using policy.\n\n\n\n"
},

{
    "location": "api/#TabularReinforcementLearning.getactionprobabilities",
    "page": "API",
    "title": "TabularReinforcementLearning.getactionprobabilities",
    "category": "Function",
    "text": "getactionprobabilities(policy::TabularReinforcementLearning.AbstractPolicy, values)\n\nReturns a array of action probabilities for a given array of values  (one value for each possible action) and policy.\n\n\n\n"
},

{
    "location": "api/#Policies-1",
    "page": "API",
    "title": "Policies",
    "category": "section",
    "text": "act(policy, values)getactionprobabilities"
},

{
    "location": "api/#TabularReinforcementLearning.callback!",
    "page": "API",
    "title": "TabularReinforcementLearning.callback!",
    "category": "Function",
    "text": "callback!(callback::AbstractCallback, learner, policy, r, a, s, isterminal)\n\nCan be used to manipulate the learner or the policy during learning, e.g. to change the learning rate or the exploration rate.\n\n\n\n"
},

{
    "location": "api/#Callbacks-1",
    "page": "API",
    "title": "Callbacks",
    "category": "section",
    "text": "callback!"
},

{
    "location": "api/#TabularReinforcementLearning.interact!",
    "page": "API",
    "title": "TabularReinforcementLearning.interact!",
    "category": "Function",
    "text": "interact!(action, environment)\n\nUpdates the environment and returns the triple state, reward, isterminal,  where state is the new state of the environment (an integer), reward is the reward obtained for the performed action and isterminal is true if the  state is terminal.\n\n\n\n"
},

{
    "location": "api/#TabularReinforcementLearning.getstate",
    "page": "API",
    "title": "TabularReinforcementLearning.getstate",
    "category": "Function",
    "text": "getstate(environment)\n\nReturns the tuple state, isterminal. See also interact!(action, environment).\n\n\n\n"
},

{
    "location": "api/#TabularReinforcementLearning.reset!",
    "page": "API",
    "title": "TabularReinforcementLearning.reset!",
    "category": "Function",
    "text": "reset!(environment)\n\nResets the environment to a possible initial state.\n\n\n\n"
},

{
    "location": "api/#api_environments-1",
    "page": "API",
    "title": "Environments",
    "category": "section",
    "text": "interact!getstatereset!"
},

{
    "location": "api/#TabularReinforcementLearning.evaluate!",
    "page": "API",
    "title": "TabularReinforcementLearning.evaluate!",
    "category": "Function",
    "text": "evaluate!(metric::TabularReinforcementLearning.AbstractEvaluationMetrics, \n		  reward, action, state, isterminal)\n\nUpdates the metric based on the experienced (reward, action, state) triplet and the boolean isterminal that is true if state is terminal.\n\n\n\n"
},

{
    "location": "api/#TabularReinforcementLearning.getvalue",
    "page": "API",
    "title": "TabularReinforcementLearning.getvalue",
    "category": "Function",
    "text": "getvalue(metric)\n\nReturns the value of a metric.\n\n\n\n"
},

{
    "location": "api/#Evaluation-Metrics-1",
    "page": "API",
    "title": "Evaluation Metrics",
    "category": "section",
    "text": "evaluate!getvalue"
},

{
    "location": "api/#TabularReinforcementLearning.isbreak!",
    "page": "API",
    "title": "TabularReinforcementLearning.isbreak!",
    "category": "Function",
    "text": "isbreak!(criterion::TabularReinforcementLearning.StoppingCriterion, r, a, s, isterminal)\n\nReturn true if criterion is matched. See ConstantNumberSteps and ConstantNumberEpisodes for builtin criterions and example for how to define new criterions.\n\n\n\n"
},

{
    "location": "api/#Stopping-Criteria-1",
    "page": "API",
    "title": "Stopping Criteria",
    "category": "section",
    "text": "isbreak!"
},

]}
