mutable struct DeepActorCritic{Tbuff, Tmodel, Tpl, Tvl, Topt}
    @common_learner_fields
    model::Tmodel
    policylayer::Tpl
    valuelayer::Tvl
    opt::Topt
    αcritic::Float64
end
export DeepActorCritic
function DeepActorCritic(model; nh = 4, na = 2, γ = .9, nsteps = 5, η = .1,
                         opt = Flux.ADAM, policylayer = Linear(nh, na),
                         valuelayer = Linear(nh, 1), T = Float64,
                         statetype = Array{Float64, 1},
                         αcritic = .1)
    θ = vcat(map(Flux.params, [model, policylayer, valuelayer])...)
    buffer = Buffer(capacity = nsteps + 1, statetype = statetype)
    DeepActorCritic(γ, buffer, model, policylayer, valuelayer, opt(θ), αcritic)
end
@inline function selectaction(learner::DeepActorCritic, policy, state)
    h = learner.model(state)
    p = learner.policylayer(h)
    selectaction(policy, p.data)
end
function update!(learner::DeepActorCritic)
    b = learner.buffer
    !isfull(b) && return
    h1 = learner.model(b.states[1])
    p1 = learner.policylayer(h1)
    v1 = learner.valuelayer(h1)
    r, γeff = discountedrewards(b.rewards, b.done, learner.γ)
    advantage = r - v1.data[1]
    if γeff > 0
        h2 = learner.model(b.states[end])
        v2 = learner.valuelayer(h2)
        advantage += γeff * v2.data[1] 
    end
    Flux.back!(advantage * (-Flux.logsoftmax(p1)[b.actions[1]]  + 
                            learner.αcritic * v1[1]))
    learner.opt()
end

