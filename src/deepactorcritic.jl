mutable struct DeepActorCritic{Tbuff, Tmodel, Tpl, Tplm, Tvl, Topt}
    @common_learner_fields
    model::Tmodel
    policylayer::Tpl
    policymodel::Tplm
    valuelayer::Tvl
    opt::Topt
    αcritic::Float64
    nenvs::Int64
end
export DeepActorCritic
function DeepActorCritic(model; nh = 4, na = 2, γ = .9, nsteps = 5, η = .1,
                         opt = Flux.ADAM, policylayer = Linear(nh, na),
                         valuelayer = Linear(nh, 1),
                         statetype = Array{Float64, 1},
                         αcritic = .1, nenvs = 1)
    θ = vcat(map(Flux.params, [model, policylayer, valuelayer])...)
    buffer = ArrayStateBuffer(capacitystates = nenvs * (nsteps + 1),
                              capacityrewards = nenvs * nsteps)
    policymodel = Flux.Chain(Flux.mapleaves(Flux.Tracker.data, model),
                             Flux.mapleaves(Flux.Tracker.data, policylayer))
    DeepActorCritic(γ, buffer, model, policylayer, policymodel,
                    valuelayer, opt(θ), αcritic, nenvs)
end
@inline function selectaction(learner::DeepActorCritic, policy, state)
    selectaction(policy, learner.policymodel(state))
end
function update!(learner::DeepActorCritic)
    b = learner.buffer
    !isfull(b) && return
    h1 = learner.model(b.states[1:learner.nenvs])
    p1 = learner.policylayer(h1)
    v1 = learner.valuelayer(h1)[:]
    advantage = similar(v1.data)
    for i in 1:learner.nenvs
        r, γeff = discountedrewards(b.rewards[i:learner.nenvs:end], 
                                    b.done[i:learner.nenvs:end], learner.γ)
        advantage[i] = r - v1.data[i]
        if γeff > 0
            h2 = learner.model(b.states[end - learner.nenvs + i])
            v2 = learner.valuelayer(h2)
            advantage[i] += γeff * v2.data[1] 
        end
    end
    Flux.back!(dot(advantage, 
                   -selecta(Flux.logsoftmax(p1), b.actions[1:learner.nenvs]) .+ 
                            learner.αcritic * v1))
    learner.opt()
end
