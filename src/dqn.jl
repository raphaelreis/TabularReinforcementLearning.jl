mutable struct DQN{Tnet,Tbuff,Topt}
    @common_learner_fields
    targetnet::Tnet
    policynet::Tnet
    updatetargetevery::Int64
    t::Int64
    nsteps::Int64
    updateevery::Int64
    opt::Topt
    startlearningat::Int64
    minibatchsize::Int64
    doubledqn::Bool
end
export DQN
function DQN(net; replaysize = 10^4, γ = .99, updatetargetevery = 500,
                  statetype = Array{Float64, 1}, nsteps = 1, 
                  startlearningat = 10^3, minibatchsize = 32,
                  opt = Flux.ADAM, updateevery = 4, doubledqn = false)
    θ = Flux.params(net)
    DQN(γ, Buffer(; capacity = replaysize, statetype = statetype),
        Flux.mapleaves(Flux.Tracker.data, net) |> Flux.gpu, net |> Flux.gpu, 
        updatetargetevery, 0, nsteps, 
        updateevery, opt(θ), startlearningat, minibatchsize, doubledqn)
end
@inline function selectaction(learner::DQN, policy, state)
    selectaction(policy, learner.policynet(Flux.gpu(state)).data)
end
function selecta(q, a)
    na, t = size(q)
    q[na * collect(0:t-1) .+ a]
end
import StatsBase
function update!(learner::DQN)
    learner.t += 1
    (learner.t <= learner.startlearningat || 
     learner.t % learner.updateevery != 0) && return
    if learner.t % learner.updatetargetevery == 0
        Flux.loadparams!(learner.targetnet, Flux.params(learner.policynet)) # this is slow
    end
    b = learner.buffer
    indices = StatsBase.sample(1:length(b.rewards), learner.minibatchsize, replace = false)
    qa = learner.policynet(Flux.gpu(lastcat(b.states[indices])))
    qat = learner.targetnet(Flux.gpu(lastcat(b.states[indices + 1])))
    q = selecta(qa, b.actions[indices])
    rs = Float64[]
    for (k, i) in enumerate(indices)
        r, γeff = discountedrewards(b.rewards[i:i + learner.nsteps - 1], 
                                    b.done[i:i + learner.nsteps - 1], learner.γ)
        if γeff > 0
            if learner.doubledqn
                r += learner.γ * qat[indmax(qa.data[:,k]), k]
            else
                r += learner.γ * maximum(qat[:, k])
            end
        end
        push!(rs, r)
    end
    Flux.back!(Flux.mse(q, rs))
    learner.opt()
end
