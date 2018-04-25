mutable struct DQN{Tnet,Tbuff,Topt}
    @common_learner_fields
    targetnet::Tnet
    trainnet::Tnet
    policynet::Tnet
    updatetargetevery::Int64
    t::Int64
    updateevery::Int64
    opt::Topt
    startlearningat::Int64
    minibatchsize::Int64
    doubledqn::Bool
    nmarkov::Int64
end
export DQN
function DQN(net; replaysize = 10^4, γ = .99, updatetargetevery = 500,
                  datatype = Float64, arraytype = AbstractArray, elemshape = (),
                  startlearningat = 10^3, minibatchsize = 32, nmarkov = 1,
                  opt = Flux.ADAM, updateevery = 4, doubledqn = false)
    net = Flux.gpu(net)
    θ = Flux.params(net)
    DQN(γ, ArrayStateBuffer(; capacity = replaysize, 
                              datatype = datatype,
                              arraytype = arraytype,
                              elemshape = elemshape),
        Flux.mapleaves(Flux.Tracker.data, deepcopy(net)), 
        net,
        Flux.mapleaves(Flux.Tracker.data, net), 
        updatetargetevery, 0,
        updateevery, opt(θ), startlearningat, minibatchsize, doubledqn, nmarkov)
end

@inline function selectaction(learner::DQN, policy, state)
    selectaction(policy, learner.policynet(view(learner.buffer.states,
                                                endof(learner.buffer.states),
                                                learner.nmarkov)))
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
        Flux.loadparams!(learner.targetnet, Flux.params(learner.policynet))
    end
    b = learner.buffer
    indices = StatsBase.sample(learner.nmarkov:length(b.rewards), 
                               learner.minibatchsize, replace = false)
    qa = learner.trainnet(view(b.states, indices, learner.nmarkov))
    qat = learner.targetnet(view(b.states, indices + 1, learner.nmarkov))
    q = selecta(qa, b.actions[indices])
    rs = Float64[]
    for (k, i) in enumerate(indices)
        r, γeff = discountedrewards(b.rewards[i], 
                                    b.done[i], learner.γ)
        if γeff > 0
            if learner.doubledqn
                r += learner.γ * qat[indmax(qa.data[:,k]), k]
            else
                r += learner.γ * maximum(qat[:, k])
            end
        end
        push!(rs, r)
    end
    Flux.back!(Flux.mse(q, Flux.gpu(rs)))
    learner.opt()
end
