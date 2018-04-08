import Knet

# type defs
abstract type Layer end
struct Id <: Layer end # used for testing
params(l::Id) = []
nparams(l::Id) = 0
(l::Id)(x) = x
(l::Id)(w, x) = x

struct Linear{T} <: Layer
    w::Array{T, 2}
end
Linear(dimin, dimout; T = Float64, initfun = zeros) = Linear(initfun(T, dimout, dimin))
params(l::Linear) = l.w
nparams(l::Linear) = 1
(l::Linear)(x) = l.w * Knet.mat(x)
(l::Linear)(w, x) = w * Knet.mat(x)

struct Dense{T,Tu} <: Layer
    w::Array{T, 2}
    b::Array{T, 1}
    unit::Tu
end
Dense(dimin, dimout; T = Float64, unit = Knet.relu, initfun = Knet.xavier) = 
    Dense(initfun(T, dimout, dimin), zeros(T, dimout), unit)

params(l::Dense) = Any[l.w, l.b]
nparams(l::Dense) = 2
(l::Dense)(x) = l.unit.(l.w * Knet.mat(x) .+ l.b)
(l::Dense)(w, x) = l.unit.(w[1] * Knet.mat(x) .+ w[2]) # needed for grad

struct Conv{T,Tu} <: Layer
    w::Array{T, 4}
    b::Array{T, 4}
    unit::Tu
    convkargs
end
Conv(xdim, ydim, inc, outc; T = Float64, unit = Knet.relu, 
     initfun = Knet.xavier, convkargs...) =
    Conv(initfun(T, xdim, ydim, inc, outc), zeros(T, 1, 1, outc, 1), unit, convkargs)

params(l::Conv) = Any[l.w, l.b]
nparams(l::Conv) = 2
(l::Conv)(x) = l.unit.(Knet.conv4(l.w, x; l.convkargs...) .+ l.b)
(l::Conv)(w, x) = l.unit.(Knet.conv4(w[1], x; l.convkargs...) .+ w[2])

struct Pooling <: Layer
    poolkargs
end
Pooling(; kargs...) = Pooling(kargs)

params(l::Pooling) = Any[]
nparams(l::Pooling) = 0
(l::Pooling)(x) = Knet.pool(x; l.poolkargs...)
(l::Pooling)(w, x) = l(x)

struct Chain
    layers::Array{Any, 1}
end
Chain(l...) = Chain(collect(l))
Chain(l::Layer) = Chain([l])
params(c::Chain) = [params(l) for l in c.layers]
nparams(c::Chain) = sum(map(nparams, c.layers))
(c::Chain)(x) = foldl((x, c) -> c(x), x, c.layers)
(c::Chain)(w, x) = foldl((x, c) -> c[2](c[1], x), x, zip(w, c.layers))

struct Model
    chain::Chain
    w::Array{Any, 1}
end
Model(c) = Model(c, params(c))
(m::Model)(x) = (m.chain)(x)
struct A2CModel{Lp, Lv}
    chain::Chain
    policylayer::Lp
    valuelayer::Lv
    w::Array{Any, 1}
end
function A2CModel(c, pl, vl)
    parc = params(c); parp = params(pl); parv = params(vl)
    A2CModel(c, pl, vl, convert(Array{Any, 1}, [parc..., parp, parv]))
end
(m::A2CModel)(x) = m(m.w, x)
function (m::A2CModel)(w, x)
    h = m.chain(w[1:end-2], x)
    m.policylayer(w[end-1], h), m.valuelayer(w[end], h)
end
export Model, A2CModel, Id, Linear, Chain, Dense, Conv, Pooling

import Knet.optimizers
optimizers(m::Union{Model, A2CModel}, opt) = optimizers(m.w, opt)

function a2closs(w, m, xstart, xend, action, rewardsums)
    pistart, vstart = m(w, xstart)
    piend, vend = m(w, xend)
    ad = rewardsums[1] + vend.value * rewardsums[2] - vstart
    -mean(ad.value .* Knet.logp(pistart,1)[action])# + mean(ad.^2)
end
a2cgradfun2 = Knet.grad(a2closs)

function forward_pass(chain, w, xstart, xend, action, rewardsums)
    tape = Knet.AutoGrad.Tape()
    w = Knet.AutoGrad.Rec(w, tape)
    ystart = chain(w, xstart)
    yend = chain(w, xend)
    ystartcopy = deepcopy(ystart)
    end_box1 = -Knet.logp(ystart[1:end-1], 1)[action]
    vestim = rewardsums[1] + yend[end].value * rewardsums[2]
    advantage = vestim - ystartcopy[end]
    end_box2 = advantage^2
    return w, end_box1, end_box2, advantage.value
end
function a2cgradfun(chain, w, xstart, xend, action, rewardsums, αcritic)
    sbox, ebox1, ebox2, ad = forward_pass(chain, w, xstart, xend, 
                                                action, rewardsums)
    x = Knet.AutoGrad.backward_pass(sbox, ebox1, ebox1.tapes[1])
    for w in x scale!(w, ad) end
    if αcritic > 0
        y = Knet.AutoGrad.backward_pass(sbox, ebox2, ebox2.tapes[1])
        for i in 1:length(x) 
            BLAS.axpy!(αcritic, y[i], x[i])
        end
    end
    x
end

mutable struct A2C{Tbuff}
    @common_learner_fields
    model::Model
    batchsize::Int64
    αcritic::Float64
    opt
end
export A2C
function A2C(chain; γ = .9, nsteps = 1, opt = Knet.Adam, replaysize = 10^4,
             batchsize = 128, αcritic = 1.,
             buffer = ReplayBuffer(nsteps = nsteps, capacity = replaysize))
    model = Model(chain)
    A2C(γ, buffer, model, batchsize, αcritic, optimizers(model, opt))
end
@inline selectaction(learner::A2C, policy, state) = 
    selectaction(policy, learner.model(state)[1:end-1])

mutable struct A2C2{Tbuff}
    @common_learner_fields
    model::A2CModel
    batchsize::Int64
    αcritic::Float64
    opt
end
export A2C2
function A2C2(chain; γ = .9, nsteps = 1, opt = Knet.Adam, replaysize = 10^4,
             batchsize = 128, αcritic = 1., na = 2, nh = 4,
             policylayer = Linear(nh, na),
             valuelayer = Linear(nh, 1),
             buffer = ReplayBuffer(nsteps = nsteps, capacity = replaysize))
    model = A2CModel(chain, policylayer, valuelayer)
    A2C2(γ, buffer, model, batchsize, αcritic, optimizers(model, opt))
end
@inline selectaction(learner::A2C2, policy, state) = 
    selectaction(policy, learner.model(state)[1]) # TODO: no need to comp. value

import StatsBase
function update!(learner::A2C)
    buffer = learner.buffer
    !isfull(buffer) && return
    indices = StatsBase.sample(1:length(buffer.rewardsums), 1, replace=false)[1]
    xstart = buffer.states[indices]
    xend = buffer.states[indices+buffer.nsteps]
    rewardsums = buffer.rewardsums[indices]
    Knet.update!(learner.model.w, 
                 a2cgradfun(learner.model.chain, learner.model.w, xstart, xend,
                            buffer.actions[indices], rewardsums, learner.αcritic),
                 learner.opt)
end
function update!(learner::A2C2)
    buffer = learner.buffer
    !isfull(buffer) && return
    indices = StatsBase.sample(1:length(buffer.rewardsums), 1, replace=false)[1]
    xstart = buffer.states[indices]
    xend = buffer.states[indices+buffer.nsteps]
    rewardsums = buffer.rewardsums[indices]
    Knet.update!(learner.model.w, 
                 a2cgradfun2(learner.model.w, learner.model, xstart, xend,
                            buffer.actions[indices], rewardsums),
                 learner.opt)
end

