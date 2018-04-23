import Knet

# type defs
abstract type Layer end
struct Id <: Layer end # used for testing
params(l::Id) = Any[]
nparams(l::Id) = 0
(l::Id)(x) = x
(l::Id)(w, x) = x

struct KLinear{T} <: Layer
    w::Array{T, 2}
end
KLinear(dimin, dimout; T = Float64, initfun = zeros) = KLinear(initfun(T, dimout, dimin))
params(l::KLinear) = Any[l.w]
nparams(l::KLinear) = 1
(l::KLinear)(x) = l.w * Knet.mat(x)
(l::KLinear)(w, x) = w[1] * Knet.mat(x)

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
export Model, A2CModel, Id, KLinear, Chain, Dense, Conv, Pooling

import Knet.optimizers
optimizers(m::Union{Model, A2CModel}, opt) = optimizers(m.w, opt)

function a2closs(w, m, xstart, xend, action, r, γeff, αcritic)
    pistart, vstart = m(w, xstart)
    ad = r - vstart.value[1]
    if  γeff > 0
        piend, vend = m(w, xend)
        ad += γeff * vend.value[1]
    end
    ad * (-Knet.logp(pistart,1)[action] + αcritic * vstart[1]) # + mean(ad.^2)
end
a2cgradfun2 = Knet.grad(a2closs)

mutable struct A2C{Tbuff}
    @common_learner_fields
    model::A2CModel
    αcritic::Float64
    opt
end
export A2C
params(l::A2C) = l.model.w
function setparams!(l::A2C, w)
    for i in 1:length(w)
        for j in 1:length(w[i])
            l.model.w[i][j] = w[i][j]
        end
    end
end
function A2C(chain; γ = .9, nsteps = 1, opt = Knet.Adam,
              αcritic = 1., na = 2, nh = 4,
              policylayer = KLinear(nh, na),
              valuelayer = KLinear(nh, 1),
              statetype = Array{Float64, 1},
              buffer = Buffer(capacity = nsteps + 1, 
                              statetype = statetype))
    model = A2CModel(chain, policylayer, valuelayer)
    A2C(γ, buffer, model, αcritic, optimizers(model, opt))
end
function selectaction(learner::A2C, policy, state)
    h = learner.model.chain(state)
    p = learner.model.policylayer(h)
    selectaction(policy, p[:])
end

function update!(learner::A2C)
    buffer = learner.buffer
    !isfull(buffer) && return
    xstart = buffer.states[1]
    xend = buffer.states[end]
    r, γeff = discountedrewards(buffer.rewards, buffer.done, learner.γ)
    Knet.update!(learner.model.w, 
                 a2cgradfun2(learner.model.w, learner.model, xstart, xend,
                            buffer.actions[1], r, γeff, learner.αcritic),
                 learner.opt)
end
function Knet.update!(w::SharedArray{Float64}, g::Array{Float64}, p::Knet.Sgd)
    Knet.gclip!(g, p.gclip)
    BLAS.axpy!(-p.lr, g, w)
end
Knet.optimizers{T<:Number}(::SharedArray{T},otype; o...)=otype(;o...)
