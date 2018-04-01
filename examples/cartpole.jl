using TabularReinforcementLearning
import TabularReinforcementLearning.interact!,
TabularReinforcementLearning.getstate,
TabularReinforcementLearning.reset!

struct CartPoleParams{T}
    gravity::T
    masscart::T
    masspole::T
    totalmass::T
    halflength::T
    polemasslength::T
    forcemag::T
    tau::T
    thetathreshold::T
    xthreshold::T
end
mutable struct CartPole{T}
    params::CartPoleParams{T}
    observation_space::TabularReinforcementLearning.Box{T}
    state::Array{T, 1}
    done::Bool
end
function CartPole(; T = Float64, gravity = T(9.8), masscart = T(1.), 
                  masspole = T(.1), halflength = T(.5), forcemag = T(10.))
    params = CartPoleParams(gravity, masscart, masspole, masscart + masspole,
                            halflength, masspole * halflength, forcemag,
                            T(.02), T(2 * 12 * π /360), T(2.4))
    high = [2 * params.xthreshold, T(1e38),
            2 * params.thetathreshold, T(1e38)]
    cp = CartPole(params, TabularReinforcementLearning.Box(-high, high), 
                  zeros(T, 4), false)
    reset!(cp)
    cp
end

function reset!(env::CartPole{T}) where T <: Number
    env.state[:] = T(.1) * rand(T, 4) - T(.05)
    env.done = false
end

function getstate(env::CartPole)
    env.state, env.done
end

function interact!(a, env::CartPole{T}) where T <: Number
    if env.done
        reset!(env)
        return env.state, 0., env.done
    end
    force = a == 2 ? env.params.forcemag : -env.params.forcemag
    x, xdot, theta, thetadot = env.state
    costheta = cos(theta)
    sintheta = sin(theta)
    tmp = (force + env.params.polemasslength * thetadot^2 * sintheta) /
        env.params.totalmass
    thetaacc = (env.params.gravity * sintheta - costheta * tmp) / 
        (env.params.halflength * 
            (4/3 - env.params.masspole * costheta^2/env.params.totalmass))
    xacc = tmp - env.params.polemasslength * thetaacc * costheta / 
        env.params.totalmass
    env.state[1] += env.params.tau * xdot
    env.state[2] += env.params.tau * xacc
    env.state[3] += env.params.tau * thetadot
    env.state[4] += env.params.tau * thetaacc
    env.done = abs(env.state[1]) > env.params.xthreshold ||
               abs(env.state[3]) > env.params.thetathreshold
    env.state, 1. - env.done, env.done
end
