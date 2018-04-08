"""
    struct NoTraces <: AbstractTraces

No eligibility traces, i.e. ``e(a, s) = 1`` for current action ``a`` and state
``s`` and zero otherwise.
"""
struct NoTraces <: AbstractTraces end
export NoTraces

for kind in (:ReplacingTraces, :AccumulatingTraces)
    @eval (struct $kind <: AbstractTraces
                λ::Float64
                γλ::Float64
                trace::Array{Float64, 2}
                minimaltracevalue::Float64
            end;
            export $kind;
            function $kind(ns, na, λ::Float64, γ::Float64; 
                           minimaltracevalue = 1e-12)
                $kind(λ, γ*λ, zeros(na, ns), minimaltracevalue)
            end)
end
@doc """
    struct ReplacingTraces <: AbstractTraces
        λ::Float64
        γλ::Float64
        trace::Array{Float64, 2}
        minimaltracevalue::Float64

Decaying traces with factor γλ. 

Traces are updated according to ``e(a, s) ←  1`` for the current action-state
pair and ``e(a, s) ←  γλ e(a, s)`` for all other pairs unless
``e(a, s) < `` `minimaltracevalue` where the trace is set to 0 
(for computational efficiency).
""" ReplacingTraces
@doc """
    ReplacingTraces(ns, na, λ::Float64, γ::Float64; minimaltracevalue = 1e-12)
""" ReplacingTraces()
@doc """
    struct AccumulatingTraces <: AbstractTraces
        λ::Float64
        γλ::Float64
        trace::Array{Float64, 2}
        minimaltracevalue::Float64

Decaying traces with factor γλ. 

Traces are updated according to ``e(a, s) ←  1 + e(a, s)`` for the current action-state
pair and ``e(a, s) ←  γλ e(a, s)`` for all other pairs unless
``e(a, s) < `` `minimaltracevalue` where the trace is set to 0 
(for computational efficiency).
""" AccumulatingTraces
@doc """
    AccumulatingTraces(ns, na, λ::Float64, γ::Float64; minimaltracevalue = 1e-12)
""" AccumulatingTraces()

function increasetrace!(traces::ReplacingTraces, state::Int, action)
    traces.trace[action, state] = 1.
end
function increasetrace!(traces::ReplacingTraces, state::Vector, action)
    traces.trace[action, :] .= state
end
function increasetrace!(traces::AccumulatingTraces, state::Int, action)
    traces.trace[action, state] += 1.
end
function increasetrace!(traces::AccumulatingTraces, state::Vector, action)
    traces.trace[action, :] .+= state
end


discounttraces!(traces) = BLAS.scale!(traces.γλ, traces.trace)
resettraces!(traces) = BLAS.scale!(0., traces.trace)

function updatetraceandparams!(traces, params, factor)
    BLAS.axpy!(factor, traces.trace, params)
    discounttraces!(traces)
end

function updatetrace!(traces, state, action)
    discounttraces!(traces)
    increasetrace!(traces, state, action)
end
