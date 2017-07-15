"""
	struct NoTraces <: AbstractTraces

No eligibility traces, i.e. ``e(a, s) = 1`` for current action ``a`` and state
``s`` and zero otherwise.
"""
struct NoTraces <: AbstractTraces
end
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

Traces are updated according to	``e(a, s) ←  1`` for the current action-state
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

Traces are updated according to	``e(a, s) ←  1 + e(a, s)`` for the current action-state
pair and ``e(a, s) ←  γλ e(a, s)`` for all other pairs unless
``e(a, s) < `` `minimaltracevalue` where the trace is set to 0 
(for computational efficiency).
""" AccumulatingTraces
@doc """
	AccumulatingTraces(ns, na, λ::Float64, γ::Float64; minimaltracevalue = 1e-12)
""" AccumulatingTraces()

function updatetrace!(traces, state, action)
	discounttraces!(traces)
	increasetrace!(traces, state, action)
end

function increasetrace!(traces::NoTraces, state, action)
end
function increasetrace!(traces::ReplacingTraces, state, action)
	traces.trace[action, state] = 1.
end
function increasetrace!(traces::AccumulatingTraces, state, action)
	traces.trace[action, state] += 1.
end

function discounttraces!(traces)
	BLAS.scale!(traces.γλ, traces.trace)
end

function resettraces!(traces::NoTraces)
end
function resettraces!(traces::Union{ReplacingTraces, AccumulatingTraces})
	BLAS.scale!(0., traces.trace)
end

function updatetraceandparams!(traces, params, factor)
	for i in 1:length(traces.trace)
		if traces.trace[i] > traces.minimaltracevalue
			params[i] += factor * traces.trace[i]
			traces.trace[i] *= traces.γλ
		elseif traces.trace[i] > 0.
			traces.trace[i] = 0.
		end
	end
end
