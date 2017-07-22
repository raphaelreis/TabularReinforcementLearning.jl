"""
	mutable struct SmallBackups <: AbstractReinforcementLearner
		γ::Float64
		maxcount::UInt64
		minpriority::Float64
		counter::Int64
		Q::Array{Float64, 2}
		Qprev::Array{Float64, 2}
		V::Array{Float64, 1}
		Nsa::Array{Int64, 2}
		Ns1a0s0::Array{Dict{Tuple{Int64, Int64}, Int64}, 1}
		queue::PriorityQueue

See [Harm Van Seijen, Rich Sutton ; Proceedings of the 30th International Conference on Machine Learning, PMLR 28(3):361-369, 2013.](http://proceedings.mlr.press/v28/vanseijen13.html)

`maxcount` defines the maximal number of backups per action, `minpriority` is
the smallest priority still added to the queue.
"""
mutable struct SmallBackups <: AbstractReinforcementLearner
	γ::Float64
	maxcount::UInt64
	minpriority::Float64
	counter::Int64
	Q::Array{Float64, 2}
	Qprev::Array{Float64, 2}
	V::Array{Float64, 1}
	Nsa::Array{Int64, 2}
	Ns1a0s0::Array{Dict{Tuple{Int64, Int64}, Int64}, 1}
	queue::PriorityQueue
end
export SmallBackups

"""
SmallBackups(; ns = 10, na = 4, γ = .9, initvalue = Inf64, maxcount = 3, 
				   minpriority = 1e-8)
"""
function SmallBackups(; ns = 10, na = 4, γ = .9, initvalue = Inf64,
					    maxcount = 3, minpriority = 1e-8)
	SmallBackups(γ, maxcount, minpriority, 0,
				 zeros(na, ns) .+ initvalue, 
				 zeros(na, ns) .+ initvalue,
				 zeros(ns),
				 zeros(na, ns),
				 [Dict{Tuple{Int64, Int64}, Int64}() for _ in 1:ns],
				 PriorityQueue(Int64[], Float64[], Base.Order.Reverse))
end

function addtoqueue!(q, s, p)
	if haskey(q, s) 
		if q[s] > p; q[s] = p; end
	else
		enqueue!(q, s, p)
	end
end

function processqueue!(learner)
	while length(learner.queue) > 0 && learner.counter < learner.maxcount
		learner.counter += 1
		s1 = dequeue!(learner.queue)
		tmp = learner.V[s1]
		for b in 1:size(learner.Q, 1)
			learner.Qprev[b, s1] = learner.Q[b, s1]
		end
		learner.V[s1] = maximumbelowInf(learner.Q[:, s1])
		ΔV = learner.V[s1] - tmp
		if length(learner.Ns1a0s0[s1]) > 0
			for ((a0, s0), n) in learner.Ns1a0s0[s1]
				learner.Q[a0, s0] += learner.γ * ΔV * n/learner.Nsa[a0, s0]
				p = abs(learner.Q[a0, s0] - learner.Qprev[a0, s0])
				if p > learner.minpriority; addtoqueue!(learner.queue, s0, p); end
			end
		end
	end
	learner.counter = 0.
end


function update!(learner::SmallBackups, r, s0, a0, s1, a1, iss0terminal)
	if iss0terminal
		learner.Nsa[1, s0] += 1
		if learner.Q[1, s0] == Inf64; learner.Q[:, s0] .= 0; end
		learner.Q[:, s0] .= (learner.Q[1, s0] * (learner.Nsa[1, s0] - 1) + r) / 
								learner.Nsa[1, s0]
	else
		learner.Nsa[a0, s0] += 1
		if haskey(learner.Ns1a0s0[s1], (a0, s0))
			learner.Ns1a0s0[s1][(a0, s0)] += 1
		else
			learner.Ns1a0s0[s1][(a0, s0)] = 1
		end
		if learner.Q[a0, s0] == Inf64; learner.Q[a0, s0] = 0.; end
		nextv = learner.γ * learner.V[s1]
		learner.Q[a0, s0] = (learner.Q[a0, s0] * (learner.Nsa[a0, s0] - 1) + 
								r + nextv) / learner.Nsa[a0, s0]
	end
	p = abs(learner.Q[a0, s0] - learner.Qprev[a0, s0])
	if p > learner.minpriority; addtoqueue!(learner.queue, s0, p); end
	processqueue!(learner)
end
