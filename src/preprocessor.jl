struct NoPreprocessor end
export NoPreprocessor
@inline preprocessstate(p::NoPreprocessor, s) = s
@inline preprocess(::NoPreprocessor, s, r, done) = (s, r, done)
@inline preprocess(p, s, r, done) = (preprocessstate(p, s), r, done)

struct Box{T}
    low::Array{T, 1}
    high::Array{T, 1}
end
struct Preprocessor{Ts, Tr, Td}
    state::Ts
    reward::Tr
    done::Td
end
struct StateAggregator
    box::Box
    nbins::Array{Int64, 1}
    offsets::Array{Int64, 1}
    perdimension::Bool
end
export StateAggregator
function StateAggregator(lb::Vector, ub::Vector, nbins::Vector;
                         perdimension = false)
    if perdimension
        offsets = [0; cumsum(nbins[1:end-1])]
    else
        offsets = foldl((x, n) -> [x...; x[end] * n], [1], nbins[1:end-1])
    end
    StateAggregator(Box(lb, ub), nbins, offsets, perdimension)
end
StateAggregator(lb::Number, ub::Number, nbins::Int, ndims::Int; perdimension = false) =
    StateAggregator(lb * ones(ndims), ub * ones(ndims), nbins * ones(ndims))

@inline indexinbox(x, l, h, n) = round(Int64, (n - 1) * (x - l)/(h - l)) |> 
                                   i -> max(min(i, n), 0)

function preprocessstate(p::StateAggregator, s)
    indices = [indexinbox(s[i], p.box.low[i], p.box.high[i], p.nbins[i]) 
               for i in 1:length(s)]
    if p.perdimension
        sp = zeros(sum(p.nbins))
        offset = 0
        for i in 1:length(s)
            sp[indices[i] + 1 + p.offsets[i]] = 1.
        end
    else
        sp = zeros((*)(p.nbins...))
        sp[dot(indices, p.offsets) + 1] = 1
    end
    sp
end

struct RadialBasisFunctions
    means::Array{Array{Float64, 1}, 1}
    sigmas::Array{Float64, 1}
    state::Array{Float64, 1}
end
export RadialBasisFunctions
function RadialBasisFunctions(box::Box, n, sigma)
    dim = length(box.low)
    means = [rand(dim) .* (box.high - box.low) .+ box.low for _ in 1:n]
    RadialBasisFunctions(means, 
                         typeof(sigma) <: Number ? fill(sigma, n) : sigma, 
                         zeros(n))
end
function preprocessstate(p::RadialBasisFunctions, s)
    @inbounds for i in 1:length(p.state)
        p.state[i] = exp(-norm(s - p.means[i])/p.sigmas[i])
    end
    p.state
end

struct RandomProjection
    w::Array{Float64, 2}
end
export RandomProjection
preprocessstate(p::RandomProjection, s) = p.w * s

struct SparseRandomProjection
    w::Array{Float64, 2}
    b::Array{Float64, 1}
end
export SparseRandomProjection
preprocessstate(p::SparseRandomProjection, s) = clamp.(p.w * s + p.b, 0, Inf)
