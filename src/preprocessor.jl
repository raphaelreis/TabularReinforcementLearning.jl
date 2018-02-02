struct NoPreprocessor end
export NoPreprocessor
preprocess(::NoPreprocessor, s) = s

struct Box{T}
    low::Array{T, 1}
    high::Array{T, 1}
end
struct StateAggregator
    box::Box
    nbins::Array{Int64, 1}
end
export StateAggregator
function StateAggregator(lb, ub, nbins, ndims)
    StateAggregator(Box(lb * ones(ndims), ub * ones(ndims)), 
                    nbins * ones(ndims))
end
function preprocess(p::StateAggregator, s)
    sp = zeros(sum(p.nbins))
    offset = 0
    for i in 1:length(s)
        index = round(Int64, (s[i] - p.box.low[i])/
                             (p.box.high[i] - p.box.low[i]) * 
                             (p.nbins[i] - 1)) + 1
        sp[index + offset] = 1.
        offset += p.nbins[i]
    end
    sp
end
