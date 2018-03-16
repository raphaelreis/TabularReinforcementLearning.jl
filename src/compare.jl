@eval TabularReinforcementLearning begin
using DataFrames

function makeuniquenames(agents)
    seen = String[]
    ids = String[]
    counts = Int64[]
    for (i, agent) in enumerate(agents)
        id = split(split(string(typeof(agent().learner)), "{")[1], ".")[end]
        seenindex = find(x -> x == id, seen)
        if length(seenindex) > 0
            counts[seenindex[1]] += 1
            push!(ids, id * "_" * string(counts[seenindex[1]]))
        else
            push!(seen, id)
            push!(counts, 0)
            push!(ids, id)
        end
    end
    ids
end

function compare(N, env, metric, stopcrit, agents...; verbose = false)
    L = length(agents)
    learnerids = makeuniquenames(agents)
    valuetype = typeof(getvalue(metric))
    results = DataFrame(learner = String[], 
                        value = valuetype[], seed = UInt64[])
    for t in 1:N
        seed = rand(1:typemax(UInt64)-1) 
        if typeof(env) <: Function
            environment = env()
        else
            environment = env
            reset!(environment)
        end
        for i in 1:L
            if verbose
                print("round $t: $(learnerids[i]) with seed $seed ")
            end
            srand(seed)
            reset!(metric)
            reset!(environment)
            if verbose; print("started ... "); end
            learn!(agents[i](), environment, metric, stopcrit)
            if verbose; println("finished"); end
            push!(results, [learnerids[i], getvalue(metric), seed])
        end
    end
    results
end
end
