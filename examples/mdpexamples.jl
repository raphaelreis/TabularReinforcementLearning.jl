using TabularReinforcementLearning

# Cliff walking (Sutton & Barto)
function getcliffwalkingmdp(; offset = 0., 
                              returnall = false, 
                              goalreward = 0.,
                              cliffreward = -100.,
                              stepreward = -1.)
    ns = 4*12; na = 4
    mdp = MDP(ns, na, init = "deterministic")
    mdp.isterminal[45] = 1
    mdp.initialstates = [1]
    mdp.state = 1
    for s in 1:ns
        for a in 1:na
            mdp.reward[a, s] = stepreward
            if a == 4 && s <= 4 || a == 1 && mod(s, na) == 0 ||
                a == 2 && s > 44 || a == 3 && s == 1
                mdp.trans_probs[a, s] = SparseVector(ns, [s], [1.]) 
            elseif mod(s, na) == 1 && s != 1 && s != 45       
                mdp.trans_probs[a, s] = SparseVector(ns, [1], [1.])
                mdp.reward[a, s] = cliffreward
            else
                if a == 1
                    mdp.trans_probs[a, s] = SparseVector(ns, [s+1], [1.])
                elseif a == 2
                    mdp.trans_probs[a, s] = SparseVector(ns, [s+4], [1.])
                elseif a == 3
                    mdp.trans_probs[a, s] = SparseVector(ns, [s-1], [1.])
                else
                    mdp.trans_probs[a, s] = SparseVector(ns, [s-4], [1.])
                end
            end
        end
    end
    mdp.reward[:, 45] .= goalreward
    mdp.reward .+= offset
    if returnall
        ones(4, 12), mdp, 45, collect(1:ns)
    else
        mdp
    end
end

# Maze
module Maze
using TabularReinforcementLearning, StatsBase
function getemptymaze(dimx, dimy)
    maze = ones(Int64, dimx, dimy)
    maze[1,:] .= maze[end,:] .= 0
    maze[:, 1] .= maze[:, end] .= 0
    maze
end

function setwall!(maze, startpos, endpos)
    dimx, dimy = startpos - endpos
    if dimx == 0
        maze[startpos[1], startpos[2]:endpos[2]] .= 0
    else
        maze[startpos[1]:endpos[1], startpos[2]] .= 0
    end
end

function indto2d(maze, pos)
    dimx = size(maze, 1)
    [rem(pos, dimx), div(pos, dimx) + 1]
end
function posto1d(maze, pos)
    dimx = size(maze, 1)
    (pos[2] - 1) * dimx + pos[1]
end

function checkpos(maze, pos)
    count = 0
    for dx in -1:1
        for dy in -1:1
            count += maze[(pos + [dx, dy])...] == 0
        end
    end
    count
end

function addrandomwall!(maze)
    startpos = rand(find(maze))
    startpos = indto2d(maze, startpos)
    starttouch = checkpos(maze, startpos) 
    if starttouch > 0
        return 0
    end
    endx, endy = startpos
    if rand(0:1) == 0 # horizontal
        while checkpos(maze, [endx, startpos[2]]) == 0
            endx += 1
        end
        if maze[endx + 1, startpos[2]] == 1 &&
            maze[endx + 1, startpos[2] + 1] == 
            maze[endx + 1, startpos[2] - 1] == 0
            endx -= 1
        end
    else
        while checkpos(maze, [startpos[1], endy]) == 0
            endy += 1
        end
        if maze[startpos[1], endy + 1] == 1 &&
            maze[startpos[1] + 1, endy + 1] == 
            maze[startpos[1] - 1, endy + 1] == 0
            endx -= 1
        end
    end
    setwall!(maze, startpos, [endx, endy])
    return 1
end

function mazetomdp(maze, ngoalstates = 1, stochastic = false)
    na = 4
    nzpos = find(maze)
    mapping = cumsum(maze[:])
    ns = length(nzpos)
    T = Array{SparseVector}(na, ns)
    goals = sort(sample(1:ns, ngoalstates, replace = false))
    R = -ones(na, ns)
    R[:, goals] .= 0.
    isterminal = zeros(Int64, ns); isterminal[goals] = 1
    isinitial = collect(1:ns); deleteat!(isinitial, goals)
    for s in 1:ns
        for (aind, a) in enumerate(([0, 1], [1, 0], [0, -1], [-1, 0]))
            pos = indto2d(maze, nzpos[s])
            nextpos = maze[(pos + a)...] == 0 ? pos : pos + a
            if stochastic
                positions = []
                push!(positions, nextpos)
                weights = [1.]
                for dir in ([0, 1], [1, 0], [0, -1], [-1, 0])
                    if maze[(nextpos + dir)...] != 0
                        push!(positions, nextpos + dir)
                        push!(weights, .05)
                    end
                end
                states = map(p -> mapping[posto1d(maze, p)], positions)
                weights /= sum(weights)
                T[aind, s] = SparseVector(ns, states, weights)
            else
                nexts = mapping[posto1d(maze, nextpos)]
                T[aind, s] = TabularReinforcementLearning.getprobvecdeterministic(ns,
                                                                       nexts,
                                                                       nexts)
            end
        end
    end
    MDP(ns, na, rand(1:ns), T, R, isinitial, isterminal), goals, nzpos
end

function breaksomewalls(m; f = 1/50, 
                        n = div(length(find(1 - m[2:end-1, 2:end-1])), 1/f))
    nx, ny = size(m)
    zeros = find(1 - m)
    i = 1
    while i < n
        candidate = rand(zeros)
        if candidate > nx && candidate < nx * (ny - 1) &&
            candidate % nx != 0 && candidate % nx != 1
            m[candidate] = 1
            i += 1
        end
    end
end
    
function getmazemdp(; nx = 40, ny = 40, returnall = false, 
                      nwalls = div(nx*ny, 10), 
                      offset = 0., stochastic = false, ngoals = 1)
    m = getemptymaze(nx, ny)
    [addrandomwall!(m) for _ in 1:nwalls]
    breaksomewalls(m)
    mdp, goals, mapping = mazetomdp(m, ngoals, stochastic)
    mdp.reward .+= offset
    if returnall
        m, mdp, goals, mapping
    else
        mdp
    end
end
export getmazemdp
end
using Maze

# this function requires
# using PyPlot, PyCall
# @pyimport matplotlib.colors as matcolors
function plotmazemdp(maze, goal, state, mapping; 
                     showvalues = false,
                     values = zeros(length(mapping)))
    maze[mapping[goal]] = 3
    maze[mapping[state]] = 2
    figure(figsize = (4, 4))
    cmap = matcolors.ListedColormap(["gray", "white", "blue", "red"], "A")
    if showvalues
        m = zeros(size(maze)...)
        m[mapping] .= values
        imshow(m, cmap = "Spectral_r")
    end
    imshow(maze, interpolation = "none", cmap = cmap, alpha = (1 - .5showvalues))
    plt[:tick_params](top="off", bottom="off",
                      labelbottom="off", labeltop="off",
                      labelleft="off", left="off")
    maze[mapping[goal]] = 1
    maze[mapping[state]] = 1
end

# using PlotlyJS
function plotmazemdp(maze, goals, state, mapping)
    m = deepcopy(maze)
    m[mapping[goals]] = 3
    m[mapping[state]] = 2
    data = heatmap(z = m, colorscale = [[0, "gray"], [1/3, "white"], 
                                        [2/3, "blue"], [1., "red"]], 
                  showscale = false)
    w, h = size(m)
    layout = Layout(autosize = false, width = 800, height = 800 * h/w)
    plot(data, layout)
end

# random MDPs
function getdetmdp(; ns = 10^4, na = 10)
    mdp = MDP(ns, na, init = "deterministic")
    mdp.reward = mdp.reward .* (mdp.reward .< -1.5)
    mdp
end
function getdettreemdp(; na = 4, depth = 5)
    mdp = treeMDP(na, depth, init = "deterministic")
end
function getdettreemdpwithinrew(; args...)
    mdp = getdettreemdp(; args...)
    nonterminals = find(1 - mdp.isterminal)
    mdp.reward[:, nonterminals] = -rand(mdp.na, length(nonterminals))
    mdp
end
function getstochtreemdp(; na = 4, depth = 4, bf = 2)
    mdp = treeMDP(na, depth, init = "random", branchingfactor = bf)
    mdp
end
getstochmdp(; na = 10, ns = 50) = MDP(ns, na)
function getabsorbingdetmdp(;ns = 10^3, na = 10)
    mdp = MDP(ns, na, init = "deterministic")
    mdp.reward .= mdp.reward .* (mdp.reward .< -.5)
    mdp.isinitial = 1:div(ns, 100)
    mdp.state = rand(mdp.isinitial)
    setterminalstates!(mdp, ns - div(ns, 100) + 1:ns)
    for s in find(mdp.isterminal)
        mdp.reward[:, s] .= 0.
    end
    mdp
end

