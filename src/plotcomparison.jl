@eval TabularReinforcementLearning begin
using DataFrames, PyPlot

smooth(vals, w) = [mean(vals[i:i+w]) for i in 1:length(vals) - w]
function plotcomparison(results; 
						labelsdict = Dict(), 
						colors = [],
						thin = .1,
						thick = 2,
						smoothingwindow = 0)
	if colors == []
		colors = plt[:rcParams]["axes.prop_cycle"][:by_key]()["color"]
	end
	groups = groupby(results, :learner)
	labels = String[]
	for g in groups
		learner = g[:learner][1]
		push!(labels, haskey(labelsdict, learner) ? labelsdict[learner] : learner)
	end
	if typeof(results[:value][1]) <: AbstractArray
		for (i, g) in enumerate(groups)
			m = mean(g[:value])
			if smoothingwindow > 0
				m = TabularReinforcementLearning.smooth(m, smoothingwindow)
			end
			plot(m, label = labels[i], linewidth = thick, color = colors[i])
			for v in g[:value]
				if smoothingwindow > 0
					v = TabularReinforcementLearning.smooth(v, smoothingwindow)
				end
				plot(v, linewidth = thin, color = colors[i])
			end
			plt[:legend]()
		end
	else
		boxplot(([groups[i][:value] for i in 1:length(groups)]...), 
				labels = labels)
	end
end
export plotcomparison
end
