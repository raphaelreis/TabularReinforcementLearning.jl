using Documenter, TabularReinforcementLearning

makedocs(modules = [TabularReinforcementLearning],
	     clean = false,
		 format = :html,
		 sitename = "Tabular Reinforcement Learning",
 		 linkcheck = !("skiplinks" in ARGS),
		 pages = [ "Introduction" => "index.md", 
				   "Usage" => "usage.md",
				   "Reference" => ["Comparison" => "comparison.md",
								   "Learning" => "learning.md",
								   "Learners" =>  "learners.md",
								   "Policies" =>  "policies.md",
								   "Environments" => "mdp.md",
								   "Evaluation Metrics" =>  "metrics.md",
								   "Stopping Criteria" =>  "stop.md",
								   "Callbacks" =>  "callbacks.md"],
				   "API" => "api.md"],
		 html_prettyurls = true
		)

deploydocs(
    repo = "github.com/jbrea/TabularReinforcementLearning.jl.git",
	julia = "0.6",
	target = "build",
    deps = nothing,
	make = nothing,
)
