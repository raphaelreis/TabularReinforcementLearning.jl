episode = [(0., 1, 1), (0., 2, 1), (0., 3, 1), (0., 4, 1), (0., 2, 1), 
		   (0., 5, 1), (0., 2, 2), (2., 6, 1), (0., 1, 1), (0., 2, 2)]
na = 2; ns = 6; γ = .9
learner = SmallBackups(na = na, ns = ns, γ = γ, maxcount = 10)
rs0a0s1a1 = [(episode[i]..., episode[i+1][2:3]..., i==8) for i in 1:length(episode)-1]
for step in rs0a0s1a1
	update!(learner, step...)
end
Nsa = zeros(Int64, na, ns)
for (r, s, a) in episode[1:end-1]; Nsa[a, s] += 1; end
@test Nsa == learner.Nsa
@test learner.Ns1a0s0[1] == Dict()
@test learner.Ns1a0s0[2] == Dict((1,1) => 2, (1, 4) => 1, (1, 5) => 1)
@test 2 * [γ^2; γ; γ^3; γ^2; γ^2; 1] == learner.V
