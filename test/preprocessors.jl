T = TabularReinforcementLearning
for perdim in [true, false]
    p = StateAggregator([0, -5, 1], [3, 9, 2], [8, 10, 12], perdimension = perdim)
    @test T.preprocessstate(p, [0, -5, 1])[1] == 1.
    @test T.preprocessstate(p, [3, 9, 2])[end] == 1.
    if perdim
        @test sum(T.preprocessstate(p, [1, -3, 1.5])) == 3
    else
        @test sum(T.preprocessstate(p, [1, -3, 1.5])) == 1
    end
end
p = RadialBasisFunctions(T.Box([-1, -1], [1, 1]), 20, 1.)
@test length(T.preprocessstate(p, randn(2))) == 20
p = RandomProjection(rand(20, 2))
@test length(T.preprocessstate(p, rand(2))) == 20
