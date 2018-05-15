T = TabularReinforcementLearning

#1D test

vect1 = [0.]
vect2 = [2.]
vect3 = [2]
myAgg = StateAggregator(vect1, vect2, vect3)

#One tiling
tiling = TilingStateAggregator(myAgg, 1)

s = [-0.5]
sp = T.preprocessstate(tiling, s)
@test sp == [1.,0.]

s = [0.]
sp = T.preprocessstate(tiling, s)
@test sp == [1.,0.]

s = [0.5]
sp = T.preprocessstate(tiling, s)
@test sp == [1.,0.]

s = [1.]
sp = T.preprocessstate(tiling, s)
@test sp == [1.,0.]

s = [1.0000001]
sp = T.preprocessstate(tiling, s)
@test sp == [0.,1.]

s = [2.0]
sp = T.preprocessstate(tiling, s)
@test sp == [0.,1.]

s = [2.5]
sp = T.preprocessstate(tiling, s)
@test sp == [0.,1.]

#Two tilings
tiling = TilingStateAggregator(myAgg, 2)
# offset, k, w = T.tilingparams(...)
offset = 0.666667
k = 0.666667
w = 1.33333

s = [-0.5]
sp = T.preprocessstate(tiling, s)
@test sp == [1.,0.,1.,0.]

s = [0.5]
sp = T.preprocessstate(tiling, s)
@test sp == [1.,0.,1.,0.]

s = [-offset+w+0.00001]
sp = T.preprocessstate(tiling, s)
@test sp == [0.,1.,1.,0.]

s = [0. + w + 0.00001]
sp = T.preprocessstate(tiling, s)
@test sp == [0.,1.,0.,1.]


#2D test

vect1 = [0.;0.]
vect2 = [2.;2.]
vect3 = [2;2]
myAgg = StateAggregator(vect1, vect2, vect3)
tiling = TilingStateAggregator(myAgg, 2)

s = [0., 0.]
sp = T.preprocessstate(tiling, s)
@test sp == [1,0,0,0,1,0,0,0]

s = [0.1, 0.1]
sp = T.preprocessstate(tiling, s)
@test sp == [1,0,0,0,1,0,0,0]
#
# s = [0., 1.-0.285714]
# sp = T.preprocessstate(tiling, s)
# @test sp == [0,0,1,0,1,0,0,0]
#
# s = [1., 1.]
# sp = T.preprocessstate(tiling, s)
# @test sp == [0,0,0,1,1,0,0,0]
#
# s = [.98, 1.02]
# sp = T.preprocessstate(tiling, s)
# @test sp == [0,0,0,1,0,0,1,0]
#
# s = [0., 1.02]
# sp = T.preprocessstate(tiling, s)
# @test sp == [0,0,1,0,0,0,1,0]
#
# #
# # vect1 = [0.;0.;0.]
# # vect2 = [2.;2.;2.]
# # vect3 = [2;2;2]
# # tiling = TilingStateAggregator(myAgg, 2)
# # s = [1.-0.181819, 0., 0.]
# # sp = T.preprocessstate(tiling, s)
# # println("sp is equal to [1,0,0,0,1,0,0,0] : $(sp == [1,0,0,0,1,0,0,0])")
