T = TabularReinforcementLearning
#1D test

vect1 = [0.]
vect2 = [2.]
vect3 = [2]
myAgg = StateAggregator(vect1, vect2, vect3)

#One tiling
tiling = TilingStateAggregator(myAgg, 1)
offset, k, w = T.tilingparams(vect2-vect1,1,vect3)
offset = offset[1]
k = k[1]
w = w[1]

s = [-0.5]
sp = T.preprocessstate(tiling, s)
@test sp == [1.,0.]

s = [offset]
sp = T.preprocessstate(tiling, s)
@test sp == [1.,0.]

s = [0.5]
sp = T.preprocessstate(tiling, s)
@test sp == [1.,0.]

s = [offset + w]
sp = T.preprocessstate(tiling, s)
@test sp == [1.,0.]

s = [offset + w + 1.0e-5]
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
offset, k, w = T.tilingparams(vect2-vect1,2,vect3)
offset = offset[1]
k = k[1]
w = w[1]

s = [-offset]
sp = T.preprocessstate(tiling, s)
@test sp == [1.,0.,1.,0.]

s = [0.5]
sp = T.preprocessstate(tiling, s)
@test sp == [1.,0.,1.,0.]

    s = [-offset+w-1.0e-5]
sp = T.preprocessstate(tiling, s)
@test sp == [1.,0.,1.,0.]

s = [-offset+w+1.0e-5]
sp = T.preprocessstate(tiling, s)
@test sp == [0.,1.,1.,0.]

s = [-offset+w+k-1.0e-5]
sp = T.preprocessstate(tiling, s)
@test sp == [0.,1.,1.,0.]

s = [-offset+w+k+1.0e-5]
sp = T.preprocessstate(tiling, s)
@test sp == [0.,1.,0.,1.]


#2D test

vect1 = [0.;0.]
vect2 = [2.;2.]
vect3 = [2;2]
myAgg = StateAggregator(vect1, vect2, vect3)
tiling = TilingStateAggregator(myAgg, 1)
offset, k, w = T.tilingparams(vect2-vect1,1,vect3)


s = vect1 - offset
sp = T.preprocessstate(tiling, s)
@test sp == [1.,0.,0.,0.]

s = [0.+w[1], 0.]
sp = T.preprocessstate(tiling, s)
@test sp == [1.,0.,0.,0.]

s = [0.+w[1]+1.0e-5, 0.]
sp = T.preprocessstate(tiling, s)
@test sp == [0.,1.,0.,0.]

s = [0., 0.+w[2]]
sp = T.preprocessstate(tiling, s)
@test sp == [1.,0.,0.,0.]

s = [0., 0.+w[2]+1.0e-5]
sp = T.preprocessstate(tiling, s)
@test sp == [0.,0.,1.,0.]


tiling = TilingStateAggregator(myAgg, 2)
offset, k, w = T.tilingparams(vect2-vect1,2,vect3)

s = [0., 0.]
sp = T.preprocessstate(tiling, s)
@test sp == [1.,0.,0.,0.,1.,0.,0.,0.]

s = [-offset[1]+w[1], 0.]
sp = T.preprocessstate(tiling, s)
@test sp == [1,0.,0.,0.,1.,0.,0.,0.]

s = [-offset[1]+w[1]+1.0e-5, 0.]
sp = T.preprocessstate(tiling, s)
@test sp == [0.,1.,0.,0.,1.,0.,0.,0.]

s = [0., -offset[2]+w[2]+1.0e-5]
sp = T.preprocessstate(tiling, s)
@test sp == [0.,0.,1.,0.,1.,0.,0.,0.]

s = [0., -offset[2]+w[2]+k[2]+1.0e-5]
sp = T.preprocessstate(tiling, s)
@test sp == [0.,0.,1.,0.,0.,0.,1.,0.]

#2D test for different bins
vect1 = [0.;0.]
vect2 = [2.;50.]
vect3 = [2;2]
myAgg = StateAggregator(vect1, vect2, vect3)
tiling = TilingStateAggregator(myAgg, 2)
offset, k, w = T.tilingparams(vect2-vect1,2,vect3)

s = [-offset[1]+w[1], 0.]
sp = T.preprocessstate(tiling, s)
@test sp == [1,0.,0.,0.,1.,0.,0.,0.]

s = [-offset[1]+w[1]+1.0e-5, 0.]
sp = T.preprocessstate(tiling, s)
@test sp == [0.,1.,0.,0.,1.,0.,0.,0.]

s = [0., -offset[2]+w[2]+1.0e-5]
sp = T.preprocessstate(tiling, s)
@test sp == [0.,0.,1.,0.,1.,0.,0.,0.]

s = [0., -offset[2]+w[2]+k[2]+1.0e-5]
sp = T.preprocessstate(tiling, s)
@test sp == [0.,0.,1.,0.,0.,0.,1.,0.]
