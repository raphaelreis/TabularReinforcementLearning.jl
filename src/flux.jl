import Flux

struct Linear{Ts}
    W::Ts
end
export Linear
function Linear(in::Int, out::Int; 
                T = Float64, initW = (x...) -> zeros(T, x...))
    Linear(Flux.param(initW(out, in)))
end
(a::Linear)(x) = a.W * x
Flux.treelike(Linear)

struct Id end # used for testing
(l::Id)(x) = x

Base.show(io::IO, l::Linear) = print(io, "Linear( $(size(l.W, 2)), $(size(l.W, 1)))")

import Flux.NNlib: conv, conv!, ∇conv_filter!, conv2d!, conv2d_grad_w!, ∇conv_filter
function conv(x, w; pad = 0, stride = 1)
    pad_, stride_ = Flux.NNlib.padtuple(x, pad), Flux.NNlib.padtuple(x, stride)
    Flux.NNlib.conv!(similar(x, Flux.NNlib.cdims(size(x), size(w), pad_, stride_)),
                     x, w, pad = pad_, stride = stride_)
end
conv!(y::AbstractArray{T,4}, x, w::AbstractArray{T,4};
      pad = 0, stride = 1) where T =
  conv2d!(y, x, w, padding = pad, stride = stride)

∇conv_filter(dy::A, x, w::A; pad = 0, stride = 1) where A<:AbstractArray =
  ∇conv_filter!(zeros(w), dy, x, w; pad = pad, stride = stride)

∇conv_filter!(dw::AbstractArray{T,4}, dy::AbstractArray{T,4}, x, w::AbstractArray{T,4};
              pad = 0, stride = 1) where T =
conv2d_grad_w!(dw, x, w, dy, padding = pad, stride = stride)
function conv2d!{T}(y::AbstractArray{T,4}, x, w::AbstractArray{T,4};
                  padding=0, stride=1, mode=0, alpha=T(1))
    if mode != 0 && mode != 1; throw(ArgumentError("conv2d only supports mode=0 or 1.")); end
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = size(w)
    if Cx!=C1; throw(DimensionMismatch()); end
    Wy,Hy,Cy,Ny = size(y)
    x2dims = im2col_dims(w,y)
    x2 = similar(x, x2dims)
    (p1,p2) = psize(padding,x)
    (s1,s2) = psize(stride,x)
    M,N,K,Y = Wy*Hy,Cy,Ww*Hw*Cx,Wy*Hy*Cy
    yidx = 1
    @inbounds for n in 1:Nx
        Flux.NNlib.im2col2d!(w, x, x2, n, p1, p2, s1, s2, mode)
        BLAS.gemm!('N','N',M,N,K,alpha,pointer(x2),pointer(w),T(0),pointer(y,yidx))
        yidx += Y
    end
    return y
end
function conv2d_grad_w!{T}(dw::AbstractArray{T,4}, x, w::AbstractArray{T,4}, dy::AbstractArray{T,4};
                   padding=0, stride=1, mode=0, alpha=1)
    # dw = x'*dy
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = size(w)
    Wy,Hy,Cy,Ny = size(dy)
    # if mode != 0 && mode != 1; throw(ArgumentError("conv2d only supports mode=0 or 1.")); end
    # @assert Cx==C1 && Cy==C2 && Ny==Nx
    x2dims = im2col_dims(w,dy)
    x2 = similar(x, x2dims)
    # op(A) is an m-by-k matrix, op(B) is a k-by-n matrix, C is an m-by-n matrix.
    Y,M,N,K = Wy*Hy*Cy,Ww*Hw*Cx,Cy,Wy*Hy
    alpha,beta = T(alpha),T(1)
    (p1,p2) = psize(padding,x)
    (s1,s2) = psize(stride,x)
    dyi = 1
    @inbounds for n in 1:Nx
        Flux.NNlib.im2col2d!(w, x, x2, n, p1, p2, s1, s2, mode)
        BLAS.gemm!('T','N',M,N,K,alpha,pointer(x2),pointer(dy,dyi),beta,pointer(dw))
        dyi += Y
    end
    return dw
end

import Requires: @require
@require CuArrays begin
    import CuArrays: conv!, ∇conv_filter!
    function conv!(y::A, x, w::A;
                   pad=0, stride=1, mode=0, alpha=1) where A<:CuArray{<:CUDNNFloat}
        cudnnConvolutionForward(y, x, w, padding=pad, stride=stride, mode=mode, alpha=alpha)
    end

    function ∇conv_filter!(dw::A, dy::A, x, w::A;
                           pad=0, stride=1, mode=0, alpha=1) where A<:CuArray{<:CUDNNFloat}
      cudnnConvolutionBackwardFilter(dw, x, w, dy, padding=pad, stride=stride, mode=mode, alpha=alpha)
    end
end
