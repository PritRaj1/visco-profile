struct KANdense{W, N} <: Lux.AbstractLuxContainerLayer{(:transform, :norm_layer)}
    transform::W
    norm_layer::N
    in_dims::Int
    out_dims::Int
end

function KANdense(in_dims::Int, out_dims::Int, wavelet_name::String, base_activation::String = "relu"; norm::Bool = false, is_2d::Bool = false)
    wavelet = create_wavelet(wavelet_name, in_dims, out_dims)
    norm_layer = norm ? Lux.BatchNorm(out_dims) : Lux.WrappedFunction(identity)
    return KANdense(wavelet, norm_layer, in_dims, out_dims)
end

function Lux.initialparameters(rng::AbstractRNG, l::KANdense)
    return (
        transform = Lux.initialparameters(rng, l.transform),
        norm_layer = Lux.initialparameters(rng, l.norm_layer),
        scale = ones(Float32, l.in_dims, l.out_dims),
        translation = zeros(Float32, l.in_dims, l.out_dims),
    )
end

_unsqueeze(x::AbstractArray{T, 2}) where {T} =
    reshape(x, size(x, 1), 1, size(x, 2))
_unsqueeze(x::AbstractArray{T, 3}) where {T} =
    reshape(x, size(x, 1), 1, size(x, 2), size(x, 3))

function (l::KANdense)(x, ps, st)
    x_exp = (_unsqueeze(x) .- ps.translation) ./ ps.scale
    y, st_t = l.transform(x_exp, ps.transform, st.transform)
    orig_size = size(y)
    y_flat = reshape(y, l.out_dims, :)
    y_n_flat, st_n = l.norm_layer(y_flat, ps.norm_layer, st.norm_layer)
    out = ndims(y) == 2 ? y_n_flat : reshape(y_n_flat, orig_size)
    return out, (transform = st_t, norm_layer = st_n)
end
