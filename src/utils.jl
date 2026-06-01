using Statistics

const ACTIVATION_MAP = Dict{String, Function}(
    "relu" => NNlib.relu,
    "leakyrelu" => NNlib.leakyrelu,
    "tanh" => NNlib.hardtanh,
    "sigmoid" => NNlib.hardsigmoid,
    "swish" => NNlib.hardswish,
    "gelu" => NNlib.gelu,
    "selu" => NNlib.selu,
)

get_activation(name::AbstractString) = ACTIVATION_MAP[name]

batch_mul(x, y) = x .* y
three_mul(x, y, z) = x .* y .* z

function node_mul(y::AbstractArray{T, 3}, w::AbstractMatrix{T}) where {T}
    in_, out_, k = size(y)
    output = reshape(w, in_, out_, 1) .* y
    flat = reshape(output, in_, out_ * k)
    summed = reshape(fill(one(T), 1, in_), 1, in_) * flat
    return reshape(summed, out_, k)
end

function node_mul(y::AbstractArray{T, 4}, w::AbstractMatrix{T}) where {T}
    in_, out_, s_, b_ = size(y)
    output = reshape(w, in_, out_, 1, 1) .* y
    flat = reshape(output, in_, out_ * s_ * b_)
    summed = reshape(fill(one(T), 1, in_), 1, in_) * flat
    return reshape(summed, out_, s_, b_)
end

const NORM_EPS = Float32(1.0e-5)

struct UnitGaussianNormaliser{T <: AbstractFloat}
    mu::T
    sigma::T
end

UnitGaussianNormaliser(x::AbstractArray) =
    UnitGaussianNormaliser(Float32(mean(x)), Float32(std(x)))

encode(n::UnitGaussianNormaliser, x) = (x .- n.mu) ./ (n.sigma .+ NORM_EPS)
decode(n::UnitGaussianNormaliser, x) = x .* (n.sigma .+ NORM_EPS) .+ n.mu

struct MinMaxNormaliser{T <: AbstractFloat}
    lo::T
    hi::T
end

MinMaxNormaliser(x::AbstractArray) = MinMaxNormaliser(Float32(minimum(x)), Float32(maximum(x)))

encode(n::MinMaxNormaliser, x) = (x .- n.lo) ./ (n.hi - n.lo)
decode(n::MinMaxNormaliser, x) = x .* (n.hi - n.lo) .+ n.lo

function loss_fcn(y_pred, y; p::Real = 2)
    return sum(abs.(y_pred .- y) .^ p)
end

function BIC(model, n_samples::Int, loss_val::Real)
    k = Lux.parameterlength(model)
    return 2 * loss_val + k * log(n_samples)
end

function log_csv(epoch, train_loss, test_loss, bic, elapsed, file_name)
    return open(file_name, "a") do f
        write(f, "$epoch,$elapsed,$train_loss,$test_loss,$bic\n")
    end
end
