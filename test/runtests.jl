include("../src/WavKANSequence.jl")
using .WavKANSequence
using Lux, Random, Test
using Lux: Training
using Optimisers
using Reactant
using MLDataDevices: reactant_device

const RNG = Xoshiro(0)
const DEV = reactant_device()

function _train_step(model, ps, st, x, y)
    ps = ps |> DEV
    st = st |> DEV
    x = x |> DEV
    y = y |> DEV
    train_state = Training.TrainState(model, ps, st, Optimisers.Adam(1.0f-3))
    objective(model, ps, st, (x, y)) = (loss_fcn(model((x, y), ps, st)[1], y), st, (;))
    _, loss, _, train_state = Training.single_train_step!(
        AutoEnzyme(), objective, (x, y), train_state,
    )
    eval_fwd = @compile model((x, y), train_state.parameters, Lux.testmode(train_state.states))
    y_pred, _ = eval_fwd((x, y), train_state.parameters, Lux.testmode(train_state.states))
    return loss, y_pred
end

@testset "Wavelets" begin
    for name in keys(WavKANSequence.WAVELET_MAP)
        w = WavKANSequence.create_wavelet(name, 3, 5)
        ps, st = Lux.setup(RNG, w)
        x = randn(Float32, 3, 5, 4)  # (in, out, batch)
        y, _ = w(x, ps, st)
        @test size(y) == (5, 4)
    end
end

@testset "KANdense" begin
    layer = WavKANSequence.KANdense(4, 3, "Morlet")
    ps, st = Lux.setup(RNG, layer)
    y, _ = layer(randn(Float32, 4, 8), ps, st)
    @test size(y) == (3, 8)
end

@testset "Attention" begin
    mha = WavKANSequence.MultiHeadAttention(6, 2, "relu")
    ps, st = Lux.setup(RNG, mha)
    x = randn(Float32, 6, 10, 2)  # (d_model, seq, batch)
    y, _ = mha((x, x, x), ps, st)
    @test size(y) == (6, 10, 2)
end

@testset "RNO forward" begin
    cfg = RNOConfig(4, 2, "relu", 1.0f-3, 10, 0.8f0, 1.0f-5, 2, 1, 2.0f0)
    model = create_model(cfg, 20)
    ps, st = Lux.setup(RNG, model)
    x = randn(Float32, 20, 2)
    y_true = randn(Float32, 20, 2)
    out, _ = model((x, y_true), ps, st)
    @test size(out) == (20, 2)
end

@testset "KAN_RNO forward" begin
    cfg = KANRNOConfig(4, 2, "relu", ["Morlet", "MexicanHat", "Shannon"], false, 1.0f-3, 10, 0.8f0, 1.0f-5, 2, 1, 2.0f0)
    model = create_model(cfg, 20)
    ps, st = Lux.setup(RNG, model)
    x = randn(Float32, 20, 2)
    y_true = randn(Float32, 20, 2)
    out, _ = model((x, y_true), ps, st)
    @test size(out) == (20, 2)
end

@testset "Transformer forward" begin
    cfg = TransformerConfig(6, 2, 12, 0.1f0, 1, 1, 100, "relu", 1.0f-3, 10, 0.8f0, 1.0f-5, 2, 1, 2.0f0)
    model = create_model(cfg, 20)
    ps, st = Lux.setup(RNG, model)
    src = randn(Float32, 20, 2)
    tgt = randn(Float32, 20, 2)
    out, _ = model((src, tgt), ps, st)
    @test size(out) == (20, 2)
end

@testset "KAN_Transformer forward" begin
    cfg = KANTransformerConfig(
        6, 2, 12, 0.1f0, 1, 1, 100, "relu",
        ["Morlet"], ["MexicanHat"], "Shannon", false, 1.0f-3, 10, 0.8f0, 1.0f-5, 2, 1, 2.0f0
    )
    model = create_model(cfg, 20)
    ps, st = Lux.setup(RNG, model)
    src = randn(Float32, 20, 2)
    tgt = randn(Float32, 20, 2)
    out, _ = model((src, tgt), ps, st)
    @test size(out) == (20, 2)
end

@testset "RNO train step" begin
    cfg = RNOConfig(4, 2, "relu", 1.0f-3, 10, 0.8f0, 1.0f-5, 2, 1, 2.0f0)
    model = create_model(cfg, 8)
    ps, st = Lux.setup(RNG, model)
    x = randn(Float32, 8, 2)
    y_true = randn(Float32, 8, 2)
    loss, y_pred = _train_step(model, ps, st, x, y_true)
    @test isfinite(loss)
    @test size(y_pred) == (8, 2)
end

@testset "KAN_RNO train step" begin
    cfg = KANRNOConfig(4, 2, "relu", ["Morlet", "MexicanHat", "Shannon"], false, 1.0f-3, 10, 0.8f0, 1.0f-5, 2, 1, 2.0f0)
    model = create_model(cfg, 8)
    ps, st = Lux.setup(RNG, model)
    x = randn(Float32, 8, 2)
    y_true = randn(Float32, 8, 2)
    loss, y_pred = _train_step(model, ps, st, x, y_true)
    @test isfinite(loss)
    @test size(y_pred) == (8, 2)
end

@testset "Transformer train step" begin
    cfg = TransformerConfig(6, 2, 12, 0.1f0, 1, 1, 100, "relu", 1.0f-3, 10, 0.8f0, 1.0f-5, 2, 1, 2.0f0)
    model = create_model(cfg, 8)
    ps, st = Lux.setup(RNG, model)
    src = randn(Float32, 8, 2)
    tgt = randn(Float32, 8, 2)
    loss, y_pred = _train_step(model, ps, st, src, tgt)
    @test isfinite(loss)
    @test size(y_pred) == (8, 2)
end

@testset "KAN_Transformer train step" begin
    cfg = KANTransformerConfig(
        6, 2, 12, 0.1f0, 1, 1, 100, "relu",
        ["Morlet"], ["MexicanHat"], "Shannon", false, 1.0f-3, 10, 0.8f0, 1.0f-5, 2, 1, 2.0f0
    )
    model = create_model(cfg, 8)
    ps, st = Lux.setup(RNG, model)
    src = randn(Float32, 8, 2)
    tgt = randn(Float32, 8, 2)
    loss, y_pred = _train_step(model, ps, st, src, tgt)
    @test isfinite(loss)
    @test size(y_pred) == (8, 2)
end
