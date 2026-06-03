struct MultiHeadAttention{Q, K, V, O} <: Lux.AbstractLuxContainerLayer{(:Wq, :Wk, :Wv, :Wo)}
    Wq::Q
    Wk::K
    Wv::V
    Wo::O
    nhead::Int
end

_linear_proj(d_model, kan_factory) =
    kan_factory === nothing ? Lux.Dense(d_model => d_model) : kan_factory(d_model, d_model)

function MultiHeadAttention(d_model::Int, nhead::Int; kan_factory = nothing)
    @assert d_model % nhead == 0 "d_model ($d_model) must be divisible by nhead ($nhead)"
    return MultiHeadAttention(
        _linear_proj(d_model, kan_factory),
        _linear_proj(d_model, kan_factory),
        _linear_proj(d_model, kan_factory),
        _linear_proj(d_model, kan_factory),
        nhead,
    )
end

function (m::MultiHeadAttention)(qkv, ps, st)
    x, y, z = qkv
    q, st_q = m.Wq(x, ps.Wq, st.Wq)
    k, st_k = m.Wk(y, ps.Wk, st.Wk)
    v, st_v = m.Wv(z, ps.Wv, st.Wv)
    attn_out, _ = NNlib.dot_product_attention(q, k, v; nheads = m.nhead)
    out, st_o = m.Wo(attn_out, ps.Wo, st.Wo)
    return out, (Wq = st_q, Wk = st_k, Wv = st_v, Wo = st_o)
end

function _make_ff(d_model, dim_ff, activation, kan_factory)
    if kan_factory !== nothing
        return kan_factory(d_model, dim_ff), kan_factory(dim_ff, d_model)
    end
    act = get_activation(activation)
    return Lux.Dense(d_model => dim_ff, act), Lux.Dense(dim_ff => d_model)
end

struct EncoderLayer{A, F1, F2, N1, N2} <: Lux.AbstractLuxContainerLayer{(:self_attn, :ff1, :ff2, :norm1, :norm2)}
    self_attn::A
    ff1::F1
    ff2::F2
    norm1::N1
    norm2::N2
end

function EncoderLayer(d_model::Int, nhead::Int, dim_ff::Int, dropout::Float32, activation::String; kan_factory = nothing)
    attn = MultiHeadAttention(d_model, nhead; kan_factory)
    ff1, ff2 = _make_ff(d_model, dim_ff, activation, kan_factory)
    return EncoderLayer(attn, ff1, ff2, Lux.LayerNorm(d_model; dims = nothing), Lux.LayerNorm(d_model; dims = nothing))
end

function (l::EncoderLayer)(x, ps, st)
    attn_out, st_a = l.self_attn((x, x, x), ps.self_attn, st.self_attn)
    x1, st_n1 = l.norm1(x .+ attn_out, ps.norm1, st.norm1)
    ff_out, st_f1 = l.ff1(x1, ps.ff1, st.ff1)
    ff_out, st_f2 = l.ff2(ff_out, ps.ff2, st.ff2)
    out, st_n2 = l.norm2(x1 .+ ff_out, ps.norm2, st.norm2)
    return out, (self_attn = st_a, ff1 = st_f1, ff2 = st_f2, norm1 = st_n1, norm2 = st_n2)
end

struct DecoderLayer{A1, A2, F1, F2, N1, N2, N3} <: Lux.AbstractLuxContainerLayer{(:self_attn, :cross_attn, :ff1, :ff2, :norm1, :norm2, :norm3)}
    self_attn::A1
    cross_attn::A2
    ff1::F1
    ff2::F2
    norm1::N1
    norm2::N2
    norm3::N3
end

function DecoderLayer(d_model::Int, nhead::Int, dim_ff::Int, dropout::Float32, activation::String; kan_factory = nothing)
    self_a = MultiHeadAttention(d_model, nhead; kan_factory)
    cross_a = MultiHeadAttention(d_model, nhead; kan_factory)
    ff1, ff2 = _make_ff(d_model, dim_ff, activation, kan_factory)
    return DecoderLayer(
        self_a, cross_a, ff1, ff2,
        Lux.LayerNorm(d_model; dims = nothing), Lux.LayerNorm(d_model; dims = nothing), Lux.LayerNorm(d_model; dims = nothing),
    )
end

function (l::DecoderLayer)(input, ps, st)
    x, memory = input
    sa_out, st_sa = l.self_attn((x, x, x), ps.self_attn, st.self_attn)
    x1, st_n1 = l.norm1(x .+ sa_out, ps.norm1, st.norm1)
    ca_out, st_ca = l.cross_attn((x1, memory, memory), ps.cross_attn, st.cross_attn)
    x2, st_n2 = l.norm2(x1 .+ ca_out, ps.norm2, st.norm2)
    ff_out, st_f1 = l.ff1(x2, ps.ff1, st.ff1)
    ff_out, st_f2 = l.ff2(ff_out, ps.ff2, st.ff2)
    out, st_n3 = l.norm3(x2 .+ ff_out, ps.norm3, st.norm3)
    return out, (self_attn = st_sa, cross_attn = st_ca, ff1 = st_f1, ff2 = st_f2, norm1 = st_n1, norm2 = st_n2, norm3 = st_n3)
end
