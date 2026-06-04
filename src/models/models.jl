include("rno.jl")
include("kan_rno.jl")
include("transformer.jl")
include("kan_transformer.jl")

function _transformer_forward(m, input, ps, st)
    src_raw, tgt_raw = input
    src, st_pe = m.pe(src_raw, ps.pe, st.pe)
    tgt, st_pe = m.pe(tgt_raw, ps.pe, st_pe)

    memory, st_enc = m.encoder(src, ps.encoder, st.encoder)
    (tgt, _), st_dec = m.decoder((tgt, memory), ps.decoder, st.decoder)

    pred, st_o = m.output_layer(tgt, ps.output_layer, st.output_layer)
    pred = reshape(pred, size(pred, 2), size(pred, 3))
    return pred, (pe = st_pe, encoder = st_enc, decoder = st_dec, output_layer = st_o)
end

function create_model(cfg::RNOConfig, input_size::Int)
    return RNO(cfg, 1, 1, input_size)
end

function create_model(cfg::KANRNOConfig, input_size::Int)
    return KANRNO(cfg, 1, 1, input_size)
end

function create_model(cfg::TransformerConfig, ::Int)
    return Transformer(cfg)
end

function create_model(cfg::KANTransformerConfig, ::Int)
    return KANTransformer(cfg)
end
