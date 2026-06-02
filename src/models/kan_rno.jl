struct KANRNO{O, H} <: Lux.AbstractLuxContainerLayer{(:output_layers, :hidden_layers)}
    output_layers::O
    hidden_layers::H
    dt::Float32
    T::Int
    n_hidden::Int
    bptt_k::Int
end

function _make_named_layers(layers::Vector)
    names = ntuple(i -> Symbol("layer_$i"), length(layers))
    return NamedTuple{names}(Tuple(layers))
end

function KANRNO(cfg::KANRNOConfig, input_dim::Int, output_dim::Int, input_size::Int)
    hidden_units = fill(cfg.n_hidden, cfg.num_layers)
    layer_output = [input_dim + output_dim + cfg.n_hidden; hidden_units; output_dim]
    layer_hidden = [cfg.n_hidden + output_dim; hidden_units; cfg.n_hidden]

    out_layers = [
        KANdense(layer_output[i], layer_output[i + 1], cfg.wavelet_names[i], cfg.activation; norm = cfg.norm)
            for i in 1:(length(layer_output) - 1)
    ]
    hid_layers = [
        KANdense(layer_hidden[i], layer_hidden[i + 1], cfg.wavelet_names[i], cfg.activation; norm = cfg.norm)
            for i in 1:(length(layer_hidden) - 1)
    ]

    dt = Float32(1 / (input_size - 1))
    return KANRNO(_make_named_layers(out_layers), _make_named_layers(hid_layers), dt, input_size, cfg.n_hidden, cfg.bptt_k)
end

function (m::KANRNO)(input, ps, st)
    x, y_true = input
    bs = size(x)[end]

    dxdt = (x[1:(m.T - 1), :] .- x[2:(m.T), :]) ./ m.dt

    y = y_true[1:1, :]
    hidden = similar(x, m.n_hidden, bs) .* 0.0f0

    st_out = st.output_layers
    st_hid = st.hidden_layers

    for t in 2:(m.T)
        if t > 2 && (t - 2) % m.bptt_k == 0
            hidden = Reactant.ignore_derivatives(hidden) # TBPTT
        end

        xprev = x[(t - 1):(t - 1), :]
        dxdt_t = dxdt[(t - 1):(t - 1), :]

        h = vcat(xprev, hidden)
        for (k, layer) in pairs(m.hidden_layers)
            h, st_hid_k = layer(h, ps.hidden_layers[k], st_hid[k])
            st_hid = merge(st_hid, NamedTuple{(k,)}((st_hid_k,)))
        end
        hidden = h .* m.dt

        output = vcat(xprev, dxdt_t, hidden)
        for (k, layer) in pairs(m.output_layers)
            output, st_out_k = layer(output, ps.output_layers[k], st_out[k])
            st_out = merge(st_out, NamedTuple{(k,)}((st_out_k,)))
        end
        y = vcat(y, output)
    end

    return y, (output_layers = st_out, hidden_layers = st_hid)
end
