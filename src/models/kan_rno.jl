struct KANRNO{O, H} <: Lux.AbstractLuxContainerLayer{(:output_layers, :hidden_layers)}
    output_layers::O
    hidden_layers::H
    dt::Float32
    T::Int
    n_hidden::Int
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
    return KANRNO(_make_named_layers(out_layers), _make_named_layers(hid_layers), dt, input_size, cfg.n_hidden)
end

function (m::KANRNO)(input, ps, st)
    x, y_true = input
    bs = size(x)[end]
    T = m.T

    dxdt = (x[1:(T - 1), :] .- x[2:T, :]) ./ m.dt

    y_init = reshape(y_true[1, :], 1, bs)
    y_rest = similar(x, T - 1, bs) .* 0.0f0

    hidden = similar(x, m.n_hidden, bs) .* 0.0f0

    state = (2, hidden, y_rest, st.output_layers, st.hidden_layers)
    @trace while first(state) <= T
        t, hidden_curr, y_curr, st_out_curr, st_hid_curr = state

        xprev = x[(t - 1):(t - 1), :]
        dxdt_t = dxdt[(t - 1):(t - 1), :]

        h = vcat(xprev, hidden_curr)
        for (k, layer) in pairs(m.hidden_layers)
            h, st_hid_k = layer(h, ps.hidden_layers[k], st_hid_curr[k])
            st_hid_curr = merge(st_hid_curr, NamedTuple{(k,)}((st_hid_k,)))
        end
        hidden_new = h .* m.dt

        output = vcat(xprev, dxdt_t, hidden_new)
        for (k, layer) in pairs(m.output_layers)
            output, st_out_k = layer(output, ps.output_layers[k], st_out_curr[k])
            st_out_curr = merge(st_out_curr, NamedTuple{(k,)}((st_out_k,)))
        end
        y_curr[(t - 1):(t - 1), :] = output

        state = (t + 1, hidden_new, y_curr, st_out_curr, st_hid_curr)
    end

    _, _, y_rest_final, st_out_final, st_hid_final = state
    return vcat(y_init, y_rest_final), (output_layers = st_out_final, hidden_layers = st_hid_final)
end
