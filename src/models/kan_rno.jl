struct KANRNO{O, H} <: Lux.AbstractLuxContainerLayer{(:output_layers, :hidden_layers)}
    output_layers::O
    hidden_layers::H
    dt::Float32
    T::Int
    n_hidden::Int
    bptt_k::Int
end

function KANRNO(cfg::KANRNOConfig, input_dim::Int, output_dim::Int, input_size::Int)
    hidden_units = fill(cfg.n_hidden, cfg.num_layers)
    layer_output = [input_dim + output_dim + cfg.n_hidden; hidden_units; output_dim]
    layer_hidden = [cfg.n_hidden + output_dim; hidden_units; cfg.n_hidden]

    out_layers = Lux.Chain(
        (
            KANdense(layer_output[i], layer_output[i + 1], cfg.wavelet_names[i], cfg.activation; norm = cfg.norm)
                for i in 1:(length(layer_output) - 1)
        )...,
    )
    hid_layers = Lux.Chain(
        (
            KANdense(layer_hidden[i], layer_hidden[i + 1], cfg.wavelet_names[i], cfg.activation; norm = cfg.norm)
                for i in 1:(length(layer_hidden) - 1)
        )...,
    )

    dt = Float32(1 / (input_size - 1))
    return KANRNO(out_layers, hid_layers, dt, input_size, cfg.n_hidden, cfg.bptt_k)
end

function (m::KANRNO)(input, ps, st)
    x, y_true = input
    bs = size(x)[end]

    dxdt = (x[1:(m.T - 1), :] .- x[2:(m.T), :]) ./ m.dt

    y_init = y_true[1:1, :]
    y_rest = fill!(similar(x, m.T - 1, bs), 0.0f0)
    hidden = fill!(similar(x, m.n_hidden, bs), 0.0f0)

    st_out = st.output_layers
    st_hid = st.hidden_layers

    # TBPTT via chunked @trace for
    for chunk_start in 2:m.bptt_k:m.T
        chunk_end = min(chunk_start + m.bptt_k - 1, m.T)
        @trace for t in chunk_start:chunk_end
            xprev = reshape(x[t - 1, :], 1, :)
            dxdt_t = reshape(dxdt[t - 1, :], 1, :)

            h, st_hid = m.hidden_layers(vcat(xprev, hidden), ps.hidden_layers, st_hid)
            hidden = hidden .+ h .* m.dt   # forward Euler accumulator

            output, st_out = m.output_layers(vcat(xprev, dxdt_t, hidden), ps.output_layers, st_out)

            y_rest[t - 1, :] = vec(output)
        end
        chunk_end < m.T && (hidden = Reactant.ignore_derivatives(hidden))
    end

    return vcat(y_init, y_rest), (output_layers = st_out, hidden_layers = st_hid)
end
