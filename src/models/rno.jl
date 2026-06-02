struct RNO{O, H} <: Lux.AbstractLuxContainerLayer{(:output_chain, :hidden_chain)}
    output_chain::O
    hidden_chain::H
    dt::Float32
    T::Int
    n_hidden::Int
    bptt_k::Int
end

function RNO(cfg::RNOConfig, input_dim::Int, output_dim::Int, input_size::Int)
    phi = get_activation(cfg.activation)
    hidden_units = fill(cfg.n_hidden, cfg.num_layers)
    layer_output = [input_dim + output_dim + cfg.n_hidden; hidden_units; output_dim]
    layer_hidden = [cfg.n_hidden + output_dim; hidden_units; cfg.n_hidden]

    out_layers = Any[Lux.Dense(layer_output[i] => layer_output[i + 1], phi) for i in 1:(length(layer_output) - 2)]
    push!(out_layers, Lux.Dense(layer_output[end - 1] => layer_output[end]))
    hid_layers = Any[Lux.Dense(layer_hidden[i] => layer_hidden[i + 1], phi) for i in 1:(length(layer_hidden) - 2)]
    push!(hid_layers, Lux.Dense(layer_hidden[end - 1] => layer_hidden[end]))

    dt = Float32(1 / (input_size - 1))
    return RNO(Lux.Chain(out_layers...), Lux.Chain(hid_layers...), dt, input_size, cfg.n_hidden, cfg.bptt_k)
end

function (m::RNO)(input, ps, st)
    x, y_true = input
    bs = size(x)[end]

    dxdt = (x[1:(m.T - 1), :] .- x[2:(m.T), :]) ./ m.dt

    y_init = y_true[1:1, :]
    y_rest = similar(x, m.T - 1, bs) .* 0.0f0
    hidden = similar(x, m.n_hidden, bs) .* 0.0f0

    st_out = st.output_chain
    st_hid = st.hidden_chain

    @trace for t in 2:(m.T)
        xprev = reshape(x[t - 1, :], 1, :)
        dxdt_t = reshape(dxdt[t - 1, :], 1, :)

        h, st_hid = m.hidden_chain(vcat(xprev, hidden), ps.hidden_chain, st_hid)
        hidden = h .* m.dt

        out, st_out = m.output_chain(vcat(xprev, dxdt_t, hidden), ps.output_chain, st_out)

        n = size(y_rest, 1)
        selector = reshape(ifelse.((0:(n - 1)) .== (t - 2), 1.0f0, 0.0f0), n, 1)
        y_rest = y_rest .+ selector .* out
    end

    return vcat(y_init, y_rest), (output_chain = st_out, hidden_chain = st_hid)
end
