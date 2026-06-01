struct RNO{O, H} <: Lux.AbstractLuxContainerLayer{(:output_chain, :hidden_chain)}
    output_chain::O
    hidden_chain::H
    dt::Float32
    T::Int
    n_hidden::Int
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
    return RNO(Lux.Chain(out_layers...), Lux.Chain(hid_layers...), dt, input_size, cfg.n_hidden)
end

function (m::RNO)(input, ps, st)
    x, y_true = input
    bs = size(x)[end]
    T = m.T

    dxdt = (x[1:(T - 1), :] .- x[2:T, :]) ./ m.dt

    y = similar(x, T, bs)
    y[1, :] .= y_true[1, :]

    hidden = fill!(similar(x, m.n_hidden, bs), 0.0f0)

    state = (2, hidden, y, st.output_chain, st.hidden_chain)
    @trace while first(state) <= T
        t, hidden_curr, y_curr, st_out_curr, st_hid_curr = state

        xprev = x[(t - 1):(t - 1), :]
        dxdt_t = dxdt[(t - 1):(t - 1), :]

        h, st_hid_new = m.hidden_chain(vcat(xprev, hidden_curr), ps.hidden_chain, st_hid_curr)
        hidden_new = h .* m.dt

        out, st_out_new = m.output_chain(
            vcat(xprev, dxdt_t, hidden_new),
            ps.output_chain, st_out_curr,
        )
        y_curr[t:t, :] = out

        state = (t + 1, hidden_new, y_curr, st_out_new, st_hid_new)
    end

    _, _, y_final, st_out_final, st_hid_final = state
    return y_final, (output_chain = st_out_final, hidden_chain = st_hid_final)
end
