include("src/WavKANSequence.jl")

using .WavKANSequence
using CSV, DataFrames, Statistics, Printf, PlotlyJS, Lux
using PlotlyJS: box, plot

const MODEL_NAMES = ["RNO", "KAN_RNO", "Transformer", "KAN_Transformer"]
const PLOT_NAMES = ["MLP RNO", "wavKAN RNO", "MLP Transformer", "wavKAN Transformer"]
const NUM_REPS = 5

train_loader, _ = get_visco_loader(1)
input_size = size(first(train_loader)[2], 1)

results = DataFrame(
    Model = String[], train_loss = String[], test_loss = String[],
    BIC = String[], time = String[], param_count = String[],
)
box_data = Dict(k => DataFrame(model = String[], value = Float64[]) for k in ["train", "test", "BIC", "time"])

for (idx, mname) in enumerate(MODEL_NAMES)
    log_dir = joinpath("logs", mname)
    tl, vl, bic_vals, times = Float64[], Float64[], Float64[], Float64[]

    for i in 1:NUM_REPS
        f = joinpath(log_dir, "repetition_$i.csv")
        isfile(f) || continue
        df = CSV.read(f, DataFrame)
        isnan(df[!, "Test Loss"][end]) && continue
        push!(tl, df[!, "Train Loss"][end])
        push!(vl, df[!, "Test Loss"][end])
        push!(bic_vals, df[!, "BIC"][end])
        push!(times, df[!, "Time (s)"][end] / 60)

        push!(box_data["train"], (model = PLOT_NAMES[idx], value = df[!, "Train Loss"][end]); promote = true)
        push!(box_data["test"], (model = PLOT_NAMES[idx], value = df[!, "Test Loss"][end]); promote = true)
        push!(box_data["BIC"], (model = PLOT_NAMES[idx], value = df[!, "BIC"][end]); promote = true)
        push!(box_data["time"], (model = PLOT_NAMES[idx], value = df[!, "Time (s)"][end] / 60); promote = true)
    end

    isempty(tl) && continue

    cfg = load_config(mname)
    model = create_model(cfg, input_size)
    n_params = Lux.parameterlength(model)

    push!(
        results, (
            Model = PLOT_NAMES[idx],
            train_loss = @sprintf("%.2g ± %.2g", mean(tl), std(tl)),
            test_loss = @sprintf("%.2g ± %.2g", mean(vl), std(vl)),
            BIC = @sprintf("%.2g ± %.2g", mean(bic_vals), std(bic_vals)),
            time = @sprintf("%.2g ± %.2g", mean(times), std(times)),
            param_count = replace(string(n_params), r"(?<=\d)(?=(\d{3})+$)" => ","),
        )
    )
end

mkpath("figures")

zebra = [iseven(i) ? "#f3f5f8" : "white" for i in 1:nrow(results)]
table_plot = plot(
    PlotlyJS.table(;
        columnwidth = [1.4, 1.0, 1.4, 1.4, 1.4, 1.1],
        header = attr(;
            values = [
                "<b>Model</b>", "<b>Params</b>", "<b>Train Loss</b>",
                "<b>Test Loss</b>", "<b>BIC</b>", "<b>Time (mins)</b>",
            ],
            align = ["left", "right", "center", "center", "center", "center"],
            line_color = "#2a3f55",
            fill_color = "#2a3f55",
            font = attr(; family = "Computer Modern, Latin Modern Roman, serif", color = "white", size = 15),
            height = 36,
        ),
        cells = attr(;
            values = [
                PLOT_NAMES[1:nrow(results)],
                results.param_count,
                results.train_loss,
                results.test_loss,
                results.BIC,
                results.time,
            ],
            align = ["left", "right", "center", "center", "center", "center"],
            line_color = "#cbd5dc",
            fill_color = [zebra for _ in 1:6],
            font = attr(; family = "Computer Modern, Latin Modern Roman, serif", size = 14, color = "#1a1a1a"),
            height = 32,
        ),
    ),
    Layout(;
        title = attr(;
            text = "Final-epoch metrics across 5 seeds",
            x = 0.5, xanchor = "center",
            font = attr(; family = "Computer Modern, Latin Modern Roman, serif", size = 18, color = "#1a1a1a"),
        ),
        font = attr(; family = "Computer Modern, Latin Modern Roman, serif", size = 13),
        margin = attr(; l = 30, r = 30, t = 80, b = 30),
        width = 1000,
        height = 80 + 36 + 32 * nrow(results) + 30,
        paper_bgcolor = "white",
    ),
)

savefig(table_plot, "figures/loss_table.png"; width = 1000, height = 80 + 36 + 32 * nrow(results) + 30)

const PALETTE = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

function make_strip(df, name; log_y::Bool)
    present = [pn for pn in PLOT_NAMES if pn in df.model]
    traces = []
    shapes = []

    for (i, pn) in enumerate(present)
        vals = Float64.(df[df.model .== pn, :value])
        n = length(vals)
        x_jit = collect(i .+ range(-0.15, 0.15; length = n))   # deterministic spread

        push!(
            traces, scatter(;
                x = x_jit, y = vals,
                mode = "markers",
                marker = attr(;
                    size = 12, opacity = 0.9, color = PALETTE[i],
                    line = attr(; width = 1.2, color = "white"),
                ),
                name = pn, showlegend = false, hoverinfo = "y+name",
            )
        )

        med = median(vals)
        push!(
            shapes, attr(;
                type = "line",
                x0 = i - 0.3, x1 = i + 0.3, y0 = med, y1 = med,
                xref = "x", yref = "y",
                line = attr(; width = 4, color = PALETTE[i]),
            )
        )
    end

    layout = Layout(;
        title = attr(;
            text = name, x = 0.5, xanchor = "center",
            font = attr(; family = "Computer Modern, Latin Modern Roman, serif", size = 18, color = "#1a1a1a"),
        ),
        xaxis = attr(;
            title = "Model",
            tickmode = "array",
            tickvals = collect(1:length(present)),
            ticktext = present,
            range = [0.5, length(present) + 0.5],
            showgrid = false,
            tickfont = attr(; family = "Computer Modern, Latin Modern Roman, serif", size = 13),
        ),
        yaxis = attr(;
            title = name,
            type = log_y ? "log" : "linear",
            gridcolor = "#e6e8eb", zerolinecolor = "#cbd5dc",
            tickfont = attr(; family = "Computer Modern, Latin Modern Roman, serif", size = 13),
        ),
        shapes = shapes,
        showlegend = false,
        font = attr(; family = "Computer Modern, Latin Modern Roman, serif", size = 14),
        width = 800, height = 500,
        margin = attr(; l = 80, r = 30, t = 80, b = 70),
        paper_bgcolor = "white", plot_bgcolor = "white",
    )

    return savefig(plot(traces, layout), "figures/$(name).png"; width = 800, height = 500)
end

make_strip(box_data["train"], "Train Loss"; log_y = true)
make_strip(box_data["test"], "Test Loss"; log_y = true)
make_strip(box_data["BIC"], "BIC"; log_y = true)
make_strip(box_data["time"], "Time (mins)"; log_y = false)
