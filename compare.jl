include("src/WavKANSequence.jl")

using .WavKANSequence
using CSV, DataFrames, Statistics, Printf, PlotlyJS, Lux
using PlotlyJS: bar, plot

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

function make_bar(df, name; log_y::Bool)
    present = [pn for pn in PLOT_NAMES if pn in df.model]
    means = [mean(Float64.(df[df.model .== pn, :value])) for pn in present]
    stds = [std(Float64.(df[df.model .== pn, :value])) for pn in present]
    labels = [@sprintf("%.2g ± %.2g", m, s) for (m, s) in zip(means, stds)]

    trace = bar(;
        x = present, y = means,
        error_y = attr(;
            type = "data", array = stds, visible = true,
            thickness = 1.6, width = 10, color = "#1a1a1a",
        ),
        marker = attr(;
            color = PALETTE[1:length(present)],
            line = attr(; color = "white", width = 1.5),
        ),
        showlegend = false,
        cliponaxis = false,
    )

    # Annotate above error-bar (mean + std)
    annotations = [
        attr(;
                x = present[i], y = means[i] + stds[i],
                xref = "x", yref = "y",
                text = labels[i], showarrow = false,
                yanchor = "bottom", yshift = 6,
                font = attr(; family = "Computer Modern, Latin Modern Roman, serif", size = 13, color = "#1a1a1a"),
            )
            for i in 1:length(present)
    ]

    layout = Layout(;
        title = attr(;
            text = name, x = 0.5, xanchor = "center",
            font = attr(; family = "Computer Modern, Latin Modern Roman, serif", size = 18, color = "#1a1a1a"),
        ),
        xaxis = attr(;
            title = "Model", showgrid = false, showline = true, linecolor = "#cbd5dc",
            tickfont = attr(; family = "Computer Modern, Latin Modern Roman, serif", size = 13),
        ),
        yaxis = attr(;
            title = name,
            type = log_y ? "log" : "linear",
            dtick = log_y ? 1 : nothing,         # log: decade-only ticks (10⁰, 10¹, …); linear: auto
            minor = log_y ? attr(; showgrid = false, ticks = "") : nothing,
            gridcolor = "#e6e8eb", griddash = "dash",
            zeroline = !log_y, zerolinecolor = "#cbd5dc",
            showline = true, linecolor = "#cbd5dc",
            tickfont = attr(; family = "Computer Modern, Latin Modern Roman, serif", size = 13),
            automargin = true,
        ),
        showlegend = false,
        font = attr(; family = "Computer Modern, Latin Modern Roman, serif", size = 14),
        width = 800, height = 500,
        margin = attr(; l = 90, r = 40, t = 90, b = 80),
        paper_bgcolor = "white", plot_bgcolor = "white",
        bargap = 0.35,
        annotations = annotations,
    )

    return savefig(plot(trace, layout), "figures/$(name).png"; width = 800, height = 500)
end

make_bar(box_data["train"], "Train Loss"; log_y = false)
make_bar(box_data["test"], "Test Loss"; log_y = false)
make_bar(box_data["BIC"], "BIC"; log_y = false)
make_bar(box_data["time"], "Time (mins)"; log_y = false)
