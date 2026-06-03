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
            train_loss = @sprintf("%.2g +/- %.2g", mean(tl), std(tl)),
            test_loss = @sprintf("%.2g +/- %.2g", mean(vl), std(vl)),
            BIC = @sprintf("%.2g +/- %.2g", mean(bic_vals), std(bic_vals)),
            time = @sprintf("%.2g +/- %.2g", mean(times), std(times)),
            param_count = string(n_params),
        )
    )
end

mkpath("figures")

table_plot = plot(
    PlotlyJS.table(;
        header = attr(;
            values = ["Model", "Train Loss", "Test Loss", "BIC", "Time (mins)", "Param Count"],
            align = "center", line_color = "darkslategray", fill_color = "grey",
            font = attr(; family = "Computer Modern", color = "white", size = 13),
        ),
        cells = attr(;
            values = [
                PLOT_NAMES[1:nrow(results)], results.train_loss, results.test_loss,
                results.BIC, results.time, results.param_count,
            ],
            line_color = "darkslategray", align = "center",
            fill_color = [["lightgrey", "white", "lightgrey", "white"]],
            font = attr(; family = "Computer Modern", size = 12, color = "black"),
        ),
    ),
    Layout(;
        autosize = true,
        title = attr(; text = "Loss and BIC for Different Models", x = 0.5),
        font = attr(; family = "Computer Modern", size = 12),
        margin = attr(; b = 0, t = 200, l = 5, r = 5),
    ),
)

savefig(table_plot, "figures/loss_table.png")

function make_box(df, name; log_y::Bool)
    data = [box(; y = Float64.(df[df.model .== pn, :value]), name = pn) for pn in PLOT_NAMES if pn in df.model]
    layout = log_y ?
        Layout(; title = name, xaxis_title = "Model", yaxis_title = name, yaxis_type = "log") :
        Layout(; title = name, xaxis_title = "Model", yaxis_title = name)
    return savefig(plot(data, layout), "figures/$(name).png")
end

make_box(box_data["train"], "Train Loss"; log_y = true)
make_box(box_data["test"], "Test Loss"; log_y = true)
make_box(box_data["BIC"], "BIC"; log_y = true)
make_box(box_data["time"], "Time (mins)"; log_y = false)
