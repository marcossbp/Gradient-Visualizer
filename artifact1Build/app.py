import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

# Import backend math logic
from math_engine import LANDSCAPES, OPTIMIZERS
from explanations import EXAMPLES, render_how_it_works

# Set page config for Streamlit layout
st.set_page_config(page_title="3D Gradient Descent Visualizer", layout="wide")
st.title("Interactive 3D Loss Landscape & Gradient Descent")

# Chrome stops drawing the 3D graph when it scrolls under Streamlit's top bar (it wrongly treats
# the solid bar as covering the whole graph). A transparent top bar avoids that bug.
st.markdown("<style>[data-testid='stHeader'] {background: transparent;}</style>",
            unsafe_allow_html=True)

# Two tabs: the interactive visualizer and the explanations of the concepts behind it
visualizer_tab, explanations_tab = st.tabs(["Visualizer", "How it works"])
with explanations_tab:
    render_how_it_works()

# Colors and marker symbols used to tell the optimizers apart on the graph
OPTIMIZER_STYLES = {
    "SGD": dict(color="#FFA421", symbol="circle"),
    "Momentum": dict(color="#FF4B4B", symbol="square"),
    "Adam": dict(color="#21C354", symbol="diamond"),
}

# Default value of every sidebar control (the keys are the widget keys).
# The widgets read their value from st.session_state, so the reset button and the examples
# only have to change these values.
DEFAULTS = {
    "landscape": list(LANDSCAPES.keys())[0],
    "w1_start": -1.0,
    "w2_start": 1.0,
    "iterations": 40,
    "optimizers": list(OPTIMIZERS.keys()),
    "sgd_lr": 0.5,
    "mom_lr": 0.1,
    "mom_beta": 0.9,
    "adam_lr": 0.05,
    "adam_b1": 0.9,
    "adam_b2": 0.999,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_controls():
    st.session_state.update(DEFAULTS)
    clear_example()


def load_example():
    # Start from the defaults and change only the controls used by the example
    name = st.session_state["example_choice"]
    st.session_state.update(DEFAULTS)
    st.session_state.update(EXAMPLES[name]["settings"])
    st.session_state["active_example"] = name


def clear_example():
    # Once the user changes the settings, the example explanation no longer applies
    st.session_state.pop("active_example", None)


# Ready-made examples that show typical optimizer behaviours with one click
st.sidebar.header("Examples")
st.sidebar.selectbox("Choose an example", list(EXAMPLES.keys()), key="example_choice")
st.sidebar.button("Load example", on_click=load_example, width="stretch")


# Sidebar Controls
# All the controls are inside a form: changing a slider does not update the graph,
# the simulation only runs again when the user presses "Apply"
with st.sidebar.form("controls"):
    st.header("1. Select Landscape")
    selected_landscape_name = st.selectbox("Loss Landscape", list(LANDSCAPES.keys()), key="landscape")
    selected_loss_fn = LANDSCAPES[selected_landscape_name]

    st.header("2. Initial Position & Iterations")
    # Shared settings: every optimizer starts at the same point and runs the same number of steps
    w1_start = st.slider("Starting Weight 1 (w1)", -2.0, 2.0, step=0.05, key="w1_start")
    w2_start = st.slider("Starting Weight 2 (w2)", -2.0, 2.0, step=0.05, key="w2_start")
    iterations = st.slider("Iterations (Epochs)", 5, 100, step=5, key="iterations")

    st.header("3. Optimizers")
    selected_optimizers = st.multiselect("Optimizers to compare", list(OPTIMIZERS.keys()),
                                         key="optimizers")

    # Hyperparameters of each optimizer
    # Adam's step size does not depend on how steep the terrain is, so it needs a smaller η.
    # SGD and Momentum allow a very high η so the user can see what happens when it diverges.
    optimizer_params = {
        "SGD": dict(),
        "Momentum": dict(),
        "Adam": dict(),
    }
    with st.expander("SGD hyperparameters"):
        optimizer_params["SGD"]["learning_rate"] = st.slider(
            "Learning Rate (η)", 0.01, 10.0, step=0.01, key="sgd_lr")
    with st.expander("Momentum hyperparameters"):
        optimizer_params["Momentum"]["learning_rate"] = st.slider(
            "Learning Rate (η)", 0.01, 10.0, step=0.01, key="mom_lr")
        optimizer_params["Momentum"]["momentum"] = st.slider(
            "Momentum (β)", 0.0, 0.99, step=0.01, key="mom_beta")
    with st.expander("Adam hyperparameters"):
        optimizer_params["Adam"]["learning_rate"] = st.slider(
            "Learning Rate (η)", 0.001, 1.0, step=0.001, key="adam_lr")
        optimizer_params["Adam"]["beta1"] = st.slider(
            "β1 (1st moment decay)", 0.0, 0.99, step=0.01, key="adam_b1")
        optimizer_params["Adam"]["beta2"] = st.slider(
            "β2 (2nd moment decay)", 0.9, 0.9999, step=0.0001, format="%.4f", key="adam_b2")

    st.form_submit_button("Apply", type="primary", on_click=clear_example)

st.sidebar.button("Reset to defaults", on_click=reset_controls)

if not selected_optimizers:
    visualizer_tab.info("Select at least one optimizer in the sidebar and press Apply to start the simulation.")
    st.stop()

# Explanation of the example that is loaded: what to look at in the graphs
if "active_example" in st.session_state:
    name = st.session_state["active_example"]
    visualizer_tab.info(f"**Example: {name}**\n\n{EXAMPLES[name]['text']}")


# Execute Optimization Loops
# runs[name] = (path_w1, path_w2, path_loss)
runs = {
    name: OPTIMIZERS[name](selected_loss_fn, w1_start, w2_start,
                           iterations=iterations, **optimizer_params[name])
    for name in selected_optimizers
}


# Numerical stability warnings
# A path shorter than iterations + 1 means the optimizer was stopped because it diverged
diverged = {}
for name, (path_w1, path_w2, path_loss) in runs.items():
    if len(path_loss) < iterations + 1:
        diverged[name] = len(path_loss) - 1
        # Every landscape has a minimum, so the only way to diverge is a learning rate that is too
        # high: every step overshoots the minimum and lands higher than before
        visualizer_tab.error(f"**{name} diverged at step {diverged[name]}**: the learning rate (η) is too high. "
                 f"Each step overshoots the minimum and lands higher than before, so the weights "
                 f"grow until they explode. Try a smaller η.")


# Comparison table with the final result of each optimizer
visualizer_tab.subheader("Optimizer comparison")
comparison_rows = []
for name, (path_w1, path_w2, path_loss) in runs.items():
    best_step = int(np.nanargmin(path_loss))
    comparison_rows.append({
        "Optimizer": name,
        "Final w1": round(path_w1[-1], 4),
        "Final w2": round(path_w2[-1], 4),
        "Final Loss": round(path_loss[-1], 4),
        "Lowest Loss": round(path_loss[best_step], 4),
        "Reached at step": best_step,
    })
visualizer_tab.dataframe(comparison_rows, hide_index=True, width="stretch")


# Creates the grid & computes mesh values
# np.linspaces divides the space between two points in x equal parts
w1_values = np.linspace(-2, 2, 100) # x-axis
w2_values = np.linspace(-2, 2, 100) # y-axis

# np.meshgrid creates two 2D arrays from two 1D vectors
# W1 contains the x coordinate of every pair of coordinates
# W2 contains the y coordinate of every pair of coordinates
W1, W2 = np.meshgrid(w1_values, w2_values)

# Calculate the loss value for each tuple of coordinates
# lossFunction -> inside this function we do matricial operations
# lossValues -> an array containing the loss value for each coordinate
lossValues = selected_loss_fn(W1, W2)

# Base surface used by the animated figure
# A single-hue colorscale keeps the colored optimizer paths easy to see on top of it
base_surface = go.Surface(z=lossValues, x=W1, y=W2, opacity=0.85,
                          colorscale="Blues", showscale=False)

# Fixed axis ranges so the view does not jump when a path leaves the landscape
z_padding = 0.1 * (lossValues.max() - lossValues.min())
z_range = [lossValues.min() - z_padding, lossValues.max() + z_padding]

# Drawing graphs & visuals logic (streamlit and embedded javascript)
try:
    is_dark = st.context.theme.type == "dark"   # Streamlit >= 1.46
except Exception:
    is_dark = (st.get_option("theme.base") == "dark")

text_color = "#FAFAFA" if is_dark else "#31333F"
grid_color = "rgba(250,250,250,0.15)" if is_dark else "rgba(49,51,63,0.15)"
transparent = "rgba(0,0,0,0)"

FIGURE_HEIGHT = 620


def build_figure():
    # Left: 3D weight trajectory. Right: 2D loss vs epochs curve.
    # Both live in the same figure so a single step slider moves them together.
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                        specs=[[{"type": "scene"}, {"type": "xy"}]],
                        subplot_titles=("Weight trajectory on the loss landscape", "Loss vs Epochs"),
                        horizontal_spacing=0.06)

    fig.add_trace(base_surface, row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[w1_start], y=[w2_start], z=[selected_loss_fn(w1_start, w2_start)],
        mode='markers', marker=dict(size=10, color=text_color),
        name='Start'), row=1, col=1)

    # Each optimizer adds four traces that move with the slider:
    # 3D path, 3D current position, 2D loss curve, 2D current loss
    animated_indices = []
    for name, (path_w1, path_w2, path_loss) in runs.items():
        style = OPTIMIZER_STYLES[name]
        fig.add_trace(go.Scatter3d(
            x=[path_w1[0]], y=[path_w2[0]], z=[path_loss[0]],
            mode='lines+markers', line=dict(color=style["color"], width=5),
            marker=dict(size=3, color=style["color"]),
            name=name, legendgroup=name), row=1, col=1)
        fig.add_trace(go.Scatter3d(
            x=[path_w1[0]], y=[path_w2[0]], z=[path_loss[0]],
            mode='markers',
            marker=dict(size=7, color=style["color"], symbol=style["symbol"],
                        line=dict(color=text_color, width=2)),
            name=f'{name} Position', legendgroup=name, showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[0], y=[path_loss[0]], mode='lines',
            line=dict(color=style["color"], width=3),
            name=f'{name} Loss', legendgroup=name, showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[0], y=[path_loss[0]], mode='markers',
            marker=dict(size=10, color=style["color"], symbol=style["symbol"],
                        line=dict(color=text_color, width=1)),
            name=f'{name} Current Loss', legendgroup=name, showlegend=False), row=1, col=2)
        animated_indices += list(range(len(fig.data) - 4, len(fig.data)))

    # Every slider step updates all the animated traces at once
    # A diverged optimizer has a shorter path, so it stays on its last recorded step
    steps = []
    for i in range(iterations + 1):
        xs, ys, zs = [], [], []
        for path_w1, path_w2, path_loss in runs.values():
            j = min(i, len(path_loss) - 1)
            epochs = list(range(j + 1))
            xs += [list(path_w1[:j + 1]), [path_w1[j]], epochs, [j]]
            ys += [list(path_w2[:j + 1]), [path_w2[j]], list(path_loss[:j + 1]), [path_loss[j]]]
            zs += [list(path_loss[:j + 1]), [path_loss[j]], None, None]
        steps.append(dict(method='restyle', label=str(i),
                          args=[{'x': xs, 'y': ys, 'z': zs}, animated_indices]))

    axis_style = dict(backgroundcolor=transparent, gridcolor=grid_color,
                      zerolinecolor=grid_color, showbackground=False,
                      color=text_color)

    fig.update_layout(
        paper_bgcolor=transparent,
        plot_bgcolor=transparent,
        font=dict(family="Source Sans Pro, sans-serif", color=text_color),
        scene=dict(
            xaxis=dict(title="Weight 1", range=[-2, 2], **axis_style),
            yaxis=dict(title="Weight 2", range=[-2, 2], **axis_style),
            zaxis=dict(title="Loss", range=z_range, **axis_style),
            bgcolor=transparent,
        ),
        # pad 'b' keeps space under the slider so its step numbers are not cut off
        sliders=[dict(active=0, pad={'t': 50, 'b': 20}, steps=steps,
                      font=dict(color=text_color),
                      currentvalue=dict(prefix="Step: ", font=dict(color=text_color)))],
        legend=dict(font=dict(color=text_color), orientation="h", x=0, y=1.1),
        height=FIGURE_HEIGHT,
        margin=dict(l=0, r=10, b=0, t=60),
    )
    fig.update_xaxes(title="Epoch", range=[0, iterations], gridcolor=grid_color,
                     zerolinecolor=grid_color, color=text_color, row=1, col=2)
    # The loss is never negative, so the loss curve starts at 0. The top is the highest loss the
    # optimizers reach (not the highest point of the surface), so the curves fill the chart.
    # A diverged path can explode, so it is capped at the top of the surface.
    highest_loss = max(max(path_loss) for _, _, path_loss in runs.values())
    loss_top = 1.1 * min(highest_loss, z_range[1])
    fig.update_yaxes(title="Loss", range=[0, loss_top], gridcolor=grid_color,
                     zerolinecolor=grid_color, color=text_color, row=1, col=2)
    fig.update_annotations(font=dict(color=text_color))
    return fig


KEEP_CAMERA_JS = """
var gd = document.getElementById('{plot_id}');
var cam = null;

gd.on('plotly_relayout', function(e) {
    if (e['scene.camera']) { cam = e['scene.camera']; }
});

function restoreCamera() {
    if (cam) { setTimeout(function() { Plotly.relayout(gd, {'scene.camera': cam}); }, 0); }
}
gd.on('plotly_sliderchange', restoreCamera);
gd.on('plotly_restyle', restoreCamera);

// The 3D canvas captures the mouse wheel, so the page could not be scrolled while the
// mouse was over the graph. Stopping the event before it reaches the canvas lets the
// browser scroll the page as usual.
gd.addEventListener('wheel', function(e) { e.stopPropagation(); }, true);

// Follow Streamlit's light/dark theme. The colors are chosen in Python when the page loads,
// so switching the theme afterwards would leave dark text on a dark page (or the opposite).
// Every half second we check the page background and recolor the graph if it changed.
var currentText = null;
function applyTheme() {
    var app = window.parent.document.querySelector('[data-testid="stApp"]');
    if (!app) { return; }
    var rgb = window.parent.getComputedStyle(app).backgroundColor.match(/\\d+/g).map(Number);
    var dark = (rgb[0] + rgb[1] + rgb[2]) / 3 < 128;
    var text = dark ? '#FAFAFA' : '#31333F';
    var grid = dark ? 'rgba(250,250,250,0.15)' : 'rgba(49,51,63,0.15)';
    if (text === currentText) { return; }
    currentText = text;

    var update = {'font.color': text, 'legend.font.color': text,
                  'sliders[0].font.color': text, 'sliders[0].currentvalue.font.color': text};
    ['xaxis', 'yaxis', 'scene.xaxis', 'scene.yaxis', 'scene.zaxis'].forEach(function(axis) {
        update[axis + '.color'] = text;
        update[axis + '.gridcolor'] = grid;
        update[axis + '.zerolinecolor'] = grid;
    });
    (gd.layout.annotations || []).forEach(function(_, i) {
        update['annotations[' + i + '].font.color'] = text;
    });
    Plotly.relayout(gd, update);

    // Start point and the outline of the position markers use the text color too
    gd.data.forEach(function(trace, i) {
        if (trace.name === 'Start') { Plotly.restyle(gd, {'marker.color': text}, [i]); }
        else if (trace.marker && trace.marker.line && trace.marker.line.color) {
            Plotly.restyle(gd, {'marker.line.color': text}, [i]);
        }
    });
}
applyTheme();
setInterval(applyTheme, 500);
"""

html = build_figure().to_html(
    full_html=False,
    include_plotlyjs="cdn",
    post_script=KEEP_CAMERA_JS,
    # scrollZoom off: scrolling over the graph scrolls the page instead of zooming the graph away
    # (zoom is still available with the zoom tool in the graph toolbar)
    config={"responsive": True, "scrollZoom": False},
    default_height=f"{FIGURE_HEIGHT}px",
)

html = "<style>html,body{margin:0;padding:0;background:transparent;}</style>" + html

with visualizer_tab:
    components.html(html, height=FIGURE_HEIGHT + 10, scrolling=False)

