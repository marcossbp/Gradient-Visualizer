import streamlit as st

# EDUCATIONAL CONTENT
# Ready-made examples and the "How it works" tab. Kept apart from app.py so the app logic
# and the teaching text can be changed independently.

NON_CONVEX = "Non-Convex Surface (Multiple Valleys)"
ROSENBROCK = "Rosenbrock (Narrow Curved Valley)"
CONVEX_BOWL = "Convex Bowl (Single Minimum)"

# Each example only lists the controls it changes; every other control goes back to its default.
# The keys are the keys of the sidebar widgets in app.py.
EXAMPLES = {
    "Saddle point trap": dict(
        settings=dict(landscape=NON_CONVEX, w1_start=-1.0, w2_start=1.0),
        text="All three optimizers start on top of a peak and roll down to **(0, 0)**, where they stop "
             "with a loss of 0.2. That point is not a valley: it is a **saddle point** between the two "
             "valleys, where the gradient is 0 so the update formula stops moving the weights. "
             "Gradient descent only knows the slope where it is standing, not where the real minimum is.",
    ),
    "The starting point decides the valley": dict(
        settings=dict(landscape=NON_CONVEX, w1_start=-0.8, w2_start=1.2, iterations=60),
        text="Starting a little to the right of the peak, every optimizer ends in the valley at **(1, 1)**. "
             "Now change the start to **w1 = -1.2, w2 = 0.8** and press Apply: they end in the other valley "
             "at **(-1, -1)**. In a real network this is why the **initial weights** matter.",
    ),
    "Learning rate too high": dict(
        settings=dict(landscape=CONVEX_BOWL, w1_start=-1.5, w2_start=1.5, sgd_lr=7.0),
        text="SGD uses **η = 7**. Each step jumps over the minimum and lands higher on the other side, so "
             "the jumps keep growing until the weights explode (the app stops it and shows a warning). "
             "Momentum and Adam, with their normal η, reach the minimum without problems.",
    ),
    "Learning rate too small": dict(
        settings=dict(landscape=CONVEX_BOWL, w1_start=-1.5, w2_start=1.5, sgd_lr=0.02),
        text="SGD uses **η = 0.02**. The steps are so small that after 40 epochs it has barely moved "
             "(look at the loss curve). It would reach the minimum eventually, but training would take "
             "far too long.",
    ),
    "Momentum overshoots": dict(
        settings=dict(landscape=CONVEX_BOWL, w1_start=-1.5, w2_start=1.5, mom_lr=0.3, mom_beta=0.95),
        text="Momentum keeps **95% of its speed** (β = 0.95). Like a heavy ball, it rolls past the minimum, "
             "climbs the other side and swings back and forth instead of stopping. Move the step slider and "
             "watch its loss curve go up and down.",
    ),
    "The Rosenbrock race": dict(
        settings=dict(landscape=ROSENBROCK, w1_start=-1.0, w2_start=1.0, iterations=100),
        text="The minimum is at **(1, 1)**, at the end of a long, narrow, curved valley. The walls are steep "
             "but the floor is almost flat, so **SGD** crawls, **Momentum** builds up speed along the floor and "
             "**Adam** goes furthest because it gives each weight its own step size.",
    ),
}


def render_how_it_works():
    st.header("What are you looking at?")
    st.markdown(
        "A neural network learns by adjusting its **weights**. This app uses the smallest possible "
        "network: **two weights** (w1 and w2), so every combination of weights can be drawn as a point "
        "on a map."
    )
    # Two inputs, each multiplied by its weight, give the prediction; comparing it with the real
    # value gives the loss
    st.graphviz_chart("""
        digraph {
            rankdir=LR; bgcolor="transparent";
            node [shape=circle, style=filled, fillcolor="#1C83E1", fontcolor=white, fontname="Helvetica"];
            edge [color="#888888", fontcolor="#888888", fontname="Helvetica"];
            x1 [label="x₁"]; x2 [label="x₂"];
            y [label="ŷ", fillcolor="#21C354"];
            L [label="Loss", shape=box, fillcolor="#FF4B4B"];
            x1 -> y [label="  w₁  ", penwidth=2];
            x2 -> y [label="  w₂  ", penwidth=2];
            y -> L [label="  compare with\\n  real value"];
        }
    """)
    st.markdown(
        "- The **two horizontal axes** of the 3D graph are the two weights, w1 and w2.\n"
        "- The **height** is the loss: how wrong the network's prediction is with those weights.\n"
        "- **Training** means moving the point downhill until the loss is as low as possible.\n\n"
        "With millions of weights the map has millions of dimensions and cannot be drawn, but the "
        "optimizers work exactly the same way."
    )

    st.header("Gradient descent")
    st.markdown(
        "The optimizer cannot see the whole map; it only knows the **slope where it is standing**. "
        "The slope in each direction is a partial derivative, and together they form the **gradient** "
        "∇L, which always points uphill. So the weights take a step in the opposite direction:"
    )
    st.latex(r"w_{\text{new}} = w_{\text{old}} - \eta \cdot \nabla L")
    st.markdown(
        "The **learning rate η** sets the size of each step: too small and training takes forever, "
        "too big and the steps jump over the minimum and can explode."
    )

    st.header("The three optimizers")
    col_sgd, col_momentum, col_adam = st.columns(3)
    with col_sgd:
        st.subheader("SGD")
        st.latex(r"w \leftarrow w - \eta \nabla L")
        st.markdown("Only looks at the current slope. Simple and reliable, but slow on flat areas "
                    "and it zig-zags in narrow valleys.")
    with col_momentum:
        st.subheader("Momentum")
        st.latex(r"v \leftarrow \beta v - \eta \nabla L \qquad w \leftarrow w + v")
        st.markdown("A ball rolling downhill: it keeps a **velocity** (β = how much speed it keeps). "
                    "Faster through flat areas, but it can overshoot the minimum.")
    with col_adam:
        st.subheader("Adam")
        st.latex(r"w \leftarrow w - \eta \frac{\hat m}{\sqrt{\hat v} + \epsilon}")
        st.markdown("Momentum (m̂, average gradient) plus an **adaptive step per weight** "
                    "(v̂, average squared gradient). Every step is about η long, whatever the slope.")

    st.header("The landscapes")
    st.markdown(
        f"- **{NON_CONVEX}**: two peaks and two valleys, with a saddle point in the middle. Shows that "
        "the result depends on where you start.\n"
        f"- **{ROSENBROCK}**: a narrow curved valley with the minimum at (1, 1). The classic test for "
        "optimizers: finding the valley is easy, moving along its flat floor is hard.\n"
        f"- **{CONVEX_BOWL}**: one minimum at (0, 0). The easiest case, good for seeing the effect of "
        "the learning rate."
    )

    st.header("How to use the visualizer")
    st.markdown(
        "1. Choose a landscape, a starting point and the hyperparameters in the sidebar, then press "
        "**Apply** (or load one of the **examples**).\n"
        "2. Move the **step slider** under the graphs to follow the optimizers epoch by epoch: the 3D path "
        "and the loss curve move together.\n"
        "3. Compare the results in the **comparison table**. A red message appears if an optimizer "
        "diverges."
    )
