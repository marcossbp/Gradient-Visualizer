# Exploration Plan: Interactive Learning Rate & Gradient Descent Visualizer

## Overview
* **Theme:** Foundations & fundamentals of deep learning (Chapters 1, 2, 4, and 5 of *Deep Learning with Python*)
* **Artifact Format:** An interactive web-based visualizer
* **Core Concept:** A real-time 3D simulation showing how a neural network uses learning rates and partial derivatives (gradients) to navigate a loss landscape and update its weights step-by-step.
* **Tech Stack:** Python, NumPy, Plotly, Streamlit

---

## Sprint 1: Math Foundations & Web MVP

### Week 1: 3D Loss Landscape & Tensors
* **Task:** Define a 2-weight Loss function $L(w_1, w_2)$ and compute a 3D surface mesh grid using NumPy and Plotly.
* **Learning:** Tensor representations, scalar loss evaluation, and mapping a 2D weight space $(w_1, w_2)$ to a 3D elevation height $L$ (Chapters 1 & 2).
* **Deliverable:** A local Python script that renders an interactive 3D loss surface plot.

### Week 2: Partial Derivatives & Weight Updates
* **Task:** Simple Vanilla SGD optimizer. Implement backpropagation using partial derivatives ($\frac{\partial L}{\partial w_1}$ and $\frac{\partial L}{\partial w_2}$) to construct the gradient vector $\nabla L$, and write the iterative weight update loop:
  $$w_{\text{new}} = w_{\text{old}} - \eta \cdot \nabla L$$
* **Learning:** Multivariable calculus of gradient vectors and how varying step size ($\eta$) modifies the trajectory across steep terrain.
* **Deliverable:** An animated 3D plot displaying the step-by-step coordinate path of a weight vector moving downhill.

### Week 3: Interactive Web MVP & Real-Time Visualization
* **Task:** Integrate the Python backend into Streamlit, add real-time step-by-step trajectory animation (play/pause controls), enable multiple loss landscape presets (e.g., saddle points, multiple local minima), and display live numeric coordinate tracking ($w_1, w_2, L$).
* **Learning:** Web UI integration for ML backends, state management for interactive web apps, and live frame-by-frame rendering mechanics.
* **Deliverable:** An interactive web application featuring hyperparameter sliders ($\eta$, iterations, starting coordinates), landscape selection, live coordinate readouts, and real-time animation.

---
readouts, and real-time animation.

---

## Sprint 2: Optimizers, Dual Views & Educational Polish

### Week 4: Advanced Optimizers (The Intentional Variation)
* **Task:** Implement alternative optimizers (Momentum, Adam) and construct UI controls to overlay their trajectories on the same loss graph across different landscape terrains.
* **Learning:** Mechanics of momentum vectors and adaptive learning rates (Chapter 5) in traversing valleys, saddle points, and local minima.
* **Deliverable:** An updated web interface allowing side-by-side trajectory comparisons between SGD, Momentum, and Adam on varied 3D loss maps.

### Weeks 5 & 6 (combined): Educational UI, Numerical Stability, User Testing & Final Showcase
> **Planning change:** the final presentation takes place on the first day of week 6, so the project has to be finished before that. That is why the work planned for week 5 and week 6 is merged into a single week. To fit everything in, the work focuses on the main goal of the artifact (as suggested in the feedback sessions) and the less important ideas move to the "Future Improvements" log.

* **Task:**
  * Build a dual-view dashboard combining the 3D Weight Trajectory plot with a 2D Loss vs. Epochs curve.
  * Implement exception handling for numerical overflow (`NaN` errors when $\eta$ is set too high) with visual warnings.
  * Apply the feedback from the review sessions (reset/apply button, graph visible when scrolling, step slider shown properly).
  * Integrate beginner-friendly educational elements: an explanations tab with examples and a simple Neural Network diagram mapping weight updates to the loss graph.
  * Conduct a short user testing session with non-experienced users to evaluate intuitive learning.
  * Compile a documented "Future Improvements" log, refactor the code, upload the complete Markdown exploration plan to GitLab and prepare the live presentation deck.
* **Learning:** Interactive UI layout design, multi-graph state synchronization, educational visualization techniques, mathematical edge-case handling, user testing methodologies, technical documentation writing, and demonstrating live technical artifacts.
* **Deliverable:** A stable, fully documented application featuring synchronous 3D/2D plots, educational explanations, automated warnings for unstable learning rates, user feedback analysis and future scope documentation, ready for the final demonstration on the first day of week 6.

### ADDITIONAL FEATURES (Outside the weekly planning) - Got from feedback
* Try javascript for the frontend and improve the visuals
* Add a reset button to let the user decide when the parameters should be change (improve user's control over the app and avoid the graph from updating unintentionally).
* Make the interactions with the graph more user-friendly.