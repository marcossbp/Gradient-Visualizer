# Gradient Visualizer — Feedback Session

Nov 2, 2026 · Marcos

## Context

Today we had a big feedback session where students from other courses came to look at the work of our learning group. I showed them the Gradient Descent Visualizer: I explained the idea of the two-weight network, let them choose a landscape and an example, and then they could play with the sliders and the optimizers by themselves.

The people who tried it also study AI, so they already knew the theory behind gradient descent. This made the session a good test to see if the artifact correctly follows the topic that I was trying to explain.

## What went well

### 1. The topic is advanced for a second year project

Several people said that the theory behind the build and the way it was visualized felt like a really advanced topic to build in the second year. For me this is a good sign that the artifact goes deeper than just using a library, because I had to understand the maths to make the visualization work and understandable.

### 2. The visualization makes the topic easy to understand

Even though the topic is advanced, people said that seeing the path move over the 3D surface and the 2D graph (Loss vs Epochs) made it easy to follow. Moreove comparing SGD, Momentum and Adam on the same surface made the difference between them much clearer than just reading the update rules.

### 3. The "How it works" tab is really helpful

People liked that the app doesn't only show the graph, but also explains what happens behind it. The "How it works" tab helped them connect what they saw on the screen with the steps the program does, like calculating the gradient and updating the weights.

### 4. The pre-loaded examples and their explanations

People really liked the six examples you can choose, like "Saddle point trap", "Learning rate too high" or "The Rosenbrock race". With one click the app sets up a situation that shows a typical problem of training, so they didn't have to look for the right settings themselves. The short explanation under each example was also helpful, because it says what to look at in the graphs and why it happens, for example why the optimizers stop at a saddle point or why the weights explode when the learning rate is too high.

## Suggestions for improvement

- **Connect it more to a real network.** Someone asked how this two-weight example relates to training a real model with millions of weights. A short section in the "How it works" tab about this would answer that question before people ask it.

- **Show overfitting and underfitting.** Someone suggested adding examples of overfitting or underfitting. Right now this is not possible, because the landscapes in the app are fixed formulas and there is no data behind them, and you need training and validation data to see overfitting. It could be possible by adding a small dataset, splitting it into training and validation data, and drawing both loss curves next to each other. Underfitting would be easy to show with only two weights, but overfitting is harder because a model with two weights can't learn much noise. So this would probably need a bigger model, which can't be drawn as a 3D surface anymore.

## Reflection

This session was useful because the feedback came from people outside my learning group who know the topic but had never seen the app. Until now most of my feedback came from teachers and my own teammates. Hearing that other AI students understood the optimizers better after using it tells me the main goal of the artifact works: make something abstract visible. Most of the suggestions are about showing more of what happens inside each step.
