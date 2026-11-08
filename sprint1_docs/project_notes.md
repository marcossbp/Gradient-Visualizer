# Learning rate and Gradient descent concepts

## 1. Structure of a deep learning model

A deep learning model is made of a neural network. A neural network is basically a combination of neurons that transforms an input into an output. Each neuron makes its own transformation over the data that it receives from the previous neurons and gives it to the next one.
Moreover each neuron has a weight associated to it. The weight is used to parameterize the transformation made in the neuron.
Indeed we can say that the output generated is the result of combining all the weights.
This whole process is made during one cycle of the learning process.

---

## 2. HOW DOES THE SYSTEM LEARNS?

The system is involved in what is called the training loop.
In order to adjust the ouput we need to adjust the weight on each neuron. The optimizer is in charge of carrying out this task.

The objective of the optimizer is to find the combination of weight that produces the less loss.
To find this, we draw a graph where you have an axis that shows the loss and other axises (for each neuron you draw an axis). The coordinates of the point drawn in the graph will tell you the loss produced to that combination of weights.

To know how the weight should be adjusted, the gradient descent and the learning rate are used.
The gradient descent applies partial derivatives to know how each independent coordinate/weight should be updated (the derivative is the tangent to the point on the graph, so depending on its inclination the point/weight on the graph should be moving in one direction or another. When the derivative is 0 you are in a local/global minimum/maximum)
The learning rate is a fixed parameter that will tell how big is the step the point will make in each iteration.

To know the new weights in each iteration we use the following formula:

> `new_weight = old_weight - (learning_rate * derivative)` -> the '-' shows the descending part

This adjustment is made independently on each neuron using backpropagation.

*In this case we will use two weights/neurons to make the visual representation of this process in a 3D graph.*

---

## 3. PROGRAM BUILD (3D graph)

### WEEK 1

The lifespan of a neural network is basically divided in two parts:
1. Training period
2. Prediction period

The training period only consists of adjusting the weight on the neurons using a dataset.
Having the weights adjusted you can use the model to predict other things.

Loss functions:
1. Mean squared error
2. Binary Cross-Entropy
...

The loss function ("mountains" on the graph) is already known before the training starts, so you can assume that you can spot the optimal values for the weight without the training.

When moving on to a more complex neural network (thousands of neurons/dimensions on the graph) it is almost impossible to find these values.

For this simple visualizer we will use a neural network with two weight, easy to understand, plot and visualize.

### WEEK 2

As we are building a lightweight program, we are going to use the numerical finite difference to calculate the partial derivatives. This is a way to approximate partial derivatives with great precision without having to import big libraries.

$$\frac{\partial L}{\partial w_1} \approx \frac{L(w_1 + \epsilon, w_2) - L(w_1 - \epsilon, w_2)}{2\epsilon}$$

The program works by recording every step along the trace and then printing it over the loss function.
To update the position each time we will use the following formula:

$$w_1 \leftarrow w_1 - \alpha \cdot dw_1$$
$$w_2 \leftarrow w_2 - \alpha \cdot dw_2$$

We substract the gradient multipllied by the learning rate (step-size) from the current position. By doing this we ensure that the we are trying to go downhill every time (the gradient vector always points uphill).
The formulas will be embedded in a loop where we will record every step.

Using this implementation the program will always find the local minima (when the gradient is 0 and the formula will always return the current location). But it heavily depends on the hyperparameters' values.

### WEEK 3

This week we will focus more on the visual aspect of the application. The mathematical basis had been set up.

To get better visuals in the web app, we will be using "streamlit".
Streamlit is an open-source python framework that allows you to create interactive web apps using only python.
Using this framework we can build the app by only using simple functions what makes it easier.

I have also divided the "frontend" logic and the maths "backend" into 2 different files to have a better organization of the code.

Regarding the mathematical backend, I have added two more loss functions to let the user try different situations.

The last thing needed for this week is to add a live visualizer to see how each step is updated. However I have some problems regarding the implementation and the compatibility between the different libraries so for now I will only add a slidebar to allow the user see how each step is updated. To do this, I added javascript code embedded in python to be able to updated the path without redrawing the graph every time.

### WEEK 4
During this week we are going to add different optimizers to the program.
The optimizer is the part of the program that is in charge of finding the new weights on each step. It is possible to look for the new weights on different ways.

Till now:
We were using the Vanilla SGD optimizer, the most simple one. It works as follow:
$w \leftarrow w - \eta \nabla L$, only looking at the current gradient.

Problems:

Always uses the maximum number of epochs available but it never actually reaches the lowest point.

No memory: it zig-zags in narrow valleys.

Slow in flat areas: a small gradient means tiny steps.

One $\eta$ (learning rate) for every weight: too big for some weights, too small for others.

We have added 2 new optimizers:

#### MOMENTUM OPTIMIZATION LOOP
It acts as a ball rolling downhill. Adds a component of velocity:

$$v \leftarrow \beta v - \eta \nabla L \qquad w \leftarrow w + v$$

The velocity $v$ builds up over the steps. $\beta = 0.9$ keeps 90% of the previous speed (fixed amount).

It speeds up in consistent directions and damps zig-zags. It can also roll over small bumps.

Downside: it can overshoot the minimum and oscillate around it.

#### ADAM OPTIMIZATION LOOP (ADAPTIVE MOMENT ESTIMATION)

If we want to obtain the new weights using this optimizer we need to compute two different parts:

$$m \leftarrow \beta_1 m + (1-\beta_1)\nabla L \quad \text{(average gradient: direction)}$$
$$v \leftarrow \beta_2 v + (1-\beta_2)(\nabla L)^2 \quad \text{(average squared gradient: steepness)}$$
$$\hat m = \tfrac{m}{1-\beta_1^t},\ \hat v = \tfrac{v}{1-\beta_2^t} \quad \text{(bias correction, because } m, v \text{ start at 0)}$$

$\beta_1$ sets how much momentum Adam has (high = more inertia and overshooting, 0 = no momentum). $\beta_2$ sets how quickly the step size adapts (low = always full-size steps, so it bounces around the minimum; high = steps shrink as the terrain flattens). $\epsilon$ only prevents division by zero; if it is too large compared to the gradients, Adam stops being adaptive and behaves like Momentum/SGD.

Relation between $\beta_1$ and $\beta_2$: each $\beta$ gives a memory of about $\frac{1}{1-\beta}$ steps, so with the default values $m$ remembers ~10 steps ($\beta_1 = 0.9$) and $v$ ~1000 steps ($\beta_2 = 0.999$). $\beta_2$ should be larger than $\beta_1$: the direction ($m$) reacts quickly while the step-size scaling ($v$) stays stable. If $\beta_1$ is much larger than $\beta_2$ (e.g. $\beta_1 = 0.99$, $\beta_2 = 0.9$), $v$ forgets old gradients faster than $m$, so $\frac{\hat m}{\sqrt{\hat v}}$ becomes much bigger than 1 and Adam takes huge steps exactly when it should slow down (near the minimum), which can make it unstable or diverge. If both have the same value, the steps are always about $\eta$ and Adam bounces around the minimum instead of settling.

To obtain the new weights we use:
$$w \leftarrow w - \eta \frac{\hat m}{\sqrt{\hat v}+\epsilon}$$

$m \rightarrow$ only measures the direction of the gradient and makes the average between the last ones.

$v \rightarrow$ It does the same as $m$ but focused on the step size.

How it works:

$$\frac{\hat m}{\sqrt{\hat v}} \approx \frac{\text{consistent direction}}{\text{typical size}}$$
Consistent gradients give a value near ±1, so the step is a full $\eta$.

Inconsistent or noisy gradients give a value near 0, so the step is small.

The step never gets much bigger than $\eta$. This is why Adam stayed controlled on the Saddle Point while SGD and Momentum ran off the map.

SUMMARY: the ratio compares the average gradient with its typical size. It’s ±1 when the gradient is consistent (low variance) and near 0 when the gradient is noisy or changes sign.

In general terms, Adam is the best optimzer within these ones when using an appropiate number of epochs. Almost always would get a better final loss than the Momentum optimizer and in less steps than the Vanilla SGD.

#### Comparison between Vanilla SGD and Adam
With a big number of epochs, the Vanilla SGD optimizer would always get to a lower point than the Adam optimizer. However, the difference between their final result could be 0.0001 and the Vanilla SGD optimizer could double the number of epochs used. So, in general terms, the Adam optimizer is more optimal.
### WEEK 5 & 6
The final presentation is on the first day of week 6, so I had to merge the work of week 5 and week 6 into one week and finish the project before that. Following the feedback from the review sessions, I focused on the main goal of the artifact (understanding how optimizers move the weights) and moved the less important ideas (for example rewriting the frontend in JavaScript) to the future improvements.

#### NUMERICAL STABILITY (LEARNING RATE TOO HIGH)
If the learning rate is too high, each step jumps over the minimum and lands higher on the other side, so the next gradient is bigger and the next jump is even bigger. The weights grow until they become `inf` or `NaN`.

Example with the Convex Bowl $L = 0.15(w_1^2 + w_2^2)$, where the gradient is $0.3w$:

$$w_{\text{new}} = w - \eta \cdot 0.3w = (1 - 0.3\eta)\,w$$

- If $0 < \eta < 3.33$ the factor is between 0 and 1: the weights shrink every step and the point goes straight to the minimum.
- If $3.33 < \eta < 6.67$ the factor is negative but smaller than 1 in size: the point jumps from one side to the other but still gets closer.
- If $\eta > 6.67$ the factor is bigger than 1 in size: every jump is bigger than the last one and the optimizer **diverges**.

In the program, every optimizer now stops when the weights are not real numbers anymore or when they are far away from the landscape (more than 10). The app shows a red warning explaining that the learning rate is too high. I also allowed a much higher learning rate for SGD and Momentum (up to 10) so the user can actually see this happen.

#### THE LOSS CAN'T BE NEGATIVE
In a real neural network the loss functions are never negative:
- Mean squared error is a sum of squares, so it is $\geq 0$.
- Binary cross-entropy is $-\log(p)$ with $0 < p \leq 1$, so it is $\geq 0$.

The best possible loss is 0 (perfect predictions). In my app two landscapes had negative values:
- The non-convex surface had valleys at -0.2. I shifted it up by 0.2. Adding a constant does not change the gradient, so the optimizers follow exactly the same paths, only the numbers on the loss axis change.
- The saddle point $0.1(w_1^2 - w_2^2)$ had no lower limit: going along $w_2$ the loss goes down forever. This cannot happen in real training, so I removed it.

#### ROSENBROCK LANDSCAPE
To replace the saddle point I added the Rosenbrock function, the classic test for optimizers:

$$L(w_1, w_2) = 0.02\left[(1 - w_1)^2 + 5\,(w_2 - w_1^2)^2\right]$$

- $(1 - w_1)^2$ gives a gentle slope that pulls $w_1$ towards 1.
- $5(w_2 - w_1^2)^2$ makes very steep walls around the curved line $w_2 = w_1^2$.
- The minimum is at $(1, 1)$, at the end of a long, narrow and almost flat curved valley ("banana" shape).
- I scaled it by 0.02 so it works with the same learning rates as the other landscapes.

Finding the valley is easy, but moving along its flat floor is hard. Starting at $(-1, 1)$ with 100 epochs:

| Optimizer | Final position | What happens |
|---|---|---|
| SGD | (0.43, 0.13) | small gradient on the floor, so tiny steps |
| Momentum | (0.73, 0.52) | builds up speed along the floor |
| Adam | (0.95, 0.90) | every step is about $\eta$, whatever the slope, so it goes furthest |

#### DUAL VIEW: LOSS VS EPOCHS
Next to the 3D graph there is now a 2D graph with the loss of each optimizer at each epoch. Both graphs move together with the same step slider. The 3D graph shows *where* the weights go and the 2D graph shows *how fast* the loss goes down. For example, on the convex bowl SGD's loss always goes down (it never overshoots with a normal $\eta$), while Momentum's loss goes up and down when it overshoots the minimum.

#### EXAMPLES AND EXPLANATIONS
To make the program easier to understand for people without experience (feedback from the review session), I added:
- **Examples**: one click loads ready-made settings and an explanation of what to look at:
  - *Saddle point trap*: starting on top of a peak, all optimizers stop at $(0, 0)$ with loss 0.2. The gradient is 0 there, so the update formula stops moving the weights, but it is not a minimum: it is a saddle point between two valleys. Gradient descent only knows the slope where it is standing.
  - *The starting point decides the valley*: starting at $(-0.8, 1.2)$ the optimizers end in the valley at $(1, 1)$, starting at $(-1.2, 0.8)$ they end at $(-1, -1)$. In a real network this is why the initial weights matter.
  - *Learning rate too high / too small*, *Momentum overshoots* and *The Rosenbrock race*.
- **"How it works" tab**: explains that the two axes of the graph are the two weights of a very small neural network (with a diagram), how gradient descent works, the three optimizers and the landscapes.

#### OTHER CHANGES (FROM THE FEEDBACK)
- **Apply / Reset buttons**: the graph only updates when the user presses Apply, so it does not change by accident while moving the sliders. Reset puts every setting back to its default value.
- **Scrolling**: scrolling over the 3D graph zoomed it out until it disappeared. Now scrolling moves the page.
- **Graph disappearing in Chrome**: in Chrome, the 3D graph stopped being drawn when it went under Streamlit's top bar. I found it by testing piece by piece: the solid top bar made Chrome think the graph was hidden. Making the top bar transparent solved it.
- **Light mode**: the graph now changes its colors when the user switches between light and dark mode.
- The comparison table is now above the graphs, and the numbers of the step slider are no longer cut off.
