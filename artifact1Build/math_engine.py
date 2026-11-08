import numpy as np

# LOSS FUNCTIONS

# We create a figure using Gaussian Bells curves [e^(-(x^2 + y^2))]
def loss_non_convex(w1, w2):
    # (w1 +- 1) -> adjust each valley/peak in its corresponding quadrant
    peak1 = np.exp(-((w1 + 1)**2 + (w2 - 1)**2))
    peak2 = np.exp(-((w1 - 1)**2 + (w2 + 1)**2))
    valley1 = np.exp(-((w1 - 1)**2 + (w2 - 1)**2))
    valley2 = np.exp(-((w1 + 1)**2 + (w2 + 1)**2))

    # Formula of the figure scaled to [-0.2, 0.2] and shifted up by 0.2 so the loss is never
    # negative (like a real loss). Adding a constant does not change the gradients, so the paths
    # of the optimizers are exactly the same.
    return 0.2 * (peak1 + peak2 - valley1 - valley2) + 0.2

# Rosenbrock landscape ("banana" valley)
def loss_rosenbrock(w1, w2):
    # (1 - w1)^2       -> pulls w1 towards 1 (gentle slope along the valley)
    # 5 (w2 - w1^2)^2  -> steep walls around the curved valley floor w2 = w1^2
    # The minimum is at (1,1), at the end of a long, narrow and almost flat curved valley.
    # Classic optimizer benchmark: finding the valley is easy, moving along it is slow.
    # Scaled by 0.02 so it works with the same learning rates as the other landscapes.
    return 0.02 * ((1 - w1)**2 + 5 * (w2 - w1**2)**2)

# Convex Bowl landscape
def loss_convex_bowl(w1, w2):
    # Simple parabolic bowl with one global minimum at (0,0)
    return 0.15 * (w1**2 + w2**2)

# Dictionary mapping names to loss functions
LANDSCAPES = {
    "Non-Convex Surface (Multiple Valleys)": loss_non_convex,
    "Rosenbrock (Narrow Curved Valley)": loss_rosenbrock,
    "Convex Bowl (Single Minimum)": loss_convex_bowl
}


# Numerical Gradient Function (Central Finite Difference)
def compute_gradients(loss_fn, w1, w2, epsilon=1e-5):
    # Partial derivative for w1
    loss_w1_north = loss_fn(w1 + epsilon, w2)
    loss_w1_south = loss_fn(w1 - epsilon, w2)
    dw1 = (loss_w1_north - loss_w1_south) / (2 * epsilon)

    # Partial derivative for w2
    loss_w2_east = loss_fn(w1, w2 + epsilon)
    loss_w2_west = loss_fn(w1, w2 - epsilon)
    dw2 = (loss_w2_east - loss_w2_west) / (2 * epsilon)

    return dw1, dw2


# NUMERICAL STABILITY
# If the learning rate is too high (or the landscape has no bottom in that direction)
# the weights can grow without limit until they become inf/NaN. We stop the run as soon as
# the point is clearly out of control, so the path is shorter than the number of iterations.
DIVERGENCE_LIMIT = 10.0

def has_diverged(w1, w2, loss):
    # Not a real number anymore (overflow -> inf, inf - inf -> NaN)
    if not np.all(np.isfinite([w1, w2, loss])):
        return True
    # Far away from the landscape we are looking at ([-2, 2] on both axes)
    return abs(w1) > DIVERGENCE_LIMIT or abs(w2) > DIVERGENCE_LIMIT


# OPTIMIZERS
# Every optimizer returns the same thing: the recorded path (w1, w2, loss) of each step,
# so the frontend can draw all of them in the same way.
# A path shorter than iterations + 1 means the optimizer diverged and was stopped early.

# Gradient Descent Optimization Loop (Vanilla SGD)
def run_gradient_descent(loss_fn, w1_start, w2_start, learning_rate, iterations):
    path_w1 = [w1_start]
    path_w2 = [w2_start]
    path_loss = [loss_fn(w1_start, w2_start)]

    curr_w1, curr_w2 = w1_start, w2_start

    for _ in range(iterations):
        # Calculate gradients at the current location
        dw1, dw2 = compute_gradients(loss_fn, curr_w1, curr_w2)

        # Update position (step downhill)
        curr_w1 -= learning_rate * dw1
        curr_w2 -= learning_rate * dw2

        # Record the new location and loss height
        path_w1.append(curr_w1)
        path_w2.append(curr_w2)
        path_loss.append(loss_fn(curr_w1, curr_w2))

        # Stop if the weights are out of control
        if has_diverged(curr_w1, curr_w2, path_loss[-1]):
            break

    return path_w1, path_w2, path_loss


# Momentum Optimization Loop
# The point behaves like a ball rolling downhill: it accumulates speed (velocity)
# from previous steps, so it can roll through flat areas and small bumps.
#   velocity = momentum * velocity - learning_rate * gradient
#   w_new    = w_old + velocity
def run_momentum(loss_fn, w1_start, w2_start, learning_rate, iterations, momentum=0.9):
    path_w1 = [w1_start]
    path_w2 = [w2_start]
    path_loss = [loss_fn(w1_start, w2_start)]

    curr_w1, curr_w2 = w1_start, w2_start
    # The ball starts without speed
    velocity_w1, velocity_w2 = 0.0, 0.0

    for _ in range(iterations):
        dw1, dw2 = compute_gradients(loss_fn, curr_w1, curr_w2)

        # Keep a fraction of the previous velocity and add the new downhill push
        velocity_w1 = momentum * velocity_w1 - learning_rate * dw1
        velocity_w2 = momentum * velocity_w2 - learning_rate * dw2

        # Move using the velocity instead of the raw gradient
        curr_w1 += velocity_w1
        curr_w2 += velocity_w2

        path_w1.append(curr_w1)
        path_w2.append(curr_w2)
        path_loss.append(loss_fn(curr_w1, curr_w2))

        # Stop if the weights are out of control
        if has_diverged(curr_w1, curr_w2, path_loss[-1]):
            break

    return path_w1, path_w2, path_loss


# Adam Optimization Loop (Adaptive Moment Estimation)
# Combines momentum (1st moment: average of the gradients) with an adaptive learning
# rate per weight (2nd moment: average of the squared gradients). Weights with big
# gradients get smaller steps and weights with small gradients get bigger steps.
#   m     = beta1 * m + (1 - beta1) * gradient
#   v     = beta2 * v + (1 - beta2) * gradient^2
#   m_hat = m / (1 - beta1^t),  v_hat = v / (1 - beta2^t)   -> bias correction
#   w_new = w_old - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
def run_adam(loss_fn, w1_start, w2_start, learning_rate, iterations,
             beta1=0.9, beta2=0.999, epsilon=1e-8):
    path_w1 = [w1_start]
    path_w2 = [w2_start]
    path_loss = [loss_fn(w1_start, w2_start)]

    curr_w1, curr_w2 = w1_start, w2_start
    # Moment estimates start at 0 for both weights
    m_w1, m_w2 = 0.0, 0.0
    v_w1, v_w2 = 0.0, 0.0

    for t in range(1, iterations + 1):
        dw1, dw2 = compute_gradients(loss_fn, curr_w1, curr_w2)

        # 1st moment: running average of the gradient (direction + momentum)
        m_w1 = beta1 * m_w1 + (1 - beta1) * dw1
        m_w2 = beta1 * m_w2 + (1 - beta1) * dw2

        # 2nd moment: running average of the squared gradient (how steep it is)
        v_w1 = beta2 * v_w1 + (1 - beta2) * dw1**2
        v_w2 = beta2 * v_w2 + (1 - beta2) * dw2**2

        # Bias correction: m and v start at 0, so the first steps are too small without it
        m_hat_w1 = m_w1 / (1 - beta1**t)
        m_hat_w2 = m_w2 / (1 - beta1**t)
        v_hat_w1 = v_w1 / (1 - beta2**t)
        v_hat_w2 = v_w2 / (1 - beta2**t)

        # Each weight gets its own effective step size
        curr_w1 -= learning_rate * m_hat_w1 / (np.sqrt(v_hat_w1) + epsilon)
        curr_w2 -= learning_rate * m_hat_w2 / (np.sqrt(v_hat_w2) + epsilon)

        path_w1.append(curr_w1)
        path_w2.append(curr_w2)
        path_loss.append(loss_fn(curr_w1, curr_w2))

        # Stop if the weights are out of control
        if has_diverged(curr_w1, curr_w2, path_loss[-1]):
            break

    return path_w1, path_w2, path_loss


# Dictionary mapping names to optimizer functions
OPTIMIZERS = {
    "SGD": run_gradient_descent,
    "Momentum": run_momentum,
    "Adam": run_adam
}