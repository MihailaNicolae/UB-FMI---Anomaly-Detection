import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

case = 1 # Alegem daca lucram in 1D sau 2D

# Parametrii modelului liniar
a, b, c = 2.0, 1.5, 0.5
n_points = 100
mu_values = [0, 0, 0, 0]
sigma_values = [0.1, 1, 2, 5]

def generate_data(a, b, mu, sigma, n_points, high_var_x=False, high_var_y=False, both_high_var=False):
    x = np.random.rand(n_points) * 10

    if high_var_x:
        x *= 5

    noise = np.random.normal(mu, sigma, n_points)
    y = a * x + b + noise

    if high_var_y:
        y += np.random.normal(0, sigma * 5, n_points)

    if both_high_var:
        x += np.random.normal(0, sigma * 5, n_points)
        y += np.random.normal(0, sigma * 5, n_points)

    return x, y

def generate_2d_data(a, b, c, mu, sigma, n_points, high_var_x1=False, high_var_x2=False, both_high_var=False):
    x1 = np.random.rand(n_points) * 10
    x2 = np.random.rand(n_points) * 10

    if high_var_x1:
        x1 *= 5

    if high_var_x2:
        x2 *= 5

    noise = np.random.normal(mu, sigma, n_points)
    y = a * x1 + b * x2 + c + noise

    if both_high_var:
        x1 += np.random.normal(0, sigma * 5, n_points)
        x2 += np.random.normal(0, sigma * 5, n_points)
        y += np.random.normal(0, sigma * 5, n_points)

    return x1, x2, y

def leverage_scores(X):
    H = X @ inv(X.T @ X) @ X.T
    return np.diag(H)

if case == 1:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    for i, sigma in enumerate(sigma_values):
        ax = axes[i]

        x_regular, y_regular = generate_data(a, b, mu_values[i], sigma, n_points)
        x_high_var_x, y_high_var_x = generate_data(a, b, mu_values[i], sigma, n_points, high_var_x=True)
        x_high_var_y, y_high_var_y = generate_data(a, b, mu_values[i], sigma, n_points, high_var_y=True)
        x_high_var_both, y_high_var_both = generate_data(a, b, mu_values[i], sigma, n_points, both_high_var=True)

        x_all = np.concatenate([x_regular, x_high_var_x, x_high_var_y, x_high_var_both])
        y_all = np.concatenate([y_regular, y_high_var_x, y_high_var_y, y_high_var_both])

        X = np.vstack([np.ones_like(x_all), x_all]).T

        scores = leverage_scores(X)

        ax.scatter(x_regular, y_regular, color='blue', label='Normal Data')
        ax.scatter(x_high_var_x, y_high_var_x, color='green', label='High Variance in x')
        ax.scatter(x_high_var_y, y_high_var_y, color='orange', label='High Variance in y')
        ax.scatter(x_high_var_both, y_high_var_both, color='red', label='High Variance in x and y')

        high_leverage_indices = np.argsort(scores)[-5:]  # Top 5 leverage points
        ax.scatter(x_all[high_leverage_indices], y_all[high_leverage_indices], color='black', edgecolor='yellow', s=100, label='High Leverage')

        ax.set_title(f'Noise Variance σ² = {sigma ** 2}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()

    plt.tight_layout()
    plt.show()
else:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    for i, sigma in enumerate(sigma_values):
        ax = axes[i]

        x1_reg, x2_reg, y_reg = generate_2d_data(a, b, c, mu_values[i], sigma, n_points)
        x1_high_x1, x2_high_x1, y_high_x1 = generate_2d_data(a, b, c, mu_values[i], sigma, n_points, high_var_x1=True)
        x1_high_x2, x2_high_x2, y_high_x2 = generate_2d_data(a, b, c, mu_values[i], sigma, n_points, high_var_x2=True)
        x1_both, x2_both, y_both = generate_2d_data(a, b, c, mu_values[i], sigma, n_points, both_high_var=True)

        x1_all = np.concatenate([x1_reg, x1_high_x1, x1_high_x2, x1_both])
        x2_all = np.concatenate([x2_reg, x2_high_x1, x2_high_x2, x2_both])
        y_all = np.concatenate([y_reg, y_high_x1, y_high_x2, y_both])

        X = np.vstack([np.ones_like(x1_all), x1_all, x2_all]).T

        scores = leverage_scores(X)

        ax.scatter(x1_reg, y_reg, color='blue', label='Normal Data')
        ax.scatter(x1_high_x1, y_high_x1, color='green', label='High Variance in x1')
        ax.scatter(x1_high_x2, y_high_x2, color='orange', label='High Variance in x2')
        ax.scatter(x1_both, y_both, color='red', label='High Variance in x1 and x2')

        high_leverage_indices = np.argsort(scores)[-5:]  # Top 5 leverage points
        ax.scatter(x1_all[high_leverage_indices], y_all[high_leverage_indices], color='black', edgecolor='yellow',
                   s=100, label='High Leverage')

        ax.set_title(f'Noise Variance σ² = {sigma ** 2}')
        ax.set_xlabel('x1')
        ax.set_ylabel('y')
        ax.legend()

    plt.tight_layout()
    plt.show()