# %%
import numpy as np
from scipy.signal import lfilter
import control
import matplotlib.pyplot as plt

%load_ext autoreload
%autoreload 2

# %%
a = np.array([1, -1.5, 0.6])
b = np.array([0.025, 0.05, 0.025])
# %%

sigma_u = 1.0
noise_scale = 0.5
N = 100000

u = np.random.normal(0, sigma_u, N)

lhs = np.array([[1,    a[1],   a[2]],
                [a[1], 1+a[2], 0],
                [a[2], a[1],   1]])
h0 = b[0]
h1 = b[1] - a[1]*h0
h2 = b[2] - a[1]*h1 - a[2]*h0

rhs = sigma_u**2 * np.array([
    b[0]*h0 + b[1]*h1 + b[2]*h2,
    b[1]*h0 + b[2]*h1,
    b[2]*h0
])

Ry = np.linalg.solve(lhs, rhs)

Rv0 = (noise_scale**2) * Ry[0]

rhs_null = -lhs[1:, 0] * Rv0
Rv = np.array([Rv0, *np.linalg.solve(lhs[1:, 1:], rhs_null)])
sigma_e = (lhs @ Rv)[0]
e = np.random.normal(0, np.sqrt(sigma_e), N)
y_arx_deterministic = lfilter(b, a, u)
y_arx_noise = lfilter([1.0], a, e)
y_arx = y_arx_deterministic + y_arx_noise


e = np.random.normal(0, noise_scale*np.sqrt(Ry[0]), N)
y_oe_deterministic = lfilter(b, a, u)
y_oe = y_oe_deterministic + e

fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(y_arx, label='ARX Output', color='blue')
ax[0].plot(y_arx_deterministic, label='ARX Deterministic Output', color='green')
ax[0].legend()
ax[0].grid()
ax[0].set_title('ARX Model Output')
ax[0].set_xlabel('Sample')
ax[0].set_ylabel('Output')
ax[1].plot(y_oe, label='OE Output', color='orange')
ax[1].plot(y_oe_deterministic, label='OE Deterministic Output', color='red', linestyle='--')
ax[1].set_title('OE Model Output')
ax[1].set_xlabel('Sample')
ax[1].set_ylabel('Output')
ax[1].legend()
ax[1].grid()
plt.tight_layout()
plt.show()

# %%
from recursive_estim import RecursiveEstimator

arx_estimator = RecursiveEstimator(model='ARX', dim=2)
arx_est_res = arx_estimator.run_batch(u, y_arx)

fig, ax = plt.subplots(3, 1, figsize=(10, 8))
ax[0].plot(y_arx, label='ARX Output', color='blue')
ax[0].plot(arx_est_res['y_pred'], label='ARX Predicted Output', color='green', linestyle='--')
ax[0].legend()
ax[0].grid()
ax[0].set_title('ARX Model Output vs Predicted')
ax[0].set_xlabel('Sample')
ax[0].set_ylabel('Output')
for i in range(arx_estimator.dim):
    ax[1].plot((1 + i) * np.ones_like(arx_est_res['theta'][:, i]), arx_est_res['theta'][:, i], ".", markersize=1)
    if i < len(a)-1:
        ax[1].plot([1 + i], a[i+1], "o", label=f'True a[{i+1}]', markersize=5)

ax[1].set_title('ARX Model Parameter a Estimation')
ax[1].set_xlabel('a Parameter Index')
ax[1].set_ylabel('Estimated Value')
ax[1].legend()
ax[1].grid()
for i in range(arx_estimator.dim + 1):
    ax[2].plot((1 + i) * np.ones_like(arx_est_res['theta'][:, arx_estimator.dim + i]), arx_est_res['theta'][:, arx_estimator.dim + i], ".", markersize=1)
    if i < len(b):
        ax[2].plot([1 + i], b[i], "o", label=f'True b[{i}]', markersize=5)
ax[2].set_title('ARX Model Parameter b Estimation')
ax[2].set_xlabel('b Parameter Index')
ax[2].set_ylabel('Estimated Value')
ax[2].legend()
ax[2].grid()
plt.tight_layout()
plt.show()


# %%
