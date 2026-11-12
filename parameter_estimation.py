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
from plot_results import plot_estimation_results


arx_estimator = RecursiveEstimator(model='ARX', dim=2)
armax_estimator = RecursiveEstimator(model='ARMAX', dim=2)
oe_estimator = RecursiveEstimator(model='OE', dim=2)
for estimator in [arx_estimator, armax_estimator, oe_estimator]:
    plot_estimation_results(estimator, u, y_arx, iterations=3, use_posterior=True, a_true=a, b_true=b)

arx_estimator = RecursiveEstimator(model='ARX', dim=3)
armax_estimator = RecursiveEstimator(model='ARMAX', dim=3)
oe_estimator = RecursiveEstimator(model='OE', dim=3)
for estimator in [arx_estimator, armax_estimator, oe_estimator]:
    plot_estimation_results(estimator, u, y_arx, iterations=3, use_posterior=True, a_true=a, b_true=b)

# %%

arx_estimator = RecursiveEstimator(model='ARX', dim=3)
armax_estimator = RecursiveEstimator(model='ARMAX', dim=3)
oe_estimator = RecursiveEstimator(model='OE', dim=3)
for estimator in [arx_estimator, armax_estimator, oe_estimator]:
    plot_estimation_results(estimator, u, y_oe, iterations=3, use_posterior=True, a_true=a, b_true=b)


# %%
