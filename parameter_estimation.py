# %%
import numpy as np
from scipy.signal import lfilter
import control
import matplotlib.pyplot as plt
from scipy.signal import welch

%load_ext autoreload
%autoreload 2

# %%
a = np.array([1, -1.5, 0.6])
b = np.array([0.025, 0.05, 0.025])

#%%
def calculate_and_plot_psd(signal, fs, nperseg=1024, show_plot=True, title="Power Spectral Density"):
    """
    Calculates and optionally plots the Power Spectral Density (PSD) of a 1D signal using Welch's method.
    
    Parameters:
    -----------
    signal : array_like
        Time series data of the signal.
    fs : float
        Sampling frequency of the signal in Hz.
    nperseg : int, optional
        Length of each segment for Welch's method. Default is 1024.
    show_plot : bool, optional
        If True, generates and displays a matplotlib plot. Default is True.
    title : str, optional
        Title for the PSD plot.
        
    Returns:
    --------
    frequencies : ndarray
        Array of sample frequencies.
    psd : ndarray
        Power spectral density or power spectrum of x.
    """
    
    # Calculate the PSD using Welch's method
    frequencies, psd = welch(signal, fs, nperseg=nperseg)
    print(np.sqrt(np.cumsum(np.diff(frequencies) * psd[1:])[-1]))
    
    # Generate the plot if requested
    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.semilogy(frequencies, psd, color='darkorange')
        plt.title(title)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V²/Hz]')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        

#%%

def gen_prbs(p_switch, T):
    rand_vals = np.random.rand(T)
    flips = np.where(rand_vals < p_switch, -1, 1)
    flips[0] = 1 
    UU = np.cumprod(flips)
    return UU

#%%
def gen_model_out(a, b, u, model="ARX", noise_scale=0.5):
    N = len(u)
    raw_out = lfilter(b, a, u)
    y_var = np.var(raw_out)
    if model=="ARX":
        lhs = np.array([[1,    a[1],   a[2]],
                    [a[1], 1+a[2], 0],
                    [a[2], a[1],   1]])
        Rv0 = (noise_scale**2) * y_var

        rhs_null = -lhs[1:, 0] * Rv0
        Rv = np.array([Rv0, *np.linalg.solve(lhs[1:, 1:], rhs_null)])
        sigma_e = (lhs @ Rv)[0]
        e = np.random.normal(0, np.sqrt(sigma_e), N)
        out_noise = lfilter([1.0], a, e)
    else:
        sigma_e = (noise_scale**2) * y_var
        out_noise = np.random.normal(0, np.sqrt(sigma_e), N)

    print(f"y_sigma {np.sqrt(y_var)} noise sigma {np.sqrt(np.var(out_noise))}")
    return raw_out + out_noise


# %%
sigma_u = 1
noise_scale = 0.5
N = 10000

u = gen_prbs(0.5, N)

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

y_arx_deterministic = lfilter(b, a, u)
Ry = np.linalg.solve(lhs, rhs)

Rv0 = (noise_scale**2) * np.var(y_arx_deterministic)

rhs_null = -lhs[1:, 0] * Rv0
Rv = np.array([Rv0, *np.linalg.solve(lhs[1:, 1:], rhs_null)])
sigma_e = (lhs @ Rv)[0]
e = np.random.normal(0, np.sqrt(sigma_e), N)
y_arx_noise = lfilter([1.0], a, e)
y_arx = y_arx_deterministic + y_arx_noise
print("var", np.var(y_arx_deterministic), Ry, np.var(y_arx_noise))
print("std", np.std(y_arx_deterministic), np.sqrt(Ry), np.std(y_arx_noise))
print("amplitude", np.mean(np.abs(y_arx_deterministic)), np.mean(np.abs(y_arx_noise)))


e = np.random.normal(0, np.sqrt(Rv0), N)
y_oe_deterministic = lfilter(b, a, u)

print("var", np.var(y_oe_deterministic), np.var(e))
print("amplitude", np.mean(np.abs(y_oe_deterministic)), np.mean(np.abs(e)))

y_oe = y_oe_deterministic + e

fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(y_arx_deterministic, label='ARX Output', color='blue')
ax[0].plot(y_arx_noise, label='ARX Propagated Noise', color='red')
ax[0].legend()
ax[0].grid()
ax[0].set_title('ARX Model Output')
ax[0].set_xlabel('Sample')
ax[0].set_ylabel('Output')
ax[1].plot(y_oe_deterministic, label='OE Output', color='blue')
ax[1].plot(e, label='OE Noise', color='red')
ax[1].set_title('OE Model Output')
ax[1].set_xlabel('Sample')
ax[1].set_ylabel('Output')
ax[1].legend()
ax[1].grid()
plt.tight_layout()
#plt.savefig("noise_to_signal_low_freq_prbs.pdf")
plt.show()

#%%
N = 10000
u_white_noise = gen_prbs(0.5, N)
u_low_freq = gen_prbs(0.2, N)
y_arx_white = gen_model_out(a, b, u_white_noise, "ARX")
y_arx_low = gen_model_out(a, b, u_low_freq, "ARX")
y_oe_white = gen_model_out(a, b, u_white_noise, "OE")
y_oe_low = gen_model_out(a, b, u_low_freq, "OE")

# %%
from recursive_estim import RecursiveEstimator
from plot_results import plot_estimation_results, plot_multiple_model_results
arx_estimator = RecursiveEstimator(model='ARX', dim=2)
oe_estimator = RecursiveEstimator(model='OE', dim=2)
arxmax_estimator = RecursiveEstimator(model='ARXMAX', dim=2)
arxmax_dim_3_estimator = RecursiveEstimator(model='ARXMAX', dim=3)

#%%
arx_est_arx_white_noise_results = arx_estimator.run_batch(u_white_noise, y_arx_white)
arx_est_arx_low_freq_results = arx_estimator.run_batch(u_low_freq, y_arx_low)
arx_est_oe_white_noise_results = arx_estimator.run_batch(u_white_noise, y_oe_white)
arx_est_oe_low_freq_results = arx_estimator.run_batch(u_low_freq, y_oe_low)

plot_multiple_model_results({"Low-Frequency":arx_est_arx_low_freq_results,
                             "White Noise": arx_est_arx_white_noise_results}, model_name="ARX", a_true=a, b_true=b, c_true=None, filename="arx_arx_convergence.pdf")
plot_multiple_model_results({"Low-Frequency":arx_est_oe_low_freq_results,
                             "White Noise": arx_est_oe_white_noise_results}, model_name="ARX", a_true=a, b_true=b, c_true=None, filename="arx_oe_convergence.pdf")
#%%
oe_est_arx_white_noise_results = oe_estimator.run_batch(u_white_noise, y_arx_white)
oe_est_arx_low_freq_results = oe_estimator.run_batch(u_low_freq, y_arx_low)
oe_est_oe_white_noise_results = oe_estimator.run_batch(u_white_noise, y_oe_white)
oe_est_oe_low_freq_results = oe_estimator.run_batch(u_low_freq, y_oe_low)

plot_multiple_model_results({"Low-Frequency":oe_est_arx_low_freq_results,
                             "White Noise": oe_est_arx_white_noise_results}, model_name="OE", a_true=a, b_true=b, c_true=None, filename="oe_arx_convergence.pdf")
plot_multiple_model_results({"Low-Frequency":oe_est_oe_low_freq_results,
                             "White Noise": oe_est_oe_white_noise_results}, model_name="OE", a_true=a, b_true=b, c_true=None, filename="oe_oe_convergence.pdf")
#%%
armax_est_arx_white_noise_results = arxmax_estimator.run_batch(u_white_noise, y_arx_white)
armax_est_arx_low_freq_results = arxmax_estimator.run_batch(u_low_freq, y_arx_low)
armax_est_oe_white_noise_results = arxmax_estimator.run_batch(u_white_noise, y_oe_white)
armax_est_oe_low_freq_results = arxmax_estimator.run_batch(u_low_freq, y_oe_low)

plot_multiple_model_results({"Low-Frequency":armax_est_arx_low_freq_results,
                             "White Noise": armax_est_arx_white_noise_results}, model_name="ARMAX", a_true=a, b_true=b, c_true=np.zeros(3), filename="armax_arx_convergence.pdf")
plot_multiple_model_results({"Low-Frequency":armax_est_oe_low_freq_results,
                             "White Noise": armax_est_oe_white_noise_results}, model_name="ARMAX", a_true=a, b_true=b, c_true=a, filename="armax_oe_convergence.pdf")

#%%
arxmax_dim_3_est_arx_white_noise_results = arxmax_dim_3_estimator.run_batch(u_white_noise, y_arx_white)
arxmax_dim_3_est_arx_low_freq_results = arxmax_dim_3_estimator.run_batch(u_low_freq, y_arx_low)
arxmax_dim_3_est_oe_white_noise_results = arxmax_dim_3_estimator.run_batch(u_white_noise, y_oe_white)
arxmax_dim_3_est_oe_low_freq_results = arxmax_dim_3_estimator.run_batch(u_low_freq, y_oe_low)

plot_multiple_model_results({"Low-Frequency":arxmax_dim_3_est_arx_low_freq_results,
                             "White Noise": arxmax_dim_3_est_arx_white_noise_results}, model_name="ARMAX", a_true=a, b_true=b, c_true=np.zeros(3), filename="arxmax_dim_3_arx_convergence.pdf")
plot_multiple_model_results({"Low-Frequency":arxmax_dim_3_est_oe_low_freq_results,
                             "White Noise": arxmax_dim_3_est_oe_white_noise_results}, model_name="ARMAX", a_true=a, b_true=b, c_true=a, filename="arxmax_dim_3_oe_convergence.pdf")

#%%
oe_post_est_arx_white_noise_results = oe_estimator.run_batch(u_white_noise, y_arx_white, use_posterior=True)
oe_post_est_arx_low_freq_results = oe_estimator.run_batch(u_low_freq, y_arx_low, use_posterior=True)
oe_post_est_oe_white_noise_results = oe_estimator.run_batch(u_white_noise, y_oe_white, use_posterior=True)
oe_post_est_oe_low_freq_results = oe_estimator.run_batch(u_low_freq, y_oe_low, use_posterior=True)

plot_multiple_model_results({"Prior":oe_est_arx_white_noise_results,
                             "Posterior": oe_post_est_arx_white_noise_results}, model_name="OE", a_true=a, b_true=b, c_true=None, filename="oe_post_white_arx_convergence.pdf")
plot_multiple_model_results({"Prior":oe_est_arx_low_freq_results,
                             "Posterior": oe_post_est_arx_low_freq_results}, model_name="OE", a_true=a, b_true=b, c_true=None, filename="oe_post_low_arx_convergence.pdf")
plot_multiple_model_results({"Prior":oe_est_oe_low_freq_results,
                             "Posterior": oe_post_est_oe_low_freq_results}, model_name="OE", a_true=a, b_true=b, c_true=None, filename="oe_post_low_oe_convergence.pdf")
plot_multiple_model_results({"Prior":oe_est_oe_white_noise_results,
                             "Posterior": oe_post_est_oe_white_noise_results}, model_name="OE", a_true=a, b_true=b, c_true=None, filename="oe_post_white_oe_convergence.pdf")
#%%
armax_post_est_arx_white_noise_results = arxmax_estimator.run_batch(u_white_noise, y_arx_white, use_posterior=True)
armax_post_est_arx_low_freq_results = arxmax_estimator.run_batch(u_low_freq, y_arx_low, use_posterior=True)
armax_post_est_oe_white_noise_results = arxmax_estimator.run_batch(u_white_noise, y_oe_white, use_posterior=True)
armax_post_est_oe_low_freq_results = arxmax_estimator.run_batch(u_low_freq, y_oe_low, use_posterior=True)

plot_multiple_model_results({"Prior": armax_est_arx_white_noise_results,
                             "Posterior": armax_post_est_arx_white_noise_results}, model_name="ARMAX", a_true=a, b_true=b, c_true=np.zeros(3), filename="armax_post_white_arx_convergence.pdf")
plot_multiple_model_results({"Prior": armax_est_arx_low_freq_results,
                             "Posterior": armax_post_est_arx_low_freq_results}, model_name="ARMAX", a_true=a, b_true=b, c_true=np.zeros(3), filename="armax_post_low_arx_convergence.pdf")
plot_multiple_model_results({"Prior":armax_est_oe_low_freq_results,
                             "Posterior": armax_post_est_oe_low_freq_results}, model_name="ARMAX", a_true=a, b_true=b, c_true=a, filename="armax_post_low_oe_convergence.pdf")
plot_multiple_model_results({"Prior":armax_est_oe_white_noise_results,
                             "Posterior": armax_post_est_oe_white_noise_results}, model_name="ARMAX", a_true=a, b_true=b, c_true=a, filename="armax_post_white_oe_convergence.pdf")

#%%
oe_iterative_oe_low_results = {}
for i in range(3):
    oe_iterative_oe_low_results[f"Pass {i+1}"] = oe_estimator.run_batch(u_low_freq[:5000], y_oe_low[:5000], reset_params=False)
oe_estimator.reset()

plot_multiple_model_results(oe_iterative_oe_low_results, model_name="OE", a_true=a, b_true=b, c_true=None, filename="oe_iterative_oe_low.pdf")

#%%
armax_iterative_oe_low_results = {}
for i in range(3):
    armax_iterative_oe_low_results[f"Pass {i+1}"] = arxmax_estimator.run_batch(u_low_freq[:5000], y_oe_low[:5000], reset_params=False)
arxmax_estimator.reset()

plot_multiple_model_results(armax_iterative_oe_low_results, model_name="ARMAX", a_true=a, b_true=b, c_true=a, filename="armax_iterative_oe_low.pdf")

#%%
N = 1000
u = gen_prbs(0.1, N)
y_arx = gen_model_out(a, b, u, "ARX")

#%%
print("Estimation without the posterior estimates")
arx_estimator = RecursiveEstimator(model='ARX', dim=2)
armax_estimator = RecursiveEstimator(model='ARMAX', dim=2)
oe_estimator = RecursiveEstimator(model='OE', dim=2)
for estimator in [arx_estimator, armax_estimator, oe_estimator]:
    plot_estimation_results(estimator, u, y_arx, iterations=3, use_posterior=False, a_true=a, b_true=b)

print("Estimation with the posterior estimates")
arx_estimator = RecursiveEstimator(model='ARX', dim=2)
armax_estimator = RecursiveEstimator(model='ARMAX', dim=2)
oe_estimator = RecursiveEstimator(model='OE', dim=2)
for estimator in [arx_estimator, armax_estimator, oe_estimator]:
    plot_estimation_results(estimator, u, y_arx, iterations=3, use_posterior=True, a_true=a, b_true=b)

print("Wrong dimension of the estimator")
arx_estimator = RecursiveEstimator(model='ARX', dim=3)
armax_estimator = RecursiveEstimator(model='ARMAX', dim=3)
oe_estimator = RecursiveEstimator(model='OE', dim=3)
for estimator in [arx_estimator, armax_estimator, oe_estimator]:
    plot_estimation_results(estimator, u, y_arx, iterations=3, use_posterior=True, a_true=a, b_true=b)

#%%
arx_estimator = RecursiveEstimator(model='ARX', dim=3)
armax_estimator = RecursiveEstimator(model='ARMAX', dim=3)
oe_estimator = RecursiveEstimator(model='OE', dim=3)
for estimator in [arx_estimator, armax_estimator, oe_estimator]:
    plot_estimation_results(estimator, u, y_oe, iterations=3, use_posterior=True, a_true=a, b_true=b)


# %%
N = 1000
u = gen_prbs(0.5, N)
y_oe = gen_model_out(a, b, u, "OE")

# %%
from recursive_estim import RecursiveEstimator
from plot_results import plot_estimation_results

print("Estimation without the posterior estimates")
arx_estimator = RecursiveEstimator(model='ARX', dim=2)
armax_estimator = RecursiveEstimator(model='ARMAX', dim=2)
oe_estimator = RecursiveEstimator(model='OE', dim=2)
for estimator in [arx_estimator, armax_estimator, oe_estimator]:
    plot_estimation_results(estimator, u, y_oe, iterations=3, use_posterior=False, a_true=a, b_true=b)

print("Estimation with the posterior estimates")
arx_estimator = RecursiveEstimator(model='ARX', dim=2)
armax_estimator = RecursiveEstimator(model='ARMAX', dim=2)
oe_estimator = RecursiveEstimator(model='OE', dim=2)
for estimator in [arx_estimator, armax_estimator, oe_estimator]:
    plot_estimation_results(estimator, u, y_oe, iterations=3, use_posterior=True, a_true=a, b_true=b)

print("Wrong dimension of the estimator")
arx_estimator = RecursiveEstimator(model='ARX', dim=3)
armax_estimator = RecursiveEstimator(model='ARMAX', dim=3)
oe_estimator = RecursiveEstimator(model='OE', dim=3)
for estimator in [arx_estimator, armax_estimator, oe_estimator]:
    plot_estimation_results(estimator, u, y_oe, iterations=3, use_posterior=True, a_true=a, b_true=b)

 #%%
N = 1000
u = gen_prbs(0.1, N)
y_arx = gen_model_out(a, b, u, "OE")

#%%
print("Estimation without the posterior estimates")
arx_estimator = RecursiveEstimator(model='ARX', dim=2)
armax_estimator = RecursiveEstimator(model='ARMAX', dim=2)
oe_estimator = RecursiveEstimator(model='OE', dim=2)
for estimator in [arx_estimator, armax_estimator, oe_estimator]:
    plot_estimation_results(estimator, u, y_oe, iterations=3, use_posterior=False, a_true=a, b_true=b)

print("Estimation with the posterior estimates")
arx_estimator = RecursiveEstimator(model='ARX', dim=2)
armax_estimator = RecursiveEstimator(model='ARMAX', dim=2)
oe_estimator = RecursiveEstimator(model='OE', dim=2)
for estimator in [arx_estimator, armax_estimator, oe_estimator]:
    plot_estimation_results(estimator, u, y_oe, iterations=3, use_posterior=True, a_true=a, b_true=b)

print("Wrong dimension of the estimator")
arx_estimator = RecursiveEstimator(model='ARX', dim=3)
armax_estimator = RecursiveEstimator(model='ARMAX', dim=3)
oe_estimator = RecursiveEstimator(model='OE', dim=3)
for estimator in [arx_estimator, armax_estimator, oe_estimator]:
    plot_estimation_results(estimator, u, y_oe, iterations=3, use_posterior=True, a_true=a, b_true=b)


#%%
calculate_and_plot_psd(gen_prbs(0.5, 100000), 2* np.pi, show_plot=True)
calculate_and_plot_psd(gen_prbs(0.1, 100000), 2* np.pi, show_plot=True)
# %%
