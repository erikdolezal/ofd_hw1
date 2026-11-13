import matplotlib.pyplot as plt
import numpy as np

def plot_estimation_results(estimator, u_arr, y_arr, iterations=1, use_posterior=True, a_true=None, b_true=None, c_true=None):
    """
    Runs the recursive parameter estimation iteratively and plots the results overlapping on the same x-axis.
    
    Args:
        estimator: An instance of RecursiveEstimator.
        u_arr, y_arr: The input and true output arrays of the dataset.
        iterations: Number of passes over the dataset (for ARMAX/OE iterative estimation).
        use_posterior: Boolean flag to use y(t|t) instead of y(t|t-1).
        a_true, b_true, c_true: True parameter arrays for reference lines.
    """
    results = []
    
    for i in range(iterations):
        if i > 0:
            # Reset histories for the new pass to clear early "polluted" data
            estimator.y_hist.fill(0)
            estimator.y_pred_hist.fill(0)
            estimator.u_hist.fill(0)
            estimator.e_hist.fill(0)
            # Reset covariance matrix (lower uncertainty than initial) while keeping learned theta
            estimator.P = np.eye(estimator.theta.shape[0]) * 10.0
            
        print(f"--- Iteration {i+1}/{iterations} ---")
        res = estimator.run_batch(u_arr, y_arr, use_posterior=use_posterior)
        results.append(res)

    model = estimator.model
    n_iters = len(results)
    n_samples = len(y_arr)
    has_c = model not in ['ARX', 'OE']
        
    # Determine grid size (4 rows for ARMAX, 3 for ARX/OE)
    n_plots = 4 if has_c else 3
    fig, ax = plt.subplots(n_plots, 1, figsize=(12, 3.5 * n_plots))
    if n_plots == 1: 
        ax = [ax]
        
    x_axis = np.arange(n_samples)
    
    # Helper to fade earlier iterations
    def get_alpha(it):
        if n_iters == 1: return 1.0
        return 0.25 + 0.75 * (it / (n_iters - 1))

    # 1. Output Plot
    ax[0].plot(x_axis, y_arr, label='True Output', color='blue')
    for it, res in enumerate(results):
        lbl = f'{model} Pred' if n_iters == 1 else f'{model} Pred (It {it+1})'
        ax[0].plot(x_axis, res['y_pred'], label=lbl, color='green', alpha=get_alpha(it))
    ax[0].set_title(f'{model} Output vs Predicted (Posterior: {use_posterior})')
    ax[0].set_ylabel('Output')
    
    # 2. A Parameters
    dim_a = results[0]['a'].shape[1]
    for i in range(dim_a):
        for it, res in enumerate(results):
            lbl = f'Est a[{i+1}]' if n_iters == 1 else f'Est a[{i+1}] (It {it+1})'
            ax[1].plot(x_axis, res['a'][:, i], color=f"C{i}", alpha=get_alpha(it), label=lbl)
            
        if a_true is not None and (i + 1) < len(a_true): 
            ax[1].plot(x_axis, np.ones_like(x_axis) * a_true[i+1], color=f"C{i}", linestyle='--', alpha=1.0, label=f'True a[{i+1}]')
    ax[1].set_title(f'{model} Parameter "a" Estimation')
    ax[1].set_ylabel('Value')
    
    # 3. B Parameters
    dim_b = results[0]['b'].shape[1]
    for i in range(dim_b):
        for it, res in enumerate(results):
            lbl = f'Est b[{i}]' if n_iters == 1 else f'Est b[{i}] (It {it+1})'
            ax[2].plot(x_axis, res['b'][:, i], color=f"C{i}", alpha=get_alpha(it), label=lbl)
            
        if b_true is not None and i < len(b_true):
            ax[2].plot(x_axis, np.ones_like(x_axis) * b_true[i], color=f"C{i}", linestyle='--', alpha=1.0, label=f'True b[{i}]')
    ax[2].set_title(f'{model} Parameter "b" Estimation')
    ax[2].set_ylabel('Value')
    
    # 4. C Parameters (ARMAX only)
    if has_c:
        dim_c = results[0]['c'].shape[1]
        for i in range(dim_c):
            for it, res in enumerate(results):
                lbl = f'Est c[{i+1}]' if n_iters == 1 else f'Est c[{i+1}] (It {it+1})'
                ax[3].plot(x_axis, res['c'][:, i], color=f"C{i}", alpha=get_alpha(it), label=lbl)
                
            if c_true is not None and (i + 1) < len(c_true): 
                ax[3].plot(x_axis, np.ones_like(x_axis) * c_true[i+1], color=f"C{i}", linestyle='--', alpha=1.0, label=f'True c[{i+1}]')
        ax[3].set_title(f'{model} Parameter "c" Estimation')
        ax[3].set_ylabel('Value')
    
    # Global formatting
    for i in range(n_plots):
        ax[i].set_xlabel('Sample Number')
        ax[i].grid(True)
        # Move legend outside the plot
        ax[i].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
            
    plt.tight_layout()
    plt.show()


def plot_multiple_model_results(results_dict, model_name="Model", a_true=None, b_true=None, c_true=None, filename=None):
    """
    Plots the results of recursive parameter estimation.
    
    Args:
        results_dict: Dictionary mapping string labels to result dictionaries (output of run_batch).
                      Example: {"RLS": res_rls, "ELS": res_els}
        model_name: String name of the model for titles.
        a_true, b_true, c_true: True parameter arrays for reference lines. 
                                Only parameters provided here will be plotted.
    """
    # Determine which parameter plots to create dynamically
    plots_to_make = ["a", "b"]
    if model_name == "ARMAX":
        plots_to_make.append('c')
        
    n_plots = len(plots_to_make)
    if n_plots == 0:
        print("No true parameters provided to plot.")
        return

    fig, ax = plt.subplots(n_plots, 1, figsize=(10, 2.5 * n_plots), sharex=True)
    if n_plots == 1: 
        ax = [ax]
        
    # Extract the first result to get dimensions
    first_res = next(iter(results_dict.values()))
    x_axis = np.arange(len(first_res['y_pred']))
    plot_idx = 0
    
    ax[0].set_title(f'{model_name} Parameter Estimation')
    # 1. A Parameters
    if a_true is not None:
        dim_a = first_res['a'].shape[1]
        for i in range(dim_a):
            # Plot GT first (zorder=1) so it sits underneath
            if (i + 1) < len(a_true): 
                ax[plot_idx].plot(x_axis, np.ones_like(x_axis) * a_true[i+1], linestyle='--', color='black', label=f'True Values', zorder=1)
            # Plot Estimates on top (zorder=2)
            for label, res in results_dict.items():
                ax[plot_idx].plot(x_axis, res['a'][:, i], label=f'{label} $a_{i+1}$', zorder=2)
                
        ax[plot_idx].set_ylabel('$a$ Parameter Value')
        plot_idx += 1
        
    # 2. B Parameters
    if b_true is not None:
        dim_b = first_res['b'].shape[1]
        for i in range(dim_b):
            # Plot GT first (zorder=1) so it sits underneath
            if i < len(b_true):
                ax[plot_idx].plot(x_axis, np.ones_like(x_axis) * b_true[i], linestyle='--', color='black', label=f'True Values', zorder=1)
            # Plot Estimates on top (zorder=2)
            for label, res in results_dict.items():
                ax[plot_idx].plot(x_axis, res['b'][:, i], label=f'{label} $b_{i}$', zorder=2)
                
        ax[plot_idx].set_ylabel('$b$ Parameter Value')
        plot_idx += 1
        
    # 3. C Parameters
    if first_res.get('c') is not None:
        dim_c = first_res['c'].shape[1]
        for i in range(dim_c):
            # Plot GT first (zorder=1) so it sits underneath
            if c_true is not None and (i + 1) < len(c_true): 
                ax[plot_idx].plot(x_axis, np.ones_like(x_axis) * c_true[i+1], linestyle='--', color='black', label=rf'True Values', zorder=1)
            # Plot Estimates on top (zorder=2)
            for label, res in results_dict.items():
                ax[plot_idx].plot(x_axis, res['c'][:, i], label=f'{label} $c_{i+1}$', zorder=2)
                
        ax[plot_idx].set_ylabel('$c$ Parameters Value')
        plot_idx += 1
    
    # Global formatting
    for i in range(n_plots):
        ax[i].grid(True)
        # Handle duplicate labels in legend (from true parameter lines)
        handles, labels = ax[i].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # Remove empty labels generated by conditional logic above
        by_label.pop("", None) 
        # Placed legend inside the plot on the right center
        ax[i].legend(by_label.values(), by_label.keys(), loc='center right')
    ax[-1].set_xlabel('Sample Number')
            
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()