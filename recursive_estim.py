import numpy as np
from tqdm import tqdm

class RecursiveEstimator:
    def __init__(self, model, dim=3, a_init=None, b_init=None, c_init=None):
        self.model = model
        self.dim = dim
        
        self.a = a_init if (a_init is not None and a_init.shape[0] == dim) else np.zeros(dim)
        self.b = b_init if (b_init is not None and b_init.shape[0] == dim+1) else np.zeros(dim+1)
        self.c = c_init if (c_init is not None and c_init.shape[0] == dim) else np.zeros(dim)
        
        if model in ['ARX', 'OE']:
            self.theta = np.concatenate((self.a, self.b))
        else:
            self.theta = np.concatenate((self.a, self.b, self.c))

        self.y_hist = np.zeros(self.dim)
        self.y_pred_hist = np.zeros(self.dim)
        self.u_hist = np.zeros(self.dim)
        self.e_hist = np.zeros(self.dim)
        
        self.P = np.eye(self.theta.shape[0]) * 1000.0

    def update(self, u, y):
        if self.model == "ARX":
            z = np.concatenate((-self.y_hist, [u], self.u_hist))
        elif self.model == "OE":
            z = np.concatenate((-self.y_pred_hist, [u], self.u_hist))
        else:
            z = np.concatenate((-self.y_hist, [u], self.u_hist, self.e_hist))

        y_pred = z @ self.theta
        error = y - y_pred

        P_z = self.P @ z
        zeta = z @ P_z
        L = P_z / (1.0 + zeta)

        self.theta += L * error
        self.P -= np.outer(L, P_z) 

        self.y_hist = np.roll(self.y_hist, 1)
        self.y_hist[0] = y
        
        self.u_hist = np.roll(self.u_hist, 1)
        self.u_hist[0] = u
        
        self.y_pred_hist = np.roll(self.y_pred_hist, 1)
        self.y_pred_hist[0] = z @ self.theta
        
        self.e_hist = np.roll(self.e_hist, 1)
        self.e_hist[0] = y - (z @ self.theta)

        if self.model in ['ARX', 'OE']:
            self.a = self.theta[:self.dim]
            self.b = self.theta[self.dim:]
        else:
            self.a = self.theta[:self.dim]
            self.b = self.theta[self.dim:2*self.dim+1]
            self.c = self.theta[2*self.dim + 1:]

        return self.y_pred_hist[0]

    def run_batch(self, u_arr, y_arr):
        theta_history = np.zeros((y_arr.shape[0], self.theta.shape[0]))
        theta_cov_history = np.zeros((y_arr.shape[0], self.theta.shape[0], self.theta.shape[0]))
        y_pred = np.zeros_like(y_arr)

        for i in tqdm(range(len(y_arr)), f"Recursive batch {self.model} model parameters estimation"):
            y_pred[i] = self.update(u_arr[i], y_arr[i])
            theta_history[i], theta_cov_history[i] = self.theta.copy(), self.P.copy()

        return {
            "theta": theta_history,
            "P": theta_cov_history,
            "y_pred": y_pred,
        }

