import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tensorly as tl
import scipy.sparse as sparse

from numpy.linalg import inv, solve, cholesky
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from scipy.stats import wishart, invwishart
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut
from scipy.stats import gamma
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import cg
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from numpy.linalg import eigh
from sklearn.ensemble import RandomForestRegressor
from tensorly.decomposition import parafac, tucker
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import NearestNeighbors

tl.set_backend('numpy')



def mvnrnd_pre(mu, Lambda):
    src = normrnd(size=(mu.shape[0],))
    return solve_ut(cholesky_upper(Lambda, overwrite_a=True, check_finite=False), 
                    src, lower=False, check_finite=False, overwrite_b=True) + mu

def cov_mat(mat, mat_bar):
    mat = mat - mat_bar
    return mat.T @ mat

def sample_precision_tau(sparse_mat, mat_hat, ind):
    var_alpha = 1e-6 + 0.5 * np.sum(ind, axis=1)
    var_beta = 1e-6 + 0.5 * np.sum(((sparse_mat - mat_hat) ** 2) * ind, axis=1)
    return np.random.gamma(var_alpha, 1 / var_beta)

def sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau, beta0=1):
    dim1, rank = W.shape
    W_bar = np.mean(W, axis=0)
    temp = dim1 / (dim1 + beta0)
    var_W_hyper = inv(np.eye(rank) + cov_mat(W, W_bar) + temp * beta0 * np.outer(W_bar, W_bar))
    var_Lambda_hyper = wishart.rvs(df=dim1 + rank, scale=var_W_hyper)
    var_mu_hyper = mvnrnd_pre(temp * W_bar, (dim1 + beta0) * var_Lambda_hyper)
    
    var1 = X.T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ tau_ind.T).reshape([rank, rank, dim1]) + var_Lambda_hyper[:, :, None]
    var4 = var1 @ tau_sparse_mat.T + (var_Lambda_hyper @ var_mu_hyper)[:, None]
    for i in range(dim1):
        W[i, :] = mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])
    return W

# =============================================================================
# Bayesian Models (Output Mean and Std Dev)
# =============================================================================

class BayesianModelBase:
    def _init_accumulators(self, dim1, dim2):
        self.mat_hat_sum = np.zeros((dim1, dim2))
        self.mat_hat_sq_sum = np.zeros((dim1, dim2))
        self.noise_var_sum = 0.0

    def _accumulate(self, W, X, tau): 
        mat_hat_it = W @ X.T
        self.mat_hat_sum += mat_hat_it
        self.mat_hat_sq_sum += mat_hat_it**2
        current_noise_var = 1.0 / np.mean(tau) 
        self.noise_var_sum += current_noise_var

    def _finalize_uq(self, gibbs_iter):
        # E[X]
        mat_hat_mean = self.mat_hat_sum / gibbs_iter
        # E[X^2] - (E[X])^2
        mat_hat_var_model = (self.mat_hat_sq_sum / gibbs_iter) - (mat_hat_mean**2)
        mat_hat_var_model = np.maximum(mat_hat_var_model, 0)
        
        avg_noise_var = self.noise_var_sum / gibbs_iter
        
        mat_hat_std = np.sqrt(mat_hat_var_model + avg_noise_var)
        
        return mat_hat_mean.reshape(self.dims), mat_hat_std.reshape(self.dims)


class PMF(BayesianModelBase):
    def __init__(self, rank, dims, **kwargs):
        self.rank = rank
        self.dims = dims

    def train(self, sparse_mat_2d, burn_iter, gibbs_iter, verbose_step=50, **kwargs):
        print(f"Initializing PMF (Rank={self.rank})...")
        dim1, dim2 = sparse_mat_2d.shape
        ind = sparse_mat_2d != 0
        W = 0.1 * np.random.randn(dim1, self.rank)
        X = 0.1 * np.random.randn(dim2, self.rank)
        tau = np.ones(dim1)
        
        self._init_accumulators(dim1, dim2)
        start_time = time.time()
        
        def sample_x_pmf(tau_sparse_mat, tau_ind, W, X):
            dim2, rank = X.shape
            X_bar = np.mean(X, axis=0)
            var_X_hyper = inv(np.eye(rank) + cov_mat(X, X_bar) + 1.0 * np.outer(X_bar, X_bar))
            var_Lambda_hyper = wishart.rvs(df=dim2 + rank, scale=var_X_hyper)
            var_mu_hyper = mvnrnd_pre(X_bar, 2.0 * var_Lambda_hyper)
            var1 = W.T
            var2 = kr_prod(var1, var1)
            var3 = (var2 @ tau_ind).reshape([rank, rank, dim2]) + var_Lambda_hyper[:, :, None]
            var4 = var1 @ tau_sparse_mat + (var_Lambda_hyper @ var_mu_hyper)[:, None]
            for t in range(dim2):
                X[t, :] = mvnrnd_pre(solve(var3[:, :, t], var4[:, t]), var3[:, :, t])
            return X

        for it in range(burn_iter + gibbs_iter):
            tau_ind = tau[:, None] * ind
            tau_sparse_mat = tau[:, None] * sparse_mat_2d
            W = sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau)
            X = sample_x_pmf(tau_sparse_mat, tau_ind, W, X)
            mat_hat = W @ X.T
            tau = sample_precision_tau(sparse_mat_2d, mat_hat, ind)
            
            if it >= burn_iter:
                self._accumulate(W, X, tau) 
            if (it + 1) % verbose_step == 0:
                print(f"[PMF] Iter {it+1} | Time: {time.time()-start_time:.1f}s")
        
        return self._finalize_uq(gibbs_iter)

class BTMF(BayesianModelBase):
    def __init__(self, rank, time_lags, dims, **kwargs):
        self.rank = rank
        self.time_lags = time_lags
        self.dims = dims

    def train(self, sparse_mat_2d, burn_iter, gibbs_iter, verbose_step=50, **kwargs):
        print(f"Initializing BTMF (Rank={self.rank})...")
        dim1, dim2 = sparse_mat_2d.shape
        ind = sparse_mat_2d != 0
        d = self.time_lags.shape[0]
        W = 0.1 * np.random.randn(dim1, self.rank)
        X = 0.1 * np.random.randn(dim2, self.rank)
        tau = np.ones(dim1)
        
        self._init_accumulators(dim1, dim2)
        start_time = time.time()

        # --- BTMF Helpers (Simplified for brevity, assume correctness from previous versions) ---
        def sample_var_coef(X, lags):
            dim, r = X.shape
            dd = len(lags)
            tmax = np.max(lags)
            Z_mat = X[tmax:dim, :]
            Q_mat = np.zeros((dim - tmax, r * dd))
            for k in range(dd): Q_mat[:, k*r : (k+1)*r] = X[tmax-lags[k] : dim-lags[k], :]
            var_Psi0 = np.eye(r * dd) + Q_mat.T @ Q_mat
            var_Psi = inv(var_Psi0)
            var_M = var_Psi @ Q_mat.T @ Z_mat
            var_S = np.eye(r) + Z_mat.T @ Z_mat - var_M.T @ var_Psi0 @ var_M
            Sigma = invwishart.rvs(df=r + dim - tmax, scale=var_S)
            return mnrnd(var_M, var_Psi, Sigma), Sigma
        def mnrnd(M, U, V):
            X0 = np.random.randn(M.shape[0], M.shape[1])
            return M + np.linalg.cholesky(U) @ X0 @ np.linalg.cholesky(V).T
        def sample_x_btmf(tau_sparse_mat, tau_ind, lags, W, X, A, inv_Sigma):
            dim2, rank = X.shape
            d = lags.shape[0]
            tmax, tmin = np.max(lags), np.min(lags)
            A0 = np.dstack([A] * d)
            for k in range(d): A0[k*rank : (k+1)*rank, :, k] = 0
            mat0 = inv_Sigma @ A.T
            mat1 = np.einsum('kij, jt -> kit', A.reshape([d, rank, rank]), inv_Sigma)
            mat2 = np.einsum('kit, kjt -> ij', mat1, A.reshape([d, rank, rank]))
            var1 = W.T
            var2 = kr_prod(var1, var1)
            var3 = (var2 @ tau_ind).reshape([rank, rank, dim2]) + inv_Sigma[:, :, None]
            var4 = var1 @ tau_sparse_mat
            for t in range(dim2):
                Mt = np.zeros((rank, rank)); Nt = np.zeros(rank); Qt = mat0 @ X[t - lags, :].reshape(rank * d)
                index = list(range(d))
                if t >= dim2 - tmax and t < dim2 - tmin: index = list(np.where(t + lags < dim2)[0])
                elif t < tmax: Qt = np.zeros(rank); index = list(np.where(t + lags >= tmax)[0])
                if t < dim2 - tmin:
                    Mt = mat2.copy()
                    temp = np.zeros((rank * d, len(index)))
                    for n, k in enumerate(index): temp[:, n] = X[t + lags[k] - lags, :].reshape(rank * d)
                    temp0 = X[t + lags[index], :].T - np.einsum('ijk, ik -> jk', A0[:, :, index], temp)
                    Nt = np.einsum('kij, jk -> i', mat1[index, :, :], temp0)
                var3_t = var3[:, :, t] + Mt
                if t < tmax: var3_t = var3_t - inv_Sigma + np.eye(rank)
                X[t, :] = mvnrnd_pre(solve(var3_t, var4[:, t] + Nt + Qt), var3_t)
            return X
        # ----------------------------------------------------------------

        for it in range(burn_iter + gibbs_iter):
            tau_ind = tau[:, None] * ind
            tau_sparse_mat = tau[:, None] * sparse_mat_2d
            W = sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau)
            A, Sigma = sample_var_coef(X, self.time_lags)
            X = sample_x_btmf(tau_sparse_mat, tau_ind, self.time_lags, W, X, A, inv(Sigma))
            mat_hat = W @ X.T
            tau = sample_precision_tau(sparse_mat_2d, mat_hat, ind)
            
            if it >= burn_iter:
                self._accumulate(W, X, tau)
            if (it + 1) % verbose_step == 0:
                print(f"[BTMF] Iter {it+1} | Time: {time.time()-start_time:.1f}s")
        
        return self._finalize_uq(gibbs_iter)

class BTRMF(BayesianModelBase):
    def __init__(self, rank, time_lags, dims, **kwargs):
        self.rank = rank
        self.time_lags = time_lags
        self.dims = dims

    def train(self, sparse_mat_2d, burn_iter, gibbs_iter, verbose_step=50, **kwargs):
        print(f"Initializing BTRMF (Rank={self.rank})...")
        dim1, dim2 = sparse_mat_2d.shape
        ind = sparse_mat_2d != 0
        d = self.time_lags.shape[0]
        W = 0.1 * np.random.randn(dim1, self.rank)
        X = 0.1 * np.random.randn(dim2, self.rank)
        theta = 0.01 * np.random.randn(d, self.rank)
        tau = np.ones(dim1)
        
        self._init_accumulators(dim1, dim2)
        start_time = time.time()
        
        # --- BTRMF Helpers (Simplified) ---
        def sample_Lambda_x(X, theta, lags):
            dim, rank = X.shape; d = lags.shape[0]; tmax = np.max(lags)
            mat = X[:tmax, :].T @ X[:tmax, :]
            temp = np.zeros((dim - tmax, rank, d))
            for k in range(d): temp[:, :, k] = X[tmax - lags[k] : dim - lags[k], :]
            new_mat = X[tmax:dim, :] - np.einsum('kr, irk -> ir', theta, temp)
            return wishart.rvs(df=dim + rank, scale=inv(np.eye(rank) + mat + new_mat.T @ new_mat))
        def sample_theta(X, theta, Lambda_x, lags, beta0=1):
            dim, rank = X.shape; d = lags.shape[0]; tmax = np.max(lags)
            theta_bar = np.mean(theta, axis=0); temp = d / (d + beta0)
            var_theta_hyper = inv(np.eye(rank) + cov_mat(theta, theta_bar) + temp * beta0 * np.outer(theta_bar, theta_bar))
            var_Lambda_hyper = wishart.rvs(df=d + rank, scale=var_theta_hyper)
            var_mu_hyper = mvnrnd_pre(temp * theta_bar, (d + beta0) * var_Lambda_hyper)
            for k in range(d):
                theta0 = theta.copy(); theta0[k, :] = 0; mat0 = np.zeros((dim - tmax, rank))
                for L in range(d): mat0 += X[tmax - lags[L] : dim - lags[L], :] @ np.diag(theta0[L, :])
                VarPi = X[tmax:dim, :] - mat0; var0 = X[tmax - lags[k] : dim - lags[k], :]
                var = np.einsum('ij, jk, ik -> j', var0, Lambda_x, VarPi)
                var_Lambda = np.einsum('ti, tj, ij -> ij', var0, var0, Lambda_x) + var_Lambda_hyper
                theta[k, :] = mvnrnd_pre(solve(var_Lambda, var + var_Lambda_hyper @ var_mu_hyper), var_Lambda)
            return theta
        def sample_x_btrmf(tau_sparse_mat, tau_ind, lags, W, X, theta, Lambda_x):
            dim2, rank = X.shape; tmax, tmin = np.max(lags), np.min(lags); d = lags.shape[0]
            A = np.zeros((d * rank, rank))
            for k in range(d): A[k * rank : (k + 1) * rank, :] = np.diag(theta[k, :])
            A0 = np.dstack([A] * d)
            for k in range(d): A0[k * rank : (k + 1) * rank, :, k] = 0
            mat0 = Lambda_x @ A.T; mat1 = np.einsum('kij, jt -> kit', A.reshape([d, rank, rank]), Lambda_x)
            mat2 = np.einsum('kit, kjt -> ij', mat1, A.reshape([d, rank, rank]))
            var1 = W.T; var2 = kr_prod(var1, var1)
            var3 = (var2 @ tau_ind).reshape([rank, rank, dim2]) + Lambda_x[:, :, None]; var4 = var1 @ tau_sparse_mat
            for t in range(dim2):
                Mt = np.zeros((rank, rank)); Nt = np.zeros(rank); Qt = mat0 @ X[t - lags, :].reshape(rank * d)
                index = list(range(d))
                if t >= dim2 - tmax and t < dim2 - tmin: index = list(np.where(t + lags < dim2)[0])
                elif t < tmax: Qt = np.zeros(rank); index = list(np.where(t + lags >= tmax)[0])
                if t < dim2 - tmin:
                    Mt = mat2.copy(); temp = np.zeros((rank * d, len(index)))
                    for n, k in enumerate(index): temp[:, n] = X[t + lags[k] - lags, :].reshape(rank * d)
                    temp0 = X[t + lags[index], :].T - np.einsum('ijk, ik -> jk', A0[:, :, index], temp)
                    Nt = np.einsum('kij, jk -> i', mat1[index, :, :], temp0)
                var3_t = var3[:, :, t] + Mt
                if t < tmax: var3_t = var3_t - Lambda_x + np.eye(rank)
                X[t, :] = mvnrnd_pre(solve(var3_t, var4[:, t] + Nt + Qt), var3_t)
            return X
        # --------------------------------------------------

        for it in range(burn_iter + gibbs_iter):
            tau_ind = tau[:, None] * ind
            tau_sparse_mat = tau[:, None] * sparse_mat_2d
            W = sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau)
            Lambda_x = sample_Lambda_x(X, theta, self.time_lags)
            theta = sample_theta(X, theta, Lambda_x, self.time_lags)
            X = sample_x_btrmf(tau_sparse_mat, tau_ind, self.time_lags, W, X, theta, Lambda_x)
            mat_hat = W @ X.T
            tau = sample_precision_tau(sparse_mat_2d, mat_hat, ind)
            
            if it >= burn_iter:
                self._accumulate(W, X, tau) # 把 tau 传进去
            if (it + 1) % verbose_step == 0:
                print(f"[BTRMF] Iter {it+1} | Time: {time.time()-start_time:.1f}s")
        
        return self._finalize_uq(gibbs_iter)




class ProbKNN:
    """
    Probabilistic K-Nearest Neighbors (Spatiotemporal).
    A non-parametric baseline.
    Mean = Weighted average of K neighbors.
    Std  = Weighted standard deviation of K neighbors + dist penalty.
    """
    def __init__(self, rank=0, dims=None, n_neighbors=50, time_scale=1.0, **kwargs):
        self.dims = dims
        self.n_neighbors = int(n_neighbors)
        self.time_scale = float(time_scale)
        print(f"ProbKNN initialized: K={self.n_neighbors}, TimeScale={self.time_scale}")

    def train(self, sparse_mat_2d, sparse_mat_val=None, verbose_step=10, **kwargs):
        # 1. Prepare Data
        # sparse_mat_2d: (Lat*Lon, Time)
        if len(self.dims) == 3:
            H, W, T = self.dims
        else:
            H, W = 100, 200; T = self.dims[1]
            
        print("Building Spatiotemporal Index (KDTree)... this may take a moment.")
        start_time = time.time()
        
        # Grid
        # x: 0..H-1, y: 0..W-1, t: 0..T-1
        # Normalize to [0, 1] roughly to make Euclidean distance meaningful
        h_coords = np.linspace(0, 1, H)
        w_coords = np.linspace(0, 1 * (W/H), W) # Keep aspect ratio
        t_coords = np.linspace(0, 1 * (T/H) * self.time_scale, T) # Adjust time importance
        
        # Grid: (H, W, T, 3)
        grid_h, grid_w, grid_t = np.meshgrid(h_coords, w_coords, t_coords, indexing='ij')
        
        # Flat coordinates: (N_total, 3)
        coords_all = np.stack([grid_h.flatten(), grid_w.flatten(), grid_t.flatten()], axis=1)
        
        # Flatten Data
        data_flat = sparse_mat_2d.reshape(-1, T).flatten() # (N*T,) order should match meshgrid?
        # Careful: sparse_mat_2d is (H*W, T). 
        # Meshgrid 'ij' -> (H, W, T). Flattening order of numpy is C (last index changes fastest).
        # sparse_mat_2d flatten order: row0_t0, row0_t1... (Pixel 0 all times, Pixel 1 all times...)
        # grid_t varies fastest in 3rd dim.
        # Let's align explicitly:
        tensor_data = sparse_mat_2d.reshape(H, W, T)
        data_flat = tensor_data.flatten() # (H*W*T)
        
        # Extract Observed Data (Train)
        mask = data_flat != 0
        X_train = coords_all[mask] # (N_obs, 3)
        y_train = data_flat[mask]  # (N_obs,)
        
        # 2. Fit KNN
        knn = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto', n_jobs=-1)
        knn.fit(X_train)
        
        print(f"KNN Index built in {time.time()-start_time:.2f}s. Querying...")
        
        # 3. Query All Points (Inference)
        batch_size = 4096 * 4
        num_points = coords_all.shape[0]
        
        mat_hat_mean = np.zeros(num_points)
        mat_hat_std = np.zeros(num_points)
        
        # Global noise estimate (from training data variance or small epsilon)
        global_std = np.std(y_train) * 0.1 
        
        for i in range(0, num_points, batch_size):
            end = min(i + batch_size, num_points)
            batch_coords = coords_all[i:end]
            
            # Find neighbors
            dists, indices = knn.kneighbors(batch_coords)
            
            # Retrieve values
            neighbor_vals = y_train[indices] # (B, K)
            
            # --- Weighted Average ---
            # Weight = 1 / (dist + epsilon)
            weights = 1.0 / (dists + 1e-6)
            
            # Normalize weights
            weights /= np.sum(weights, axis=1, keepdims=True)
            
            # Mean
            mean = np.sum(neighbor_vals * weights, axis=1)
            
            # Std (Weighted)
            # Var = Sum( w * (x - mu)^2 )
            variance = np.sum(weights * (neighbor_vals - mean[:, None])**2, axis=1)
            
            # Add distance penalty to Std (Data Sparsity Uncertainty)
            dist_penalty = np.mean(dists, axis=1) * global_std
            
            mat_hat_mean[i:end] = mean
            mat_hat_std[i:end] = np.sqrt(variance) + dist_penalty + 1e-3 # Add epsilon
            
            if i % (batch_size * 10) == 0:
                print(f"Processed {i}/{num_points} points...")

        # Reshape to (N_spatial, T)
        # Note: coords_all was (H, W, T) flattened.
        # Output needs to be (H*W, T)
        # tensor_data.flatten() is C-order (H, W, T).
        # sparse_mat_2d is (H*W, T). It matches.
        
        mat_hat_mean = mat_hat_mean.reshape(H*W, T)
        mat_hat_std = mat_hat_std.reshape(H*W, T)
        
        return mat_hat_mean, mat_hat_std



class MVN_EM:
    """
    Multivariate Normal Imputation via Expectation-Maximization.
    Models the data as N independent samples from a T-dimensional Gaussian N(mu, Sigma).
    
    - Captures pure Temporal Correlations.
    - Ignores Spatial Correlations (treats pixels as i.i.d. samples).
    - Provides EXACT Analytical Uncertainty (Conditional Variance).
    """
    def __init__(self, rank=0, dims=None, max_iter=50, tol=1e-4, **kwargs):
        self.dims = dims
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        print(f"MVN-EM initialized. Max Iter: {self.max_iter}")

    def train(self, sparse_mat_2d, verbose_step=5, **kwargs):
        """
        sparse_mat_2d: (Lat*Lon, Time) -> (N, T)
        Input should be CENTERED (though MVN estimates mu, centering helps stability).
        """
        # 1. Data Prep
        if len(self.dims) == 3: H, W, T = self.dims
        else: H, W = 100, 200; T = self.dims[1]
        
        # Y: (N, T)
        Y = sparse_mat_2d 
        N, _ = Y.shape
        mask = (Y != 0)
        
        # 2. Initialization
        # Simple mean imputation for initial covariance estimation
        # Column-wise mean (Temporal mean)
        col_mean = np.sum(Y, axis=0) / (np.sum(mask, axis=0) + 1e-6)
        
        Y_filled = Y.copy()
        # Initial fill: if missing, use column mean
        for t in range(T):
            Y_filled[~mask[:, t], t] = col_mean[t]
            
        # Initial Parameters
        mu = np.mean(Y_filled, axis=0) # (T,)
        Sigma = np.cov(Y_filled, rowvar=False) + 1e-4 * np.eye(T) # (T, T)
        
        start_time = time.time()
        
        # 3. EM Loop
        for it in range(self.max_iter):
            prev_Y_filled = Y_filled.copy()
            
            # --- E-Step: Impute missing values based on current mu, Sigma ---
            # For speed in Python, we use a simplified linear projection for the bulk update
            # Y_miss = Mu_miss + Sigma_mo * Sigma_oo^-1 * (Y_obs - Mu_obs)
            # Doing this per-pixel is slow in loop.
            # Fast approximation for M-step: 
            # Re-estimate using updated mu/Sigma on the filled matrix?
            # Standard EM for Gaussian:
            # S = 1/N * Sum (E[y y.T])
            # mu = 1/N * Sum (E[y])
            
            # Let's use the iterative imputation formulation (simpler to implement):
            # For each column t (time), regress on all other columns to fill missing.
            # But that's slow.
            
            # Vectorized Matrix Multiplication approach:
            # We want to project Observed -> Missing using Sigma correlation.
            # Since mask patterns vary per pixel, full vectorization is hard.
            # But T=31 is small. We can iterate over T? No.
            
            # Let's stick to the standard iterative approach used in packages like 'fancyimpute':
            # 1. Estimate Mu, Sigma from Y_filled
            mu = np.mean(Y_filled, axis=0)
            diff = Y_filled - mu
            Sigma = (diff.T @ diff) / N + 1e-5 * np.eye(T) # Covariance
            
            # 2. Update Y_filled (Imputation)
            # This is the bottleneck. We solve (Sigma_oo) x = Sigma_om for each missing pattern?
            # To make it fast for this baseline, we can use a "Ridge Regression" approximation
            # treating Sigma as the weights.
            # Or simplified: Y_new = Y_old + ...
            
            # Let's use the precise row-by-row update but optimize for groups of masks?
            # Given N=20000, maybe just looping is okay if we use Numba, but here pure Python.
            # Let's try a simplified update: 
            # Y_hat = Y_filled @ (I - Sigma_inv / diag(Sigma_inv)) ... equivalent to GMRF?
            
            # Let's use the inverse covariance (Precision Matrix) Lambda = Sigma^-1
            # Conditional mean of y_i given y_{-i} is -Lambda_{ij}/Lambda_{ii} * y_j
            # This allows iterating columns.
            
            try:
                Lambda = inv(Sigma)
            except:
                Lambda = inv(Sigma + 1e-3 * np.eye(T))
            
            # Update each column t conditional on others
            for t in range(T):
                # Identify rows where t is missing
                missing_idx = ~mask[:, t]
                if np.sum(missing_idx) == 0: continue
                
                # y_t = -1/L_tt * Sum_{j!=t} L_tj * y_j
                # Vectorized over all N samples
                # target = (Y_filled - mu) @ Lambda[:, t]
                # We want (Y_filled - mu)_t. 
                # (y - mu)^T Lambda (y - mu) -> min w.r.t y_t
                # deriv: 2 * Lambda_tt * (y_t - mu_t) + 2 * Sum_{j!=t} Lambda_tj * (y_j - mu_j) = 0
                # y_t - mu_t = - (1/Lambda_tt) * Sum_{j!=t} Lambda_tj * (y_j - mu_j)
                
                # Compute interaction from other columns
                # Y_centered @ Lambda_col_t
                # But Y_centered[:, t] is currently old estimate. 
                # We subtract its contribution: Y_c @ Col - Y_c[:,t] * L_tt
                
                Y_centered = Y_filled - mu
                interaction = Y_centered @ Lambda[:, t] - Y_centered[:, t] * Lambda[t, t]
                
                # Update missing
                y_t_new = mu[t] - interaction / Lambda[t, t]
                Y_filled[missing_idx, t] = y_t_new[missing_idx]

            # Convergence
            change = np.linalg.norm(Y_filled - prev_Y_filled) / (np.linalg.norm(prev_Y_filled) + 1e-9)
            
            if (it + 1) % verbose_step == 0:
                print(f"[MVN-EM] Iter {it+1} | Diff: {change:.6f} | Time: {time.time()-start_time:.1f}s")
            
            if change < self.tol:
                break
        
        # 4. Final Uncertainty Quantification (Exact Conditional Variance)
        # We need to compute Var(y_miss | y_obs) for each pixel.
        # This requires inverting Sigma_obs_obs for each pixel.
        # Since T=31, inversion is cheap. We loop over 20,000 pixels.
        
        print("Calculating MVN Uncertainty (Exact)...")
        mat_hat_mean = Y_filled
        mat_hat_var = np.zeros_like(Y_filled)
        
        # Pre-allocate global noise (if any, typically 0 for pure MVN, but let's assume Sigma captures it)
        # Conditional Variance formula: Sigma_mm - Sigma_mo * Sigma_oo^-1 * Sigma_om
        # This returns a matrix. We only need the diagonal (marginal variance for each day).
        
        # Optimization: Group pixels by missing pattern to batch inversion? 
        # Too complex for a baseline script. 
        # Just loop. 20,000 inversions of (k x k) where k < 31 takes < 5 seconds.
        
        for i in range(N):
            m_i = ~mask[i] # missing bools
            o_i = mask[i]  # observed bools
            
            if np.sum(m_i) == 0: continue # Nothing missing
            
            # If everything missing, return diagonal of Sigma (Prior variance)
            if np.sum(o_i) == 0:
                mat_hat_var[i, m_i] = np.diag(Sigma)[m_i]
                continue
                
            # Partition Sigma
            # We only need diagonal of the conditional covariance block
            Sigma_mm = Sigma[np.ix_(m_i, m_i)]
            Sigma_mo = Sigma[np.ix_(m_i, o_i)]
            Sigma_oo = Sigma[np.ix_(o_i, o_i)]
            
            # Schur Complement
            # Var = diag(Sigma_mm - Sigma_mo @ inv(Sigma_oo) @ Sigma_mo.T)
            
            try:
                Sigma_oo_inv = inv(Sigma_oo)
                Cond_Cov = Sigma_mm - Sigma_mo @ Sigma_oo_inv @ Sigma_mo.T
                mat_hat_var[i, m_i] = np.diag(Cond_Cov)
            except np.linalg.LinAlgError:
                # Fallback
                mat_hat_var[i, m_i] = np.diag(Sigma_mm)

        mat_hat_std = np.sqrt(np.maximum(mat_hat_var, 0))
        
        # Reshape to (Lat*Lon, T) -> (Space, Time) 
        # It is already (N, T)
        return mat_hat_mean, mat_hat_std



class PPCA:
    """
    Probabilistic Principal Component Analysis (PPCA).
    Solved via EM algorithm for missing data.
    
    Model: y ~ N(mu, W W^T + sigma^2 I)
    Latent: y = Wx + mu + noise
    
    Provides EXACT conditional mean and variance for missing values.
    """
    def __init__(self, rank=10, dims=None, max_iter=100, tol=1e-4, **kwargs):
        self.rank = int(rank)
        self.dims = dims
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        print(f"PPCA initialized: Rank={self.rank}")

    def train(self, sparse_mat_2d, verbose_step=10, **kwargs):
        """
        sparse_mat_2d: (N_samples, T_features) -> (20000, 31)
        Input is assumed to be roughly centered (main.py handles mean centering).
        """
        # 1. Data Prep
        if len(self.dims) == 3: H, W_dim, T = self.dims
        else: H, W_dim = 100, 200; T = self.dims[1]
        
        Y = sparse_mat_2d # (N, T)
        N, D = Y.shape    # D is Time (31)
        mask = (Y != 0)
        
        # 2. Initialization
        # Fill missing with column means for initial SVD
        col_mean = np.sum(Y, axis=0) / (np.sum(mask, axis=0) + 1e-6)
        Y_filled = Y.copy()
        for t in range(D):
            Y_filled[~mask[:, t], t] = col_mean[t]
            
        # Init W using SVD of filled data
        # Y ~ U S V.T -> W approx V * S / sqrt(N)
        # Random init is also fine, but SVD is faster
        u, s, vh = np.linalg.svd(Y_filled, full_matrices=False)
        # W: (D, K)
        W = vh[:self.rank, :].T * np.sqrt(s[:self.rank]**2 / N) 
        sigma2 = np.var(Y_filled - Y_filled @ (vh[:self.rank].T @ vh[:self.rank])) # Init noise
        
        start_time = time.time()
        
        # 3. EM Algorithm
        for it in range(self.max_iter):
            prev_Y_filled = Y_filled.copy()
            
            # --- E-Step & Imputation ---
            # For PPCA with missing data, the exact E-step requires per-sample inversion 
            # if missing patterns differ.
            # M_inv = inv(W.T W + sigma2 I)
            # x_n = M_inv W.T (y_n - mu) ?? No, this is for fully observed.
            
            # Simplified EM for Missing Data (Iterative Imputation approach):
            # 1. Project current filled data to latent space
            #    x = (W.T W + sigma2 I)^-1 W.T y_filled
            # 2. Reconstruct
            #    y_recon = W x
            # 3. Fill missing
            #    y_new[miss] = y_recon[miss]
            
            # Compute M matrix (K x K)
            M = W.T @ W + sigma2 * np.eye(self.rank)
            M_inv = inv(M)
            
            # Latent expectations: E[x] = M_inv @ W.T @ Y.T -> (K, N)
            Ex = M_inv @ W.T @ Y_filled.T
            
            # Reconstruction: Y_hat = (W @ Ex).T -> (N, D)
            Y_recon = (W @ Ex).T
            
            # Update missing values
            Y_filled[~mask] = Y_recon[~mask]
            
            # --- M-Step ---
            # Update W: Y_filled.T @ Ex.T @ inv(Ex @ Ex.T + N * sigma2 * M_inv)
            # Standard PPCA M-step formula adapted
            
            # Compute E[x x^T] term
            # Sum_n E[x_n x_n^T] = N * sigma2 * M_inv + Ex @ Ex.T
            Exx = N * sigma2 * M_inv + Ex @ Ex.T
            
            # New W = (Sum_n y_n E[x_n]^T) @ (Sum_n E[x_n x_n^T])^-1
            #       = (Y_filled.T @ Ex.T) @ inv(Exx)
            W_new = (Y_filled.T @ Ex.T) @ inv(Exx)
            
            # Update sigma2
            # sigma2 = 1/(N*D) * sum ||y_n - W x_n||^2 ... roughly
            # More precise: 1/(N*D) * tr(Y Y^T - 2 Y Ex^T W^T + W Exx W^T)
            # Simplified: Mean squared error of reconstruction + variance correction
            
            # Using the reconstruction error on filled data
            recon_error = np.sum((Y_filled - Y_recon)**2)
            # Add trace term from latent uncertainty
            trace_term = N * np.trace((M_inv @ W_new.T @ W_new)) # Approximation
            sigma2_new = (recon_error + trace_term * sigma2) / (N * D)
            
            W = W_new
            sigma2 = sigma2_new
            
            # Convergence
            diff = np.linalg.norm(Y_filled - prev_Y_filled) / (np.linalg.norm(prev_Y_filled) + 1e-9)
            if (it + 1) % verbose_step == 0:
                print(f"[PPCA] Iter {it+1} | Diff: {diff:.6f} | Sigma2: {sigma2:.4f} | Time: {time.time()-start_time:.1f}s")
            
            if diff < self.tol:
                break
        
        # 4. Final Uncertainty Quantification (Exact Conditional Variance)
        # Covariance Matrix C = W W^T + sigma2 I  (D x D)
        print("Calculating PPCA Uncertainty...")
        
        C = W @ W.T + sigma2 * np.eye(D)
        
        mat_hat_mean = Y_filled
        mat_hat_var = np.zeros_like(Y_filled)
        
        # Calculate conditional variance for each pixel
        # Var(y_m | y_o) = C_mm - C_mo * C_oo^-1 * C_om
        
        # Optimization: Loop is fine for N=20000, D=31
        for i in range(N):
            m_i = ~mask[i] # Missing indices
            o_i = mask[i]  # Observed indices
            
            if np.sum(m_i) == 0: continue
            
            # If everything missing (rare), return diagonal of C
            if np.sum(o_i) == 0:
                mat_hat_var[i, m_i] = np.diag(C)[m_i]
                continue
            
            # Partition C
            C_mm = C[np.ix_(m_i, m_i)]
            C_mo = C[np.ix_(m_i, o_i)]
            C_oo = C[np.ix_(o_i, o_i)]
            
            try:
                # Schur Complement
                # Using solve is more stable than inv: C_mo @ solve(C_oo, C_om)
                # C_om = C_mo.T
                term = C_mo @ solve(C_oo, C_mo.T)
                cond_cov = C_mm - term
                mat_hat_var[i, m_i] = np.diag(cond_cov)
            except np.linalg.LinAlgError:
                # Fallback
                mat_hat_var[i, m_i] = np.diag(C_mm)

        # Ensure positive
        mat_hat_var = np.maximum(mat_hat_var, 0)
        mat_hat_std = np.sqrt(mat_hat_var)
        
        return mat_hat_mean, mat_hat_std







class BKCP:
    def __init__(self, rank, dims, length_scale=[3.0, 3.0, 1.5], **kwargs):
        self.rank = rank
        self.dims = dims
        self.length_scale = length_scale
        
        # Accumulators
        self.mat_hat_sum = None
        self.mat_hat_sq_sum = None
        
        # 新增：累积噪声方差 (1/tau)
        self.noise_var_sum = 0.0

    def _init_accumulators(self, dim1, dim2):
        self.mat_hat_sum = np.zeros((dim1, dim2))
        self.mat_hat_sq_sum = np.zeros((dim1, dim2))
        self.noise_var_sum = 0.0

    def _accumulate(self, tensor_hat_2d, tau):
        if self.mat_hat_sum is None:
            self._init_accumulators(tensor_hat_2d.shape[0], tensor_hat_2d.shape[1])
        
        # 1. 累积模型预测值 (用于计算 Epistemic Uncertainty)
        self.mat_hat_sum += tensor_hat_2d
        self.mat_hat_sq_sum += tensor_hat_2d**2
        
        # 2. 累积观测噪声方差 (用于计算 Aleatoric Uncertainty)
        # sigma^2 = 1 / tau
        self.noise_var_sum += (1.0 / tau)

    def _finalize_uq(self, gibbs_iter):
        # 1. Epistemic Variance (模型参数带来的不确定性)
        mat_hat_mean = self.mat_hat_sum / gibbs_iter
        mat_hat_var_epistemic = (self.mat_hat_sq_sum / gibbs_iter) - (mat_hat_mean**2)
        mat_hat_var_epistemic = np.maximum(mat_hat_var_epistemic, 0)
        
        # 2. Aleatoric Variance (数据固有的观测噪声)
        # Average noise variance
        avg_noise_var = self.noise_var_sum / gibbs_iter
        
        # 3. Total Standard Deviation
        # Std = sqrt( Var_model + Var_noise )
        mat_hat_std = np.sqrt(mat_hat_var_epistemic + avg_noise_var)

        
        return mat_hat_mean.reshape(self.dims), mat_hat_std.reshape(self.dims)

    def _kernel_rbf(self, x1, x2, length_scale, variance=1.0):
        sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        K = variance * np.exp(-0.5 * sqdist / (length_scale**2))
        return K + 1e-6 * np.eye(len(x1)) 
    

    def _kernel_matern32(self, x1, x2, length_scale, variance=1.0):
        # Matérn 3/2 Kernel: k(r) = var * (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)
        # 这种核允许数据不那么平滑，非常适合气象/环境数据
        dist = np.sqrt(np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T))
        # 避免除以0
        dist = np.maximum(dist, 1e-10) 
        
        sqrt3_d_l = np.sqrt(3.0) * dist / length_scale
        K = variance * (1.0 + sqrt3_d_l) * np.exp(-sqrt3_d_l)
        return K + 1e-6 * np.eye(len(x1))
    
    
    def _compute_prior_precisions(self):
        precisions = []
        for mode in range(len(self.dims)):
            n = self.dims[mode]
            scale = self.length_scale[mode]
            coords = np.arange(n).reshape(-1, 1)
            
            # --- 修改开始 ---
            if mode == 2:  # Time Dimension
                print(f"Mode {mode} (Time): Using Matérn-3/2 Prior (Physics-aware)")
                # 使用 Matérn 核捕捉时序相关性
                K = self._kernel_matern32(coords, coords, length_scale=scale) 
            else:          # Spatial Dimensions (Lat, Lon)
                K = self._kernel_rbf(coords, coords, length_scale=scale)
            # --- 修改结束 ---
            
            try:
                precisions.append(inv(K))
            except np.linalg.LinAlgError:
                precisions.append(inv(K + 1e-4 * np.eye(n)))
        
        return precisions

    def _ten2mat(self, tensor, mode):
        return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')

    def train(self, sparse_mat_2d, burn_iter, gibbs_iter, verbose_step=50, **kwargs):
        print(f"Initializing GP-CP (Rank={self.rank})...")
        print(f"Lengthscales (Lat, Lon, Time): {self.length_scale}")
        
        # Prepare Data
        sparse_tensor = sparse_mat_2d.reshape(self.dims)
        mask = (sparse_tensor != 0)
        mean_val = np.sum(sparse_tensor) / np.sum(mask)
        dense_tensor = sparse_tensor.copy()
        dense_tensor[~mask] = mean_val


        # Init Factors
        U = [0.1 * np.random.randn(d, self.rank) for d in self.dims]
        K_invs = self._compute_prior_precisions()
        tau = 1.0 
        
        self._init_accumulators(sparse_mat_2d.shape[0], sparse_mat_2d.shape[1])
        start_time = time.time()

        for it in range(burn_iter + gibbs_iter):
            # --- Sample Factors ---
            for k in range(len(self.dims)):
                dim_k = self.dims[k]
                Y_k = self._ten2mat(dense_tensor, k)
                
                idx_others = [i for i in range(len(self.dims)) if i != k]
                V = U[idx_others[-1]]
                for i in range(len(idx_others)-2, -1, -1):
                    V = kr_prod(V, U[idx_others[i]])
                
                VTV = np.ones((self.rank, self.rank))
                for i in idx_others:
                    VTV *= (U[i].T @ U[i])
                
                Proj = Y_k @ V 
                Prior_Prec = K_invs[k] 

                for r in range(self.rank):
                    interaction = (U[k] @ VTV[:, r]) - (U[k][:, r] * VTV[r, r])
                    lambda_lik = tau * VTV[r, r]
                    
                    Post_Prec = Prior_Prec.copy()
                    np.fill_diagonal(Post_Prec, Post_Prec.diagonal() + lambda_lik)
                    target = tau * (Proj[:, r] - interaction)
                    
                    try:
                        L = cholesky(Post_Prec)
                        mu = solve(L.T, solve(L, target))
                        z = np.random.randn(dim_k)
                        U[k][:, r] = mu + solve(L.T, z)
                    except np.linalg.LinAlgError:
                        U[k][:, r] = solve(Post_Prec, target)

            # --- Reconstruct ---
            # V_rec = U[2]
            # V_rec = kr_prod(V_rec, U[1])
            # tensor_mat = U[0] @ V_rec.T
            # tensor_hat = tensor_mat.reshape(self.dims, order='F')

            tensor_hat = np.einsum('ir,jr,kr->ijk', U[0], U[1], U[2])
            dense_tensor[~mask] = tensor_hat[~mask]
            
            # --- Sample tau (Noise Precision) ---
            error = dense_tensor[mask] - tensor_hat[mask]
            sse = np.sum(error ** 2)
            # Sample tau from posterior Gamma
            tau = np.random.gamma(1e-6 + 0.5 * np.sum(mask), 1.0 / (1e-6 + 0.5 * sse))

            # --- Accumulate (传递 tau) ---
            if it >= burn_iter:
                mat_hat_2d = tensor_hat.reshape(sparse_mat_2d.shape)
                self._accumulate(mat_hat_2d, tau) # <--- 关键修改：传入 tau

            if (it + 1) % verbose_step == 0:
                print(f"[GP-CP] Iter {it+1} | Time: {time.time()-start_time:.1f}s | Tau: {tau:.4f}")

        return self._finalize_uq(gibbs_iter)


    """
    Gaussian Process CP Decomposition.
    Epistemic Uncertainty Only (No Aleatoric Noise).
    Uses Matérn-3/2 kernel for Time dimension to capture non-smooth dynamics.
    """
    def __init__(self, rank, dims, length_scale=[3.0, 3.0, 1.5], **kwargs):
        self.rank = rank
        self.dims = dims
        self.length_scale = length_scale
        
        # Accumulators
        self.mat_hat_sum = None
        self.mat_hat_sq_sum = None
        
        # [Removed] self.noise_var_sum = 0.0

    def _init_accumulators(self, dim1, dim2):
        self.mat_hat_sum = np.zeros((dim1, dim2))
        self.mat_hat_sq_sum = np.zeros((dim1, dim2))

    def _accumulate(self, tensor_hat_2d):
        # [Modified] No tau argument needed
        if self.mat_hat_sum is None:
            self._init_accumulators(tensor_hat_2d.shape[0], tensor_hat_2d.shape[1])
        
        # Only accumulate model predictions
        self.mat_hat_sum += tensor_hat_2d
        self.mat_hat_sq_sum += tensor_hat_2d**2

    def _finalize_uq(self, gibbs_iter):
        # 1. Epistemic Variance (Variance of the mean prediction)
        mat_hat_mean = self.mat_hat_sum / gibbs_iter
        mat_hat_var_epistemic = (self.mat_hat_sq_sum / gibbs_iter) - (mat_hat_mean**2)
        mat_hat_var_epistemic = np.maximum(mat_hat_var_epistemic, 0)
        
        # [Modified] Total Std = sqrt(Var_model_only)
        mat_hat_std = np.sqrt(mat_hat_var_epistemic)
        
        return mat_hat_mean.reshape(self.dims), mat_hat_std.reshape(self.dims)

    def _kernel_rbf(self, x1, x2, length_scale, variance=1.0):
        sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        K = variance * np.exp(-0.5 * sqdist / (length_scale**2))
        return K + 1e-6 * np.eye(len(x1)) 
    
    def _kernel_matern32(self, x1, x2, length_scale, variance=1.0):
        dist = np.sqrt(np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T))
        dist = np.maximum(dist, 1e-10) 
        sqrt3_d_l = np.sqrt(3.0) * dist / length_scale
        K = variance * (1.0 + sqrt3_d_l) * np.exp(-sqrt3_d_l)
        return K + 1e-6 * np.eye(len(x1))
    
    def _compute_prior_precisions(self):
        precisions = []
        for mode in range(len(self.dims)):
            n = self.dims[mode]
            scale = self.length_scale[mode]
            coords = np.arange(n).reshape(-1, 1)
            
            if mode == 2:  # Time Dimension
                print(f"Mode {mode} (Time): Using Matérn-3/2 Prior")
                K = self._kernel_matern32(coords, coords, length_scale=scale) 
            else:          # Spatial Dimensions
                K = self._kernel_rbf(coords, coords, length_scale=scale)
            
            try:
                precisions.append(inv(K))
            except np.linalg.LinAlgError:
                precisions.append(inv(K + 1e-4 * np.eye(n)))
        return precisions

    def _ten2mat(self, tensor, mode):
        return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')

    def train(self, sparse_mat_2d, burn_iter, gibbs_iter, verbose_step=50, **kwargs):
        print(f"Initializing GP-CP (Epistemic Only, Rank={self.rank})...")
        print(f"Lengthscales: {self.length_scale}")
        
        sparse_tensor = sparse_mat_2d.reshape(self.dims)
        mask = (sparse_tensor != 0)
        mean_val = np.sum(sparse_tensor) / np.sum(mask)
        dense_tensor = sparse_tensor.copy()
        dense_tensor[~mask] = mean_val

        U = [0.1 * np.random.randn(d, self.rank) for d in self.dims]
        K_invs = self._compute_prior_precisions()
        tau = 1.0 
        
        self._init_accumulators(sparse_mat_2d.shape[0], sparse_mat_2d.shape[1])
        start_time = time.time()

        for it in range(burn_iter + gibbs_iter):
            # --- Sample Factors ---
            for k in range(len(self.dims)):
                dim_k = self.dims[k]
                Y_k = self._ten2mat(dense_tensor, k)
                
                idx_others = [i for i in range(len(self.dims)) if i != k]
                # V calc order matching matricization
                # If mode 0: V = U2 kr U1 ?? No.
                # ten2mat(0) -> U0 @ (U2 kr U1).T if dims are 0,1,2
                # Let's stick to standard loop construction
                V = U[idx_others[-1]]
                for i in range(len(idx_others)-2, -1, -1):
                    V = kr_prod(V, U[idx_others[i]])
                
                VTV = np.ones((self.rank, self.rank))
                for i in idx_others:
                    VTV *= (U[i].T @ U[i])
                
                Proj = Y_k @ V 
                Prior_Prec = K_invs[k] 

                for r in range(self.rank):
                    interaction = (U[k] @ VTV[:, r]) - (U[k][:, r] * VTV[r, r])
                    lambda_lik = tau * VTV[r, r]
                    
                    Post_Prec = Prior_Prec.copy()
                    np.fill_diagonal(Post_Prec, Post_Prec.diagonal() + lambda_lik)
                    target = tau * (Proj[:, r] - interaction)
                    
                    try:
                        L = cholesky(Post_Prec)
                        mu = solve(L.T, solve(L, target))
                        z = np.random.randn(dim_k)
                        U[k][:, r] = mu + solve(L.T, z)
                    except np.linalg.LinAlgError:
                        U[k][:, r] = solve(Post_Prec, target)

            # --- Reconstruct ---
            tensor_hat = np.einsum('ir,jr,kr->ijk', U[0], U[1], U[2])
            dense_tensor[~mask] = tensor_hat[~mask]
            
            # --- Sample tau ---
            error = dense_tensor[mask] - tensor_hat[mask]
            sse = np.sum(error ** 2)
            tau = np.random.gamma(1e-6 + 0.5 * np.sum(mask), 1.0 / (1e-6 + 0.5 * sse))

            # --- Accumulate ---
            if it >= burn_iter:
                mat_hat_2d = tensor_hat.reshape(sparse_mat_2d.shape)
                # [Modified] No tau passed
                self._accumulate(mat_hat_2d)

            if (it + 1) % verbose_step == 0:
                print(f"[GP-CP] Iter {it+1} | Time: {time.time()-start_time:.1f}s | Tau: {tau:.4f}")

        return self._finalize_uq(gibbs_iter)