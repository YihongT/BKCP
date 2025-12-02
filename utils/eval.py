import numpy as np
from scipy.stats import norm


def compute_metrics(y_true, y_pred_mean, y_pred_std, mask):
    """
    Compute a full set of deterministic and probabilistic (UQ) evaluation metrics.

    Deterministic:
        - RMSE: Root Mean Squared Error (sensitive to large errors)
        - MAE:  Mean Absolute Error (average absolute deviation)
        - MAPE: Mean Absolute Percentage Error (relative error)
        - R2:   Coefficient of Determination (goodness of fit)
        - Bias: Mean Forecast Error (systematic error; >0 overestimation, <0 underestimation)

    Probabilistic (Uncertainty Quantification):
        - CRPS: Continuous Ranked Probability Score (distribution-aware scoring)
        - NLL:  Negative Log Likelihood (fit of probabilistic model)
        - PICP: Prediction Interval Coverage Probability (ideally ~0.95 for 95% CI)
        - IS:   Interval Score (penalizes both width and coverage; lower is better)
    """
    if np.sum(mask) == 0:
        return {}

    # Select masked values
    true_vals = y_true[mask]
    pred_mean = y_pred_mean[mask]

    # ==========================================
    # 1. Deterministic Metrics
    # ==========================================

    # RMSE
    rmse = np.sqrt(np.mean((true_vals - pred_mean) ** 2))

    # MAE
    mae = np.mean(np.abs(true_vals - pred_mean))

    # Bias (pred − true)
    bias = np.mean(pred_mean - true_vals)

    # MAPE (ignore zero true values)
    non_zero = true_vals != 0
    if np.sum(non_zero) > 0:
        mape = np.mean(
            np.abs((true_vals[non_zero] - pred_mean[non_zero]) / true_vals[non_zero])
        )
    else:
        mape = np.nan

    # R2 Score
    ss_res = np.sum((true_vals - pred_mean) ** 2)
    ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))

    # ==========================================
    # 2. Probabilistic Metrics
    # ==========================================

    # Deterministic model: no std -> return deterministic metrics + degenerate CRPS
    if y_pred_std is None or np.all(y_pred_std[mask] < 1e-9):
        crps = mae  # Degenerate CRPS becomes MAE
        return {
            "RMSE": rmse, "MAE": mae, "Bias": bias, "MAPE": mape, "R2": r2,
            "CRPS": crps, "NLL": np.nan, "PICP": np.nan,
            "MPIW": np.nan, "IS": np.nan
        }

    pred_std = y_pred_std[mask]

    # A. Gaussian CRPS
    z_score = (true_vals - pred_mean) / pred_std
    pdf = norm.pdf(z_score)
    cdf = norm.cdf(z_score)
    crps = pred_std * (
        z_score * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi)
    )
    mean_crps = np.mean(crps)

    # B. Gaussian NLL
    eps = 1e-6
    nll = 0.5 * np.log(2 * np.pi * pred_std**2 + eps) + \
          (true_vals - pred_mean)**2 / (2 * pred_std**2 + eps)
    mean_nll = np.mean(nll)

    # C. 95% Prediction Interval Metrics
    alpha = 0.05
    z = 1.96  # z-score for 95% interval

    lower = pred_mean - z * pred_std
    upper = pred_mean + z * pred_std

    # PICP: Coverage probability
    hits = (true_vals >= lower) & (true_vals <= upper)
    picp = np.mean(hits)

    # D. Interval Score (IS)
    width = upper - lower
    penalty_lower = (2 / alpha) * (lower - true_vals) * (true_vals < lower)
    penalty_upper = (2 / alpha) * (true_vals - upper) * (true_vals > upper)
    interval_score = np.mean(width + penalty_lower + penalty_upper)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "MAPE": mape,
        "R2": r2,
        "CRPS": mean_crps,
        "NLL": mean_nll,
        "PICP": picp,
        "IS": interval_score
    }
