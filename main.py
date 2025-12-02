import os
import sys
import yaml
import json
import time
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from utils.data import load_and_prepare_data
from models.models import (
    BTMF, BTRMF, PMF, ProbKNN, MVN_EM, PPCA, BKCP
)
from models.nn import Prob3DMAE 
from utils.viz import run_visualization

from utils.eval import compute_metrics 

RESULTS_DIR = "results"

MODEL_MAP = {
    "ProbKNN": ProbKNN,
    "MVN_EM": MVN_EM,
    "PPCA": PPCA,

    # Matrix Factorization Baselines
    "BTMF": BTMF,
    "BTRMF": BTRMF,
    "PMF": PMF,
    
    # Neural Networks
    "Prob3DMAE": Prob3DMAE,
    
    # Ours
    "BKCP": BKCP,
}

def main(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load Config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 2. Load Data
    data_dict = load_and_prepare_data(
        cfg['data']['path'], 
        val_ratio=cfg['data']['val_ratio'],
        seed=cfg['data'].get('seed', 1000)
    )
    
    train_tensor_3d = data_dict['train_tensor']
    dims = train_tensor_3d.shape # (Lat, Lon, Time)
    sparse_mat_2d = train_tensor_3d.reshape([dims[0] * dims[1], dims[2]])

    # print(f'train_tensor_3d: {train_tensor_3d.shape}')
    # print(f'sparse_mat_2d: {sparse_mat_2d.shape}')


    # ==========================================
    # Mean Centering
    # ==========================================
    print("Applying Mean Centering to remove static background...")
    
    mask_2d = (sparse_mat_2d != 0)
    row_sum = np.sum(sparse_mat_2d, axis=1)
    row_count = np.sum(mask_2d, axis=1)
    row_count[row_count == 0] = 1 
    row_mean = row_sum / row_count
    
    sparse_mat_centered = sparse_mat_2d.copy()
    row_mean_expanded = row_mean[:, None] 
    np.putmask(sparse_mat_centered, mask_2d, sparse_mat_2d - row_mean_expanded)

    sparse_mat_val_centered = None
    if args.model in ['MLP', 'Prob3DMAE', 'ProbImputeFormer']:
        val_tensor_3d = data_dict['val_tensor']
        sparse_mat_val_2d = val_tensor_3d.reshape([dims[0] * dims[1], dims[2]])
        mask_val_2d = (sparse_mat_val_2d != 0)
        
        sparse_mat_val_centered = sparse_mat_val_2d.copy()
        np.putmask(sparse_mat_val_centered, mask_val_2d, sparse_mat_val_2d - row_mean_expanded)
        print("Prepared centered validation set for Neural Net early stopping.")

    # --- Ground Truth Visualization (Optional) ---
    if args.viz_gt:
        print("Visualizing Data Splits...")
        run_visualization(data_dict['train_tensor'], "DataSplit", "Train Set (Observed)", RESULTS_DIR, mask_zeros=True)
        run_visualization(data_dict['test_tensor'], "DataSplit", "Test Set (Masked)", RESULTS_DIR, mask_zeros=True)
        
        gt_tensor = data_dict['train_tensor'] + data_dict['test_tensor'] + data_dict['val_tensor']
        run_visualization(gt_tensor, "DataSplit", "Full Ground Truth", RESULTS_DIR, mask_zeros=True)
        
        # gt_centered = gt_tensor.reshape(-1, dims[2]) - row_mean_expanded
        # run_visualization(gt_centered.reshape(dims), "DataSplit", "Centered GT", RESULTS_DIR)
        
        print("Visualization done. Exiting.")
        sys.exit(0)

    # --- Model Selection & Init ---
    model_name = args.model
    if model_name not in MODEL_MAP:
        print(f"Error: Unknown model {model_name}")
        sys.exit(1)
    if model_name not in cfg:
        print(f"Error: Config for {model_name} not found.")
        sys.exit(1)

    print(f"=== Experiment: {model_name} ===")
    model_params = cfg[model_name]
    model_class = MODEL_MAP[model_name]
    
    init_kwargs = {k: v for k, v in model_params.items() if k not in ['rank', 'time_lags']}
    
    if model_name in ['Prob3DMAE', 'ProbImputeFormer']:
        model = model_class(dims=dims, **init_kwargs)
        
    elif 'time_lags' in model_params: # BTMF, TRMF, BTRMF
        model = model_class(rank=model_params['rank'], time_lags=np.array(model_params['time_lags']), dims=dims, **init_kwargs)
        
    else: # PMF, BayesianCP, GP_CP
        model = model_class(rank=model_params['rank'], dims=dims, **init_kwargs)
    
    # --- Train ---
    train_kwargs = {}
    
    if model_name in ['MLP', 'Prob3DMAE', 'ProbImputeFormer']:
        train_kwargs['sparse_mat_val'] = sparse_mat_val_centered 
    
    else:
        train_kwargs['burn_iter'] = int(cfg['train'].get('burn_in', 100))
        train_kwargs['gibbs_iter'] = int(cfg['train'].get('gibbs_iter', 50))
        train_kwargs['tensor_max_iter'] = int(cfg['train'].get('tensor_max_iter', 100))
        train_kwargs['tensor_tol'] = float(cfg['train'].get('tensor_tol', 1e-4))
    
    start_time = time.time()
    tensor_mean_centered, tensor_std = model.train(
        sparse_mat_centered, 
        verbose_step=int(cfg['train'].get('verbose_step', 10)),
        **train_kwargs
    )
    end_time = time.time()
    
    # --- Restore Mean ---
    print("Restoring Mean...")
    
    mat_mean_centered = tensor_mean_centered.reshape([dims[0] * dims[1], dims[2]])
    mat_mean_final = mat_mean_centered + row_mean_expanded
    tensor_mean = mat_mean_final.reshape(dims)
    if tensor_std is not None:
        tensor_std = tensor_std.reshape(dims)

    # --- Evaluation ---
    print("\n>>> Evaluation...")
    tensor_mean[tensor_mean < 0] = 0 
    
    # 1. Validation Set
    metrics_val = compute_metrics(data_dict['val_tensor'], tensor_mean, tensor_std, data_dict['mask_val'])
    print(f"Val  | RMSE: {metrics_val['RMSE']:.4f}, CRPS: {metrics_val['CRPS']:.4f}, PICP: {metrics_val['PICP']:.4f}, NLL: {metrics_val['NLL']:.4f}")
    
    # 2. Test Set
    metrics_test = compute_metrics(data_dict['test_tensor'], tensor_mean, tensor_std, data_dict['mask_test'])
    print(f"Test | RMSE: {metrics_test['RMSE']:.4f}, CRPS: {metrics_test['CRPS']:.4f}, PICP: {metrics_test['PICP']:.4f}, NLL: {metrics_test['NLL']:.4f}")

    # Save Metrics to JSON
    metrics = {
        "model": args.model,
        "validation": metrics_val,
        "test": metrics_test,
        "time": end_time - start_time
    }
    json_path = os.path.join(RESULTS_DIR, f"{model_name}_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {json_path}")

    # ==========================================
    # Visualization
    # ==========================================
    print("\n>>> Visualization...")
    
    run_visualization(tensor_mean, model_name, "Prediction Mean", RESULTS_DIR)
    if tensor_std is not None and np.max(tensor_std) > 1e-5:
        run_visualization(tensor_std, model_name, "Uncertainty StdDev", RESULTS_DIR, is_std=True, custom_range=(None, None))
    
    # 3. Error Map Visualization (Residuals)
    print("Generating Error Maps...")
    gt_full = data_dict['train_tensor'] + data_dict['test_tensor'] + data_dict['val_tensor']
    mask_full = (gt_full != 0)
    
    error_tensor = tensor_mean - gt_full
    error_tensor[~mask_full] = 0 
    
    ERROR_RANGE = (-15, 15) 
    
    run_visualization(
        error_tensor, 
        model_name, 
        "Prediction Error", 
        RESULTS_DIR, 
        mask_zeros=True,
        custom_range=ERROR_RANGE
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Spatiotemporal Imputation Models")
    
    parser.add_argument("--model", type=str, default='BTMF', help="Choose model to run")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--viz_gt", action="store_true", help="Visualize ground truth data and exit")
    parser.add_argument("--viz_data", action="store_true", help="Visualize train/test split only") # Alias for viz_gt logic if needed
    
    args = parser.parse_args()
    
    if args.viz_gt:
        pass
    else:
        pass
        
    main(args)