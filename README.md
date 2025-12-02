# Bayesian Kernelized CP for Spatiotemporal Imputation

This project implements a probabilistic tensor factorization framework, **Bayesian Kernelized CP (BKCP)**, for spatiotemporal data imputation. It is designed to reconstruct missing values in environmental datasets (e.g., MODIS Land Surface Temperature) while providing calibrated uncertainty estimates.

The repository includes a suite of state-of-the-art baselines ranging from matrix factorization to deep learning models, along with comprehensive evaluation metrics for both deterministic accuracy and probabilistic calibration.

-----

## 🚀 Key Features

  * **State-of-the-Art Model**: Implementation of **BKCP** (Bayesian Kernelized CP Decomposition), which combines the structural advantages of tensor decomposition with the flexibility of Gaussian Process priors.
  * **Comprehensive Baselines**: Includes implementations of 7+ baseline models (Matrix Factorization, Tensor Decomposition, Deep Learning).
  * **Probabilistic Evaluation**: Beyond RMSE, we evaluate using rigorous probabilistic metrics like CRPS, PICP, and Interval Score.
  * **Visualization**: Tools for visualizing 3D spatiotemporal slices, error maps, and uncertainty distributions.

-----

## 📊 Evaluation Metrics

The framework supports a wide range of metrics to evaluate both point estimation accuracy and uncertainty quantification quality:

**Deterministic Metrics (Accuracy):**

  * **RMSE** (Root Mean Square Error)
  * **MAE** (Mean Absolute Error)
  * **MAPE** (Mean Absolute Percentage Error)
  * **R²** (Coefficient of Determination)
  * **Bias** (Mean Forecast Error)

**Probabilistic Metrics (Uncertainty):**

  * **CRPS** (Continuous Ranked Probability Score)
  * **NLL** (Negative Log-Likelihood)
  * **PICP** (Prediction Interval Coverage Probability)
  * **Interval Score** (Winkler Score)

-----

## 📦 Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/YihongT/BKCP
    cd BKCP
    ```

2.  Create a virtual environment (optional but recommended):

    ```bash
    conda create -n bkcp python=3.8
    conda activate bkcp
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

-----

## 🧪 Usage

### 1\. Data Preparation

Ensure your data files are in the root directory. The project expects `.mat` files:

  * `MODIS_Aug.mat`: Should contain `training_tensor` and `test_tensor`.

**Data Format:**

  * **Dimensions:** $100 \times 200 \times 31$ (Latitude $\times$ Longitude $\times$ Day).
  * **Unit:** Kelvin.
  * **Missing Values:** Represented as `0`.

### 2\. Running Models

You can run any implemented model using the `main.py` script. The configuration for hyperparameters is handled via `config.yaml`.

**Run the Proposed Model (BKCP):**

```bash
python main.py --model BKCP
```

**Run Baselines:**

```bash
python main.py --model PMF          # Probabilistic Matrix Factorization
python main.py --model BTMF         # Bayesian Temporal MF
python main.py --model BTRMF        # Bayesian Temporal Regularized MF
python main.py --model TRMF         # Temporal Regularized MF
python main.py --model PPCA         # Probabilistic PCA
python main.py --model MVN_EM       # Multivariate Normal EM
python main.py --model ProbKNN      # Probabilistic KNN
python main.py --model Prob3DMAE    # Probabilistic 3D Masked Autoencoder
```

### 3\. Visualization

To visualize the Ground Truth data splits (Train/Validation/Test) or check data quality:

```bash
python main.py --viz_gt
```

-----

## 🤖 Supported Models (Baselines)

| Category | Model | Description |
| :--- | :--- | :--- |
| **Proposed** | **BKCP** | **Bayesian Kernelized CP Decomposition**. Combines CP with mode-specific GP priors (RBF/Matérn) for SOTA accuracy and uncertainty. |
| **Matrix Factorization** | **PMF** | Probabilistic Matrix Factorization with Gaussian priors. |
| | **BTMF** | Bayesian Temporal Matrix Factorization (Vector Autoregressive priors). |
| | **BTRMF** | Bayesian Temporal Regularized MF (Element-wise AR priors). |
| | **TRMF** | Temporal Regularized MF (Regularized optimization). |
| **Statistical / Classical** | **PPCA** | Probabilistic PCA solved via EM algorithm. |
| | **MVN\_EM** | Multivariate Normal imputation (Time-only covariance). |
| | **ProbKNN** | Probabilistic K-Nearest Neighbors (Local weighted averaging). |
| **Deep Learning** | **Prob3DMAE** | Probabilistic 3D Masked Autoencoder (3D CNN + Self-supervised Masking). |

-----

## 📁 Project Structure

```
./
├── MODIS_Aug.mat          # Dataset
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── config.yaml            # Hyperparameter configuration
├── main.py                # Main entry point for training and evaluation
├── data_analysis.py       # Scripts for preliminary data analysis (SVD, Correlation)
├── readout.py             # Helper to parse results from logs
├── models/
│   ├── models.py          # Classical and Bayesian model implementations (BKCP, BTMF, etc.)
│   └── nn.py              # Neural network architectures (Prob3DMAE)
├── utils/
│   ├── data.py            # Data loading and preprocessing
│   ├── eval.py            # Evaluation metrics implementation
│   └── viz.py             # Visualization utilities
└── results/               # Directory for saved metrics and plots
```

-----

## 📜 License

This project is licensed under the MIT License.