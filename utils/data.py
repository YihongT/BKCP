import numpy as np
import scipy.io as sio


def load_and_prepare_data(
    mat_path,
    val_ratio=0.1,
    seed=0
):
    """
    Load MODIS LST tensor data and automatically split into train / val / test.

    Args:
        mat_path: Path to .mat file (containing training_tensor, test_tensor)
        val_ratio: Ratio of validation samples taken from observed entries in training tensor
        seed: Random seed

    Returns:
        train_tensor: Training tensor (validation and missing positions set to 0)
        val_tensor: Validation tensor (only validation positions kept, others set to 0)
        test_tensor: Test tensor (original test values)
        mask_train: Boolean mask of training usable positions (train != 0 and not chosen for validation)
        mask_val: Boolean mask for validation positions
        mask_test: Boolean mask for test positions (test_tensor != 0)
        original_missing_mask: Positions missing in both train and test (real missing, e.g., clouds)
        test_region_mask: Positions where train == 0 but test != 0 (artificial occlusions)
    """
    np.random.seed(seed)

    # ------------------------
    # 1. Load data
    # ------------------------
    mat = sio.loadmat(mat_path)
    train = mat['training_tensor'].astype(float)
    test = mat['test_tensor'].astype(float)

    # ------------------------
    # 2. Construct masks
    # ------------------------
    mask_observed = (train != 0)
    mask_missing = (train == 0)
    mask_test = (test != 0)

    # Real missing values (missing in both train and test)
    original_missing_mask = (train == 0) & (test == 0)

    # Artificially masked test region (train has 0 but test has valid values)
    test_region_mask = (train == 0) & (test != 0)

    # ------------------------
    # 3. Select validation samples from observed entries
    # ------------------------
    obs_idx = np.where(mask_observed)
    N = len(obs_idx[0])

    num_val = int(N * val_ratio)
    perm = np.random.permutation(N)
    val_sel = perm[:num_val]

    mask_val = np.zeros_like(train, dtype=bool)
    mask_val[
        obs_idx[0][val_sel],
        obs_idx[1][val_sel],
        obs_idx[2][val_sel]
    ] = True

    # ------------------------
    # 4. Build the new training tensor (set validation positions to 0)
    # ------------------------
    train_new = train.copy()
    train_new[mask_val] = 0

    # ------------------------
    # 5. Training mask = observed minus validation
    # ------------------------
    mask_train = (train_new != 0)

    # ------------------------
    # 6. Construct validation tensor (only validation positions kept)
    # ------------------------
    val_tensor = np.zeros_like(train)
    val_tensor[mask_val] = train[mask_val]

    print("========= Data Summary =========")
    print("Train observed count:", np.sum(mask_train))
    print("Val count:", np.sum(mask_val))
    print("Test count:", np.sum(mask_test))
    print("Original missing:", np.sum(original_missing_mask))
    print("Test region (train=0,test!=0):", np.sum(test_region_mask))
    print("Train tensor:", train_new.shape)
    print("Val tensor:", val_tensor.shape)
    print("Test tensor:", test.shape)
    print("================================")

    return {
        "train_tensor": train_new,
        "val_tensor": val_tensor,
        "test_tensor": test,
        "mask_train": mask_train,
        "mask_val": mask_val,
        "mask_test": mask_test,
        "original_missing_mask": original_missing_mask,
        "test_region_mask": test_region_mask
    }
