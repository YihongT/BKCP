import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import copy
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# Distribution Heads (Encapsulated Probabilistic Outputs)
# =============================================================================

class DistributionHead(nn.Module):
    """Base class for probabilistic distribution heads."""
    def __init__(self, in_channels):
        super().__init__()
    
    def forward(self, x):
        """Return distribution parameters."""
        raise NotImplementedError
    
    def nll(self, target, *params):
        """Compute negative log-likelihood."""
        raise NotImplementedError
        
    def get_mean_std(self, *params):
        """Convert distribution parameters into mean & standard deviation."""
        raise NotImplementedError


# =============================================================================
# Gaussian Distribution Head
# =============================================================================

class NormalHead(DistributionHead):
    def __init__(self, in_channels):
        super().__init__(in_channels)
        self.conv_mu = nn.Conv3d(in_channels, 1, 1)
        self.conv_logvar = nn.Conv3d(in_channels, 1, 1)
        
    def forward(self, x):
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        # Clamp log-variance to avoid numerical overflow/underflow
        std = torch.exp(0.5 * torch.clamp(logvar, min=-10, max=5))
        return mu, std
    
    def nll(self, y, mu, std):
        var = std ** 2
        nll = 0.5 * torch.log(2 * np.pi * var) + 0.5 * (y - mu)**2 / var
        return nll
    
    def get_mean_std(self, mu, std):
        return mu, std


# =============================================================================
# Skew-Normal Distribution Head
# =============================================================================

class SkewNormalHead(DistributionHead):
    def __init__(self, in_channels):
        super().__init__(in_channels)
        self.conv_loc = nn.Conv3d(in_channels, 1, 1)
        self.conv_scale = nn.Conv3d(in_channels, 1, 1)
        self.conv_alpha = nn.Conv3d(in_channels, 1, 1)
        
    def forward(self, x):
        loc = self.conv_loc(x)
        scale = F.softplus(self.conv_scale(x)) + 1e-4  # Ensure positive scale
        alpha = self.conv_alpha(x)
        return loc, scale, alpha
    
    def nll(self, y, loc, scale, alpha):
        z = (y - loc) / scale
        log_phi = -0.5 * np.log(2 * np.pi) - 0.5 * z**2
        log_Phi = torch.special.log_ndtr(alpha * z)
        log_pdf = np.log(2) - torch.log(scale) + log_phi + log_Phi
        return -log_pdf
    
    def get_mean_std(self, loc, scale, alpha):
        delta = alpha / torch.sqrt(1 + alpha**2)
        const = np.sqrt(2 / np.pi)
        mean = loc + scale * delta * const
        var = (scale**2) * (1 - const**2 * delta**2)
        return mean, torch.sqrt(torch.clamp(var, min=1e-6))


# =============================================================================
# Skewed Generalized Error Distribution (SGED)
# =============================================================================

class SGEDHead(DistributionHead):
    """Skewed Generalized Error Distribution (Theodossiou, 1998)."""
    def __init__(self, in_channels):
        super().__init__(in_channels)
        self.conv_mu = nn.Conv3d(in_channels, 1, 1)
        self.conv_sigma = nn.Conv3d(in_channels, 1, 1)
        self.conv_lambda = nn.Conv3d(in_channels, 1, 1)
        self.conv_p = nn.Conv3d(in_channels, 1, 1)
        
    def forward(self, x):
        mu = self.conv_mu(x)
        sigma = F.softplus(self.conv_sigma(x)) + 1e-4
        lam = torch.tanh(self.conv_lambda(x))   # skewness ∈ (-1,1)
        p = F.softplus(self.conv_p(x)) + 0.1     # shape parameter > 0
        return mu, sigma, lam, p
    
    def nll(self, y, mu, sigma, lam, p):
        """
        SGED negative log likelihood (split-scale formulation)
        """
        u = y - mu
        sigma_L = sigma * (1 - lam)
        sigma_R = sigma * (1 + lam)

        log_K = torch.log(p) - torch.log(sigma_L + sigma_R) - torch.lgamma(1/p)

        mask_neg = (u < 0).float()
        scale_eff = sigma_L * mask_neg + sigma_R * (1 - mask_neg)
        
        term = (torch.abs(u) / scale_eff) ** p
        log_pdf = log_K - term
        return -log_pdf
    
    def get_mean_std(self, mu, sigma, lam, p):
        """
        Approximate SGED mean and variance.
        """
        g1 = torch.exp(torch.lgamma(1/p))
        g2 = torch.exp(torch.lgamma(2/p))
        g3 = torch.exp(torch.lgamma(3/p))
        
        A = g2 / g1
        mean = mu + 2 * sigma * lam * A
        
        var = (sigma**2) * (1 + 3 * lam**2) * (g3/g1) - (2 * sigma * lam * A)**2
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        return mean, std


# =============================================================================
# Johnson SU Distribution
# =============================================================================

class JohnsonSUHead(DistributionHead):
    def __init__(self, in_channels):
        super().__init__(in_channels)
        self.conv_gamma = nn.Conv3d(in_channels, 1, 1)
        self.conv_delta = nn.Conv3d(in_channels, 1, 1)
        self.conv_xi = nn.Conv3d(in_channels, 1, 1)
        self.conv_lambda = nn.Conv3d(in_channels, 1, 1)
        
    def forward(self, x):
        gamma = self.conv_gamma(x)
        delta = F.softplus(self.conv_delta(x)) + 0.1
        xi = self.conv_xi(x)
        lam = F.softplus(self.conv_lambda(x)) + 1e-4
        return gamma, delta, xi, lam
    
    def nll(self, y, gamma, delta, xi, lam):
        z = (y - xi) / lam
        r = gamma + delta * torch.asinh(z)
        log_pdf = (
            torch.log(delta) - torch.log(lam) 
            - 0.5 * torch.log(z**2 + 1)
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * r**2
        )
        return -log_pdf
    
    def get_mean_std(self, gamma, delta, xi, lam):
        """
        Closed-form Johnson SU mean and variance.
        """
        w = torch.exp(delta.pow(-2))
        mean = xi - lam * torch.sqrt(w) * torch.sinh(gamma / delta)
        var = 0.5 * (lam**2) * (w - 1) * (w * torch.cosh(2 * gamma / delta) + 1)
        return mean, torch.sqrt(torch.clamp(var, min=1e-6))


# =============================================================================
# 3D U-Net Backbone
# =============================================================================

class Prob3DUNet_Backbone(nn.Module):
    """
    3D U-Net backbone supporting various distribution heads.
    kernel_size may be:
        - int
        - tuple (kt, kh, kw)
    """
    def __init__(self, dist_type='Normal', kernel_size=(3, 3, 3)):
        super().__init__()
        
        # Normalize kernel size input
        if isinstance(kernel_size, int):
            self.k = (kernel_size, kernel_size, kernel_size)
        else:
            self.k = tuple(kernel_size)
            
        # SAME padding for each dimension
        self.pad = tuple((k - 1) // 2 for k in self.k)
        
        print(f"Prob3DUNet Backbone: Kernel={self.k}, Padding={self.pad}, Dist={dist_type}")
            
        # Encoder
        self.enc1 = self._conv_block(6, 32)      # Input: Data + Mask + 4 positional embeddings
        self.pool1 = nn.MaxPool3d((1, 2, 2))     # No pooling on time in first layer
        
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool3d((2, 2, 2))     # Pool time dimension starting here
        
        self.enc3 = self._conv_block(64, 128)
        
        # Decoder
        self.up2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        self.dec2 = self._conv_block(128 + 64, 64)
        
        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.dec1 = self._conv_block(64 + 32, 32)
        
        # Distribution head selection
        if dist_type == 'Normal':
            self.head = NormalHead(32)
        elif dist_type == 'SkewNormal':
            self.head = SkewNormalHead(32)
        elif dist_type == 'SGED':
            self.head = SGEDHead(32)
        elif dist_type == 'JohnsonSU':
            self.head = JohnsonSUHead(32)
        else:
            raise ValueError(f"Unknown distribution: {dist_type}")

    def _conv_block(self, in_c, out_c):
        """Two consecutive 3D convolutions with SAME padding."""
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=self.k, padding=self.pad),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=self.k, padding=self.pad),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        
        # Decoder
        d2 = self.up2(e3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.head(d1)


# =============================================================================
# Spatiotemporal Embeddings
# =============================================================================

def get_embeddings(T, H, W, device):
    """
    Generate sinusoidal time embeddings and linear spatial embeddings.
    
    Outputs: 4 tensors of shape [1, 1, T, H, W]
        - sin(time)
        - cos(time)
        - x-coordinate (normalized to [-1,1])
        - y-coordinate (normalized to [-1,1])
    """
    # Time embeddings
    t_range = torch.arange(T, device=device).float()
    t_norm = t_range / T * 2 * np.pi
    t_sin = torch.sin(t_norm).view(1, 1, T, 1, 1).expand(1, 1, T, H, W)
    t_cos = torch.cos(t_norm).view(1, 1, T, 1, 1).expand(1, 1, T, H, W)
    
    # Spatial embeddings
    x_range = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, 1, W).expand(1, 1, T, H, W)
    y_range = torch.linspace(-1, 1, H, device=device).view(1, 1, 1, H, 1).expand(1, 1, T, H, W)
    
    return t_sin, t_cos, x_range, y_range


# =============================================================================
# Main Wrapper: Probabilistic 3D Masked Autoencoder
# =============================================================================

class Prob3DMAE:
    """
    Probabilistic 3D Masked Autoencoder with distributional output heads.
    Supports self-supervised training via dynamic mask ratios and block masking.
    """
    def __init__(self, rank=0, dims=None, learning_rate=0.001, epochs=200, 
                 patience=20, mask_ratio=0.2, dist_type='JohnsonSU',
                 kernel_size=[3, 3, 3], **kwargs):

        self.dims = dims
        self.lr = float(learning_rate)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.dist_type = dist_type
        self.kernel_size = kernel_size
        
        if isinstance(mask_ratio, (float, int)):
            self.mask_range = [float(mask_ratio), float(mask_ratio)]
        else:
            self.mask_range = [float(mask_ratio[0]), float(mask_ratio[1])]

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Prob3DMAE ({dist_type}) initialized on {self.device}. Mask Ratio={self.mask_range}")

    # -------------------------------------------------------------------------
    # Dynamic mask generator (Mixed random + block masking)
    # -------------------------------------------------------------------------
    def _generate_mask(self, data, current_ratio):
        """
        Create a mixed masking pattern:
            50% probability → random pixel masking
            50% probability → random block masking (simulates clouds)
        """
        B, C, T, H, W = data.shape
        
        if np.random.rand() > 0.5:
            # Random pixel-wise masking
            mask = torch.rand_like(data) > current_ratio
            return mask.float()
        else:
            # Random block masking to enforce spatial reasoning
            mask = torch.ones_like(data)
            total_pixels = T * H * W
            masked_pixels = 0
            target_masked = int(total_pixels * current_ratio)
            
            for _ in range(100):
                if masked_pixels >= target_masked:
                    break
                
                t = np.random.randint(0, T)
                h_size = np.random.randint(H // 10, H // 4)
                w_size = np.random.randint(W // 10, W // 4)
                
                h_start = np.random.randint(0, H - h_size)
                w_start = np.random.randint(0, W - w_size)
                
                mask[:, :, t, h_start:h_start+h_size, w_start:w_start+w_size] = 0.0
                masked_pixels += h_size * w_size
            
            return mask


    # -------------------------------------------------------------------------
    # Training procedure
    # -------------------------------------------------------------------------
    def train(self, sparse_mat_2d, sparse_mat_val=None, verbose_step=10, **kwargs):
        """
        Main training loop for probabilistic 3D MAE.
        Input matrices are assumed to be mean-centered prior to training.
        """
        # 1. Unpack dimensions
        if len(self.dims) == 3:
            H, W, T = self.dims
        else:
            H, W = 100, 200
            T = self.dims[1]

        # Convert data to 3D tensor (T,H,W)
        tensor_data = sparse_mat_2d.reshape(H, W, T)
        
        # Format for Conv3D: [1,1,T,H,W]
        data_t = torch.FloatTensor(tensor_data).to(self.device).unsqueeze(0).unsqueeze(0)
        original_mask_t = (data_t != 0).float()
        
        _, _, T_dim, H_dim, W_dim = data_t.shape
        emb_list = get_embeddings(T_dim, H_dim, W_dim, self.device)
        embeddings = torch.cat(emb_list, dim=1)
        
        # Validation data
        val_data_t = None
        val_mask_t = None
        if sparse_mat_val is not None:
            val_tensor = sparse_mat_val.reshape(H, W, T)
            val_data_t = torch.FloatTensor(val_tensor).to(self.device).unsqueeze(0).unsqueeze(0)
            val_mask_t = (val_data_t != 0).float()

        # 2. Build model
        model = Prob3DUNet_Backbone(
            dist_type=self.dist_type,
            kernel_size=self.kernel_size
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        # 3. Training loop
        start_time = time.time()
        best_val_nll = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()

            # --- Dynamic Masking ---
            curr_ratio = np.random.uniform(*self.mask_range)
            artificial_mask = self._generate_mask(data_t, curr_ratio)
            input_mask = original_mask_t * artificial_mask
            input_data = data_t * input_mask
            
            # Model Input = data + mask + positional embeddings
            x_in = torch.cat([input_data, input_mask, embeddings], dim=1)
            
            # Forward pass
            params = model(x_in)
            nll_map = model.head.nll(data_t, *params)
            
            # Compute loss on all truly observed pixels
            loss_mask = original_mask_t
            loss = (nll_map * loss_mask).sum() / (loss_mask.sum() + 1e-6)
            
            loss.backward()
            optimizer.step()
            
            # --- Validation ---
            if val_data_t is not None:
                model.eval()
                with torch.no_grad():
                    x_val_in = torch.cat([data_t, original_mask_t, embeddings], dim=1)
                    val_params = model(x_val_in)
                    
                    val_nll_map = model.head.nll(val_data_t, *val_params)
                    val_loss = (val_nll_map * val_mask_t).sum() / (val_mask_t.sum() + 1e-6)
                    val_loss = val_loss.item()
                
                # Early stopping
                if val_loss < best_val_nll:
                    best_val_nll = val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                
                if (epoch + 1) % verbose_step == 0 or epoch == 0:
                    print(f"[Prob3DMAE-{self.dist_type}] Epoch {epoch+1}/{self.epochs} | "
                          f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | "
                          f"Time: {time.time()-start_time:.1f}s")
                
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            else:
                if (epoch + 1) % verbose_step == 0:
                    print(f"[Prob3DMAE] Epoch {epoch+1} | Loss: {loss.item():.4f}")

        # 4. Final inference
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        model.eval()
        with torch.no_grad():
            x_final = torch.cat([data_t, original_mask_t, embeddings], dim=1)
            final_params = model(x_final)
            
            mu_t, std_t = model.head.get_mean_std(*final_params)
            
            mat_hat_mean = mu_t.squeeze().cpu().numpy().reshape(-1, T)
            mat_hat_std = std_t.squeeze().cpu().numpy().reshape(-1, T)
            
        return mat_hat_mean, mat_hat_std
