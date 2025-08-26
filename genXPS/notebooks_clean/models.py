
import torch
import torch.nn as nn
import torch.nn.functional as F


class STN1D(nn.Module):
    def __init__(self, input_size):
        """
        A simple Spatial Transformer for 1D signals.
        Args:
            input_size (int): Number of features (length of the spectrum).
        """
        super(STN1D, self).__init__()
        self.localization = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 2)  
        )

        # Initialize the last layer to predict the identity transform.
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(torch.tensor([1.0, 0.0], dtype=torch.float))

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (B, input_size)
        Returns:
            x_transformed (Tensor): spatially transformed input, shape (B, input_size)
            theta (Tensor): the affine transformation matrices, shape (B, 2, 3)
        """
        B, input_size = x.size()
        # Predict affine transform parameters.
        theta_params = self.localization(x)  # (B, 2)
        # Extract scale and translation for x-axis.
        a = theta_params[:, 0].unsqueeze(1)  # scaling
        t = theta_params[:, 1].unsqueeze(1)  # translation

        # Build a 2x3 affine transformation matrix.
        # We keep the second row fixed to [0, 1, 0] so that only horizontal adjustments occur.
        theta = torch.cat([a, torch.zeros_like(a), t,
                           torch.zeros_like(a), torch.ones_like(a), torch.zeros_like(a)], dim=1)
        theta = theta.view(-1, 2, 3)  # shape (B, 2, 3)

        # Reshape input x to 4D tensor: (B, C, H, W) where H=1 and W=input_size.
        x_reshaped = x.unsqueeze(1).unsqueeze(2)
        grid = F.affine_grid(theta, x_reshaped.size(), align_corners=False)
        x_transformed = F.grid_sample(x_reshaped, grid, align_corners=False)
        # Squeeze out the extra dimensions to revert back to shape (B, input_size).
        x_transformed = x_transformed.squeeze(1).squeeze(1)
        return x_transformed, theta

class MultiClassifierV2_STN(nn.Module):
    def __init__(self, input_features=6601, output_features=40):
        """
        A neural network for XPS spectra classification that first applies a Spatial Transformer
        to automatically align the spectrum, then passes the aligned input through fully connected layers.
        Args:
            input_features (int): Length of input 1D spectrum.
            output_features (int): Number of output classes.
        """
        super(MultiClassifierV2_STN, self).__init__()
        # Add the STN module.
        self.stn = STN1D(input_size=input_features)
        
        # The main classifier layers.
        self.layer_1 = nn.Linear(input_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer_2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer_3 = nn.Linear(128, output_features)
        
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # First, apply the STN to align the input spectrum.
        x_aligned, theta = self.stn(x)
        
        # Pass the aligned spectrum through the classification layers.
        x_out = self.activation(self.bn1(self.layer_1(x_aligned)))
        x_out = self.dropout(x_out)
        x_out = self.activation(self.bn2(self.layer_2(x_out)))
        x_out = self.dropout(x_out)
        x_out = self.layer_3(x_out)  # Output layer (logits)
        
        # --- CHANGE THIS LINE ---
        # Original: return x
        # New: return both the final output and the aligned spectrum
        return x_out, x_aligned
    


class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, cond_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        # Encoder 
        self.encoder_layers = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu_logvar = nn.Linear(256, 2 * latent_dim)

        #  Decoder
        self.decoder = nn.Sequential(
            # Input: latent_dim + cond_dim
            nn.Linear(latent_dim + cond_dim, 512), 
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 1024), 
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, input_dim), # Output layer
            nn.Softplus()
        )

    # encode, reparameterize, decode, forward methods
    def encode(self, x, c):
        combined_input = torch.cat([x, c], 1)
        hidden = self.encoder_layers(combined_input)
        mu_logvar = self.fc_mu_logvar(hidden)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        combined_input = torch.cat([z, c], 1)
        return self.decoder(combined_input)

    def forward(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, c)
        return recon_x, mu, log_var