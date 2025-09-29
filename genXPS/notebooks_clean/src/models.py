
import torch
import torch.nn as nn
import torch.nn.functional as F


# Component 1: The standalone MLP classifier
class MLPClassifier(nn.Module):
    """
    A simple Multi-Layer Perceptron for classification.
    """
    def __init__(self, input_features=6601, output_features=40):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=input_features, out_features=256)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer_2 = nn.Linear(in_features=256, out_features=128)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer_3 = nn.Linear(in_features=128, out_features=output_features)
        
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.activation(self.bn1(self.layer_1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.layer_2(x)))
        x = self.dropout(x)
        x = self.layer_3(x)  # Output layer (logits)
        return x


# STN layer that only learns a horizontal shift
class STN1D(nn.Module):
    """
    A Spatial Transformer for 1D signals that learns ONLY a horizontal shift.
    The scaling factor is fixed to 1.
    
    Args:
        input_size (int): Number of features (length of the spectrum).
    """
    def __init__(self, input_size):
        super(STN1D, self).__init__()
        
        # The localization network predicts one parameter: the horizontal shift 't'.
        self.localization = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1) # Output is a single value for translation
        )

        # Initialize the last layer to predict a zero shift at the start.
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.fill_(0.0)

    def forward(self, x):
        """ Applies the spatial transformation. """
        # Predict only the translation parameter 't'.
        t = self.localization(x)
        
        # The scaling factor 'a' is always fixed to 1.0 for a pure shift.
        a = torch.ones_like(t)

        # Build the 2x3 affine transformation matrix: [[1, 0, t], [0, 1, 0]]
        theta = torch.cat([a, torch.zeros_like(a), t,
                           torch.zeros_like(a), torch.ones_like(a), torch.zeros_like(a)], dim=1)
        theta = theta.view(-1, 2, 3)

        # Apply the transformation
        x_reshaped = x.unsqueeze(1).unsqueeze(2)
        grid = F.affine_grid(theta, x_reshaped.size(), align_corners=False)
        x_transformed = F.grid_sample(x_reshaped, grid, align_corners=False)
        x_transformed = x_transformed.squeeze(1).squeeze(1)
        
        return x_transformed, theta
    
class STN1D_ScaleShift(nn.Module):
    """
    A Spatial Transformer for 1D signals that learns both a horizontal
    SHIFT (translation) and a SQUEEZE (scaling).
    
    Args:
        input_size (int): Number of features (length of the spectrum).
    """
    def __init__(self, input_size):
        super(STN1D_ScaleShift, self).__init__()
        
        # --- CHANGE 1: The localization network now predicts TWO parameters. ---
        # The first will be for scaling ('a'), the second for translation ('t').
        self.localization = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 2) # <<< Outputs two values: one for scale, one for shift
        )

        # --- CHANGE 2: Initialize to the identity transform. ---
        # A bias of [1.0, 0.0] corresponds to a=1 (no scale) and t=0 (no shift).
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(torch.tensor([1.0, 0.0], dtype=torch.float))

    def forward(self, x):
        """ Applies the spatial transformation. """
        
        # --- CHANGE 3: Extract both scale and shift from the network output. ---
        theta_params = self.localization(x) # Shape: (B, 2)
        a = theta_params[:, 0].unsqueeze(1) # Scaling parameter
        t = theta_params[:, 1].unsqueeze(1) # Translation parameter

        # Build the 2x3 affine transformation matrix: [[a, 0, t], [0, 1, 0]]
        theta = torch.cat([a, torch.zeros_like(a), t,
                           torch.zeros_like(a), torch.ones_like(a), torch.zeros_like(a)], dim=1)
        theta = theta.view(-1, 2, 3)

        # Apply the transformation
        x_reshaped = x.unsqueeze(1).unsqueeze(2)
        grid = F.affine_grid(theta, x_reshaped.size(), align_corners=False)
        x_transformed = F.grid_sample(x_reshaped, grid, align_corners=False)
        x_transformed = x_transformed.squeeze(1).squeeze(1)
        
        return x_transformed, theta

# Component 3: The final model that combines the STN and MLP
class STNClassifier(nn.Module):
    """
    A model that first applies a 1D Spatial Transformer (shift-only) and 
    then passes the result through an MLP classifier.
    """
    def __init__(self, input_features=6601, output_features=40):
        super().__init__()
        # Instantiate the two components.
        self.stn = STN1D(input_size=input_features)
        self.classifier = MLPClassifier(input_features=input_features, output_features=output_features)

    def forward(self, x):
        # 1. Apply the STN to align the input spectrum.
        x_aligned, theta = self.stn(x)
        
        # 2. Pass the aligned spectrum through the classifier.
        x_out = self.classifier(x_aligned)
        
        # Return both the final output and the aligned spectrum for analysis.
        return x_out, x_aligned
    

# --- NEW: Component 4: A CNN feature extractor for 1D signals ---
class CNN1DFeatureExtractor(nn.Module):
    """
    A simple 1D Convolutional layer to act as a feature extractor.
    It maintains the same output dimension as the input.
    """
    def __init__(self, input_features):
        super().__init__()
        # Conv1d layer to find local patterns. kernel_size=3 is a common choice.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        # Adaptive pooling ensures the output length is the same as the input length.
        self.pool = nn.AdaptiveAvgPool1d(input_features)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Reshape for Conv1d: [Batch, Length] -> [Batch, Channels, Length]
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool(x)
        # Flatten back to [Batch, Length] for the MLP
        x = self.flatten(x)
        return x

# --- NEW: Component 5: The CNN model that combines the CNN and MLP ---
class CNNClassifier(nn.Module):
    """
    A model that first applies a 1D Convolutional layer and then passes the
    result through an MLP classifier. This uses the same MLP backbone as the STNClassifier.
    """
    def __init__(self, input_features=6601, output_features=40):
        super().__init__()
        # Instantiate the two components.
        self.feature_extractor = CNN1DFeatureExtractor(input_features=input_features)
        self.classifier = MLPClassifier(input_features=input_features, output_features=output_features)

    def forward(self, x):
        # 1. Apply the CNN to extract features from the input spectrum.
        x_features = self.feature_extractor(x)
        
        # 2. Pass the extracted features through the classifier.
        x_out = self.classifier(x_features)
        
        # Return the final output.
        return x_out



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
    

## older more (unfactored) classifier models:

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
    


class STN1D_ShiftOnly(nn.Module):
    """
    A Spatial Transformer for 1D signals that learns ONLY a horizontal shift.
    The scaling factor is fixed to 1.
    Args:
        input_size (int): Number of features (length of the spectrum).
    """
    def __init__(self, input_size):
        super(STN1D_ShiftOnly, self).__init__()

        # --- CHANGE 1: The localization network now predicts only ONE parameter (the shift, t). ---
        self.localization = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1) # <<< Outputs a single value for translation
        )

        # --- CHANGE 2: Initialize the last layer to predict a zero shift. ---
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.fill_(0.0) # Bias is 0.0 for no initial shift

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (B, input_size)
        Returns:
            x_transformed (Tensor): spatially transformed input, shape (B, input_size)
            theta (Tensor): the affine transformation matrices, shape (B, 2, 3)
        """
        # Predict the single affine transform parameter (translation t).
        t = self.localization(x)  # Shape: (B, 1)

        # --- CHANGE 3: Manually build the 2x3 affine matrix with scaling fixed to 1. ---
        # The scaling factor 'a' is now hardcoded to 1.0.
        a = torch.ones_like(t) # Create a tensor of ones with the same shape and device as t

        # Build the 2x3 affine transformation matrix.
        # The first row is [1, 0, t] for pure horizontal shift.
        theta = torch.cat([a, torch.zeros_like(a), t,
                           torch.zeros_like(a), torch.ones_like(a), torch.zeros_like(a)], dim=1)
        theta = theta.view(-1, 2, 3)  # shape (B, 2, 3)

        # Reshape input x for grid sampling.
        x_reshaped = x.unsqueeze(1).unsqueeze(2) # (B, 1, 1, input_size)
        
        # Apply the transformation.
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
    

# 1. Construct a model class that subclasses nn.Module
class MultiClassifierV1(nn.Module):
    def __init__(self, input_features = 6601 , output_features = 40):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=input_features, out_features=256)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer_2 = nn.Linear(in_features=256, out_features=128)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer_3 = nn.Linear(in_features=128, out_features=output_features)
        
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.activation(self.bn1(self.layer_1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.layer_2(x)))
        x = self.dropout(x)
        x = self.layer_3(x)  # Output layer (logits)
        return x



class MultiClassifierConvFinal(nn.Module):
    def __init__(self, input_features=6601, output_features=40):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(input_features)  # Ensure output stays the same size as input_features
        self.flatten = nn.Flatten()

        self.layer_1 = nn.Linear(in_features=input_features, out_features=256)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer_2 = nn.Linear(in_features=256, out_features=128)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer_3 = nn.Linear(in_features=128, out_features=output_features)

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, 6601]
        x = self.conv1(x)   # [B, 1, 6601]
        x = self.pool(x)    # keep same shape
        x = self.flatten(x) # [B, 6601]
        x = self.activation(self.bn1(self.layer_1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.layer_2(x)))
        x = self.dropout(x)
        x = self.layer_3(x)
        return x

