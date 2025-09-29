"""
Utility functions for processing XPS spectral data and training machine learning models.
Includes functions for data augmentation, normalization, data loading, model training,
and performance evaluation.
"""

# --- 1. Imports ---
import os
import random
from pathlib import Path
import lzma
import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# --- 2. General Helper Functions ---

def create_uniform_energy_grid(start_energy, end_energy, increment=0.1):
    """
    Create a uniform energy grid with a fixed increment.
    
    Args:
        start_energy (float): Starting energy value.
        end_energy (float): Ending energy value.
        increment (float): Energy increment (default is 0.1 eV).
    
    Returns:
        np.ndarray: Uniform energy grid.
    """
    return np.arange(start_energy, end_energy + increment, increment)

def get_label_dict(path_to_file):
    """
    Extract the list of functional groups from an Excel file.
    
    Args:
        path_to_file (str or Path): Path to the .xlsx file.

    Returns:
        list: A list of functional group names.
    """
    spreadsheet_f = pd.ExcelFile(path_to_file)
    df_f = pd.read_excel(spreadsheet_f)
    return list(df_f['Functional groups'])

def fg_checker(label):
    """Converts a quantitative label array to a binary (presence/absence) list."""
    bin_fg_list = []
    for val in label:
        if float(val) == 0:
            bin_fg_list.append(0)
        else:
            bin_fg_list.append(1)
    return bin_fg_list

# --- 3. Data Augmentation and Processing Functions ---

def normalize_spectrum_by_area(spectrum):
    """
    Normalize a 1D XPS spectrum by area (integral under the curve).
    This version handles both PyTorch Tensors and NumPy arrays.
    
    Args:
        spectrum (torch.Tensor or np.ndarray): The input spectrum to normalize.
    
    Returns:
        np.ndarray: Normalized spectrum with area under the curve equal to 1.
    """
    # Convert to NumPy array if it's a PyTorch tensor
    if isinstance(spectrum, torch.Tensor):
        spectrum = spectrum.cpu().detach().numpy()  # Ensure it's on CPU and convert

    # Compute the area under the spectrum (sum of intensities)
    area = np.sum(spectrum)

    # Avoid division by zero
    if area == 0:
        return spectrum

    # Normalize the spectrum
    return spectrum / area

def apply_horizontal_shift(spectrum, max_shift):
    """
    Applies a random horizontal shift to the entire spectrum sequence.
    
    Args:
        spectrum (np.ndarray): The input spectrum.
        max_shift (int): Maximum number of indices by which the data can shift.
                         Set to zero for no shift.
    
    Returns:
        np.ndarray: The shifted spectrum.
    """
    shift = random.randint(-max_shift, max_shift)
    shifted_spectrum = np.zeros_like(spectrum)

    if shift > 0: # Shift to the right
        shifted_spectrum[shift:] = spectrum[:-shift]
    elif shift < 0: # Shift to the left
        shifted_spectrum[:shift] = spectrum[-shift:]
    else: # No shift
        shifted_spectrum = spectrum.copy()
        
    return shifted_spectrum

def correct_spectrum(spectrum, amount):
    """
    Applies a fixed horizontal shift to the spectrum, padding with zeros.

    Args:
        spectrum (np.ndarray): The input spectrum array.
        amount (int): The number of indices to shift. A positive value
                      shifts to the right, a negative value shifts to the left.

    Returns:
        np.ndarray: The shifted spectrum.
    """
    shifted_spectrum = np.zeros_like(spectrum)

    if amount > 0: # Shift to the right
        shifted_spectrum[amount:] = spectrum[:-amount]
    elif amount < 0: # Shift to the left
        shifted_spectrum[:amount] = spectrum[-amount:]
    else: # No shift
        shifted_spectrum = spectrum.copy()

    return shifted_spectrum

def gaussian_broadening(spectrum, sigma_sigma):
    """
    Applies random Gaussian broadening to a spectrum.
    
    Args:
        spectrum (np.ndarray): Input (unbroadened) spectrum.
        sigma_sigma (float): The standard deviation for the distribution 
                             determining the sigma for the Gaussian kernel.

    Returns:
        np.ndarray: Spectrum with peak broadening applied.
    """
    sigma = np.abs(np.random.normal(0, sigma_sigma)) # Half-normal distribution
    convolved_spectrum = gaussian_filter1d(spectrum, sigma)
    return convolved_spectrum

def shift_test_batch(X_test_tensor, max_shift_indices):
    """
    Applies a random horizontal shift to each spectrum in a PyTorch tensor batch.
    
    Args:
        X_test_tensor (torch.Tensor): The batch of spectra (e.g., shape [64, 6601]).
        max_shift_indices (int): The maximum shift in terms of array indices.
        
    Returns:
        torch.Tensor: A new tensor with each spectrum randomly shifted.
    """
    X_test_numpy = X_test_tensor.cpu().numpy()
    shifted_spectra_list = [apply_horizontal_shift(s, max_shift_indices) for s in X_test_numpy]
    shifted_batch_numpy = np.array(shifted_spectra_list)
    shifted_batch_tensor = torch.from_numpy(shifted_batch_numpy).float().to(X_test_tensor.device)
    return shifted_batch_tensor

def label_converter(label, label_dict):
    '''
    Converts the encoded label lists into a readable format

    '''
    fg_counts = []
    fg_count = "" 

    for i, e in enumerate(label):
        if pd.isna(e):
            e = 0
        if e != 0.0:
            fg_count = f'{e:.2f}  {label_dict[i]}'
            fg_counts.append(fg_count)

    return fg_counts

def label_normaliser(label):
    total = sum(label)
    if total == 0:
        return [0 for _ in label]  # Return an array of zeros if total is zero
    else:
        return [value / total for value in label]

# --- 4. Data Loading Pipeline ---

def preprocess_data_for_classifier(
    spectra_dir: str,
    labels_dir: str,
    max_shift: float = 0,
    max_broadening_sigma: float = 2.0,
    valence_range: float = 0, 
    energy_resolution: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 64
):
    """
    Loads spectral and label data, applies augmentations, and prepares PyTorch DataLoaders.
    
    Args:
        spectra_dir (str): Path to directory containing spectral data files.
        labels_dir (str): Path to directory containing label files.
        max_shift (float): Maximum horizontal shift in eV (default: 0).
        max_broadening_sigma (float): Maximum Gaussian broadening sigma (default: 2.0).
        valence_range (float): Range to exclude for valence electrons in eV (default: 40).
        energy_resolution (float): Energy resolution in eV per data point (default: 0.1).
        test_size (float): Proportion of data for testing (default: 0.2).
        random_state (int): Random seed for reproducibility (default: 42).
        batch_size (int): Batch size for DataLoader (default: 64).
        
    Returns:
        tuple: (train_loader, test_loader, input_features, output_features)
    """
    spectra_dir = Path(spectra_dir)
    labels_dir = Path(labels_dir)
    
    max_shift_index = int(max_shift / energy_resolution)
    valence_region_index = int(valence_range / energy_resolution)
    
    # Load and process spectra
    all_spectra = sorted(spectra_dir.glob('*'), key=lambda x: int(x.stem.split('_')[1]))
    
    spectrum_list = []
    for spectra_path in all_spectra:
        spectrum = np.load(spectra_path)
        
        shifted_spectrum = apply_horizontal_shift(spectrum, max_shift_index)
        shifted_spectrum = shifted_spectrum[valence_region_index:]
        shifted_spectrum = np.maximum(shifted_spectrum, 0)
        # --- Using the more robust, singular version of the function ---
        normalized_spectrum = normalize_spectrum_by_area(shifted_spectrum)
        broadened_spectrum = gaussian_broadening(normalized_spectrum, max_broadening_sigma)
        
        spectrum_list.append(broadened_spectrum)
    
    spectrum_array = np.array(spectrum_list)
    spectrum_tensor = torch.tensor(spectrum_array, dtype=torch.float32)
    
    # Load and process labels
    all_labels = sorted(labels_dir.glob('*'), key=lambda x: int(x.stem.split('_')[1]))
    
    label_list = [np.load(label_path) for label_path in all_labels]
    
    label_array = np.array(label_list)
    bin_label_array = np.array([fg_checker(i) for i in label_array])
    label_tensor = torch.tensor(bin_label_array, dtype=torch.float32)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        spectrum_tensor,
        label_tensor,
        test_size=test_size,
        random_state=random_state
    )
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_features = spectrum_tensor.shape[1]
    output_features = label_tensor.shape[1]
    
    return train_loader, test_loader, input_features, output_features


def preprocess_data_for_cvae(
    spectra_dir: str,
    labels_dir: str,
    max_shift: float = 0,
    energy_resolution: float = 0.1,
    test_size: float = 0.2,
    random_seed: int = 42,
    batch_size: int = 64
):
    """
    Loads and processes spectral and label data for training/testing a CVAE.
    Differs from classifier pipeline: labels are continuous and normalized,
    spectra are not Gaussian-broadened, and valence is assumed pre-excluded.

    Args:
        spectra_dir (str): Path to directory containing spectral data files.
        labels_dir (str): Path to directory containing label files.
        max_shift (float): Maximum horizontal shift in eV (default: 0).
        energy_resolution (float): Energy resolution in eV per data point (default: 0.1).
        test_size (float): Proportion of data for testing (default: 0.2).
        random_state (int): Random seed for reproducibility (default: 42).
        batch_size (int): Batch size for DataLoader (default: 64).

    Returns:
        tuple: (train_loader, test_loader, input_features, output_features)
    """
    spectra_dir = Path(spectra_dir)
    labels_dir = Path(labels_dir)

    max_shift_index = int(max_shift / energy_resolution)
    valence_region_index = 0  # valence already excluded in data generation

    # --- Load and process spectra ---
    all_spectra_paths = sorted(
        spectra_dir.glob("spectrum_*.npy"),
        key=lambda x: int(x.stem.split("_")[1])
    )
    if not all_spectra_paths:
        raise FileNotFoundError(f"No spectrum files found in {spectra_dir}")

    processed_spectrum_list = []
    for i, spectra_path in enumerate(all_spectra_paths):
        spectrum = np.load(spectra_path)
        shifted = apply_horizontal_shift(spectrum, max_shift_index)

        if valence_region_index < len(shifted):
            processed = shifted[valence_region_index:]
        else:
            processed = shifted

        processed = np.maximum(processed, 0)
        normalized = normalize_spectrum_by_area(processed)
        processed_spectrum_list.append(normalized)

    first_len = len(processed_spectrum_list[0])
    if not all(len(s) == first_len for s in processed_spectrum_list):
        raise ValueError("Processed spectra have inconsistent lengths.")

    spectrum_array = np.stack(processed_spectrum_list, axis=0)
    spectrum_tensor = torch.tensor(spectrum_array, dtype=torch.float32)

    # --- Load and process labels ---
    all_label_paths = sorted(
        labels_dir.glob("label_*.npy"),
        key=lambda x: int(x.stem.split("_")[1])
    )
    if not all_label_paths:
        raise FileNotFoundError(f"No label files found in {labels_dir}")
    if len(all_spectra_paths) != len(all_label_paths):
        raise ValueError(f"Mismatch: {len(all_spectra_paths)} spectra vs {len(all_label_paths)} labels.")

    label_list = [np.load(label_path) for label_path in all_label_paths]
    label_array = np.stack(label_list, axis=0)

    normalised_label_array = np.array([label_normaliser(lbl) for lbl in label_array])
    label_tensor = torch.tensor(normalised_label_array, dtype=torch.float32)

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        spectrum_tensor,
        label_tensor,
        test_size=test_size,
        random_state=random_seed
    )

    # --- Create datasets & loaders ---
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_features = X_train.shape[1]
    output_features = y_train.shape[1] if y_train.dim() > 1 else 1

    return train_loader, test_loader, input_features, output_features




# --- 5. Model Training & Evaluation ---

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=30, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(
    model,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    device,
    epochs=10000,
    print_freq=1,
    scheduler=None, 
    seed=42
):
    """
    Training and evaluation loop for PyTorch models.
    
    Returns:
        dict: A dictionary containing training history and the best model state.
    """
    torch.manual_seed(seed)
    early_stopping = EarlyStopping(patience=15, min_delta=0.0001)
    
    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': []
    }
    best_test_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    
    model = model.to(device)
    
    for epoch in range(epochs):
        # --- Training Step ---
        model.train()
        train_loss, train_correct, train_samples = 0, 0, 0
            
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            y_output = model(X_batch)
            y_logits = y_output[0].squeeze() if isinstance(y_output, tuple) else y_output.squeeze() # STN returns a tuple (prediction tensor, transformed spectrum tensor)
            loss = loss_fn(y_logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            y_pred = torch.round(torch.sigmoid(y_logits))
            train_loss += loss.item() * len(y_batch)
            train_correct += (y_batch == y_pred).all(axis=1).sum().item()
            train_samples += len(y_batch)

        epoch_train_loss = train_loss / train_samples
        epoch_train_acc = (train_correct / train_samples) * 100
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        
        # --- Evaluation Step ---
        model.eval()
        test_loss, test_correct, test_samples = 0, 0, 0
        
        with torch.inference_mode():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                

                test_output = model(X_batch)
                test_logits = test_output[0].squeeze() if isinstance(test_output, tuple) else test_output.squeeze() # STN returns a tuple (prediction tensor, transformed spectrum tensor)
                test_pred = torch.round(torch.sigmoid(test_logits))
                
                test_loss += loss_fn(test_logits, y_batch).item() * len(y_batch)
                test_correct += (y_batch == test_pred).all(axis=1).sum().item()
                test_samples += len(y_batch)
        
        epoch_test_loss = test_loss / test_samples
        epoch_test_acc = (test_correct / test_samples) * 100
        history['test_loss'].append(epoch_test_loss)
        history['test_acc'].append(epoch_test_acc)

        # Check for best model
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            best_model_state = model.state_dict()
            best_epoch = epoch
        
        # Early stopping check
        if epoch >= 20:
            early_stopping(epoch_test_loss)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Scheduler step
        if scheduler:
            scheduler.step(epoch_test_loss)
        
        # Print progress
        if epoch % print_freq == 0:
            print(f"Epoch: {epoch:4d} | "
                  f"Train Loss: {epoch_train_loss:.5f}, Acc: {epoch_train_acc:.2f}% | "
                  f"Test Loss: {epoch_test_loss:.5f}, Acc: {epoch_test_acc:.2f}%")
    
    print(f"\nBest performance at epoch {best_epoch} with Test Loss: {best_test_loss:.5f}")
    
    # Load best model state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)

    return history, model

def compute_class_metrics(y_true, y_pred, class_names=None):
    """
    Compute performance metrics and confusion matrices for each class.
    
    Args:
        y_true (np.ndarray): True labels (n_samples × n_classes)
        y_pred (np.ndarray): Predicted labels (n_samples × n_classes)
        class_names (list, optional): List of class names.
        
    Returns:
        tuple: (metrics_df, confusion_matrices, normalized_matrices)
    """
    n_classes = y_true.shape[1]
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]
    
    metrics_data = []
    confusion_matrices = []
    normalized_matrices = []
    
    for i, name in enumerate(class_names):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        confusion_matrices.append(cm)
        
        # Normalize by true labels (rows)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized) # Replace nan with 0
        normalized_matrices.append(cm_normalized)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score = 2 * (precision * recall) / (precision + recall)

        metrics_data.append({
            'class': name,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': np.nan_to_num(precision),
            'recall': np.nan_to_num(recall),
            'f1_score': np.nan_to_num(f1_score),
            'support': tp + fn
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    return metrics_df, confusion_matrices, normalized_matrices
