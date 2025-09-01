
import numpy as np
import torch
import random

def create_uniform_energy_grid(start_energy, end_energy, increment=0.1):
    """
    Create a uniform energy grid with a fixed increment.
    
    Args:
        start_energy: Starting energy value.
        end_energy: Ending energy value.
        increment: Energy increment (default is 0.1 eV).
    
    Returns:
        energy_range: Uniform energy grid as a NumPy array.
    """
    return np.arange(start_energy, end_energy + increment, increment)


def apply_horizontal_shift(spectrum, max_shift):
    '''
    Applies a random horizontal shift to the entire spectrum sequence.
    Args:
        spectrum
        max_shift: maximum number of indices by which the data can shift (set to zero for no shift allowed)
    Returns:
        shifted_spectrum
    '''
    shift = random.randint(-max_shift, max_shift)
    # Create an array of zeros with the same length as the original spectrum
    shifted_spectrum = np.zeros_like(spectrum)

    if shift > 0:
        # Shift to the right
        shifted_spectrum[shift:] = spectrum[:-shift]
    elif shift < 0:
        # Shift to the left
        shifted_spectrum[:shift] = spectrum[-shift:]
    else:
        # No shift, return the original spectrum
        shifted_spectrum = spectrum.copy()

    shifted_spectrum = shifted_spectrum[:len(spectrum)]
    return shifted_spectrum


def shift_test_batch(X_test_tensor, max_shift_indices):
    """
    Applies a random horizontal shift to each spectrum in a batch.
    
    Args:
        X_test_tensor (torch.Tensor): The batch of spectra (e.g., shape [64, 6601]).
        max_shift_indices (int): The maximum shift in terms of array indices.
        
    Returns:
        torch.Tensor: A new tensor with each spectrum randomly shifted.
    """
    # 1. Move tensor to CPU and convert to a NumPy array to work with your function
    X_test_numpy = X_test_tensor.cpu().numpy()
    
    # 2. Create a list to hold the shifted spectra
    shifted_spectra_list = []
    
    # 3. Loop through each spectrum in the batch
    for spectrum in X_test_numpy:
        shifted_spec = apply_horizontal_shift(spectrum, max_shift_indices)
        shifted_spectra_list.append(shifted_spec)
        
    # 4. Stack the list of arrays back into a single NumPy array
    shifted_batch_numpy = np.array(shifted_spectra_list)
    
    # 5. Convert back to a PyTorch tensor and send it to the original device
    shifted_batch_tensor = torch.from_numpy(shifted_batch_numpy).float().to(X_test_tensor.device)
    
    return shifted_batch_tensor


def normalize_spectra_by_area(spectrum):
    """
    Normalize a 1D XPS spectrum by area (integral under the curve).
    
    Args:
        spectrum (torch.Tensor or np.ndarray): The input spectrum to normalize.
    
    Returns:
        np.ndarray: Normalized spectrum with area under the curve equal to 1.
    """
    # Convert to NumPy array if it's a PyTorch tensor
    if isinstance(spectrum, torch.Tensor):
        spectrum = spectrum.cpu().detach().numpy()  # Ensure it's on CPU and convert to NumPy

    # Compute the area under the spectrum (sum of intensities)
    area = np.sum(spectrum)

    # Avoid division by zero (if area is zero, return the original spectrum)
    if area == 0:
        return spectrum

    # Normalize the spectrum
    return spectrum / area



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
    # Create an array of zeros with the same shape as the original
    shifted_spectrum = np.zeros_like(spectrum)

    if amount > 0:
        # Shift to the right, leaving zeros at the beginning
        shifted_spectrum[amount:] = spectrum[:-amount]
    elif amount < 0:
        # Shift to the left, leaving zeros at the end
        shifted_spectrum[:amount] = spectrum[-amount:]
    else:
        # No shift, return a copy of the original
        shifted_spectrum = spectrum.copy()

    return shifted_spectrum