import os
import random
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from rdkit import Chem


def create_random_compositions(elements, num_of_composition):
    '''
    Creates the random compositions for the training data
    Args
        elements: The elements to include in the training set
        num_of_compositions: the number of training data examples to create
    Returns:
        composition: a dictionary of the emements present and their percentages
    '''
    bool_list = [True, False]
    is_present = []
    composition =[]
    for i in range(num_of_composition):
        tmp = []
        for j in range(len(elements)):
            tmp.append(random.choice(bool_list))

        if True not in tmp:
            tmp[random.randint(0,len(elements)-1)] = True
        is_present.append(tmp)

    for i in range(num_of_composition):
        tmp = {}
        total = 0
        for j in range(len(elements)):
            if not is_present[i][j]:
                continue
            x =  random.random()
            tmp[elements[j]] = x
            total += x

        tmp = {k:(v/total) for k,v in tmp.items()}
        composition.append(tmp)

    return composition


def chop_and_shift(spectrum, ranges, max_shift):
    '''
    Cuts out the regions of the spectrum from the main spectrum and applys a random shift +/- to the entire spectrum
    Args:
        spectrum
        ranges: the indices of the ranges to take out of the spectrum
        max_shift: maximum number of indices by which the data can shift (set to zero for no shift allowed)
    Returns:
        spectrum
    '''

    indices = []
    shift = random.randint(-max_shift, max_shift)
    for i in range(len(ranges)):
        inds = list(range(ranges[i][0]+shift, ranges[i][1]+shift))
        for ind in inds:
            indices.append(ind)
    return spectrum[indices]


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


def get_spec(path_to_file):
    """
    Extract the sequential intensity values (CPS) from Excel files.
    """
    df_f = pd.read_excel(path_to_file, sheet_name='All')
    return np.array(df_f['CPS'])

def get_binding_energies(path_to_file):
    """
    Extract the binding energy column from Excel files.
    """
    df_f = pd.read_excel(path_to_file, sheet_name='All')
    return np.array(df_f['Binding Energy'])

def get_label(path_to_file):
    """
    Extract the labels from Excel files.
    """
    df_f = pd.read_excel(path_to_file)
    labels = np.array(df_f['Number'])
    return np.nan_to_num(labels)

def get_SMILES(path_to_file):
    '''
    extract SMILES string from Excel fil
    '''
    df_f = pd.read_excel(path_to_file)
    SMILES = str((df_f.columns[0]))
    return SMILES

def get_label_dict(path_to_file):
    '''
    Extract the list of functional groups
    '''
    spreadsheet_f = pd.ExcelFile(path_to_file)
    df_f = pd.read_excel(spreadsheet_f)
    return list(df_f['Functional groups'])

def label_converter(label):
    '''
    Converts the encoded label lists into a readable format

    '''
    CEL_path = os.path.join(data_path, 'cellulose (CEL)', 'CEL_FG.xlsx') 
    label_dict = get_label_dict(CEL_path)
    fg_counts = []
    fg_count = "" 

    for i, e in enumerate(label):
        if pd.isna(e):
            e = 0
        if e != 0.0:
            fg_count = f'{e:.2f}  {label_dict[i]}'
            fg_counts.append(fg_count)

    return fg_counts

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

def fill_spectral_data(cps_data, be_data, start_energy, end_energy, increment=0.1):
    """
    Fill gaps in the spectral data with zeroes to ensure uniform energy increments.
    
    Args:
        cps_data: Dictionary of CPS values for each polymer.
        be_data: Dictionary of binding energy values for each polymer.
        start_energy: Starting energy value for the uniform grid.
        end_energy: Ending energy value for the uniform grid.
        increment: Energy increment (default is 0.1 eV).
    
    Returns:
        filled_data: Dictionary with filled CPS data for each polymer.
    """
    # Create the uniform energy grid
    energy_range = create_uniform_energy_grid(start_energy, end_energy, increment)
    filled_data = {}

    for polymer, cps_values in cps_data.items():
        # Initialize a new array filled with zeroes
        filled_cps = np.zeros_like(energy_range)

        # Fill the new array with existing CPS values
        for i, energy in enumerate(energy_range):
            # Find the index of the energy in the BE data
            index = np.where(np.isclose(be_data[polymer], energy, atol=0.01))[0]  # Use tolerance for floating-point comparison
            if index.size > 0:  # If the energy exists in the BE data
                filled_cps[i] = cps_values[index[0]]  # Assign the corresponding CPS value

        # Store the filled CPS data for the polymer
        filled_data[polymer] = filled_cps

    return filled_data

def open_all_xps_files(data_path, elements=None):
    """
    Open and read in the data from the experimental spectra.
    
    Args:
        data_path: Path to the data directory.
        elements: List of specific polymers to use (if None, all polymers are used).
    
    Returns:
        spectral_data: Dictionary of CPS values for each polymer.
        label_data: Dictionary of labels for each polymer.
        ids: List of polymer IDs.
        be_data: Dictionary of binding energy values for each polymer.
    """
    if not elements:
        all_folders = os.listdir(data_path)
    else:
        all_folders = [e for e in elements]

    spectral_data = {}
    label_data = {}
    be_data = {}
    SMILES_data = {}
    ids = []
    failed_polymers = []


    for folder in all_folders:
        folder_spec_format = folder[:folder.rfind(' (')]
        folder_label_format = folder[folder.rfind(' (') + 2:folder.rfind(')')]

        filename_spec = folder_spec_format + '.xlsx'
        filename_label = folder_label_format + '_FG.xlsx'
        filename_SMILES = folder_label_format + '_SMILES.xlsx'

        spec_path = os.path.join(data_path, folder, filename_spec)
        label_path = os.path.join(data_path, folder, filename_label)
        SMILES_path = os.path.join(data_path, folder, filename_SMILES)

        if not os.path.isfile(label_path):
            filename_label = folder + '_FG.xlsx'
            label_path = os.path.join(data_path, folder, filename_label)

            filename_SMILES = folder + '_SMILES.xlsx'
            SMILES_path =  os.path.join(data_path, folder, filename_SMILES)

        if not os.path.isfile(spec_path):
            filename_spec = folder + '.xlsx'
            spec_path = os.path.join(data_path, folder, filename_spec)

        if os.path.isfile(spec_path) and os.path.isfile(label_path) and os.path.isfile(SMILES_path):
            print('Material ID:', folder)
            data_spec = get_spec(spec_path)
            data_label = get_label(label_path)
            data_be = get_binding_energies(spec_path)
            data_SMILES = get_SMILES(SMILES_path)

            data_label = np.nan_to_num(data_label)
            ids.append(folder)

            spectral_data[folder] = data_spec
            label_data[folder] = data_label
            be_data[folder] = data_be
            SMILES_data[folder] = data_SMILES
        else:
            failed_polymers.append(folder)
    
    if failed_polymers: 
        print(f'failed to load polymers: {failed_polymers}') 

    return spectral_data, label_data, ids, be_data, SMILES_data

def count_carbons(mol):
    """
    Counts non-aromatic sp3 carbon atoms bonded ONLY to Carbon or Hydrogen atoms.

    Args:
        mol: RDKit molecule object.

    Returns:
        int: Number of plain sp3 alkane carbon atoms.
    """
    plain_alkane_carbon_count = 0
    for atom in mol.GetAtoms():
        # Initial checks: Is it a non-aromatic sp3 Carbon?
        if (atom.GetAtomicNum() == 6 and \
            atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3 and \
            not atom.GetIsAromatic()):

            # Assume it IS a plain alkane carbon until proven otherwise
            is_plain_alkane = True

            # Check its neighbors
            for neighbor in atom.GetNeighbors():
                neighbor_atomic_num = neighbor.GetAtomicNum()
                # If any neighbor is NOT Carbon (6) and NOT Hydrogen (1)
                if neighbor_atomic_num != 6 and neighbor_atomic_num != 1:
                    is_plain_alkane = False
                    break # No need to check other neighbors for this atom

            # If all neighbors were C or H, increment the count
            if is_plain_alkane:
                plain_alkane_carbon_count += 1

    return plain_alkane_carbon_count

def gaussian_broadening(spectrum, sigma_sigma):
    '''
    Applies random Gaussian broadening to spectra 
    Args:
        spectrum = input (unbroadened spectrum)
        sigma_sigma = the standard deviation for the distrobution determining sigma for gaussian kernal

    returns:
        broadened_spectrum: spectrum with peak broadening 
    
    '''
    sigma = np.abs(np.random.normal(0, sigma_sigma)) # half normal distribution centered at zero
    convolved_spectrum = gaussian_filter1d(spectrum,sigma)
    return convolved_spectrum



def normalize_spectrum_by_area(spectrum):
    """
    Normalize a 1D XPS spectrum by area (integral under the curve).
    
    Args:
        spectrum (np.ndarray): 1D array representing the spectrum.
    
    Returns:
        np.ndarray: Normalized spectrum with area under the curve equal to 1.
    """
    # Compute the area under the spectrum (sum of intensities)
    area = np.sum(spectrum)
    
    # Avoid division by zero (if area is zero, return the original spectrum)
    if area == 0:
        return spectrum
    
    # Normalize the spectrum by its area
    normalized_spectrum = spectrum / area
    
    return normalized_spectrum

def generate_mixture_xps(all_spectra, all_labels, sample_ids, max_materials, debug=False):
    '''
    Returns:
        spectrum: a fractionally weighted combination of the spectra
        labels: a fractionally weighted combination of the labels
    '''
    
    labels = np.zeros(shape=all_labels['cellulose (CEL)'].shape)
    spectrum = np.zeros(shape=all_spectra['cellulose (CEL)'].shape)

    num_materials = random.randint(2, max_materials)
    fractions = np.random.rand(num_materials)
    fractions = fractions/np.sum(fractions)
    for _i in range(num_materials):
        material_id = sample_ids[random.randint(0, len(sample_ids)-1)]

        added_spectrum = all_spectra[material_id]
        added_spectrum = added_spectrum[:spectrum.shape[0]]
        spectrum += all_spectra[material_id] * fractions[_i]
        labels += all_labels[material_id] * fractions[_i]
    
        spectrum = normalize_spectrum_by_area(spectrum)
    if debug:
        print(fractions)
        
        
    return spectrum, labels, fractions


def shift_spectrum(spectrum: np.ndarray, shift_amount):
        """
        Aligns shifted spectrum to original using cross-correlation
        Args:
            original: Reference spectrum (1D array)
            shifted: Shifted spectrum to realign (1D array)
            max_shift_index: Maximum allowed shift in index units
        Returns:
            realigned: Corrected spectrum
            shift_amount: Detected shift amount (positive = right shift)
        """
        
        # Apply correction using roll (circular) + zero-padding for valid region
        if shift_amount == 0:
            return spectrum.copy()
            
        realigned = np.roll(spectrum, shift_amount)
        
        # Zero out the wrapped regions
        if shift_amount > 0:
            realigned[:shift_amount] = 0
        else:
            realigned[shift_amount:] = 0
            
        return realigned

def shift_to_reference(
    spectrum: np.ndarray,
    energy_resolution: float = 0.1,  # eV per index
    carbon_region: tuple = (285.1, 286.5),  # eV range
    reference_peak_eV: float = 285.1,
    valence_region_index: int = 400,  # number of indices removed
    fallback_no_shift: bool = True
) -> np.ndarray:
    """
    Aligns the given spectrum to a reference based on the lowest-energy peak in the carbon region.

    Args:
        spectrum (np.ndarray): The unaligned spectrum.
        reference (np.ndarray): The reference spectrum (used to locate peak).
        energy_resolution (float): eV per index step.
        carbon_region (tuple): Energy range (eV) of the carbon region.
        reference_peak_eV (float): Expected eV position of reference carbon peak.
        valence_region_index (int): Number of leading indices (typically valence region) removed from spectrum.
        fallback_no_shift (bool): If True, return unshifted spectrum when no peaks are found.

    Returns:
        aligned_spectrum (np.ndarray): spectrum aligned to reference alkane carbon peak
        has_lower_alkane_peak(boolean): whether or not there is a peak at lower energy than reference alkane  
    """
    region_min_eV, region_max_eV = carbon_region

    # Convert carbon region eV range to indices
    region_start_idx = int((region_min_eV - valence_region_index * energy_resolution) / energy_resolution)
    region_end_idx = int((region_max_eV - valence_region_index * energy_resolution) / energy_resolution)

    unaligned_region = spectrum[region_start_idx:region_end_idx]

    # Convert reference peak position to index within carbon region
    ref_peak_index_in_region = int((reference_peak_eV - region_min_eV) / energy_resolution)
    ref_peak_index_in_region = np.clip(ref_peak_index_in_region, 0, region_end_idx - region_start_idx - 1)

     # Peak picking
    peaks, _ = find_peaks(unaligned_region,
                            prominence=0.0000005,
                            height = 0.001)
    

    significant_peaks, intensity = find_peaks(unaligned_region,
                            prominence=0.05,
                            height = 0.001)
    # Check if any of the detected peaks occur before (i.e. at a lower index than) the expected reference peak
    has_lower_alkane_peak = any(peak < ref_peak_index_in_region for peak in significant_peaks)
    
    
    if len(peaks) == 0:
        if fallback_no_shift:
            return spectrum.copy(), has_lower_alkane_peak
        else:
            raise ValueError("No peaks found in unaligned carbon region.")

    alkane_peak_index_in_region = peaks[0]

    # Calculate required shift in full spectrum
    full_ref_index = region_start_idx + ref_peak_index_in_region
    full_alkane_index = region_start_idx + alkane_peak_index_in_region
    shift = full_ref_index - full_alkane_index

    aligned_spectrum = shift_spectrum(spectrum, shift)

    return aligned_spectrum, has_lower_alkane_peak
