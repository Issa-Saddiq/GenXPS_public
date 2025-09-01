
import numpy as np
import torch
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt


def get_model_predictions(shifted_spectrum_tensor, true_labels, fg_list, classifier):
    """
    Gets model predictions for a given spectrum and prints a summary.
    """
    with torch.inference_mode():
        logits, aligned_spectrum_tensor = classifier(shifted_spectrum_tensor)
        predictions = torch.round(torch.sigmoid(logits)).squeeze()

    # --- Create and Display a Prediction Summary DataFrame ---
    true_labels_np = true_labels.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    
    df = pd.DataFrame({
        'Functional Group': fg_list,
        'Actual': true_labels_np,
        'Predicted': predictions_np
    })
    summary_df = df[(df['Actual'] == 1) | (df['Predicted'] == 1)]
    print(f"--- Prediction Summary for Sample ---")
    print(summary_df.to_string(index=False))
    print("-" * 45)

    positive_class_indices = torch.where(predictions == 1)[0]
    aligned_spectrum_np = aligned_spectrum_tensor.detach().cpu().squeeze().numpy()
    
    return predictions_np, positive_class_indices, aligned_spectrum_np


def generate_counterfactual_spectra(predictions_np, positive_class_indices, cvae_model, latent_dim, device):
    """
    Generates counterfactual spectra using a CVAE by leaving one FG out at a time.
    """
    if len(positive_class_indices) == 0:
        print("\nNo positive classes were predicted for this spectrum.")
        return {}
        
    print(f"\nFound {len(positive_class_indices)} positive predictions. Generating explanations.")
    
    base_prediction_vector = predictions_np / len(positive_class_indices)
    base_prediction_tensor = torch.tensor(base_prediction_vector, dtype=torch.float32).to(device)
    z = torch.randn(1, latent_dim).to(device)
    generated_spectra_dict = {}

    for fg_idx in positive_class_indices:
        label_loo = base_prediction_tensor.clone()
        label_loo[fg_idx] = 0
        if label_loo.sum() > 0:
            label_loo = label_loo / label_loo.sum()
        
        cvae_condition_vector = label_loo.unsqueeze(0)
        
        with torch.no_grad():
            spectrum = cvae_model.decode(z, cvae_condition_vector)
        
        spectrum_np = spectrum.cpu().squeeze().numpy()
        spectrum_np[spectrum_np < 0] = 0
        generated_spectra_dict[fg_idx] = spectrum_np
        
    return generated_spectra_dict


def assign_peaks_from_counterfactuals(aligned_spectrum_np, generated_spectra_dict):
    """
    Finds peaks in the aligned spectrum and assigns them based on the generated spectra.
    """
    peaks, _ = find_peaks(aligned_spectrum_np, height=0.001, width=1)
    peak_assignments = {}
    
    for peak_idx in peaks:
        min_value = np.inf
        assigned_fg = None
        for fg_idx, spectrum in generated_spectra_dict.items():
            value_at_peak = spectrum[peak_idx]
            if value_at_peak < min_value:
                min_value = value_at_peak
                assigned_fg = fg_idx
        peak_assignments[peak_idx] = assigned_fg
        
    return peaks, peak_assignments


def plot_annotated_regions(original_spectrum, shifted_spectrum, aligned_spectrum, peaks, peak_assignments, be_values, fg_list, predictions, sample_index):
    """
    Automatically finds active regions and plots the annotated spectra in subplots.
    """
    # A. Define active regions
    if len(peaks) > 0:
        peak_energies = be_values[peaks]
        regions = sorted([(be - 3, be + 3) for be in peak_energies])
        merged_regions = [regions[0]]
        for current_start, current_end in regions[1:]:
            last_start, last_end = merged_regions[-1]
            if current_start <= last_end:
                merged_regions[-1] = (last_start, max(last_end, current_end))
            else:
                merged_regions.append((current_start, current_end))
        print(f"Identified {len(merged_regions)} active regions for plotting.")
    else:
        print("No peaks found to define active regions.")
        return

    # B. Create dynamic subplots
    widths = [r[1] - r[0] for r in merged_regions]
    num_regions = len(merged_regions)
    fig, axes = plt.subplots(1, num_regions, figsize=(5 * num_regions, 6), sharey=True, gridspec_kw={'width_ratios': widths})
    if num_regions == 1: # Ensure axes is always a list
        axes = [axes]
    fig.subplots_adjust(wspace=0.05)

    # C. Determine Alkane override
    alkane_predicted = (predictions[39] == 1) # Simplified boolean check
    lowest_energy_alkane_peak_idx = None
    if alkane_predicted:
        alkane_region_peaks = {p_idx: be_values[p_idx] for p_idx in peak_assignments.keys() if 285.0 <= be_values[p_idx] <= 286.0}
        if alkane_region_peaks:
            lowest_energy_alkane_peak_idx = min(alkane_region_peaks, key=alkane_region_peaks.get)
            print(f"Lowest energy peak in alkane region identified at index: {lowest_energy_alkane_peak_idx}")

    # D. Loop through regions to plot and annotate
    for i, ax in enumerate(axes):
        region = merged_regions[i]
        ax.plot(be_values, original_spectrum, color="Orange", linewidth=2, linestyle=':', label="Original Spectrum")
        ax.plot(be_values, shifted_spectrum, color="black", linewidth=2, linestyle=':', label="Shifted Spectrum")
        ax.plot(be_values, aligned_spectrum, color="green", linewidth=2, label="Aligned Spectrum")
        ax.set_xlim(*region)

        for peak_idx, fg_idx in peak_assignments.items():
            peak_energy = be_values[peak_idx]
            if region[0] <= peak_energy <= region[1]:
                fg_name = 'alkane' if peak_idx == lowest_energy_alkane_peak_idx else fg_list[fg_idx]
                peak_height = aligned_spectrum[peak_idx]
                ax.annotate(
                    text=fg_name,
                    xy=(peak_energy, peak_height),
                    xytext=(peak_energy, peak_height + 0.12 * aligned_spectrum.max()),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5),
                    fontsize=11, color='red', ha='center', va='bottom'
                )
        if i > 0:
            ax.tick_params(left=False, labelleft=False)
        plt.grid(True, linestyle='--', alpha=0.6)

    # E. Final plot styling
    max_height_in_regions = aligned_spectrum[np.any([(be_values >= r[0]) & (be_values <= r[1]) for r in merged_regions], axis=0)].max()
    axes[0].set_ylim(bottom=aligned_spectrum.min() * 1.1, top=max_height_in_regions * 1.4)
    fig.supxlabel("Binding Energy (eV)", fontsize=14, y=0.02)
    fig.supylabel("Relative Intensity", fontsize=14, x=0.06)
    fig.suptitle(f"Automatic Peak Annotation for Sample", fontsize=16, y=0.98)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.95))
    plt.show()


def plot_counterfactuals_for_debugging(aligned_spectrum, generated_spectra_dict, be_values, fg_list):
    """
    Plots the aligned spectrum and all the generated 'leave-one-out' spectra.
    """
    plt.figure(figsize=(15, 7))
    
    # Plot the main aligned spectrum as a solid reference line
    plt.plot(be_values, aligned_spectrum, color="green", linewidth=2.5, label="Aligned Spectrum (Reference)")
    
    # Get a colormap to assign unique colors to each counterfactual
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(generated_spectra_dict)))
    
    # Plot each generated spectrum
    for i, (fg_idx, spectrum) in enumerate(generated_spectra_dict.items()):
        fg_name = fg_list[fg_idx]
        plt.plot(be_values, spectrum, color=colors[i], linestyle='--', alpha=0.8,
                 label=f"Generated without '{fg_name}'")
                 
    plt.title("Debugging View: Aligned Spectrum vs. Counterfactuals", fontsize=16)
    plt.xlabel("Binding Energy (eV)", fontsize=12)
    plt.ylabel("Relative Intensity", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()