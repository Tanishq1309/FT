import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# Function to perform FFT and identify the individual sine waves
def analyze_signal_with_fft(sampling_rate, signal):
    # Perform FFT
    N = len(signal)
    fft_result = fft(signal)
    fft_magnitudes = np.abs(fft_result)[:N//2] * 2 / N  # Take the magnitude of the first half
    frequencies = fftfreq(N, 1/sampling_rate)[:N//2]    # Get corresponding frequencies
    
    # Identify the significant frequencies and their corresponding amplitudes
    threshold = np.max(fft_magnitudes) * 0.1  # Threshold for significant peaks
    significant_frequencies = frequencies[fft_magnitudes > threshold]
    significant_amplitudes = fft_magnitudes[fft_magnitudes > threshold]
    
    # Calculate the phase difference
    significant_indices = np.where(fft_magnitudes > threshold)[0]
    phase_differences = np.angle(fft_result[significant_indices])
    
    return significant_frequencies, significant_amplitudes, phase_differences, fft_result

# Function to reconstruct the signal from FFT results
def reconstruct_signal_from_fft(t, frequencies, amplitudes, phase_differences):
    reconstructed_signal = np.zeros_like(t)
    for R, f, pd in zip(amplitudes, frequencies, phase_differences):
        reconstructed_signal += R * np.sin(2 * np.pi * f * t + pd)
    return reconstructed_signal

def plot_fft(fft_result, n, sampling_rate, max_freq=1000):
    """
    Plots the FFT results, limiting the frequency range to a maximum frequency.

    Parameters:
    - fft_result: The FFT result of the signal.
    - n: The number of samples in the signal.
    - sampling_rate: The sampling rate used for the signal.
    - max_freq: The maximum frequency to display on the x-axis (default is 1000 Hz).
    """
    # Compute the frequency axis
    frequency = fftfreq(n, 1 / sampling_rate)
    
    # Limit the frequency axis to the max_freq
    mask = frequency[:n // 2] <= max_freq

    # Plot the FFT results
    plt.figure(figsize=(14, 6))
    plt.plot(frequency[:n // 2][mask], np.abs(fft_result)[:n // 2][mask])
    plt.title("FFT of Combined Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()