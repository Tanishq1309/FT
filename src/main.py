import numpy as np
import matplotlib.pyplot as plt
import signal_generator.sine_wave_generator as swg
import fast_fourier_transformation.fft as fft
import audio.operations as ao

def main(sampling_rate=44100, duration=5, num_waves=5):
    # Generate combined signal
    t, combined_signal = swg.generate_combined_signal(sampling_rate, duration, num_waves)

    # Perform FFT
    significant_frequencies, amplitudes, phase_differences, fft_result = fft.analyze_signal_with_fft(sampling_rate, combined_signal)
    
    # Plot the FFT results
    fft.plot_fft(fft_result, len(t), sampling_rate)

    # Reconstruct signal
    reconstructed_signal = fft.reconstruct_signal_from_fft(t, significant_frequencies, amplitudes, phase_differences)

    # Plot the original and reconstructed signals
    plot_signals(t, combined_signal, reconstructed_signal)
    
    # Print the sine wave equations
    equations = swg.format_sine_wave_equations(significant_frequencies, amplitudes, phase_differences)
    
    for i, equation in enumerate(equations):
        print(f"Wave {i+1}: {equation}")
    
    # Save the reconstructed signal as an audio file
    ao.create_audio_file_from_waves(sampling_rate, reconstructed_signal, "reconstructed_signal")
    ao.create_audio_file_from_waves(sampling_rate, combined_signal, "original_combined_signal")
    
    # Combine original sine waves and pi/2 phase shifted reconstructed sine waves
    phase_shifted_reconstructed_signal = fft.reconstruct_signal_from_fft(t, significant_frequencies, amplitudes, phase_differences + np.pi)
    combined_phase_shifted_signal = combined_signal + phase_shifted_reconstructed_signal
    
    # Save the combined signal as an audio file
    ao.create_audio_file_from_waves(sampling_rate, combined_phase_shifted_signal, "combined_phase_shifted_signal")
    
    # Plot the resultant signal
    plot_signal(t, combined_phase_shifted_signal)

def plot_signal(t, signal):
    # Plot the signal
    plt.figure(figsize=(14, 6))
    plt.plot(t, signal, label="Signal")
    plt.title("Combined Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
    plt.show()

def plot_signals(t, original_signal, reconstructed_signal):
    # Plot the original and reconstructed signals side by side
    plt.figure(figsize=(14, 6))

    # Original Signal
    plt.subplot(1, 2, 1)
    plt.plot(t, original_signal, label="Original Signal")
    plt.title("Original Combined Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()

    # Reconstructed Signal
    plt.subplot(1, 2, 2)
    plt.plot(t, reconstructed_signal, label="Reconstructed Signal", color='orange')
    plt.title("Reconstructed Signal from FFT")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    sampling_rate = input("Enter sampling rate: ") if input("Use default sampling rate? (y/n): ") == "n" else 44100
    duration = input("Enter duration: ")
    num_waves = input("Enter number of waves: ")
    main(sampling_rate=int(sampling_rate), duration=float(duration), num_waves=int(num_waves))