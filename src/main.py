import signal_generator.sine_wave_generator as swg
import fast_fourier_transformation.fft as fft
import audio.operations as ao
import graph_plotting.plot as plot

def main(program_run):
    if program_run == 1:
        sampling_rate = input("Enter sampling rate: ") if input("Use default sampling rate? (y/n): ") == "n" else 44100
        duration = input("Enter duration: ")
        num_waves = input("Enter number of waves: ")
        fft_main(sampling_rate=int(sampling_rate), duration=float(duration), num_waves=int(num_waves))
    else:
        dft_main()

def fft_main(sampling_rate=44100, duration=5, num_waves=5):
    # Generate combined signal
    t, combined_signal = swg.generate_combined_signal(sampling_rate, duration, num_waves)

    # Perform FFT
    significant_frequencies, amplitudes, phase_differences, fft_result = fft.analyze_signal_with_fft(sampling_rate, combined_signal)
    
    # Plot the FFT results
    fft.plot_fft(fft_result, len(t), sampling_rate)

    # Reconstruct signal
    reconstructed_signal = fft.reconstruct_signal_from_fft(t, significant_frequencies, amplitudes, phase_differences)

    # Plot the original and reconstructed signals
    plot.plot_original_and_reconstructed_signals(t, combined_signal, reconstructed_signal)
    
    # Print the sine wave equations
    equations = swg.format_sine_wave_equations(significant_frequencies, amplitudes, phase_differences)
    
    for i, equation in enumerate(equations):
        print(f"Wave {i+1}: {equation}")
    
    # Save the reconstructed signal as an audio file
    ao.create_audio_file_from_waves(sampling_rate, reconstructed_signal, "reconstructed_signal")
    ao.create_audio_file_from_waves(sampling_rate, combined_signal, "original_combined_signal")
    
    # Invert the reconstructed signal
    inverted_reconstructed_signal = -reconstructed_signal
    
    # Combine the original and inverted signals
    combined_inverted_signal = combined_signal + inverted_reconstructed_signal
    
    # Plot the resultant signal
    plot.plot_signal(t, combined_inverted_signal, "Combined Original and Inverted Signal")
    
    # Save the combined inverted signal as an audio file
    ao.create_audio_file_from_waves(sampling_rate, combined_inverted_signal, "combined_inverted_signal")

def dft_main():
    pass
    
if __name__ == "__main__":
    program_run = int(input("Enter 1 for FFT or 2 for DFT: "))
    main(program_run)