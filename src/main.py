import signal_generator.sine_wave_generator as swg
import fast_fourier_transformation.fft as fft
import audio.operations as ao
import graph_plotting.plot as plot
import audio.piano_frequency_extractor as pfe
import wave_interpreter.song_analysis as sa
import wave_interpreter.song_recreator as sr

def main(program_run):
    match program_run:
        case 1:
            fft_main()
        case 2:
            dft_main()
        case 3:
            song_analysis_and_recreation_main()
        case _:
            print("Invalid input. Please enter 1, 2 or 3.")

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

def song_analysis_and_recreation_main():
    dataset_path = "src\\audio\\piano_notes_jobro"
    note_frequencies = pfe.extract_frequencies_from_notes(dataset_path)
    
    # Example: Print the frequencies for each note
    for note, frequency in note_frequencies.items():
        print(f"Note: {note}, Frequency: {frequency:.2f} Hz")
    
    song_path = "src\\audio\\audio_files\\giorno_theme_gb.wav"
    recreated_song_name = f"recreated_{song_path.split('\\')[-1].replace('.wav', '')}"
    
    # Extract dominant frequencies and timings from the song
    dominant_frequencies_and_timings = sa.extract_dominant_frequencies_and_timings(song_path)

    # Match these frequencies to the closest piano notes
    identified_notes_and_times = sa.match_frequencies_with_context(dominant_frequencies_and_timings, note_frequencies)

    # Get the top 5 most frequent notes (optional, you may not need this for synthesis)
    top_5_notes = sa.get_top_5_notes([note for _, note in identified_notes_and_times])

    # Print the top 5 most frequent notes
    print("Top 5 Most Frequent Notes in the Song:")
    for note, count in top_5_notes:
        print(f"Note: {note}, Count: {count}")
        
    # Extract just the note names from identified_notes_and_times
    note_names = {note for _, note in identified_notes_and_times}
    
    # Load the corresponding .wav files for the identified notes
    note_wavs = sr.load_note_wavs(note_names)

    # Synthesize the new song using these notes and timings
    synthesized_song = sr.synthesize_song_with_timing(identified_notes_and_times, note_wavs, 44100)

    # Save the synthesized song as an audio file
    ao.create_audio_file_from_waves(44100, synthesized_song, recreated_song_name)

if __name__ == "__main__":
    program_run = int(input("Enter 1 for FFT, 2 for DFT and 3 for Song Analysis: "))
    main(program_run)