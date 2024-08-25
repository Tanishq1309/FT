import os
import json
import librosa
import numpy as np
from collections import Counter
import re

# Load the pre-extracted piano frequencies
def load_piano_frequencies(frequencies_file='piano_frequencies.json'):
    """
    Loads the piano frequencies from a JSON file.

    Parameters:
    - frequencies_file: Path to the JSON file containing the piano frequencies.

    Returns:
    - A dictionary mapping file names (notes) to their fundamental frequencies.
    """
    if not os.path.exists(frequencies_file):
        raise FileNotFoundError(f"Frequencies file {frequencies_file} not found.")
    
    with open(frequencies_file, 'r') as f:
        note_frequencies = json.load(f)
    
    return note_frequencies

# Analyze the song to extract dominant frequencies
def extract_dominant_frequencies_and_timings(song_path, sr=44100):
    """
    Analyzes a song to extract the dominant frequencies and their timings using STFT and onset detection.

    Parameters:
    - song_path: Path to the .wav file of the song.
    - sr: Sampling rate to load the song (default is 44100 Hz).

    Returns:
    - A list of tuples where each tuple contains (time, frequency) corresponding to the dominant frequency at each onset.
    """
    y, sr = librosa.load(song_path, sr=sr)

    # Perform Short-Time Fourier Transform (STFT)
    stft_result = np.abs(librosa.stft(y))
    frequencies = librosa.fft_frequencies(sr=sr)

    # Onset detection to find when each note occurs
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')

    # Get the dominant frequency for each onset time
    dominant_frequencies_and_times = []
    for onset_time in onset_times:
        frame_index = librosa.time_to_frames(onset_time, sr=sr)
        dominant_frequency = frequencies[np.argmax(stft_result[:, frame_index])]
        dominant_frequencies_and_times.append((onset_time, dominant_frequency))

    return dominant_frequencies_and_times

# Match the extracted frequencies to the closest piano notes
def match_frequencies_with_context(dominant_frequencies_and_times, note_frequencies, target_octave=4, base_octave_penalty=1.0, context_weight=0.5):
    """
    Matches the dominant frequencies and their timings to the best possible piano note,
    with an adaptive bias towards a target octave and context-aware smoothing.

    Parameters:
    - dominant_frequencies_and_times: A list of tuples where each tuple contains (time, frequency).
    - note_frequencies: A dictionary mapping notes to their fundamental frequencies.
    - target_octave: The octave that the matching process should generally favor (e.g., 4 for middle C).
    - base_octave_penalty: A base penalty for notes that are far from the target octave.
    - context_weight: A weight factor to consider the previous note's octave when making a match.

    Returns:
    - A list of tuples where each tuple contains (time, identified_note).
    """
    identified_notes_and_times = []
    previous_octave = target_octave
    
    for time, freq in dominant_frequencies_and_times:
        best_match_note = None
        best_match_score = float('inf')
        
        for note, note_freq in note_frequencies.items():
            # Extract the octave number using regex
            match = re.search(r'\d+', note)
            if match:
                note_octave = int(match.group(0))
            else:
                # Default to the target octave if no number is found
                note_octave = target_octave
            
            # Calculate the difference between the song frequency and the note's fundamental frequency
            fundamental_diff = abs(note_freq - freq)
            
            # Calculate the octave difference with adaptive penalty
            octave_diff = abs(note_octave - target_octave)
            context_diff = abs(note_octave - previous_octave)
            total_diff = (fundamental_diff + 
                          base_octave_penalty * octave_diff +
                          context_weight * context_diff)
            
            # Select the note with the minimum score
            if total_diff < best_match_score:
                best_match_score = total_diff
                best_match_note = note
        
        # Update previous_octave to the current note's octave
        identified_notes_and_times.append((time, best_match_note))
        previous_octave = int(re.search(r'\d+', best_match_note).group(0)) if re.search(r'\d+', best_match_note) else previous_octave
    
    return identified_notes_and_times

# Count the most frequent notes in the song and get the top 5
def get_top_5_notes(identified_notes):
    """
    Identifies the top 5 most frequent notes in the identified notes.

    Parameters:
    - identified_notes: A list of notes identified in the song.

    Returns:
    - A list of the top 5 most frequent notes with their counts.
    """
    note_counter = Counter(identified_notes)
    return note_counter.most_common(5)