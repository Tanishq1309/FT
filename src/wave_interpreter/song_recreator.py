import os
import json
import librosa
import numpy as np

def load_note_wavs(note_filenames, note_dir='src\\audio\\piano_notes_jobro'):
    """Loads the .wav files for the notes."""
    note_wavs = {}
    for note in note_filenames:
        file_path = os.path.join(note_dir, note)
        y, sr = librosa.load(file_path, sr=None)
        note_wavs[note] = (y, sr)
    return note_wavs

def synthesize_song_with_timing(identified_notes_and_times, note_wavs, sr):
    """Recreates the song using the identified notes and their timings."""
    max_time = max(time for time, _ in identified_notes_and_times)
    synthesized_song = np.zeros(int(sr * (max_time + 1)))  # +1 to ensure the last note fits

    for time, note in identified_notes_and_times:
        note_wave, note_sr = note_wavs[note]

        # Resample if the note's sample rate doesn't match the song's sample rate
        if sr != note_sr:
            note_wave = librosa.resample(note_wave, orig_sr=note_sr, target_sr=sr)

        # Determine where to place this note in the synthesized song
        start_idx = int(sr * time)
        end_idx = start_idx + len(note_wave)

        # Add the note to the synthesized song, taking care of potential overlaps
        synthesized_song[start_idx:end_idx] += note_wave[:min(len(note_wave), len(synthesized_song) - start_idx)]

    return synthesized_song