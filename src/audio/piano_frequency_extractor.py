import os
import json
import librosa
import numpy as np

FREQUENCIES_FILE = 'src\\audio\\piano_frequencies.json'

def extract_frequencies_from_notes(dataset_path):
    """
    Extracts the fundamental frequency of each note in the dataset and saves it to a JSON file.

    Parameters:
    - dataset_path: Path to the directory containing the .wav files.

    Returns:
    - A dictionary mapping file names (notes) to their fundamental frequencies.
    """
    if os.path.exists(FREQUENCIES_FILE):
        # Load frequencies from the saved JSON file
        with open(FREQUENCIES_FILE, 'r') as f:
            note_frequencies = json.load(f)
        print("Loaded frequencies from existing file.")
    else:
        # Extract frequencies and save them
        note_frequencies = {}
        for filename in os.listdir(dataset_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(dataset_path, filename)
                y, sr = librosa.load(file_path)
                f0 = librosa.yin(y, fmin=librosa.note_to_hz('A0'), fmax=librosa.note_to_hz('C8'))
                f0 = f0[~np.isnan(f0)]
                if len(f0) > 0:
                    fundamental_freq = np.median(f0)  # Use the median frequency
                    note_frequencies[filename] = fundamental_freq
        
        # Save the frequencies to a JSON file
        with open(FREQUENCIES_FILE, 'w') as f:
            json.dump(note_frequencies, f, indent=4)
        print("Frequencies extracted and saved to file.")

    return note_frequencies

def main():
    dataset_path = 'src\\audio\\piano_notes_jobro'  # Update with your dataset path
    note_frequencies = extract_frequencies_from_notes(dataset_path)
    
    # Example: Print the frequencies for each note
    for note, frequency in note_frequencies.items():
        print(f"Note: {note}, Frequency: {frequency:.2f} Hz")

if __name__ == "__main__":
    main()