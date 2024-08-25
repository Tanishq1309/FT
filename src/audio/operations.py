import numpy as np
from scipy.io.wavfile import write

def create_audio_file_from_waves(sampling_rate, signal, filename):
    signal_normalized = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal
    write(f"src\\audio\\audio_files\\{filename}.wav", sampling_rate, signal_normalized.astype(np.float32))
    print(f"Audio file '{filename}.wav' saved successfully!")