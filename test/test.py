import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the original and recreated audio files
original_audio_path = 'src\\audio\\audio_files\\giorno_theme_gb.wav'
recreated_audio_path = 'src\\audio\\audio_files\\recreated_giorno_theme_gb.wav'

original_y, original_sr = librosa.load(original_audio_path, sr=None)
recreated_y, recreated_sr = librosa.load(recreated_audio_path, sr=None)

# Plot the waveforms for comparison
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.title("Original Audio Waveform")
plt.plot(original_y, label="Original")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(2, 1, 2)
plt.title("Recreated Audio Waveform")
plt.plot(recreated_y, label="Recreated", color='orange')
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()

# Perform STFT on both signals for frequency comparison
original_stft = np.abs(librosa.stft(original_y, dtype=np.complex128))
recreated_stft = np.abs(librosa.stft(recreated_y, dtype=np.complex128))

# Plot the STFT results for comparison
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.title("Original Audio STFT Magnitude")
librosa.display.specshow(librosa.amplitude_to_db(original_stft, ref=np.max), sr=original_sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')

plt.subplot(2, 1, 2)
plt.title("Recreated Audio STFT Magnitude")
librosa.display.specshow(librosa.amplitude_to_db(recreated_stft, ref=np.max), sr=recreated_sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.show()
