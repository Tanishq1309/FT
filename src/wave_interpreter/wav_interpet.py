import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import librosa.display

# Load the audio file
file_path = 'src\\audio\\bohemian_rhapsody.wav'
y, sr = librosa.load(file_path, sr=None)  # y is the audio time series, sr is the sampling rate

# Print some basic information
print(f"Audio data shape: {y.shape}")
print(f"Sampling rate: {sr} Hz")

plt.figure(figsize=(14, 6))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform of Bohemian Rhapsody")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.show()

# Number of samples in the audio file
n = len(y)

# Perform FFT
fft_result = fft(y)

# Generate the frequency axis
frequencies = fftfreq(n, 1/sr)

# Take the positive half of the spectrum (since the FFT is symmetric)
positive_frequencies = frequencies[:n // 2]
positive_fft = np.abs(fft_result[:n // 2])

# Plot the magnitude spectrum
plt.figure(figsize=(14, 6))
plt.plot(positive_frequencies, positive_fft)
plt.title("Frequency Spectrum of Bohemian Rhapsody")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 5000)  # Limiting the x-axis for better visibility (most musical content is within 0-5000 Hz)
plt.show()

# Generate the spectrogram
D = np.abs(librosa.stft(y))**2
S_db = librosa.amplitude_to_db(D, ref=np.max)

# Display the spectrogram
plt.figure(figsize=(14, 6))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram of Bohemian Rhapsody')
plt.show()