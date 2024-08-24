import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Parameters
sampling_rate = 1000  # samples per second
t = np.linspace(0, 1, sampling_rate)  # 1 second of data & required to generate the sine wave over time and not a constant value

# Generate sine waves with different frequencies
sine_wave_1 = np.sin(2 * np.pi * 5 * t)  # 5 Hz
sine_wave_2 = np.sin(2 * np.pi * 20 * t) # 20 Hz
sine_wave_3 = np.sin(2 * np.pi * 50 * t) # 50 Hz

# Combine the sine waves
combined_wave = sine_wave_1 + sine_wave_2 + sine_wave_3

# Plot the combined wave
plt.plot(t, combined_wave)
plt.title("Combined Sine Wave")
plt.show()

# Compute the FFT of the combined signal
fft_result = fft(combined_wave)
n = len(t)
frequency = fftfreq(n, 1 / sampling_rate)

# Only plot the positive frequencies
plt.plot(frequency[:n // 2], np.abs(fft_result)[:n // 2])
plt.title("FFT of Combined Sine Wave")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()