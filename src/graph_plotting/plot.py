import matplotlib.pyplot as plt

def plot_signal(t, signal, title):
    # Plot the signal
    plt.figure(figsize=(14, 6))
    plt.plot(t, signal, label="Signal")
    plt.title(f"{title}")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
    plt.show()

def plot_original_and_reconstructed_signals(t, original_signal, reconstructed_signal):
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