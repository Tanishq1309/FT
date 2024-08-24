import numpy as np

def generate_random_sine_wave(sampling_rate, duration):
    """
    Generates a random sine wave signal.

    Parameters:
    sampling_rate (float): The number of samples per second.
    duration (float): The duration of the sine wave in seconds.

    Returns:
    numpy.ndarray: A numpy array containing the generated sine wave.

    The function generates a sine wave with:
    - A random amplitude between 0.1 and 1.0.
    - A random frequency between 300 and 700 Hz.
    - A random phase difference between 0 and 2π radians.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    
    R = np.random.uniform(0.1, 1.0)  # Random amplitude between 0.1 and 1.0
    f = np.random.uniform(300, 700)  # Random frequency between 300 and 700 Hz
    pd = np.random.uniform(0, 2*np.pi) # Random phase difference between 0 and 2π radians
    
    sine_wave = R * np.sin(2 * np.pi * f * t + pd)
    return sine_wave

def generate_combined_signal(sampling_rate, duration, num_waves=5):
    """
    Generates a combined signal by summing multiple random sine waves.

    Parameters:
    sampling_rate (float): The number of samples per second.
    duration (float): The duration of the combined signal in seconds.
    num_waves (int): The number of sine waves to combine. Default is 5.

    Returns:
    tuple: A tuple containing:
        - t (numpy.ndarray): The time array corresponding to the signal.
        - combined_signal (numpy.ndarray): The combined signal created by summing the random sine waves.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    
    combined_signal = np.zeros_like(t)
    for _ in range(num_waves):
        sine_wave = generate_random_sine_wave(sampling_rate, duration)
        combined_signal += sine_wave
    
    return t, combined_signal

def format_sine_wave_equations(frequencies, amplitudes, phase_differences):
    """
    Formats the sine wave parameters into equations.

    Parameters:
    frequencies (numpy.ndarray): An array of frequencies corresponding to the sine waves.
    amplitudes (numpy.ndarray): An array of amplitudes corresponding to the sine waves.
    phase_differences (numpy.ndarray): An array of phase differences corresponding to the sine waves.

    Returns:
    list: A list of strings, where each string is a formatted sine wave equation.
    """
    equations = []
    for R, f, pd in zip(amplitudes, frequencies, phase_differences):
        equation = f"y(t) = {R:.2f} * sin(2 * pi * {f:.2f} * t + {pd:.2f})"
        equations.append(equation)
    return equations

def phase_shift_signal(signal, phase_shift):
    """
    Phase shifts a signal by a given angle.

    Parameters:
    signal (numpy.ndarray): The input signal to be phase shifted.
    phase_shift (float): The phase shift angle in radians.

    Returns:
    numpy.ndarray: The phase shifted signal.
    """
    return signal * np.exp(1j * phase_shift)
