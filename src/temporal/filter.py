from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import firwin

class FIRFilter:
    """
    Implements a Finite Impulse Response (FIR) filter.

    Args:
        filter_type (str): Type of the filter ('lowpass', 'highpass', 'bandpass').
        order (int): The order of the filter.
        fs (float): Sampling frequency in Hz.
        cutoff (float or tuple): Cutoff frequency/frequencies in Hz. Single value for 'lowpass'/'highpass', tuple for 'bandpass'.
    """

    def __init__(self, filter_type: str, order: int, fs: float, cutoff: [float, tuple]):
        """
        Initialize the FIR filter by calculating its coefficients.

        Args:
            filter_type (str): Type of the filter ('lowpass', 'highpass', 'bandpass').
            order (int): The order of the filter.
            fs (float): Sampling frequency in Hz.
            cutoff (float or tuple): Cutoff frequency/frequencies in Hz.
        """
        self.order = order
        self.fs = fs

        # Normalize the cutoff frequency
        if filter_type in ['lowpass', 'highpass']:
            w = cutoff / (fs / 2)
        else:  # bandpass
            w = [c / (fs / 2) for c in cutoff]

        # Compute the filter coefficients
        self.coeffs = signal.firwin(order, w, pass_zero=(filter_type != 'highpass'), window='hamming')

        # Initialize the buffer
        self.buffer = np.zeros(order)

    def update_buffer(self, new_sample: float) -> None:
        """
        Update the buffer with a new sample.

        Args:
            new_sample (float): The new incoming sample.
        """
        self.buffer = np.roll(self.buffer, -1)
        self.buffer[-1] = new_sample

    def apply_filter(self) -> float:
        """
        Apply the filter to the current buffer.

        Returns:
            float: The filtered sample.
        """
        return np.dot(self.coeffs, self.buffer)

    def process_sample(self, sample: float) -> float:
        """
        Process an incoming sample through the filter and return the filtered result.

        Args:
            sample (float): The new incoming sample.

        Returns:
            float: Filtered sample.
        """
        self.update_buffer(sample)
        return self.apply_filter()


class RCFilter:
    def __init__(self, R, C, fs, initial_input=0, initial_output=0):
        self.R = R
        self.C = C
        self.tau = R * C  # Time constant
        self.fc = 1 / (2 * np.pi * R * C)  # Cutoff frequency
        self.fs = fs
        self.previous_input = initial_input
        self.previous_output = initial_output

    def frequency_response(self, f):
        # Calculate the transfer function H(f)
        H = 1 / (1 + 1j * 2 * np.pi * f * self.tau)
        return H

    def plot_frequency_response(self, max_freq):
        # Create a set of frequencies for the plot
        freqs = np.linspace(0, max_freq, 1000)
        # Calculate the frequency response
        H = self.frequency_response(freqs)

        plt.figure(figsize=(10, 6))
        plt.plot(freqs, 20 * np.log10(np.abs(H)))  # Gain in dB
        plt.xscale('log')
        plt.title("Frequency Response of the RC Filter")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Gain (dB)")
        plt.grid(which='both', linestyle='--')
        plt.show()

    def process_sample(self, input_signal):
        # Apply the filter to a single sample (real-time filtering)
        alpha = (self.R * self.C) / (self.R * self.C + 1.0 / self.fs)
        output_signal = alpha * self.previous_output + (1 - alpha) * input_signal
        self.previous_input = input_signal
        self.previous_output = output_signal
        return output_signal


class LowPassFilter:
    def __init__(self, order, fs, cutoff, n):
        assert cutoff <= fs / 2, "Nyquist limit not respected"
        # n is the number of filters to implement
        self.filters = [FIRFilter('lowpass', order, fs, cutoff) for _ in range(n)]
        # init_input = [0, 0, 20]
        # init_output = [0, 0, 20]
        # self.filters = [RCFilter(1e3, 1e-3, fs, init_input[i], init_output[i]) for i in range(n)]

    def predict(self, values):
        assert len(values) == len(self.filters)

        filtered_values = []
        for i, value in enumerate(values):
            filtered_value = self.filters[i].process_sample(value)
            filtered_values.append(filtered_value)

        return filtered_values
