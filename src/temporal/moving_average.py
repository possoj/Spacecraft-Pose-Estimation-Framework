import numpy as np

class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.data = []

    def reset(self):
        self.data = []

    def add_data(self, data_list):
        self.data.append(data_list)
        if len(self.data) > self.window_size:
            self.data.pop(0)

    def predict(self):
        averages = [sum(x) / len(x) for x in zip(*self.data)]
        return averages


class ExponentialMovingAverage:
    def __init__(self, alpha):
        """
        Initialize the ExponentialMovingAverage instance.

        Args:
            alpha (float): Smoothing factor (between 0 and 1). Higher alpha discounts older observations faster.
        """
        self.alpha = alpha
        self.ema = None

    def reset(self):
        """
        Reset the EMA to its initial state.
        """
        self.ema = None

    def add_data(self, new_pdf):
        """
        Update the EMA with a new PDF.

        Args:
            new_pdf (list or np.ndarray): The new probability density function (PDF) to add.
        """
        if self.ema is None:
            # Initialize EMA with the first input PDF
            self.ema = np.array(new_pdf, dtype=np.float32)
        else:
            # Update EMA using the new PDF
            self.ema = self.alpha * np.array(new_pdf, dtype=np.float32) + (1 - self.alpha) * self.ema

    def predict(self):
        """
        Get the current EMA of the PDFs.

        Returns:
            np.ndarray: The smoothed PDF (EMA).
        """
        return self.ema