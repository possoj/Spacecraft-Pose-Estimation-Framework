"""
Copyright (c) 2025 Julien Posso
"""

import numpy as np
from typing import Tuple


class TemporalPDF:
    def __init__(self, n: float = 1.0, alpha: float = 1.0, distance_metric: str = 'l2'):
        """
        Temporal PDF Filter using adaptive weighting based on the difference between PDFs.

        Args:
            n (float): Scaling factor for the influence of the current PDF.
            alpha (float): Scaling factor controlling the sensitivity of the weight to the distance.
            distance_metric (str): The distance metric to use ('l2', 'kl', 'js', 'hellinger', 'tv', 'wasserstein').
        """
        self.n = n
        self.alpha = alpha
        self.distance_metric = distance_metric.lower()
        self.previous_pdf = None

    def reset(self) -> None:
        """
        Reset the filter's state.

        This method clears the stored previous PDF, effectively restarting the filter.
        """
        self.previous_pdf = None

    def compute_distance(self, pdf1: np.ndarray, pdf2: np.ndarray) -> float:
        """
        Compute the distance between two PDFs using the specified metric.

        Args:
            pdf1 (np.ndarray): The first probability density function.
            pdf2 (np.ndarray): The second probability density function.

        Returns:
            float: The computed distance between pdf1 and pdf2.

        Raises:
            ValueError: If an unsupported distance metric is specified.
        """
        # Normalize PDFs to ensure they are valid probability distributions
        pdf1 = pdf1 / np.sum(pdf1)
        pdf2 = pdf2 / np.sum(pdf2)

        if self.distance_metric == 'l2':
            # Euclidean (L2) distance
            distance = np.linalg.norm(pdf1 - pdf2)
        elif self.distance_metric == 'kl':
            # Kullback-Leibler divergence
            epsilon = 1e-12  # Small value to avoid log(0)
            pdf1_safe = pdf1 + epsilon
            pdf2_safe = pdf2 + epsilon
            distance = np.sum(pdf1_safe * np.log(pdf1_safe / pdf2_safe))
        elif self.distance_metric == 'js':
            # Jensen-Shannon divergence
            m = 0.5 * (pdf1 + pdf2)
            distance = 0.5 * (np.sum(pdf1 * np.log(pdf1 / m)) + np.sum(pdf2 * np.log(pdf2 / m)))
            distance = np.sqrt(distance)  # Taking the square root for metric property
        elif self.distance_metric == 'hellinger':
            # Hellinger distance
            distance = np.sqrt(0.5 * np.sum((np.sqrt(pdf1) - np.sqrt(pdf2)) ** 2))
        elif self.distance_metric == 'tv':
            # Total Variation distance
            distance = 0.5 * np.sum(np.abs(pdf1 - pdf2))
        elif self.distance_metric == 'wasserstein':
            # Wasserstein distance (1D case)
            cdf1 = np.cumsum(pdf1)
            cdf2 = np.cumsum(pdf2)
            distance = np.sum(np.abs(cdf1 - cdf2)) / len(pdf1)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

        return distance

    def compute_weight(self, distance: float) -> float:
        """
        Compute the weight based on the distance between PDFs.

        Args:
            distance (float): The computed distance between two PDFs.

        Returns:
            float: Weight in the range [0, 1].
        """
        weight = np.exp(-self.alpha * distance)
        weight = np.clip(weight, 0.0, 1.0)
        return weight

    def update_pdf(self, current_pdf: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Update the PDF estimate based on the current PDF and previous PDF using adaptive weighting.

        Args:
            current_pdf (np.ndarray): Current PDF from the neural network.

        Returns:
            tuple[np.ndarray, float]:
                - Updated PDF after applying the temporal filter.
                - Computed distance between the current and previous PDFs.
        """
        # Normalize the current PDF
        current_pdf = current_pdf / np.sum(current_pdf)

        if self.previous_pdf is None:
            # First frame, initialize the previous PDF
            self.previous_pdf = current_pdf
            updated_pdf = current_pdf
            distance = 0.0
        else:
            # Compute distance between current and previous PDFs
            distance = self.compute_distance(current_pdf, self.previous_pdf)

            # Compute weight based on distance
            weight = self.compute_weight(distance)

            # Update PDF using adaptive weighting
            updated_pdf = weight * self.n * current_pdf + (1 - weight) * self.previous_pdf

            # Note: The following line was used for a specific experimental setup.
            # updated_pdf = current_pdf

            # Normalize updated PDF
            updated_pdf = updated_pdf / np.sum(updated_pdf)

            # Update the previous PDF for the next frame
            self.previous_pdf = updated_pdf

        return updated_pdf, distance

