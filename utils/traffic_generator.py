import math
from typing import Optional


class BackgroundTrafficGenerator:
    def __init__(self, peak_time: int = 15, amplitude: int = 200, intensity: Optional[int] = None):
        self.peak_time = peak_time
        # Allow intensity alias to match TrendSim config naming
        self.amplitude = intensity if intensity is not None else amplitude

    def get_noise_volume(self, t: int) -> int:
        """
        Compute background noise volume for a given time step.
        Growth until peak_time, plateau for 5 steps, then decay.
        """
        if t <= self.peak_time:
            prob = math.exp(0.5 * (t - self.peak_time))
        elif t <= self.peak_time + 5:
            prob = 1.0
        else:
            prob = (t - self.peak_time - 4) ** (-1.5)

        return int(prob * self.amplitude)
