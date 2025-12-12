import numpy as np


class HeatDither:
    """
    累积误差抖动器，将连续热度转为离散发声次数，残差顺延。
    """

    def __init__(self, unit_heat: float = 1.0):
        self.unit_heat = max(1e-9, unit_heat)
        self.accumulated_error = 0.0

    def quantize(self, target_continuous_heat: float) -> int:
        total_potential = target_continuous_heat + self.accumulated_error
        num_agents = int(np.floor(total_potential / self.unit_heat))
        self.accumulated_error = total_potential - (num_agents * self.unit_heat)
        if self.accumulated_error < 0:
            self.accumulated_error = 0.0
        return num_agents


class PIDController:
    """
    简单离散 PID，用于校正累计热度漂移。
    """

    def __init__(self, kp: float = 0.1, ki: float = 0.05, kd: float = 0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, setpoint: float, measured: float) -> float:
        error = setpoint - measured
        self.integral += error
        derivative = error - self.prev_error
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        return output
