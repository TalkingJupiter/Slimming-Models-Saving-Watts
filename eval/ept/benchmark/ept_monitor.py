import time
import threading
from typing import List, Optional

import pynvml


class EnergyMonitor:
    def __init__(self, sample_interval: float = 0.1, gpu_indices: Optional[List[int]] = None):
        self.sample_interval = sample_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = False
        self.energy_joules = 0.0

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        if gpu_indices is None:
            self.gpu_indices = list(range(device_count))
        else:
            for idx in gpu_indices:
                if idx < 0 or idx >= device_count:
                    raise ValueError(f"GPU index {idx} out of range (0..{device_count-1})")
            self.gpu_indices = list(gpu_indices)

        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.gpu_indices]

        self._use_total_energy_counter = True
        try:
            for h in self.handles:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        except pynvml.NVMLError:
            self._use_total_energy_counter = False

        self._start_energy_mj = 0.0

    def _sample_loop(self):
        last_time = time.time()
        while not self._stop_flag:
            now = time.time()
            dt = now - last_time
            last_time = now

            total_power_watts = 0.0
            for h in self.handles:
                p_mw = pynvml.nvmlDeviceGetPowerUsage(h)
                total_power_watts += p_mw / 1000.0

            self.energy_joules += total_power_watts * dt
            time.sleep(self.sample_interval)

    def start(self):
        if self._use_total_energy_counter:
            self._start_energy_mj = sum(
                float(pynvml.nvmlDeviceGetTotalEnergyConsumption(h)) for h in self.handles
            )
            return

        if self._thread is not None:
            return

        self._stop_flag = False
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._use_total_energy_counter:
            end_energy_mj = sum(
                float(pynvml.nvmlDeviceGetTotalEnergyConsumption(h)) for h in self.handles
            )
            self.energy_joules = (end_energy_mj - self._start_energy_mj) / 1000.0
            pynvml.nvmlShutdown()
            return

        if self._thread is not None:
            self._stop_flag = True
            self._thread.join()
            self._thread = None

        pynvml.nvmlShutdown()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()