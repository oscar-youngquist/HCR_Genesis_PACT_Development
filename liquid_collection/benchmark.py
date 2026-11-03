"""Whole-process GPU sampling includes Genesis allocations outside Torch."""
import os
import subprocess
import threading


class MemoryMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.peak_mib = None
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                    '--format=csv,noheader,nounits'], text=True, timeout=3, stderr=subprocess.DEVNULL)
                for row in output.splitlines():
                    pid, memory = row.split(',')
                    if int(pid) == os.getpid():
                        self.peak_mib = max(self.peak_mib or 0, int(memory))
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.error = str(exc)
            self.stop.wait(.5)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=4)
        return dict(process_gpu_peak_sampled_mib=self.peak_mib,
                    process_gpu_sampling_interval_s=.5, process_gpu_monitor_error=self.error)
