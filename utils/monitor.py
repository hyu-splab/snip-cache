import csv
import os
import threading
import time
import logging
import psutil


class ResourceMonitor:
    def __init__(self, interval=1.0, output_file="cpu_usage_log.csv"):
        self.interval = interval
        self.output_file = output_file
        self._stop_event = threading.Event()
        self._thread = None
        self.start_time = 0.0
        self.current_phase = "idle"
        self.records = []
        self._lock = threading.Lock()

    def set_output_file(self, output_file: str):
        self.output_file = output_file

    def start(self):
        self.start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background thread gracefully"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 1.0)
        self._save_to_csv()

    def mark_phase(self, phase_name):
        """Immediately record phase change"""
        with self._lock:
            self.current_phase = phase_name
            if self.start_time:
                elapsed = time.time() - self.start_time
                cpu = psutil.cpu_percent(interval=0)  # non-blocking
                self.records.append((elapsed, cpu, phase_name))

    def _run(self):
        """Background sampler loop"""
        while not self._stop_event.is_set():
            start_loop = time.time()
            cpu = psutil.cpu_percent(interval=0)  # non-blocking
            elapsed = time.time() - self.start_time
            with self._lock:
                self.records.append((elapsed, cpu, self.current_phase))
            while (time.time() - start_loop) < self.interval:
                if self._stop_event.is_set():
                    return
                time.sleep(0.05)

    def _save_to_csv(self):
        dir_path = os.path.dirname(self.output_file)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(self.output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time_sec", "cpu_percent", "phase"])
            with self._lock:
                writer.writerows(self.records)
        print(f"[INFO] Saved {len(self.records)} records to {self.output_file}")


monitor = ResourceMonitor()
