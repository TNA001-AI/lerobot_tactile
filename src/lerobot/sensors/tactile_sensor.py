#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tactile sensor interface for LeRobot (wraps the flexitac package)."""

import logging
import multiprocessing as mp
import threading
import time
from typing import Optional

import cv2
import numpy as np
from flexitac import FlexiTacSensor


def _visualization_worker(queue: mp.Queue, window_name: str, shape: tuple):
    """Subprocess entry point: receives colormap frames and displays via OpenCV."""
    window_width = shape[1] * 60
    window_height = shape[0] * 60
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)
    while True:
        try:
            frame = queue.get(timeout=1.0)
        except Exception:
            continue
        if frame is None:
            break
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


class TactileSensor:
    """LeRobot wrapper around ``flexitac.FlexiTacSensor`` with threading + cv2 viz."""

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud_rate: int = 2_000_000,
        shape: tuple[int, int] = (12, 32),
        baseline: float | None = None,
        init_frames: int = 30,
        enable_visualization: bool = True,
        window_name: str = "Tactile Sensor",
        threshold: float = 25.0,
        noise_scale: float = 30.0,
        temporal_alpha: float = 0.2,
    ):
        """Initialize tactile sensor.

        Args:
            port: USB serial port.
            baud_rate: Serial baud rate (matches flashed firmware).
            shape: (rows, cols) of the tactile grid.
            baseline: If set, skips the calibration phase on startup. The scalar
                (e.g. ``20.0``) is applied to every pixel as a fixed baseline.
                Leave as ``None`` to auto-calibrate on first read.
            init_frames: Number of frames collected during auto-calibration.
            enable_visualization: Launch cv2 viz subprocess on startup.
            window_name: Viz window title.
            threshold: Contact-detection threshold (ADC counts above baseline).
            noise_scale: Divisor used to normalize sub-threshold readings.
            temporal_alpha: EMA blending factor for smoothing the viz (0-1).
        """
        self.port = port
        self.baud_rate = baud_rate
        self.shape = shape
        self.rows, self.cols = shape

        self._sensor = FlexiTacSensor(
            port,
            rows=self.rows,
            cols=self.cols,
            baud=baud_rate,
            threshold=threshold,
            noise_scale=noise_scale,
            init_frames=init_frames,
            baseline=baseline,
        )
        self.is_connected = False

        # Threading for continuous data collection
        self._stop_event = threading.Event()
        self._data_thread: Optional[threading.Thread] = None
        self._latest_data: Optional[np.ndarray] = None
        self._data_lock = threading.Lock()

        # Visualization settings
        self.enable_visualization = enable_visualization
        self.window_name = window_name
        self.temporal_alpha = temporal_alpha
        self._prev_frame: Optional[np.ndarray] = None
        self._viz_queue: Optional[mp.Queue] = None
        self._viz_process: Optional[mp.Process] = None
        self._visualization_initialized = False

        self.connect()
        if self.enable_visualization:
            self._init_visualization()

    @property
    def baseline(self) -> Optional[np.ndarray]:
        return self._sensor.baseline

    @property
    def is_calibrated(self) -> bool:
        return self._sensor.baseline is not None

    def wait_for_calibration(self, timeout_s: float = 30.0, poll_s: float = 0.1) -> bool:
        """Block until baseline is set or ``timeout_s`` elapses.

        Useful when auto-calibration runs in the background continuous-read
        thread and a caller (e.g. dataset recorder) needs the baseline to be
        present before proceeding. Returns True if calibrated, False on timeout.
        """
        deadline = time.monotonic() + timeout_s
        while self._sensor.baseline is None:
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_s)
        return True

    def metadata(self) -> dict:
        """Return a JSON-serializable snapshot of sensor config + current baseline.

        Intended for embedding in dataset metadata (e.g. ``info.json``) so the
        normalization regime that produced recorded frames is reproducible.
        """
        baseline = self._sensor.baseline
        return {
            "port": self.port,
            "baud_rate": self.baud_rate,
            "rows": self.rows,
            "cols": self.cols,
            "threshold": float(self._sensor.threshold),
            "noise_scale": float(self._sensor.noise_scale),
            "init_frames": int(self._sensor.init_frames),
            "is_calibrated": baseline is not None,
            "baseline": baseline.tolist() if baseline is not None else None,
        }

    def connect(self) -> bool:
        try:
            self._sensor.open()
            self.is_connected = True
            logging.info(f"Connected to tactile sensor on {self.port}")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to tactile sensor on {self.port}: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        self.stop_continuous_read()
        self.close_visualization()
        self._sensor.close()
        self.is_connected = False
        logging.info("Disconnected from tactile sensor")

    def calibrate(self, num_samples: int = 30) -> bool:
        try:
            self._sensor.calibrate(n=num_samples)
            logging.info("Tactile sensor calibration completed")
            return True
        except Exception as e:
            logging.error(f"Calibration failed: {e}")
            return False

    def read_data(self) -> Optional[np.ndarray]:
        """Read normalized tactile data (0-1 range) from the sensor."""
        try:
            frame = self._sensor.read_latest()
            return frame.normalized
        except Exception as e:
            logging.warning(f"Read error: {e}")
            return None

    def start_continuous_read(self):
        if self._data_thread and self._data_thread.is_alive():
            logging.warning("Continuous reading already started")
            return
        self._stop_event.clear()
        self._data_thread = threading.Thread(target=self._continuous_read_loop, daemon=True)
        self._data_thread.start()
        logging.info("Started continuous tactile data reading")

    def stop_continuous_read(self):
        if self._data_thread and self._data_thread.is_alive():
            self._stop_event.set()
            self._data_thread.join(timeout=1.0)
            logging.info("Stopped continuous tactile data reading")

    def _continuous_read_loop(self):
        while not self._stop_event.is_set():
            data = self.read_data()
            if data is not None:
                with self._data_lock:
                    self._latest_data = data.copy()
                self.update_visualization(data)
            else:
                time.sleep(0.01)

    def get_latest_data(self) -> Optional[np.ndarray]:
        with self._data_lock:
            return self._latest_data.copy() if self._latest_data is not None else None

    def _init_visualization(self):
        self._viz_queue = mp.Queue(maxsize=2)
        display_shape = (self.rows, self.cols)
        self._viz_process = mp.Process(
            target=_visualization_worker,
            args=(self._viz_queue, self.window_name, display_shape),
            daemon=True,
        )
        self._viz_process.start()
        self._prev_frame = np.zeros(display_shape, dtype=np.float32)
        self._visualization_initialized = True
        logging.info(f"Visualization process started for '{self.window_name}'")

    def _temporal_filter(self, new_frame: np.ndarray) -> np.ndarray:
        if self._prev_frame is None:
            self._prev_frame = np.zeros_like(new_frame)
        filtered = self.temporal_alpha * new_frame + (1 - self.temporal_alpha) * self._prev_frame
        self._prev_frame = filtered.copy()
        return filtered

    def update_visualization(self, data: Optional[np.ndarray] = None) -> bool:
        if not self.enable_visualization:
            return False
        if not self._visualization_initialized:
            self._init_visualization()
        if data is None:
            data = self.get_latest_data()
        if data is None:
            return False

        filtered = self._temporal_filter(data)
        scaled = (filtered * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)

        if self._viz_queue is not None:
            try:
                self._viz_queue.put_nowait(colormap)
            except Exception:
                pass
        return True

    def close_visualization(self):
        if self._visualization_initialized:
            if self._viz_queue is not None:
                try:
                    self._viz_queue.put_nowait(None)
                except Exception:
                    pass
            if self._viz_process is not None and self._viz_process.is_alive():
                self._viz_process.join(timeout=2.0)
                if self._viz_process.is_alive():
                    self._viz_process.terminate()
            self._viz_queue = None
            self._viz_process = None
            self._visualization_initialized = False
            logging.info(f"Visualization process for '{self.window_name}' stopped")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass
