#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tactile sensor interface for LeRobot"""

import logging
import time
import threading
import multiprocessing as mp
from typing import Optional

import cv2
import numpy as np
import serial


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
    """Interface for 16x32 tactile sensor array via USB serial communication"""

    # Binary protocol constants
    MAGIC = b"\xAA\x55"
    ROWS, COLS = 16, 32
    FRAME_BYTES = ROWS * COLS  # 512

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud_rate: int = 2000000,
        timeout: float = 0.01,
        shape: tuple[int, int] = (16, 32),
        auto_calibrate: bool = True,
        enable_visualization: bool = True,
        window_name: str = "Tactile Sensor",
        threshold: float = 25.0,
        noise_scale: float = 30.0,
        temporal_alpha: float = 0.2,
        display_rows: int = 12,
    ):
        """
        Initialize tactile sensor interface

        Args:
            port: USB serial port for sensor communication
            baud_rate: Serial communication baud rate
            timeout: Serial communication timeout in seconds
            shape: Expected tactile array shape (height, width)
            auto_calibrate: Whether to automatically calibrate sensor on initialization
            enable_visualization: Whether to enable real-time visualization
            window_name: Name of the visualization window
            threshold: Threshold for contact detection
            noise_scale: Scale factor for normalizing low-pressure readings
            temporal_alpha: Blending factor for temporal smoothing (0-1)
            display_rows: Number of rows to show in the visualization window.
                Set to 12 to hide the 4 silent rows at the top of a 16x32 sensor.
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.shape = shape

        self.serial_conn: Optional[serial.Serial] = None
        self.baseline: Optional[np.ndarray] = None
        self.is_connected = False
        self.is_calibrated = False

        # Binary protocol buffers
        self._ring_buffer = bytearray()
        self._frame_buffer = bytearray(self.FRAME_BYTES)

        # Threading for continuous data collection
        self._stop_event = threading.Event()
        self._data_thread: Optional[threading.Thread] = None
        self._latest_data: Optional[np.ndarray] = None
        self._data_lock = threading.Lock()

        # Visualization settings
        self.enable_visualization = enable_visualization
        self.window_name = window_name
        self.threshold = threshold
        self.noise_scale = noise_scale
        self.temporal_alpha = temporal_alpha
        self.display_rows = min(display_rows, self.ROWS)
        self._prev_frame: Optional[np.ndarray] = None
        self._viz_queue: Optional[mp.Queue] = None
        self._viz_process: Optional[mp.Process] = None
        self._visualization_initialized = False

        # Connect and calibrate
        self.connect()
        if auto_calibrate:
            self.calibrate()

        # Initialize visualization if enabled
        if self.enable_visualization:
            self._init_visualization()

    def connect(self) -> bool:
        """Connect to tactile sensor"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
            )

            # Clear any existing data in buffer
            self.serial_conn.flush()
            self.serial_conn.reset_input_buffer()

            self.is_connected = True
            logging.info(f"Connected to tactile sensor on {self.port}")
            return True

        except serial.SerialException as e:
            logging.error(f"Failed to connect to tactile sensor on {self.port}: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            logging.error(f"Unexpected error connecting to tactile sensor: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Disconnect from tactile sensor"""
        self.stop_continuous_read()
        self.close_visualization()

        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.is_connected = False
            logging.info("Disconnected from tactile sensor")

    def _read_exact(self, n: int) -> Optional[bytearray]:
        """Read exactly n bytes from serial (blocking up to timeout loops)."""
        buf = bytearray(n)
        mv = memoryview(buf)
        got = 0
        max_retries = 100
        retries = 0

        while got < n and retries < max_retries:
            r = self.serial_conn.readinto(mv[got:])
            if r is None:
                r = 0
            if r == 0:
                retries += 1
                time.sleep(0.001)
                continue
            got += r
            retries = 0

        if got < n:
            return None
        return buf

    def read_raw_data(self) -> Optional[np.ndarray]:
        """Read raw data from sensor (binary format with magic header 0xAA 0x55)"""
        if not self.is_connected or not self.serial_conn:
            logging.warning("Sensor not connected")
            return None

        try:
            # Read chunks and search for magic header
            while True:
                # Read available data
                chunk = self.serial_conn.read(8192)
                if chunk:
                    self._ring_buffer.extend(chunk)

                # Keep buffer bounded
                if len(self._ring_buffer) > 50000:
                    self._ring_buffer = self._ring_buffer[-50000:]

                # Try to parse all available frames in buffer
                # Search for magic header
                idx = self._ring_buffer.find(self.MAGIC)
                if idx < 0:
                    # Keep last byte in case marker splits across chunks
                    if len(self._ring_buffer) > 1:
                        self._ring_buffer = self._ring_buffer[-1:]
                    continue

                # Drop bytes before marker
                if idx > 0:
                    del self._ring_buffer[:idx]

                # Need at least magic header + frame data
                if len(self._ring_buffer) < 2 + self.FRAME_BYTES:
                    continue

                # Consume magic header
                del self._ring_buffer[:2]

                # Extract frame bytes (may need to read more if not enough in buffer)
                if len(self._ring_buffer) >= self.FRAME_BYTES:
                    self._frame_buffer[:] = self._ring_buffer[:self.FRAME_BYTES]
                    del self._ring_buffer[:self.FRAME_BYTES]
                else:
                    # Partial frame in buffer - need to read remaining bytes
                    have = len(self._ring_buffer)
                    self._frame_buffer[:have] = self._ring_buffer[:have]
                    del self._ring_buffer[:have]
                    rem = self.FRAME_BYTES - have
                    rest = self._read_exact(rem)
                    if rest is None:
                        self._ring_buffer.clear()
                        continue
                    self._frame_buffer[have:] = rest

                # Convert to numpy array and return
                frame = np.frombuffer(self._frame_buffer, dtype=np.uint8).reshape((self.ROWS, self.COLS)).astype(np.float32)
                return frame

        except serial.SerialException as e:
            logging.warning(f"Serial error: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error reading sensor data: {e}")
            return None

    def calibrate(self, num_samples: int = 30) -> bool:
        """Calibrate sensor by taking baseline reading using median"""
        if not self.is_connected:
            logging.error("Cannot calibrate: sensor not connected")
            return False

        logging.info(f"Calibrating tactile sensor with {num_samples} samples...")

        samples = []
        for i in range(num_samples):
            data = self.read_raw_data()
            if data is not None:
                samples.append(data)
            else:
                logging.warning(f"Failed to read calibration sample {i + 1}/{num_samples}")

            # Small delay between samples
            time.sleep(0.01)

        if len(samples) < num_samples * 0.5:  # Require at least 50% success rate
            logging.error(f"Calibration failed: only {len(samples)}/{num_samples} valid samples")
            return False

        # Compute baseline as median of samples (more robust to outliers)
        self.baseline = np.median(samples, axis=0).astype(np.float32)
        self.is_calibrated = True

        logging.info("Tactile sensor calibration completed")
        return True

    def read_data(self) -> Optional[np.ndarray]:
        """Read processed tactile data (raw - baseline - threshold)"""
        raw_data = self.read_raw_data()
        if raw_data is None:
            return None

        if not self.is_calibrated or self.baseline is None:
            logging.warning("Sensor not calibrated, returning raw data")
            return raw_data.astype(np.float32)

        # Subtract baseline and threshold, then clip (matching fast-32-16.py line 164)
        processed_data = raw_data.astype(np.float32) - self.baseline - self.threshold
        processed_data = np.clip(processed_data, 0, 100)

        return processed_data

    def start_continuous_read(self):
        """Start continuous data reading in background thread"""
        if self._data_thread and self._data_thread.is_alive():
            logging.warning("Continuous reading already started")
            return

        self._stop_event.clear()
        self._data_thread = threading.Thread(target=self._continuous_read_loop)
        self._data_thread.daemon = True
        self._data_thread.start()

        logging.info("Started continuous tactile data reading")

    def stop_continuous_read(self):
        """Stop continuous data reading"""
        if self._data_thread and self._data_thread.is_alive():
            self._stop_event.set()
            self._data_thread.join(timeout=1.0)
            logging.info("Stopped continuous tactile data reading")

    def _continuous_read_loop(self):
        """Background loop for continuous data reading and visualization"""
        while not self._stop_event.is_set():
            data = self.read_data()
            if data is not None:
                with self._data_lock:
                    self._latest_data = data.copy()
                self.update_visualization(data)


    def get_latest_data(self) -> Optional[np.ndarray]:
        """Get latest data from continuous reading"""
        with self._data_lock:
            return self._latest_data.copy() if self._latest_data is not None else None

    def _init_visualization(self):
        """Launch a subprocess to display the tactile visualization window."""
        self._viz_queue = mp.Queue(maxsize=2)
        display_shape = (self.display_rows, self.shape[1])
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
        """
        Apply temporal smoothing filter.

        Args:
            new_frame: Current frame data

        Returns:
            Temporally smoothed frame
        """
        if self._prev_frame is None:
            self._prev_frame = np.zeros_like(new_frame)
        filtered = self.temporal_alpha * new_frame + (1 - self.temporal_alpha) * self._prev_frame
        self._prev_frame = filtered.copy()
        return filtered

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize tactile data for visualization.

        Args:
            data: Processed tactile data (already baseline and threshold subtracted)

        Returns:
            Normalized data in range [0, 1]
        """
        max_val = np.max(data)
        if max_val < self.threshold:
            # Low pressure - use noise scale normalization
            normalized = data / self.noise_scale
        else:
            # High pressure - normalize by max value
            normalized = data / (max_val + 1e-6)

        return np.clip(normalized, 0, 1)


    def update_visualization(self, data: Optional[np.ndarray] = None) -> bool:
        """
        Update the visualization with current tactile data.

        Args:
            data: Tactile data to visualize. If None, uses latest data from continuous reading.

        Returns:
            True if visualization was updated successfully
        """
        if not self.enable_visualization:
            return False

        if not self._visualization_initialized:
            self._init_visualization()

        # Get data if not provided
        if data is None:
            data = self.get_latest_data()

        if data is None:
            return False

        # Crop to display_rows before visualization (hides silent rows at the top)
        data = data[-self.display_rows:, :]

        # Normalize the data
        normalized = self._normalize_data(data)

        # Apply temporal filtering
        filtered = self._temporal_filter(normalized)

        # Scale to 0-255 and convert to uint8
        scaled = (filtered * 255).astype(np.uint8)

        # Apply color map
        colormap = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)

        # Send to visualization subprocess (non-blocking; drop frame if process is busy)
        if self._viz_queue is not None:
            try:
                self._viz_queue.put_nowait(colormap)
            except Exception:
                pass  # Queue full — skip frame, don't block

        return True

    def close_visualization(self):
        """Stop the visualization subprocess."""
        if self._visualization_initialized:
            if self._viz_queue is not None:
                try:
                    self._viz_queue.put_nowait(None)  # sentinel to stop worker
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
        """Cleanup on deletion"""
        try:
            self.disconnect()
        except:
            pass  # Ignore cleanup errors
