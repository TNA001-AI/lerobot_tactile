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
from typing import Optional

import cv2
import numpy as np
import serial
from scipy.ndimage import gaussian_filter


class TactileSensor:
    """Interface for 16x32 tactile sensor array via USB serial communication"""
    
    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud_rate: int = 2000000,
        timeout: float = 1.0,
        shape: tuple[int, int] = (16, 32),
        auto_calibrate: bool = True,
        enable_visualization: bool = True,
        window_name: str = "Tactile Sensor",
        threshold: float = 30.0,
        noise_scale: float = 50.0,
        temporal_alpha: float = 0.2,
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
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.shape = shape

        self.serial_conn: Optional[serial.Serial] = None
        self.baseline: Optional[np.ndarray] = None
        self.is_connected = False
        self.is_calibrated = False

        # Buffer for building current frame from text lines
        self._current_frame: list = []
        self._last_complete_frame: Optional[np.ndarray] = None

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
        self._prev_frame: Optional[np.ndarray] = None
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
            
            # Wait for connection to stabilize
            time.sleep(0.1)
            
            # Clear any existing data in buffer
            self.serial_conn.flushInput()
            self.serial_conn.flushOutput()
            
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
    
    def read_raw_data(self) -> Optional[np.ndarray]:
        """Read raw data from sensor (ASCII text format, line by line)"""
        if not self.is_connected or not self.serial_conn:
            logging.warning("Sensor not connected")
            return None

        try:
            # Read lines until we have a complete frame
            max_attempts = 100  # Prevent infinite loop
            attempts = 0

            while attempts < max_attempts:
                if self.serial_conn.in_waiting > 0:
                    try:
                        line = self.serial_conn.readline().decode('utf-8').strip()
                    except Exception:
                        line = ""

                    # Short line indicates end of frame
                    if len(line) < 10:
                        if self._current_frame is not None and len(self._current_frame) == self.shape[0]:
                            # Complete frame received
                            self._last_complete_frame = np.array(self._current_frame, dtype=np.float32)
                        self._current_frame = []

                        if self._last_complete_frame is not None:
                            return self._last_complete_frame.copy()
                        continue

                    # Parse the line as space-separated integers
                    if self._current_frame is not None:
                        try:
                            str_values = line.split()
                            int_values = [int(val) for val in str_values]
                            self._current_frame.append(int_values)
                        except ValueError:
                            # Skip malformed lines
                            pass

                attempts += 1
                time.sleep(0.001)  # Small delay to avoid busy waiting

            return self._last_complete_frame.copy() if self._last_complete_frame is not None else None

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
        """Read processed tactile data (raw - baseline)"""
        raw_data = self.read_raw_data()
        if raw_data is None:
            return None

        if not self.is_calibrated or self.baseline is None:
            logging.warning("Sensor not calibrated, returning raw data")
            return raw_data.astype(np.float32)

        # Subtract baseline and ensure non-negative values
        processed_data = raw_data.astype(np.float32) - self.baseline
        processed_data = np.maximum(processed_data, 0)

        # Update visualization if enabled
        if self.enable_visualization:
            self.update_visualization(processed_data)

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
        """Background loop for continuous data reading"""
        while not self._stop_event.is_set():
            data = self.read_data()
            if data is not None:
                with self._data_lock:
                    self._latest_data = data.copy()
                # Update visualization if enabled
                if self.enable_visualization:
                    self.update_visualization(data)
            time.sleep(0.01)  # 100 Hz reading rate
    
    def get_latest_data(self) -> Optional[np.ndarray]:
        """Get latest data from continuous reading"""
        with self._data_lock:
            return self._latest_data.copy() if self._latest_data is not None else None

    def _init_visualization(self):
        """Initialize the visualization window"""
        window_width = self.shape[1] * 30
        window_height = self.shape[0] * 30
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, window_width, window_height)
        self._prev_frame = np.zeros(self.shape, dtype=np.float32)
        self._visualization_initialized = True
        logging.info(f"Visualization window '{self.window_name}' initialized")

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
            data: Raw or processed tactile data (already baseline-subtracted)

        Returns:
            Normalized data in range [0, 1]
        """
        # Apply threshold (data already has baseline subtracted)
        contact_data = data - self.threshold
        contact_data = np.clip(contact_data, 0, 100)

        # Normalize based on max value
        max_val = np.max(contact_data)
        if max_val < self.threshold:
            # Low pressure - use noise scale normalization
            normalized = contact_data / self.noise_scale
        else:
            # High pressure - normalize by max value
            normalized = contact_data / max_val

        return np.clip(normalized, 0, 1)

    def _apply_gaussian_blur(self, data: np.ndarray, sigma: float = 0.5) -> np.ndarray:
        """
        Apply Gaussian blur to smooth the visualization.

        Args:
            data: Input data array
            sigma: Standard deviation for Gaussian kernel

        Returns:
            Blurred data array
        """
        return gaussian_filter(data, sigma=sigma)

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

        # Normalize the data
        normalized = self._normalize_data(data)

        # Apply temporal filtering
        filtered = self._temporal_filter(normalized)

        # Scale to 0-255 and convert to uint8
        scaled = (filtered * 255).astype(np.uint8)

        # Apply color map
        colormap = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)

        # Display
        cv2.imshow(self.window_name, colormap)
        cv2.waitKey(1)

        return True

    def close_visualization(self):
        """Close the visualization window"""
        if self._visualization_initialized:
            cv2.destroyWindow(self.window_name)
            self._visualization_initialized = False
            logging.info(f"Visualization window '{self.window_name}' closed")

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