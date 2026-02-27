#!/usr/bin/env python
"""Test script for updated tactile sensor driver with binary protocol"""

import sys
import time
import numpy as np

# Add src to path
sys.path.insert(0, '/home/tao/lerobot_tactile/src')

from lerobot.sensors.tactile_sensor import TactileSensor


def main():
    print("Testing updated tactile sensor with binary protocol...")
    print("=" * 60)

    # Initialize sensor with visualization enabled
    sensor = TactileSensor(
        port="/dev/ttyUSB0",
        baud_rate=2_000_000,
        timeout=0.01,
        auto_calibrate=True,
        enable_visualization=True,
        window_name="Tactile Test - Binary Protocol",
        threshold=25.0,
        noise_scale=30.0,
        temporal_alpha=0.2,
    )

    if not sensor.is_connected:
        print("Failed to connect to sensor")
        return

    print(f"Connected: {sensor.is_connected}")
    print(f"Calibrated: {sensor.is_calibrated}")
    print(f"Shape: {sensor.shape}")
    print(f"Baseline shape: {sensor.baseline.shape if sensor.baseline is not None else 'None'}")
    print("=" * 60)

    # Start continuous reading
    sensor.start_continuous_read()

    print("\nReading data for 30 seconds...")
    print("Press Ctrl+C to stop early")
    print("=" * 60)

    try:
        start_time = time.time()
        frame_count = 0
        last_print_time = start_time

        while time.time() - start_time < 30:
            data = sensor.get_latest_data()

            if data is not None:
                frame_count += 1

                # Print stats every second
                now = time.time()
                if now - last_print_time >= 1.0:
                    fps = frame_count / (now - start_time)
                    print(f"Time: {now - start_time:.1f}s | Frames: {frame_count} | FPS: {fps:.1f} | "
                          f"Min: {data.min():.1f} | Max: {data.max():.1f} | Mean: {data.mean():.1f}")
                    last_print_time = now

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    finally:
        # Cleanup
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0

        print("=" * 60)
        print(f"Total frames: {frame_count}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        print("=" * 60)

        sensor.disconnect()
        print("Sensor disconnected")


if __name__ == "__main__":
    main()
