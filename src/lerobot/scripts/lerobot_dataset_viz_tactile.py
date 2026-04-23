#!/usr/bin/env python
"""Visualize a LeRobotDataset episode in Rerun with tactile-frame support.

This is a tactile-aware sibling of ``lerobot_dataset_viz.py``. In addition to
the standard cameras / state / action logging, every feature whose key starts
with ``observation.tactile`` and whose declared shape is 2D is logged as a
colormapped depth image, so you can scrub tactile frames in sync with the
rest of the episode.

Examples:

- Spawn a local viewer on a dataset stored on disk:
```
lerobot-dataset-viz-tactile \
    --repo-id so101_tactile_pens_bag_dev \
    --root /home/tao/Downloads/so101_tactile_pens_bag_dev \
    --episode-index 0
```

- Save to .rrd for later viewing:
```
lerobot-dataset-viz-tactile \
    --repo-id so101_tactile_pens_bag_dev \
    --root /home/tao/Downloads/so101_tactile_pens_bag_dev \
    --episode-index 0 \
    --save 1 \
    --output-dir /tmp/rrd
```
"""

import argparse
import gc
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, OBS_TACTILE, REWARD
from lerobot.utils.utils import init_logging

# Mirror the live-viz pipeline in `sensors/tactile_sensor.py`:
# adaptive normalization + EMA temporal filter + COLORMAP_VIRIDIS.
# We reuse cv2.applyColorMap purely as a colormap function; the result is
# rendered inside the rerun viewer via rr.Image — no cv2 window is opened.
TACTILE_THRESHOLD = 25.0
TACTILE_NOISE_SCALE = 30.0
TACTILE_TEMPORAL_ALPHA = 0.2
TACTILE_PIXEL_SCALE = 60  # each sensor cell becomes a SCALE x SCALE block


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def tactile_keys(dataset: LeRobotDataset) -> list[str]:
    """Return dataset feature keys that store 2D tactile frames per step."""
    keys = []
    for key, ft in dataset.features.items():
        if key != OBS_TACTILE and not key.startswith(f"{OBS_TACTILE}."):
            continue
        if len(ft.get("shape", ())) != 2:
            continue
        keys.append(key)
    return keys


def _normalize_tactile(data: np.ndarray) -> np.ndarray:
    """Adaptive normalization matching TactileSensor._normalize_data."""
    max_val = float(np.max(data))
    if max_val < TACTILE_THRESHOLD:
        normalized = data / TACTILE_NOISE_SCALE
    else:
        normalized = data / (max_val + 1e-6)
    return np.clip(normalized, 0.0, 1.0)


def build_tactile_rgb(frame: np.ndarray, prev: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (RGB uint8 image, updated prev frame) mimicking the live cv2 viz.

    Pipeline matches `sensors/tactile_sensor.py`:
      adaptive normalize -> EMA temporal filter -> uint8 -> viridis -> upscale.
    The RGB array is rendered inside rerun via rr.Image (no cv2 window).
    """
    normalized = _normalize_tactile(frame)
    if prev is None:
        prev = np.zeros_like(normalized)
    filtered = TACTILE_TEMPORAL_ALPHA * normalized + (1.0 - TACTILE_TEMPORAL_ALPHA) * prev

    scaled = (filtered * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if TACTILE_PIXEL_SCALE > 1:
        h, w = rgb.shape[:2]
        rgb = cv2.resize(
            rgb,
            (w * TACTILE_PIXEL_SCALE, h * TACTILE_PIXEL_SCALE),
            interpolation=cv2.INTER_NEAREST,
        )
    return rgb, filtered


def log_tactile_frame(entity_path: str, frame: torch.Tensor, prev: np.ndarray | None) -> np.ndarray:
    """Log one (H, W) tactile frame as a viridis RGB image in rerun."""
    arr = frame.detach().cpu().to(torch.float32).numpy()
    rgb, filtered = build_tactile_rgb(arr, prev)
    rr.log(entity_path, rr.Image(rgb))
    return filtered


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    grpc_port: int = 9876,
    save: bool = False,
    output_dir: Path | None = None,
    display_compressed_images: bool = False,
    **kwargs,
) -> Path | None:
    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id
    tac_keys = tactile_keys(dataset)
    if tac_keys:
        logging.info(f"Detected tactile features: {tac_keys}")
    else:
        logging.warning("No tactile features detected; falling back to standard visualization.")

    logging.info("Loading dataloader")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    # TODO(rcadene): remove `gc.collect` when rerun version 0.16 is out, which includes a fix
    gc.collect()

    if mode == "distant":
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        logging.info(f"Connect to a Rerun Server: rerun rerun+http://IP:{grpc_port}/proxy")
        rr.serve_web_viewer(open_browser=False, web_port=web_port, connect_to=server_uri)

    logging.info("Logging to Rerun")

    first_index = None
    tactile_prev: dict[str, np.ndarray] = {}
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        if first_index is None:
            first_index = batch["index"][0].item()
        # iterate over the batch
        for i in range(len(batch["index"])):
            rr.set_time("frame_index", sequence=batch["index"][i].item() - first_index)
            rr.set_time("timestamp", timestamp=batch["timestamp"][i].item())

            # display each camera image
            for key in dataset.meta.camera_keys:
                img = to_hwc_uint8_numpy(batch[key][i])
                img_entity = rr.Image(img).compress() if display_compressed_images else rr.Image(img)
                rr.log(key, entity=img_entity)

            # display each tactile frame as a viridis RGB image (matches live viz)
            for key in tac_keys:
                if key not in batch:
                    continue
                tactile_prev[key] = log_tactile_frame(key, batch[key][i], tactile_prev.get(key))

            # display each dimension of action space (e.g. actuators command)
            if ACTION in batch:
                for dim_idx, val in enumerate(batch[ACTION][i]):
                    rr.log(f"{ACTION}/{dim_idx}", rr.Scalars(val.item()))

            # display each dimension of observed state space (e.g. agent position in joint space)
            if OBS_STATE in batch:
                for dim_idx, val in enumerate(batch[OBS_STATE][i]):
                    rr.log(f"state/{dim_idx}", rr.Scalars(val.item()))

            if DONE in batch:
                rr.log(DONE, rr.Scalars(batch[DONE][i].item()))

            if REWARD in batch:
                rr.log(REWARD, rr.Scalars(batch[REWARD][i].item()))

            if "next.success" in batch:
                rr.log("next.success", rr.Scalars(batch["next.success"][i].item()))

    if mode == "local" and save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        return rrd_path

    elif mode == "distant":
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mode", type=str, default="local", choices=["local", "distant"])
    parser.add_argument("--web-port", type=int, default=9090)
    parser.add_argument("--grpc-port", type=int, default=9876)
    parser.add_argument("--save", type=int, default=0)
    parser.add_argument("--tolerance-s", type=float, default=1e-4)
    parser.add_argument("--display-compressed-images", action="store_true")

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")

    init_logging()
    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, episodes=[args.episode_index], root=root, tolerance_s=tolerance_s)

    visualize_dataset(dataset, **kwargs)


if __name__ == "__main__":
    main()
