# LeflexiTac

A fork of [LeRobot](https://github.com/huggingface/lerobot) that adds **tactile sensing** to robot policies. We integrate FlexiTac tactile sensors into the LeRobot stack — data collection, dataset format, and policy training/inference (ACT, Diffusion, Pi0.5, SmolVLA) — so policies can learn from touch in addition to vision and proprioception.

🌐 **Project website (overview, demos, and full step-by-step guide):** [https://tna001-ai.github.io/LeFlexiTac/docs.html](https://tna001-ai.github.io/LeFlexiTac/docs.html)

The website is the primary reference for usage, hardware setup, data collection, and training recipes. This README only covers what you need to get the environment running.

## Demo

https://github.com/user-attachments/assets/5335e61f-08e4-48a0-aae2-5739038b5505

## Environment Setup

You need two components: the LeRobot library (this repo) and the `pyflexitac` driver for the tactile sensor.

### 1. LeRobot (from source)

> [!IMPORTANT]
> This fork is **not** available on PyPI — `pip install lerobot` will install the upstream version without tactile support. You must install from source.

Follow the official [LeRobot installation guide](https://huggingface.co/docs/lerobot/installation) for system prerequisites (ffmpeg, conda, etc.), then install **this** repo from source instead of the PyPI package:

```bash
# Create the conda env (Python 3.12, as recommended by LeRobot)
conda create -y -n lerobot python=3.12
conda activate lerobot

# Install ffmpeg via conda (see the LeRobot installation guide for details)
conda install -y -c conda-forge ffmpeg

# Clone and install this fork in editable mode
git clone https://github.com/TNA001-AI/lerobot_tactile.git
cd lerobot_tactile
pip install -e .
```

### 2. PyFlexiTac (tactile sensor driver)

The tactile sensors are driven by [PyFlexiTac](https://github.com/WT-MM/PyFlexiTac). Install it into the same conda env:

```bash
# In the same `lerobot` env
conda activate lerobot

# From PyPI
pip install flexitac

# From source
git clone https://github.com/WT-MM/PyFlexiTac.git
cd PyFlexiTac
pip install -e .
```

Refer to the PyFlexiTac README for sensor wiring, permissions, flashing, and a quick connectivity test before running any LeRobot scripts.

### Verify

```bash
conda activate lerobot
python -c "import lerobot, pyflexitac; print(lerobot.__file__); print(pyflexitac.__file__)"
```

Both imports should succeed and point into your local clones.

## Next Steps

For data collection, training, and evaluation with tactile sensors, follow the guide on the [project website](https://tna001-ai.github.io/LeFlexiTac/docs.html).

## Citation

If you use this work, please cite both the upstream LeRobot project and the tactile extension.

```bibtex
@misc{cadene2024lerobot,
  author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
  title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
  howpublished = "\url{https://github.com/huggingface/lerobot}",
  year = {2024}
}

@article{tao2026leflexitac,
  author = {Naian Tao and Yifan He and Wesley Maa and Binghao Huang and Yunzhu Li},
  title = {{LeFlexiTac}: Giving Robots a Sense of Touch},
  journal = {Columbia University RoboPIL Blog},
  year = {2026},
  note = {\url{https://tna001-ai.github.io/tactile-lerobot-website/}},
}
```
