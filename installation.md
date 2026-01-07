# Installation Guide for TrackVLA / EVT-Bench

This guide covers the setup process for running TrackVLA on a headless Linux server with NVIDIA GPU.

## Prerequisites

- Linux server with NVIDIA GPU
- NVIDIA driver installed (tested with 535.x)
- Miniconda or Anaconda

## Step 1: Create Conda Environment

```bash
conda create -n evt_bench python=3.9 cmake=3.14.0 -y
conda activate evt_bench
```

## Step 2: Install habitat-sim (Headless Version)

You must install the headless version for server environments without display:

```bash
conda install habitat-sim=0.3.1=py3.9_headless_bullet_linux_3d6d67d6deae4ab2472cc84df7a3cef1503f606d -c aihabitat -c conda-forge -y
```

## Step 3: Install Python Dependencies

```bash
pip install "hydra-core>=1.2.0" "omegaconf>=2.2.3" "gym>=0.22.0,<0.23.1" "opencv-python>=3.3.0,<4.10"
pip install "numpy>=1.20.0,<1.24.0" --force-reinstall
```

## Step 4: Clone and Install TrackVLA

```bash
git clone https://github.com/wsakobe/TrackVLA.git
cd TrackVLA
pip install -e habitat-lab
```

## Step 5: Configure NVIDIA EGL

Create the NVIDIA EGL vendor configuration file:

```bash
sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json > /dev/null << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF
```

## Step 6: Set Environment Variables

Add the following to your `~/.bashrc`:

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0
```

Then reload:

```bash
source ~/.bashrc
```

## Step 7: Prepare Datasets

Download HM3D dataset and create symlink:

```bash
ln -s /path/to/your/scene_datasets data/scene_datasets
```

Expected structure:

```
data/scene_datasets/
├── hm3d/
│   ├── train/
│   ├── val/
│   └── minival/
└── mp3d/ (optional)
```

Download humanoid data:

```bash
python download_humanoid_data.py
```

## Step 8: Verify Installation

```bash
conda activate evt_bench
python -c "import habitat_sim; import habitat; print('OK')"
```

## Step 9: Run Evaluation

```bash
bash eval_baseline.sh
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| GL::Context: cannot retrieve OpenGL version | Set LD_PRELOAD as shown in Step 6 |
| No module named 'hydra' | Run pip install commands from Step 3 |
| NumPy version conflict | Force reinstall numpy<1.24.0 |
| EGL initialization failed | Create 10_nvidia.json as shown in Step 5 |