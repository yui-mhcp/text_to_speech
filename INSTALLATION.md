# :yum: Installation guide

This guide sets up a **GPU-enabled environment** (NVIDIA driver, CUDA, `mamba`, and the
deep-learning backends) shared by every `yui-mhcp` project. It is **project-agnostic**:
once the environment is ready, install a given project with the short *Installation*
section of its own `README.md` (typically `pip install -e .[<extras>]`).

> **Versions.** The exact versions below (CUDA 13.1, `torch` cu130, `tensorrt_llm < 1.3`,
> the Blackwell TensorFlow wheel) are a **known-working combination tested on 2026-07**.
> Adjust them to your GPU / CUDA using the linked archives.

> **TensorRT-LLM.** Any `tensorrt_llm < 1.3` works. `1.2.1` is tested but its `int8` path
> is broken on Blackwell (RTX 50xx) ; `1.1.0` is fully tested and works everywhere.

> **Distribution.** Commands target **Debian 13**; adapt the package-manager steps on other
> distributions.

> **Backends.** This guide installs *all* backends (PyTorch, TensorRT-LLM, TensorFlow) for a
> full workstation. You only need the one(s) your target project uses — see its `README.md`.

## 1. System packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y wget rsync git ffmpeg linux-headers-$(uname -r)

# Python build dependencies
sudo apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
    libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev

# Required by some Python libraries (plotting, PDF, audio I/O)
sudo apt install -y graphviz poppler-utils lzma portaudio19-dev

# Required for TensorRT-LLM only
sudo apt-get install -y openmpi-bin libopenmpi-dev git-lfs
```

## 2. NVIDIA driver

### Blackwell generation (RTX 50xx)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

sudo apt install -y linux-headers-amd64 dkms
sudo apt install -y nvidia-open

nvidia-smi
```

Verify the kernel module is loaded:

```bash
lsmod | grep nvidia
# if the list is empty:
sudo modprobe nvidia
```

### Alternative — manual installer

Download the driver from the [NVIDIA website](https://www.nvidia.com/Download/index.aspx)
(GeForce → *Proprietary*), then:

```bash
sudo sh <driver_file>.run
```

## 3. CUDA toolkit

Pick your version from the [CUDA archive](https://developer.nvidia.com/cuda-toolkit-archive).
Example for **CUDA 13.1** on **Debian 13**:

```bash
wget https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda_13.1.0_590.44.01_linux.run
sudo sh cuda_13.1.0_590.44.01_linux.run
```

## 4. Mamba (conda-forge)

```bash
mkdir -p installations
wget -O installations/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh
bash installations/Miniforge3.sh -b -u -p ~/.conda

source ~/.conda/bin/activate
conda init --all
conda config --set auto_activate false

mamba update -n base -c conda-forge conda mamba -y
mamba shell init --shell bash --root-prefix ~/.conda
source ~/.bashrc
mamba deactivate
```

### Create the environment

```bash
mamba create -n yui python=3.12
mamba activate yui
```

Wire CUDA into the environment so it is on the path **only** while the env is active. The
snippets below write the resolved CUDA path into the activate/deactivate hooks:

```bash
CUDA_PATH=/usr/local/cuda-13.1

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/initializer.sh" <<EOF
#! /bin/bash
export PATH="\$PATH:$CUDA_PATH/bin"
export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:\$CONDA_PREFIX/lib:$CUDA_PATH/lib64"
EOF

mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/cleaner.sh" <<EOF
#! /bin/bash
export PATH=\$(echo \$PATH | sed -e "s|:$CUDA_PATH/bin||")
export LD_LIBRARY_PATH=\$(echo \$LD_LIBRARY_PATH | sed -e "s|:\$CONDA_PREFIX/lib:$CUDA_PATH/lib64||")
EOF
```

Re-activate to apply: `mamba deactivate && mamba activate yui`.

## 5. Deep-learning backends

```bash
pip install jupyter jupyterlab notebook

# PyTorch built for CUDA 13.x
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu130

# TensorRT-LLM (from the NVIDIA index). --ignore-installed avoids re-pulling torch.
# See the "Versions" note above re: 1.2.1 int8 on Blackwell vs 1.1.0.
pip install "tensorrt_llm<1.3" --extra-index-url https://pypi.nvidia.com --ignore-installed
# Only if `import tensorrt_llm` fails afterwards:
pip install pynvml==11.5.3

# TensorFlow with CUDA. Install AFTER TRT-LLM to keep matching cuda/cudnn versions.
pip install --upgrade tensorflow[and-cuda]
```

### Blackwell GPUs (RTX 50xx)

TensorFlow 2.20 is not yet compiled with Blackwell support — install a custom wheel:

```bash
# python 3.12 wheel:
pip install tensorflow-2.20.0dev0+selfbuild-cp312-cp312-linux_x86_64.whl
```

Download it from [this release](https://github.com/mypapit/tensorflowRTX50/releases)
(3.11 / 3.12 / 3.13 builds also available
[here](https://github.com/nhsmit/tensorflow-rtx-50-series/releases/tag/2.20.0dev)).

## 6. Verify the GPU

```bash
mamba activate yui
```

```python
import tensorflow as tf
print('GPU available (tensorflow):', tf.config.list_physical_devices('GPU') != [])

# If installed:
# import torch;         print('GPU available (torch):', torch.cuda.is_available())
# import tensorrt_llm
```

Start Jupyter:

```bash
jupyter lab --ip 0.0.0.0 --allow-root
```

---

Your environment is ready. To install a specific project, follow the *Installation* section
of its `README.md` (e.g. `pip install -e .[image,tf]`).
