#!/bin/bash

# Set Conda path
export PATH="/miniconda/bin:$PATH"

# Update Conda base environment
#conda update -n base -c defaults conda

# Install dependencies using Conda with Python version 3.9
#conda install -c conda-forge python=3.9 -y
#conda install -c conda-forge numpy pandas scikit-learn matplotlib jupyter jupyterlab notebook statsmodels -y

# Uncomment and install additional Python packages if needed
# pip install -U csaps
# pip install import_ipynb

# Install TensorRT bindings and libs
python3 -m pip install --extra-index-url https://pypi.nvidia.com tensorrt-bindings==8.6.1 tensorrt-libs==8.6.1

# Install TensorFlow with GPU support
python3 -m pip install -U tensorflow[and-cuda]