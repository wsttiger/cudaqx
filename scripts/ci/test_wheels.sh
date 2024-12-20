#!/bin/sh

# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Exit immediately if any command returns a non-zero status
set -e

# Installing dependencies
python_version=3.10
python=python${python_version}

apt-get update && apt-get install -y --no-install-recommends \
        libgfortran5 python${python_version} python$(echo ${python_version} | cut -d . -f 1)-pip

${python} -m pip install --no-cache-dir pytest nvidia-cublas-cu11

cd /cuda-qx

${python} -m pip install wheels/cuda_quantum_cu12-0.0.0-cp310-cp310-manylinux_2_28_x86_64.whl

# QEC library
# ======================================

${python} -m pip install wheels/cudaq_qec-0.0.1-cp310-cp310-*.whl
${python} -m pytest libs/qec/python/tests/

# Solvers library
# ======================================

${python} -m pip install wheels/cudaq_solvers-0.0.1-cp310-cp310-*.whl
${python} -m pytest libs/solvers/python/tests/

