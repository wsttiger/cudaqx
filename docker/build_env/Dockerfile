# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

FROM ghcr.io/nvidia/cuda-quantum-devdeps:ext-cu11.8-gcc11-main

RUN  apt-get update && apt-get install -y gfortran libblas-dev libcusolver-dev-11-8 \
  && python3 -m pip install cmake --user \
  && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/nvidia/cuda-quantum \
  && cd cuda-quantum && mkdir build && cd build \
  && ~/.local/bin/cmake -G Ninja .. -DLLVM_DIR=/opt/llvm/lib/cmake \
     -DCUDAQ_ENABLE_PYTHON=TRUE -DCMAKE_INSTALL_PREFIX=$HOME/.cudaq \
  && ninja install 
 


