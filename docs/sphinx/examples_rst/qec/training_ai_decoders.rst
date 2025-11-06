Training and Deploying Neural Network Decoders with TensorRT
-------------------------------------------------------------

Starting with CUDA-Q QEC v0.5.0, a GPU-accelerated TensorRT-based decoder is included with the
CUDA-Q QEC library. The TensorRT decoder (``trt_decoder``) enables users to leverage custom neural network
models for quantum error correction, providing a flexible framework for deploying trained models
with optimized inference performance on NVIDIA GPUs.

Unlike traditional algorithmic decoders, neural network decoders can be trained on specific error
models and code structures, potentially achieving superior performance for certain noise regimes.
The TensorRT decoder supports loading models in ONNX format and provides configurable precision
modes (fp16, bf16, int8, fp8, tf32) to balance accuracy and inference speed.

This tutorial demonstrates the complete workflow for training a simple multi-layer perceptron (MLP)
to decode surface code syndromes using PyTorch and Stim, exporting the model to ONNX format, and
deploying it with the TensorRT decoder for accelerated inference.

Overview of the Training-to-Deployment Pipeline
++++++++++++++++++++++++++++++++++++++++++++++++

The workflow consists of three main stages:

1. **Data Generation**: Use Stim to generate synthetic quantum error correction data by simulating
   surface code circuits with realistic noise models. This produces detector measurements (syndromes)
   and observable flips (logical errors) that serve as training data.

2. **Model Training**: Train a neural network (in this case, an MLP) using PyTorch to learn the
   mapping from syndromes to logical error predictions. The model is trained with standard deep
   learning techniques including dropout regularization, learning rate scheduling, and validation monitoring.

3. **ONNX Export and Deployment**: Export the trained PyTorch model to ONNX format, which can then
   be loaded by the TensorRT decoder for optimized GPU inference in production QEC workflows.

Training a Neural Network Decoder with PyTorch and Stim
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The following example shows how to generate training data using Stim's built-in surface code
generator, train an MLP decoder with PyTorch, and export the model to ONNX format:

.. literalinclude:: ../../examples/qec/python/train_mlp_decoder.py
   :language: python
   :start-after: [Begin Documentation]

Using the TensorRT Decoder in CUDA-Q QEC
+++++++++++++++++++++++++++++++++++++++++

Once you have a trained ONNX model, you can load it with the TensorRT decoder for accelerated
inference. The decoder can be used in both C++ and Python workflows.

**Loading from ONNX (with automatic TensorRT optimization)**:

.. tab:: Python

   .. code-block:: python

      import cudaq_qec as qec
      import numpy as np

      # Load your parity check matrix
      H = np.array([[1, 0, 0, 1, 0, 1, 1],
                    [0, 1, 0, 1, 1, 0, 1],
                    [0, 0, 1, 0, 1, 1, 1]], dtype=np.uint8)

      # Create TensorRT decoder from ONNX model
      decoder = qec.get_decoder("trt_decoder", H,
                                onnx_load_path="surface_code_decoder.onnx",
                                precision="fp16")

      # Decode a syndrome
      syndrome = np.array([1, 0, 1], dtype=np.uint8)
      result = decoder.decode(syndrome)
      print(f"Predicted error: {result}")

.. tab:: C++

   .. code-block:: cpp

      #include <cudaq/qec.h>

      int main() {
          // Load parity check matrix
          std::vector<std::vector<uint8_t>> H = {
              {1, 0, 0, 1, 0, 1, 1},
              {0, 1, 0, 1, 1, 0, 1},
              {0, 0, 1, 0, 1, 1, 1}
          };

          // Create decoder parameters
          cudaq::heterogeneous_map params;
          params.insert("onnx_load_path", "surface_code_decoder.onnx");
          params.insert("precision", "fp16");

          // Create TensorRT decoder
          auto decoder = cudaq::qec::get_decoder("trt_decoder", H, params);

          // Decode syndrome
          std::vector<uint8_t> syndrome = {1, 0, 1};
          auto result = decoder->decode(syndrome);

          return 0;
      }

**Loading a pre-built TensorRT engine (for fastest initialization)**:

If you've already converted your ONNX model to a TensorRT engine using the provided utility script,
you can load it directly:

.. tab:: Python

   .. code-block:: python

      decoder = qec.get_decoder("trt_decoder", H,
                                engine_load_path="surface_code_decoder.trt")

Converting ONNX Models to TensorRT Engines
+++++++++++++++++++++++++++++++++++++++++++

For production deployments where initialization time is critical, you can pre-build a TensorRT
engine from your ONNX model using the ``trtexec`` command-line tool that comes with TensorRT:

.. code-block:: bash

   # Build with FP16 precision
   trtexec --onnx=surface_code_decoder.onnx \
           --saveEngine=surface_code_decoder.trt \
           --fp16

   # Build with best precision for your GPU
   trtexec --onnx=surface_code_decoder.onnx \
           --saveEngine=surface_code_decoder.trt \
           --best

   # Build with specific input shape (optional, for optimization)
   trtexec --onnx=surface_code_decoder.onnx \
           --saveEngine=surface_code_decoder.trt \
           --fp16 \
           --shapes=detectors:1x24

Pre-built engines offer several advantages:

- **Faster initialization**: Engine loading is significantly faster than ONNX parsing and optimization
- **Reproducible optimization**: The same optimization decisions are made every time
- **Version control**: Engines can be versioned alongside code for reproducible deployments


Dependencies and Requirements
++++++++++++++++++++++++++++++

The TensorRT decoder requires:

- **TensorRT**: Version 10.13.3.9 or higher
- **CUDA**: Version 12.0 or higher
- **GPU**: NVIDIA GPU with compute capability 6.0+ (Pascal architecture or newer)

For training:

- **PyTorch**: Version 2.0+ recommended
- **Stim**: For quantum circuit simulation and data generation

See Also
++++++++

- :class:`cudaq_qec.Decoder` - Base decoder interface
- `ONNX <https://onnx.ai/>`_ - Open Neural Network Exchange format
- `TensorRT Documentation <https://docs.nvidia.com/deeplearning/tensorrt/>`_ - NVIDIA TensorRT
- `Stim Documentation <https://github.com/quantumlib/Stim>`_ - Fast stabilizer circuit simulator

