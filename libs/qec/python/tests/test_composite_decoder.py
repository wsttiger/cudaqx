# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                           #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.   #
# ============================================================================ #
"""
Composite Decoder Tests

Tests for the composite_decoder (TRT pre-decoder -> residual syndrome ->
global decoder, e.g. pymatching).

- Parameter validation and config (fast, no GPU required when plugin is loaded).
- Optional integration tests with a small TRT engine (require CUDA/TensorRT).
"""
import os
import tempfile
import pytest
import numpy as np
import cudaq_qec as qec


def _composite_decoder_available():
    """True if the composite_decoder plugin is registered (TRT plugin built)."""
    try:
        H = np.zeros((2, 3), dtype=np.uint8)
        H[0, 0] = H[0, 2] = 1
        H[1, 1] = H[1, 2] = 1
        qec.get_decoder(
            "composite_decoder",
            H,
            global_decoder="pymatching",
            onnx_load_path="/nonexistent.onnx",
        )
        return True
    except RuntimeError as e:
        if "invalid decoder requested" in str(e).lower():
            return False
        return True
    except Exception:
        return True


COMPOSITE_DECODER_AVAILABLE = _composite_decoder_available()


def _make_simple_H(syndrome_size=2, block_size=3):
    """Minimal valid PCM for tests."""
    H = np.zeros((syndrome_size, block_size), dtype=np.uint8)
    H[0, 0] = H[0, 2] = 1
    H[1, 1] = H[1, 2] = 1
    return H


# -----------------------------------------------------------------------------
# Parameter validation (require plugin only)
# -----------------------------------------------------------------------------

@pytest.mark.skipif(
    not COMPOSITE_DECODER_AVAILABLE,
    reason="composite_decoder plugin not built or not loaded",
)
class TestCompositeDecoderParameterValidation:
    """Parameter validation for composite_decoder (no GPU/engine required)."""

    def test_missing_global_decoder_raises(self):
        H = _make_simple_H()
        with pytest.raises(RuntimeError) as exc_info:
            qec.get_decoder(
                "composite_decoder",
                H,
                onnx_load_path="/any.onnx",
            )
        assert "global_decoder" in str(exc_info.value).lower()

    def test_unsupported_global_decoder_raises(self):
        H = _make_simple_H()
        with pytest.raises(RuntimeError) as exc_info:
            qec.get_decoder(
                "composite_decoder",
                H,
                global_decoder="other",
                onnx_load_path="/any.onnx",
            )
        assert "pymatching" in str(exc_info.value).lower()

    def test_no_onnx_or_engine_path_raises(self):
        H = _make_simple_H()
        with pytest.raises(RuntimeError) as exc_info:
            qec.get_decoder(
                "composite_decoder",
                H,
                global_decoder="pymatching",
            )
        assert "onnx" in str(exc_info.value).lower() or "engine" in str(
            exc_info.value
        ).lower()

    def test_both_onnx_and_engine_path_raises(self):
        H = _make_simple_H()
        with pytest.raises(RuntimeError) as exc_info:
            qec.get_decoder(
                "composite_decoder",
                H,
                global_decoder="pymatching",
                onnx_load_path="/a.onnx",
                engine_load_path="/b.engine",
            )
        assert "both" in str(exc_info.value).lower() or "cannot" in str(
            exc_info.value
        ).lower()

    def test_accepts_onnx_path_and_global_decoder(self):
        """With valid params but bad file, we get a file/TRT error, not param validation."""
        H = _make_simple_H()
        try:
            qec.get_decoder(
                "composite_decoder",
                H,
                global_decoder="pymatching",
                onnx_load_path="/nonexistent.onnx",
            )
        except RuntimeError as e:
            if "invalid decoder requested" in str(e).lower():
                pytest.fail("composite_decoder plugin not loaded")
            assert "global_decoder" not in str(e) or "required" not in str(e).lower()

    def test_accepts_engine_path_and_global_decoder(self):
        """With valid params but bad file, we get a file/TRT error, not param validation."""
        H = _make_simple_H()
        try:
            qec.get_decoder(
                "composite_decoder",
                H,
                global_decoder="pymatching",
                engine_load_path="/nonexistent.engine",
            )
        except RuntimeError as e:
            if "invalid decoder requested" in str(e).lower():
                pytest.fail("composite_decoder plugin not loaded")
            assert "global_decoder" not in str(e) or "required" not in str(e).lower()


# Realtime config (composite_decoder_config) is tested in test_decoding_config.py
# with the other decoder config types (trt_decoder_config, nv_qldpc, etc.).

# -----------------------------------------------------------------------------
# Integration: decode with a real TRT engine (identity residual) + pymatching
# -----------------------------------------------------------------------------

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


def _is_cuda_available():
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=2
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _build_identity_residual_engine(syndrome_size, engine_path):
    """Build a minimal TRT engine: input [syndrome_size] -> output [syndrome_size] (identity)."""
    if not TRT_AVAILABLE:
        return False
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)

        input_tensor = network.add_input(
            name="input",
            dtype=trt.float32,
            shape=(1, syndrome_size),
        )
        if hasattr(network, "add_identity"):
            identity = network.add_identity(input_tensor)
            out = identity.get_output(0)
        else:
            zeros = network.add_constant(
                (1, syndrome_size), np.zeros((1, syndrome_size), dtype=np.float32)
            )
            add_layer = network.add_elementwise(
                input_tensor, zeros.get_output(0), trt.ElementWiseOperation.SUM
            )
            out = add_layer.get_output(0)
        out.name = "output"
        network.mark_output(out)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            return False
        with open(engine_path, "wb") as f:
            f.write(serialized)
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not COMPOSITE_DECODER_AVAILABLE or not TRT_AVAILABLE or not _is_cuda_available(),
    reason="composite_decoder + TensorRT + CUDA required",
)
class TestCompositeDecoderIntegration:
    """Integration tests: composite_decoder with a real TRT engine and pymatching."""

    def test_decode_single_and_batch(self):
        """Build identity residual engine; composite_decoder decode() and decode_batch() return valid results."""
        syndrome_size = 3
        block_size = 5
        H = np.array(
            [
                [1, 0, 1, 0, 1],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 1, 1],
            ],
            dtype=np.uint8,
        )
        assert H.shape == (syndrome_size, block_size)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine_path = os.path.join(tmpdir, "identity.trt")
            if not _build_identity_residual_engine(syndrome_size, engine_path):
                pytest.skip("Could not build TRT identity engine")

            decoder = qec.get_decoder(
                "composite_decoder",
                H,
                global_decoder="pymatching",
                engine_load_path=engine_path,
            )
            assert decoder is not None
            assert decoder.get_syndrome_size() == syndrome_size
            assert decoder.get_block_size() == block_size

            syndrome = [0.0, 1.0, 1.0]
            result = decoder.decode(syndrome)
            assert hasattr(result, "converged")
            assert hasattr(result, "result")
            assert result.converged
            assert len(result.result) == block_size
            assert all(0 <= x <= 1 for x in result.result)

            results_batch = decoder.decode_batch([syndrome, [1.0, 0.0, 0.0]])
            assert len(results_batch) == 2
            for r in results_batch:
                assert r.converged
                assert len(r.result) == block_size
