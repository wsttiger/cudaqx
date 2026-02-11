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


# -----------------------------------------------------------------------------
# composite_decoder_config (realtime config)
# -----------------------------------------------------------------------------

FIELDS_COMPOSITE_DECODER = {
    "global_decoder": (str, "pymatching", "pymatching"),
    "onnx_load_path": (str, "/path/to/model.onnx", "/other/model.onnx"),
    "engine_load_path": (str, "/path/to/engine.trt", "/other/engine.trt"),
    "engine_save_path": (str, "/path/to/save.trt", "/other/save.trt"),
    "precision": (str, "fp16", "fp32"),
    "memory_workspace": (int, 1073741824, 2147483648),
    "use_cuda_graph": (bool, True, False),
    "error_rate_vec": (list, [0.01, 0.02, 0.03], [0.1, 0.1, 0.1]),
    "merge_strategy": (str, "disallow", "independent"),
}


def test_composite_decoder_config_required_global_decoder():
    cfg = qec.composite_decoder_config()
    assert hasattr(cfg, "global_decoder")
    cfg.global_decoder = "pymatching"
    assert cfg.global_decoder == "pymatching"


@pytest.mark.parametrize(
    "name,meta",
    [(k, v) for k, v in FIELDS_COMPOSITE_DECODER.items() if k != "global_decoder"],
)
def test_composite_decoder_config_optional_fields(name, meta):
    cfg = qec.composite_decoder_config()
    py_type, sample_val, alt_val = meta
    setattr(cfg, name, sample_val)
    assert getattr(cfg, name) == sample_val
    setattr(cfg, name, alt_val)
    assert getattr(cfg, name) == alt_val


def test_composite_decoder_config_yaml_roundtrip():
    cfg = qec.composite_decoder_config()
    cfg.global_decoder = "pymatching"
    cfg.engine_load_path = "/path/to/engine.trt"
    cfg.precision = "fp16"

    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "composite_decoder"
    dc.block_size = 5
    dc.syndrome_size = 3
    dc.H_sparse = [0, 1, 2, -1, 1, 2, 3, -1, 2, 3, 4, -1]
    dc.O_sparse = [-1]
    dc.D_sparse = qec.generate_timelike_sparse_detector_matrix(
        dc.syndrome_size, 2, include_first_round=False
    )
    dc.set_decoder_custom_args(cfg)

    yaml_text = dc.to_yaml_str()
    assert isinstance(yaml_text, str) and len(yaml_text) > 0

    dc2 = qec.decoder_config.from_yaml_str(yaml_text)
    assert dc2.id == 0
    assert dc2.type == "composite_decoder"
    assert dc2.block_size == 5
    assert dc2.syndrome_size == 3

    cfg2 = dc2.decoder_custom_args
    assert cfg2 is not None
    assert cfg2.global_decoder == "pymatching"
    assert cfg2.engine_load_path == "/path/to/engine.trt"
    assert cfg2.precision == "fp16"


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
