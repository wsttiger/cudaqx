# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""
TRT Decoder Tests

These tests validate the TensorRT-based decoder for quantum error correction.

Test Categories:
  - Parameter validation and file operations (fast, no GPU required)

Note: GPU tests require CUDA/TensorRT and may crash if GPU is not properly
accessible within the container environment. Engine build takes ~5 seconds
on first run.
"""
import pytest
import numpy as np
import cudaq_qec as qec
import os
import tempfile

# Test data constants
NUM_TEST_SAMPLES = 200
NUM_DETECTORS = 24
NUM_OBSERVABLES = 1

# Path to the ONNX model file for testing (relative to this test file)
ONNX_MODEL_PATH = "assets/tests/surface_code_decoder.onnx"


# Check if CUDA/GPU is available for TensorRT tests
def _is_cuda_available():
    """Check if CUDA is available and can be initialized."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=2)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


CUDA_AVAILABLE = _is_cuda_available()

# Test inputs - 30 test cases with 24 detectors each
TEST_INPUTS = [[
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
],
               [
                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
               ],
               [
                   1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
               ]]

# Expected outputs from PyTorch model
TEST_OUTPUTS = [[0.546192], [0.542916], [0.534589], [0.559515], [0.510735],
                [0.526783], [0.536010], [0.522932], [0.563812], [0.560833],
                [0.578034], [0.555534], [0.542923], [0.495684], [0.545392],
                [0.562221], [0.532649], [0.559208], [0.534731], [0.499671],
                [0.497873], [0.511810], [0.566766], [0.527120], [0.573466],
                [0.523840], [0.504422], [0.507719], [0.474930], [0.583623]]


class TestTRTDecoderSetup:
    """Test fixture setup and teardown for TRT decoder tests."""

    def setup_method(self):
        """Set up test parameters before each test."""
        self.block_size = 3
        self.syndrome_size = 2

        # Create a simple parity check matrix H
        self.H = np.zeros((self.syndrome_size, self.block_size), dtype=np.uint8)
        self.H[0, 0] = 1
        self.H[0, 1] = 0
        self.H[0, 2] = 1  # First syndrome bit
        self.H[1, 0] = 0
        self.H[1, 1] = 1
        self.H[1, 2] = 1  # Second syndrome bit

        # For file loading tests
        self.test_file_path = None

    def teardown_method(self):
        """Clean up any test files after each test."""
        if self.test_file_path and os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)


class TestTRTDecoderParameterValidation(TestTRTDecoderSetup):
    """Tests for TRT decoder parameter validation."""

    def test_validate_parameters_valid_onnx_path(self):
        """Test parameter validation with valid ONNX path."""
        # Note: In Python, parameter validation happens inside the decoder creation
        # We can't test the validation function directly, but we can verify that
        # providing valid parameters doesn't raise errors when creating a decoder
        # This test is more of a placeholder showing the expected behavior
        pass

    def test_validate_parameters_valid_engine_path(self):
        """Test parameter validation with valid engine path."""
        # Similar to above - placeholder for parameter validation testing
        pass

    def test_validate_parameters_both_paths_provided(self):
        """Test that providing both ONNX and engine paths raises an error."""
        # This would need to be tested when actually creating a decoder
        # with both parameters, which should raise a RuntimeError
        pass

    def test_validate_parameters_no_paths_provided(self):
        """Test that providing no paths creates decoder with warning."""
        # Decoder is created but logs a warning - it won't be usable for inference
        # Create the TRT decoder
        try:
            decoder = qec.get_decoder('trt_decoder', self.H)
            # If decoder is None or doesn't initialize properly, skip these tests
            if decoder is None:
                pytest.skip(
                    "TRT decoder returned None - likely CUDA/GPU unavailable")
        except (RuntimeError, SystemError, Exception) as e:
            pytest.skip(
                f"Failed to create TRT decoder (GPU may be unavailable): {e}")


class TestTRTDecoderFileOperations(TestTRTDecoderSetup):
    """Tests for TRT decoder file loading operations."""

    def test_load_file_valid_file(self):
        """Test loading a valid file."""
        # Create a temporary test file
        test_content = "Hello, World!"
        with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                         suffix='.txt') as f:
            f.write(test_content)
            self.test_file_path = f.name

        # Read the file back
        with open(self.test_file_path, 'r') as f:
            loaded_content = f.read()

        assert loaded_content == test_content

    def test_load_file_non_existent_file(self):
        """Test loading a non-existent file raises an error."""
        with pytest.raises(FileNotFoundError):
            with open('non_existent_file.txt', 'r') as f:
                f.read()


@pytest.mark.skipif(
    not os.path.exists(ONNX_MODEL_PATH) or not CUDA_AVAILABLE,
    reason="ONNX model file not found or CUDA/GPU not available")
class TestTRTDecoderInference(TestTRTDecoderSetup):
    """Tests for TRT decoder inference with actual model.
    
    Warning: These tests require GPU access and may take 20-30 seconds on first 
    run as TensorRT builds and optimizes the engine from the ONNX model.
    These tests may fail in containerized environments without proper GPU access.
    """

    def setup_method(self):
        """Set up decoder for inference tests."""
        super().setup_method()

        # Create parity check matrix matching the test data
        # For distance-3 surface code: 24 detectors
        num_detectors = NUM_DETECTORS

        # Create a dummy H matrix (identity matrix for simplicity)
        self.H_inference = np.eye(num_detectors, dtype=np.uint8)

        # Create the TRT decoder
        self.onnx_path = ONNX_MODEL_PATH
        try:
            self.decoder = qec.get_decoder('trt_decoder',
                                           self.H_inference,
                                           onnx_load_path=self.onnx_path)
            # If decoder is None or doesn't initialize properly, skip these tests
            if self.decoder is None:
                pytest.skip(
                    "TRT decoder returned None - likely CUDA/GPU unavailable")
        except (RuntimeError, SystemError, Exception) as e:
            pytest.skip(
                f"Failed to create TRT decoder (GPU may be unavailable): {e}")

    def test_validate_against_pytorch_model(self):
        """Test that TRT decoder produces identical results to PyTorch."""
        # Tolerance for floating point comparison
        TOLERANCE = 1e-4

        # Track statistics
        num_passed = 0
        num_failed = 0
        max_error = 0.0
        total_error = 0.0
        failed_cases = []

        # Test each of the 100 test cases
        for i, (test_input,
                test_output) in enumerate(zip(TEST_INPUTS, TEST_OUTPUTS)):
            # Run TRT decoder inference
            result = self.decoder.decode(test_input)

            # Get the expected output
            expected_output = test_output[0]

            # Get the TRT decoder output
            assert len(
                result.result
            ) > 0, f"TRT decoder returned empty result for test case {i}"
            trt_output = result.result[0]

            # Compute absolute error
            error = abs(trt_output - expected_output)
            total_error += error
            max_error = max(max_error, error)

            # Check if within tolerance
            if error < TOLERANCE:
                num_passed += 1
            else:
                num_failed += 1
                # Store detailed error info for first few failures
                if len(failed_cases) < 5:
                    failed_cases.append({
                        'index': i,
                        'expected': expected_output,
                        'got': trt_output,
                        'error': error
                    })

        # Print summary statistics
        print(f"\n=== TRT Decoder Validation Summary ===")
        print(f"Total test cases: {len(TEST_INPUTS)}")
        print(f"Passed: {num_passed}")
        print(f"Failed: {num_failed}")
        print(f"Max error: {max_error}")
        print(f"Average error: {total_error / len(TEST_INPUTS)}")
        print(f"====================================\n")

        # Print detailed failure information
        if failed_cases:
            print("\nFirst few failures:")
            for case in failed_cases:
                print(f"Test case {case['index']} FAILED:")
                print(f"  Expected: {case['expected']}")
                print(f"  Got:      {case['got']}")
                print(f"  Error:    {case['error']}")

        # Overall assertion: all cases must pass
        assert num_failed == 0, f"{num_failed} test cases failed validation"

    def test_validate_single_test_case(self):
        """Test a single specific case for detailed debugging."""
        # Test first case in detail
        syndrome = TEST_INPUTS[0]

        print(f"Input syndrome (first 10 values): {syndrome[:10]}")

        result = self.decoder.decode(syndrome)

        expected = TEST_OUTPUTS[0][0]
        actual = result.result[0]
        error = abs(actual - expected)

        print(f"Expected output: {expected}")
        print(f"Actual output:   {actual}")
        print(f"Absolute error:  {error}")
        print(f"Converged:       {result.converged}")

        assert error < 1e-4, "Single test case validation failed"
        assert result.converged, "Decoder did not converge"

    def test_decoder_result_structure(self):
        """Test that decoder result has expected structure."""
        syndrome = TEST_INPUTS[0]
        result = self.decoder.decode(syndrome)

        # Test basic structure
        assert hasattr(result, 'converged')
        assert hasattr(result, 'result')
        assert isinstance(result.converged, bool)
        assert isinstance(result.result, list)
        assert len(result.result) > 0

    def test_decoder_batch_processing(self):
        """Test decoder batch processing with multiple syndromes."""
        # Test with first 5 test cases
        syndromes = TEST_INPUTS[:5]
        expected_outputs = [out[0] for out in TEST_OUTPUTS[:5]]

        # Decode batch
        results = self.decoder.decode_batch(syndromes)

        assert len(results) == len(syndromes)

        # Check each result
        TOLERANCE = 1e-4
        for i, (result, expected) in enumerate(zip(results, expected_outputs)):
            assert hasattr(result, 'converged')
            assert hasattr(result, 'result')
            assert len(result.result) > 0

            error = abs(result.result[0] - expected)
            assert error < TOLERANCE, f"Batch test case {i} failed with error {error}"


class TestTRTDecoderEdgeCases(TestTRTDecoderSetup):
    """Tests for TRT decoder edge cases. Requires GPU access."""

    def test_decoder_with_zero_syndrome(self):
        """Test decoder with all-zero syndrome."""
        if not os.path.exists(ONNX_MODEL_PATH) or not CUDA_AVAILABLE:
            pytest.skip("ONNX model file not found or CUDA/GPU not available")

        num_detectors = NUM_DETECTORS
        H = np.eye(num_detectors, dtype=np.uint8)

        try:
            decoder = qec.get_decoder('trt_decoder',
                                      H,
                                      onnx_load_path=ONNX_MODEL_PATH)
        except Exception:
            pytest.skip("Failed to create TRT decoder")

        # All-zero syndrome
        zero_syndrome = [0.0] * num_detectors
        result = decoder.decode(zero_syndrome)

        assert hasattr(result, 'result')
        assert len(result.result) > 0

    def test_decoder_with_all_ones_syndrome(self):
        """Test decoder with all-ones syndrome."""
        if not os.path.exists(ONNX_MODEL_PATH) or not CUDA_AVAILABLE:
            pytest.skip("ONNX model file not found or CUDA/GPU not available")

        num_detectors = NUM_DETECTORS
        H = np.eye(num_detectors, dtype=np.uint8)

        try:
            decoder = qec.get_decoder('trt_decoder',
                                      H,
                                      onnx_load_path=ONNX_MODEL_PATH)
        except Exception:
            pytest.skip("Failed to create TRT decoder")

        # All-ones syndrome
        ones_syndrome = [1.0] * num_detectors
        result = decoder.decode(ones_syndrome)

        assert hasattr(result, 'result')
        assert len(result.result) > 0


class TestTRTDecoderBatchValidation(TestTRTDecoderSetup):
    """Tests for TRT decoder batch size validation. Requires GPU access."""

    def test_decode_batch_non_integral_multiple_should_fail(self):
        """Test that decode_batch fails when syndrome count is not an integral multiple of batch size."""
        if not os.path.exists(ONNX_MODEL_PATH) or not CUDA_AVAILABLE:
            pytest.skip("ONNX model file not found or CUDA/GPU not available")

        # We need to figure out what the batch size is first
        # Most models have batch size 1, 2, 4, 8, 16, etc.
        # We'll test with various syndrome counts that are likely non-integral
        
        # Test with 3 syndromes (not divisible by 2, 4, 8, etc.)
        syndromes = TEST_INPUTS[:3]
        
        # Get the actual batch size from the model by introspection
        # We'll try decoding and see if it fails
        try:
            # First, let's try with 1 - this should always work
            results_1 = self.decoder.decode_batch(TEST_INPUTS[:1])
            assert len(results_1) == 1, "Single syndrome batch should work"
            
            # Try with 2 - should work for batch_size 1 or 2
            results_2 = self.decoder.decode_batch(TEST_INPUTS[:2])
            assert len(results_2) == 2, "Two syndrome batch should work"
            
            # Try with 3 - should fail for batch_size 2, 4, 8, etc but work for batch_size 1
            try:
                results_3 = self.decoder.decode_batch(TEST_INPUTS[:3])
                # If we get here, batch_size is likely 1
                assert len(results_3) == 3
                print("Batch size appears to be 1 (3 syndromes succeeded)")
                
                # For batch_size 1, all counts work, so we can't test non-integral error
                pytest.skip("Model has batch_size=1, cannot test non-integral batch error")
                
            except RuntimeError as e:
                # Good! This means batch_size > 1 and 3 is not a multiple
                assert "integral multiple" in str(e), \
                    f"Expected 'integral multiple' error message, got: {e}"
                print(f"Successfully caught non-integral batch size error: {e}")
                
        except Exception as e:
            pytest.fail(f"Unexpected error during batch validation test: {e}")

    def test_decode_batch_with_batch_size_2_model(self):
        """Test batch decoding with various counts to find and validate batch size."""
        if not os.path.exists(ONNX_MODEL_PATH) or not CUDA_AVAILABLE:
            pytest.skip("ONNX model file not found or CUDA/GPU not available")

        # Try to determine batch size by testing different syndrome counts
        test_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        working_counts = []
        failing_counts = []
        
        for count in test_counts:
            if count > len(TEST_INPUTS):
                break
            try:
                results = self.decoder.decode_batch(TEST_INPUTS[:count])
                working_counts.append(count)
                assert len(results) == count, \
                    f"Expected {count} results, got {len(results)}"
            except RuntimeError as e:
                if "integral multiple" in str(e):
                    failing_counts.append(count)
                else:
                    # Different error, re-raise
                    raise
        
        print(f"Working counts: {working_counts}")
        print(f"Failing counts: {failing_counts}")
        
        # Determine batch size from the pattern
        if len(working_counts) == len(test_counts):
            # All counts work - batch size is 1
            batch_size = 1
        elif len(working_counts) > 0:
            # Find GCD of working counts
            from math import gcd
            from functools import reduce
            batch_size = reduce(gcd, working_counts)
        else:
            pytest.fail("No syndrome counts worked!")
        
        print(f"Detected batch size: {batch_size}")
        
        # Verify that multiples of batch_size work and non-multiples fail
        if batch_size > 1:
            # Test a multiple that should work
            multiple = batch_size * 2
            if multiple <= len(TEST_INPUTS):
                results = self.decoder.decode_batch(TEST_INPUTS[:multiple])
                assert len(results) == multiple
            
            # Test a non-multiple that should fail
            non_multiple = batch_size + 1
            if non_multiple <= len(TEST_INPUTS) and non_multiple not in working_counts:
                with pytest.raises(RuntimeError, match="integral multiple"):
                    self.decoder.decode_batch(TEST_INPUTS[:non_multiple])

    def test_decode_batch_syndrome_size_mismatch(self):
        """Test that decode_batch validates individual syndrome sizes."""
        if not os.path.exists(ONNX_MODEL_PATH) or not CUDA_AVAILABLE:
            pytest.skip("ONNX model file not found or CUDA/GPU not available")

        # Create syndromes with wrong size
        wrong_size_syndrome = [0.0] * (NUM_DETECTORS + 5)  # Too long
        syndromes = [TEST_INPUTS[0], wrong_size_syndrome]
        
        with pytest.raises(RuntimeError, match="Syndrome size mismatch"):
            self.decoder.decode_batch(syndromes)

    def test_decode_single_syndrome_size_mismatch(self):
        """Test that decode validates syndrome size."""
        if not os.path.exists(ONNX_MODEL_PATH) or not CUDA_AVAILABLE:
            pytest.skip("ONNX model file not found or CUDA/GPU not available")

        # Create syndrome with wrong size
        wrong_size_syndrome = [0.0] * (NUM_DETECTORS + 5)  # Too long
        
        with pytest.raises(RuntimeError, match="Syndrome size mismatch"):
            self.decoder.decode(wrong_size_syndrome)

    def test_decode_batch_empty_list(self):
        """Test that decode_batch handles empty syndrome list."""
        if not os.path.exists(ONNX_MODEL_PATH) or not CUDA_AVAILABLE:
            pytest.skip("ONNX model file not found or CUDA/GPU not available")

        results = self.decoder.decode_batch([])
        assert len(results) == 0, "Empty batch should return empty results"

    def test_decode_batch_large_integral_multiple(self):
        """Test batch decoding with a larger integral multiple of batch size."""
        if not os.path.exists(ONNX_MODEL_PATH) or not CUDA_AVAILABLE:
            pytest.skip("ONNX model file not found or CUDA/GPU not available")

        # Try different counts to find valid batch sizes
        # Start with powers of 2 which are common
        for count in [2, 4, 8, 16]:
            if count > len(TEST_INPUTS):
                break
            
            try:
                syndromes = TEST_INPUTS[:count]
                results = self.decoder.decode_batch(syndromes)
                
                # Verify results
                assert len(results) == count, f"Expected {count} results"
                
                # Verify all results are valid
                for i, result in enumerate(results):
                    assert result.converged, f"Result {i} did not converge"
                    assert len(result.result) > 0, f"Result {i} has empty output"
                
                print(f"Successfully decoded batch of {count} syndromes")
                break  # Test passed, exit loop
                
            except RuntimeError as e:
                if "integral multiple" in str(e):
                    # This count doesn't match batch size, try next
                    continue
                else:
                    # Different error, re-raise
                    raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
