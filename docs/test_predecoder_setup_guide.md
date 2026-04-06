# Running test_realtime_predecoder_w_pymatching (d13_r104)

## Quick Start

```bash
# Download the bundle
wget https://urm.nvidia.com/artifactory/sw-cuda-qx-generic-local/predecoder-assets-tmp/predecoder_d13_r104_bundle.tar.gz

# Extract the bundle
tar xzf predecoder_d13_r104_bundle.tar.gz -C /path/to/data

# Copy ONNX model into the build tree (or wherever ONNX_MODEL_DIR points)
cp /path/to/data/models/predecoder_memory_d13_T104_X.onnx \
   libs/qec/lib/realtime/

# Build
cd build && ninja test_realtime_predecoder_w_pymatching

# Run (104 µs injection rate, 20 seconds, with real Stim data)
./libs/qec/unittests/test_realtime_predecoder_w_pymatching \
    d13_r104 104 20 \
    --data-dir /path/to/data/test_data/d13_T104_X/
```

---

## Command Line

```
./test_realtime_predecoder_w_pymatching d13_r104 [rate_us] [duration_s] [--data-dir <path>]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `rate_us` | `0` (open-loop) | Inter-arrival time in µs between syndrome submissions |
| `duration_s` | `5` | Test duration in seconds |
| `--data-dir` | (none) | Path to Stim-generated test data for correctness verification |

---

## Distribution Bundle

**File:** `predecoder_d13_r104_bundle.tar.gz` (5.1 MB)

Contents:

```
models/
  predecoder_memory_d13_T104_X.onnx    3.8 MB   TensorRT predecoder model (d=13, T=104, X basis)

test_data/d13_T104_X/
  detectors.bin                        67 MB    1000 Stim-generated syndrome samples (17472 detectors each)
  observables.bin                       4 KB    Ground truth logical observables (1000 × 1)
  H_csr.bin                           792 KB    Parity check matrix in CSR format (17472 × 94021, 185060 nnz)
  O_csr.bin                           2.9 KB    Observable matrix in CSR format (1 × 94021, 735 nnz)
  priors.bin                          735 KB    Edge error probabilities for PyMatching (94021 doubles)
  metadata.txt                        258 B     Generation parameters (distance, rounds, noise model, etc.)
```

### Test Data Generation Parameters

From `metadata.txt`:

| Parameter | Value |
|-----------|-------|
| Distance | 13 |
| Rounds | 104 |
| Basis | X |
| Code rotation | XV |
| Physical error rate | 0.003 |
| Noise model | 25-param |
| Samples | 1000 |
| Detectors per sample | 17472 |

### Binary File Formats

**detectors.bin / observables.bin:**
```
[uint32 num_rows][uint32 num_cols][int32 data × (rows × cols)]
```

**H_csr.bin / O_csr.bin (sparse CSR):**
```
[uint32 nrows][uint32 ncols][uint32 nnz][int32 indptr × (nrows+1)][int32 indices × nnz]
```

**priors.bin:**
```
[uint32 nedges][float64 priors × nedges]
```

---

## Pipeline Configuration (d13_r104)

| Parameter | Value |
|-----------|-------|
| Ring buffer slots | 16 |
| Predecoder streams | 8 |
| PyMatching worker threads | 16 |
| Model input | 17,472 × uint8 detectors |
| Model output | 17,473 × uint8 (1 logical prediction + 17,472 residual detectors) |
| Slot size | 32,768 bytes |
| Queue depth per predecoder | 1 |

---

## What Happens at Runtime

1. **First run**: TRT compiles the ONNX model into a cached `.engine` file (~15s). Subsequent runs load the engine directly (~4s).
2. **With `--data-dir`**: Loads real Stim data, builds PyMatching decoders from the full parity check matrix `H` with edge priors. Reports LER against ground truth.
3. **Without `--data-dir`**: Uses random Bernoulli(0.01) syndromes, falls back to per-slice PyMatching with `cudaq-qec` surface code `H_z`. No LER verification.

---

## Expected Results

```
  Submitted:          192,309
  Completed:          192,309
  Throughput:         9,610 req/s
  Backpressure stalls: ~6M

  Latency (µs):
    min    =      193
    p50    =      352
    mean   =      391
    p90    =      511
    p95    =      594
    p99    =    1,240
    max    =    3,798

  PyMatching decode:       223 µs avg
  Syndrome density reduction: 98.3%
  Pipeline LER:            0.0020
```

---

## System Requirements

| Dependency | Notes |
|------------|-------|
| CUDA Toolkit 12.0+ | CUDA runtime, CUDA graphs |
| TensorRT 10.x+ | Neural network inference engine |
| cudaq-realtime library | Host dispatcher, ring buffer API (installed at `/home/.cudaq_realtime/`) |
| cudaq-qec + PyMatching plugin | QEC framework and MWPM decoder (installed at `/home/.cudaqx/`) |
| NVIDIA GPU (sm_70+) | Grace Blackwell (GB200) is the primary target |
| ARM aarch64 or x86_64 | ARM (Grace) is the primary platform |
