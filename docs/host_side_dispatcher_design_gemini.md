# Host-Side Spin-Polling Dispatcher with Dynamic Worker Pool

## Design Specification

**Component**: `cudaq-qec` Realtime Decoding Subsystem
**Status**: Implemented
**Supersedes**: Device-side persistent kernel dispatcher (`dispatch_kernel_with_graph`) and Statically-mapped Host Dispatcher
**Target Platforms**: NVIDIA Grace Hopper (GH200), Grace Blackwell (GB200)
**Shared-Memory Model**: libcu++ `cuda::std::atomic` with `thread_scope_system`
**Last Updated**: 2026-03-17

---

## 1. System Context & Motivation

### 1.1 The Pipeline
The system performs real-time quantum error correction (QEC). An FPGA streams syndrome measurements into a host-device shared ring buffer continuously (~104 µs cadence for d=13, T=104). 
1. **Predecoding (GPU)**: TensorRT neural network inference (~88 µs pure GPU compute for d=13/T=104 with FP16; ~146 µs p50 in pipeline with DMA and dispatch overhead).
2. **Global Decoding (CPU)**: PyMatching (MWPM) (~224 µs average for d=13/T=104 with full 17,472-detector parity check matrix).

### 1.2 The Problem
The legacy architecture used a persistent GPU kernel to launch child CUDA graphs using `cudaStreamGraphFireAndForget`. This hit a hardcoded CUDA runtime limit of 128 cumulative launches, causing fatal crashes. A naive host-side port mapping FPGA slots 1:1 to GPU streams caused **Head-of-Line (HOL) blocking**: a single slow PyMatching decode would stall the sequential dispatcher, backing up the ring buffer and violating strict quantum coherence latency budgets.

### 1.3 The Solution
This document defines a **Host-Side Dispatcher with a Dynamic Worker Pool**. 
* The dispatcher runs on a dedicated CPU core.
* Predecoder streams and CPU workers act as an interchangeable pool.
* Inflight jobs are tagged with their origin slot, allowing out-of-order execution and completion.
* Synchronization relies exclusively on Grace Blackwell's NVLink-C2C hardware using libcu++ system-scope atomics.
* **Decoupled architecture**: PyMatching decode runs in a separate thread pool from the predecoder workers, allowing GPU streams to be released immediately after inference completion rather than blocking on CPU decode.

---

## 2. Core Architecture: Dynamic Worker Pool

Instead of mapping predecoder streams statically to incoming data, the host dispatcher maintains a bitmask of available workers (`idle_mask`).

1. **Allocate**: When `rx_flags[slot]` indicates new data, the dispatcher finds the first available worker stream using a hardware bit-scan (`__builtin_ffsll`).
2. **Tag**: The dispatcher records the original `slot` in a tracking array (`inflight_slot_tags[worker_id]`) so the response can be routed correctly.
3. **Pre-launch DMA**: If a `pre_launch_fn` callback is registered on the worker, the dispatcher calls it to issue a `cudaMemcpyAsync` (DMA engine copy) of the input payload from the ring buffer to the TRT input buffer before graph launch.
4. **Dispatch**: The dispatcher launches the CUDA graph on the assigned worker's stream and clears its availability bit.
5. **Free**: The predecoder worker thread (not the PyMatching thread) restores the worker's availability bit in the `idle_mask` after copying inference output and enqueuing the PyMatching job. Slot completion is deferred to the PyMatching thread pool.

---

## 3. Memory & Synchronization Model

**CRITICAL DIRECTIVE**: The ARM Neoverse architecture (Grace) is **weakly ordered**. Code generated from this document MUST NOT use `volatile`, `__threadfence_system()`, or `std::atomic_thread_fence`. 

All shared state must use **libcu++ system-scope atomics** allocated in mapped pinned memory (`cudaHostAllocMapped`).

### 3.1 Shared State Variables

| Variable | Type | Memory Location | Purpose |
| :--- | :--- | :--- | :--- |
| `rx_flags[NUM_SLOTS]` | `atomic<uint64_t, thread_scope_system>` | Mapped Pinned | FPGA writes data ptr; CPU polls (Acquire). |
| `tx_flags[NUM_SLOTS]` | `atomic<uint64_t, thread_scope_system>` | Mapped Pinned | CPU writes response; FPGA polls (Release). |
| `ready_flags[1]` | `atomic<int, thread_scope_system>` | Mapped Pinned | GPU signals TRT done; CPU polls (Release/Acquire). Queue depth = 1. |
| `idle_mask` | `atomic<uint64_t, thread_scope_system>` | Host CPU Mem | Bitmask of free workers. 1 = free, 0 = busy. |
| `inflight_slot_tags[NUM_WORKERS]`| `int` (Plain array) | Host CPU Mem | Maps `worker_id` -> original FPGA `slot`. |
| `mailbox_bank[NUM_WORKERS]` | `void*` (Plain array) | Mapped Pinned | Dispatcher writes device ptr for pre-launch callback. |
| `h_ring_ptrs[1]` | `void*` (Mapped Pinned) | Mapped Pinned | Pre-launch callback writes slot device ptr for CPU worker readback. |
| `h_predecoder_outputs_[1]` | `void*` (Mapped Pinned) | Mapped Pinned | GPU output copied here via DMA; CPU worker reads inference results. |

**NUM_SLOTS**: 16 (ring buffer capacity).
**NUM_WORKERS**: 8 (predecoder streams, each with a dedicated CPU poller thread).
**Queue depth**: 1 per predecoder (single in-flight inference per stream).

---

## 4. Host Dispatcher Thread (Producer)

The dispatcher loop is a tight spin-polling loop running on a dedicated CPU core. It is implemented in `realtime/lib/daemon/dispatcher/host_dispatcher.cu` as `cudaq_host_dispatcher_loop()`.

### 4.1 cudaq_host_dispatch_worker_t Structure

Each worker in the pool has the following fields:

```cpp
typedef struct {
    cudaGraphExec_t graph_exec;
    cudaStream_t stream;
    uint32_t function_id;
    void (*pre_launch_fn)(void* user_data, void* slot_dev, cudaStream_t stream);
    void* pre_launch_data;
    void (*post_launch_fn)(void* user_data, void* slot_dev, cudaStream_t stream);
    void* post_launch_data;
} cudaq_host_dispatch_worker_t;
```

The `pre_launch_fn` callback enables the dispatcher to issue a `cudaMemcpyAsync` (using the DMA copy engine) for the input payload before each graph launch, without baking application-specific logic into the generic dispatcher. The `post_launch_fn` callback is used in GPU-only mode to enqueue a `cudaLaunchHostFunc` that signals slot completion without CPU worker threads.

### 4.2 Dispatcher Logic (Pseudocode)
```cpp
void cudaq_host_dispatcher_loop(const cudaq_host_dispatcher_config_t *config) {
    size_t current_slot = 0;

    while (config.shutdown_flag->load(acquire) == 0) {
        uint64_t rx_value = config.rx_flags[current_slot].load(acquire);
        if (rx_value == 0) { QEC_CPU_RELAX(); continue; }

        void* slot_host = reinterpret_cast<void*>(rx_value);

        // Optional: parse RPC header and lookup function table
        if (use_function_table) {
            ParsedSlot parsed = parse_slot_with_function_table(slot_host, config);
            if (parsed.drop) { clear_and_advance(); continue; }
        }

        // Wait for an available worker (spin if all busy)
        int worker_id = acquire_graph_worker(config, ...);
        if (worker_id < 0) { QEC_CPU_RELAX(); continue; }

        // Mark worker busy, tag with origin slot
        config.idle_mask->fetch_and(~(1ULL << worker_id), release);
        config.inflight_slot_tags[worker_id] = current_slot;

        // Translate host ptr to device ptr, write to mailbox
        ptrdiff_t offset = (uint8_t*)slot_host - config.rx_data_host;
        void* data_dev = config.rx_data_dev + offset;
        config.h_mailbox_bank[worker_id] = data_dev;
        __sync_synchronize();

        // Pre-launch callback: DMA copy input to TRT buffer
        if (worker.pre_launch_fn)
            worker.pre_launch_fn(worker.pre_launch_data, data_dev, worker.stream);

        // Launch graph
        cudaError_t err = cudaGraphLaunch(worker.graph_exec, worker.stream);
        if (err != cudaSuccess) {
            tx_flags[current_slot].store(0xDEAD|err, release);
            idle_mask->fetch_or(1ULL << worker_id, release);
        } else {
            tx_flags[current_slot].store(0xEEEEEEEEEEEEEEEEULL, release);
        }

        // Post-launch callback (GPU-only mode: enqueue cudaLaunchHostFunc)
        if (worker.post_launch_fn)
            worker.post_launch_fn(worker.post_launch_data, data_dev, worker.stream);

        // Consume slot and advance
        rx_flags[current_slot].store(0, release);
        current_slot = (current_slot + 1) % num_slots;
    }
    for (auto& w : config.workers) cudaStreamSynchronize(w.stream);
}
```

---

## 5. GPU Graph Composition & Data Transfer

### 5.1 DMA-Based Data Movement

Data copies between the ring buffer and TRT inference buffers use the GPU's DMA copy engine rather than SM-based kernels, freeing compute resources for inference.

**Input copy (ring buffer -> TRT input)**: Issued by the host dispatcher via `pre_launch_fn` callback as a `cudaMemcpyAsync(DeviceToDevice)` on the worker's stream *before* `cudaGraphLaunch`. The source address is dynamic (determined at dispatch time from the ring buffer slot at offset `CUDAQ_RPC_HEADER_SIZE` = 24 bytes), so it cannot be baked into the captured graph.

**Output copy (TRT output -> host-mapped outputs)**: Captured inside the CUDA graph as a `cudaMemcpyAsync(DeviceToDevice)`. Both source (`d_trt_output_`) and destination (`d_predecoder_outputs_`) are fixed addresses, so this is captured at graph instantiation time.

### 5.2 Captured CUDA Graph Contents

The CUDA graph for each predecoder contains (in order):

1. **TRT inference** (`context_->enqueueV3(stream)`) -- or `passthrough_copy_kernel` if `SKIP_TRT` is set.
2. **Output DMA copy** (`cudaMemcpyAsync` D2D) -- copies TRT output to host-mapped predecoder output buffer (`h_predecoder_outputs_`).
3. **Signal kernel** (`predecoder_signal_ready_kernel<<<1,1>>>`) -- a single-thread kernel that performs `d_ready_flags[0].store(1, release)` to notify the CPU worker.

The graph is instantiated with `cudaGraphInstantiate(&graph_exec_, graph, 0)` for host-launch mode. Input data arrives exclusively via the pre-launch DMA copy callback; no input-copy kernel exists in the graph.

### 5.3 Source Files

The `ai_predecoder_service.cu` implementation contains only two device kernels:

- `predecoder_signal_ready_kernel` -- single-thread kernel that atomically stores `1` to the ready flag with system-scope release semantics.
- `passthrough_copy_kernel` -- vectorized identity copy (`uint4` 16-byte loads/stores, 256 threads) used when `SKIP_TRT` is set, substituting for TRT inference.

### 5.4 Passthrough Copy Kernel (SKIP_TRT mode)

When `SKIP_TRT` is set, the `passthrough_copy_kernel` substitutes for TRT inference, providing a deterministic identity function for testing and benchmarking the infrastructure overhead. In SKIP_TRT mode, the `AIDecoderService` constructor sets `input_size_ = output_size_ = 1600 * sizeof(float)` (6400 bytes) without loading any model file.

---

## 6. Decoupled Worker Architecture

The CPU-side processing uses a **two-tier decoupled architecture** that separates GPU polling from CPU-intensive decode:

### 6.1 Tier 1: Predecoder Workers (GPU Polling + Copy)

Each predecoder has a dedicated worker thread in the `RealtimePipeline`. These threads:

1. **Poll** `ready_flags[0]` via `compare_exchange_strong(1, 2, acquire, relaxed)` (CAS claiming).
2. **Copy** inference output from `h_predecoder_outputs_` to a per-slot buffer (`deferred_outputs[origin_slot]`).
3. **Compute** syndrome density metrics (input vs. output nonzero detector counts).
4. **Release** the GPU predecoder slot via `release_job(slot_idx)` → `ready_flags[0].store(0, release)`.
5. **Enqueue** a `PyMatchJob` to the `PyMatchQueue`.
6. **Return** `DEFERRED_COMPLETION` to the pipeline, which releases `idle_mask` but does NOT set `tx_flags`.

### 6.2 Tier 2: PyMatching Workers (CPU Decode + Completion)

A separate thread pool (16 workers for d13_r104) dequeues from `PyMatchQueue` and:

1. **Decode** using a per-thread PyMatching decoder instance (lock-free `thread_local` acquisition).
2. **Write** the RPC response (`DecodeResponse`) directly into the ring buffer slot.
3. **Signal** slot completion via `pipeline.complete_deferred(origin_slot)`, which stores the slot host address into `tx_flags[origin_slot]`.

### 6.3 Why Decouple?

With the coupled architecture, a single slow PyMatching decode (up to 139 ms tail latency) would hold the predecoder worker busy, preventing the GPU stream from being dispatched new work. This caused:
- Severe head-of-line blocking on `idle_mask`
- ~41M backpressure stalls
- Tail latencies: p90 = 970 µs, p99 = 1,767 µs

The decoupled architecture reduces predecoder worker hold time from ~214 µs to ~10 µs, dropping:
- Backpressure stalls from 41M to 6.2M (85% reduction)
- p90 from 970 µs to 515 µs (47% reduction)
- p99 from 1,767 µs to 1,249 µs (29% reduction)

### 6.4 DEFERRED_COMPLETION Protocol

```
Pipeline Worker Thread:         PyMatching Thread:
  poll_next_job() → CAS 1→2      (blocked on queue)
  copy output to deferred_buf          |
  release_job() → store 0             |
  enqueue PyMatchJob ──────────►  pop PyMatchJob
  return DEFERRED_COMPLETION      decode with PyMatching
  pipeline sets idle_mask ✓       write RPC response
  pipeline skips tx_flags ✗       complete_deferred(slot)
                                   └──► tx_flags[slot].store(addr)
```

### 6.5 PyMatchQueue

Thread-safe MPSC queue using `std::mutex` + `std::condition_variable`:

```cpp
struct PyMatchJob {
    int origin_slot;
    uint64_t request_id;
    void *ring_buffer_ptr;
};

class PyMatchQueue {
    std::mutex mtx_;
    std::condition_variable cv_;
    std::queue<PyMatchJob> jobs_;
    bool stop_ = false;
public:
    void push(PyMatchJob &&j);
    bool pop(PyMatchJob &out);  // blocks until job available or shutdown
    void shutdown();
};
```

### 6.6 Ready-Flag State Machine (Atomic Claiming)

With queue depth 1, the poller must **claim** each completion exactly once.

**States** (per-worker ready flag):

| Value | State      | Meaning |
| :---  | :---       | :---    |
| 0     | Idle       | Waiting for GPU, or worker has called `release_job`. |
| 1     | Ready      | GPU finished; signal kernel stored 1. |
| 2     | Processing | CPU poller claimed the job; copying output. |

**Poller**: Use `compare_exchange_strong(expected=1, desired=2, memory_order_acquire, memory_order_relaxed)`. Only the thread that wins the CAS proceeds. Use **relaxed on failure** so spin-polling does not add barriers that delay seeing the GPU's store(1).

**Worker**: When output is copied and job is enqueued, call `release_job(slot_idx)` which does `ready_flags[0].store(0, release)` so the slot is Idle for the next launch.

---

## 7. Out-of-Order Consumer

The consumer thread harvests completions **out-of-order** by scanning all active slots on every iteration, rather than waiting for a sequential `next_harvest` counter. This eliminates head-of-line blocking where a slow request in slot N would prevent harvesting faster completions in slot N+1.

### 7.1 Consumer Logic (Pseudocode)
```cpp
while (!consumer_stop) {
    bool found_any = false;
    for (uint32_t s = 0; s < NUM_SLOTS; ++s) {
        if (!slot_occupied[s]) continue;

        cudaq_tx_status_t status = cudaq_host_ringbuffer_poll_tx_flag(&rb, s, &err);

        if (status == CUDAQ_TX_READY) {
            int rid = slot_request[s];
            complete_ts[rid] = now();
            completed[rid] = true;
            total_completed++;

            slot_occupied[s] = 0;      // Reset occupancy FIRST
            __sync_synchronize();       // ARM memory fence
            cudaq_host_ringbuffer_clear_slot(&rb, s);  // Then clear tx_flags
            found_any = true;
        }
    }
    if (!found_any) QEC_CPU_RELAX();
}
```

### 7.2 Consumer-Producer Race Fix

On ARM's weakly ordered memory model, the consumer must reset `slot_occupied[s] = 0` **before** clearing `tx_flags[s]` (via `cudaq_host_ringbuffer_clear_slot`), with a `__sync_synchronize()` fence between them. Without this ordering:
1. Consumer clears `tx_flags[s]` (slot appears free to producer)
2. Producer writes new `slot_occupied[s] = 1` 
3. Consumer's delayed `slot_occupied[s] = 0` clobbers the producer's write

This race caused exactly one request to get "stuck" indefinitely, eventually stalling the entire pipeline through backpressure.

---

## 8. RealtimePipeline Scaffolding

The low-level dispatcher, consumer, and worker threads are wrapped by a higher-level `RealtimePipeline` class (`libs/qec/include/cudaq/qec/realtime/pipeline.h`) that hides all ring buffer management, atomics, and thread lifecycle. Application code provides three callbacks:

1. **GPU stage factory** (`GpuStageFactory`): Called once per worker during `start()`. Returns the `cudaGraphExec_t`, `cudaStream_t`, `pre_launch_fn`, `post_launch_fn`, `function_id`, and an opaque `user_context` for each worker.
2. **CPU stage callback** (`CpuStageCallback`): Called by each worker thread when GPU inference completes. Receives `CpuStageContext` containing `gpu_output`, `gpu_output_size`, `response_buffer`, and the `user_context`. Returns the number of bytes written, `0` if no result ready (poll again), or `DEFERRED_COMPLETION` to release the worker without signaling slot completion.
3. **Completion callback** (`CompletionCallback`): Called by the consumer thread for each completed (or errored) request with a `Completion` struct.

```cpp
RealtimePipeline pipeline(config);
pipeline.set_gpu_stage([&](int worker_id) -> GpuWorkerResources { ... });
pipeline.set_cpu_stage([&](const CpuStageContext& ctx) -> size_t { ... });
pipeline.set_completion_handler([&](const Completion& c) { ... });
auto injector = pipeline.create_injector();
pipeline.start();
injector.submit(function_id, payload, payload_size, request_id);
// ...
pipeline.stop();
```

### 8.1 DEFERRED_COMPLETION

When the CPU stage callback returns `DEFERRED_COMPLETION` (= `SIZE_MAX`), the pipeline:
- Sets the worker's bit in `idle_mask` (worker is free for next dispatch)
- Does NOT write to `tx_flags[origin_slot]` (slot stays IN_FLIGHT)

The caller is responsible for eventually calling `pipeline.complete_deferred(slot)`, which stores the slot host address into `tx_flags[slot]` with release semantics, making the completion visible to the consumer.

### 8.2 GPU-Only Mode

If no `CpuStageCallback` is registered, the pipeline operates in **GPU-only mode**: no CPU worker threads are spawned. Instead, the dispatcher's `post_launch_fn` enqueues a `cudaLaunchHostFunc` on each worker stream. When the GPU finishes, the CUDA runtime calls the host function, which stores into `tx_flags` and restores the `idle_mask` bit — all from the CUDA callback thread.

### 8.3 RingBufferInjector

The `RingBufferInjector` class (created via `pipeline.create_injector()`) encapsulates the host-side submission logic for testing without FPGA hardware. It provides:

- `try_submit()`: Non-blocking, returns false on backpressure.
- `submit()`: Blocking spin-wait until a slot becomes available.
- `backpressure_stalls()`: Counter of spin iterations during backpressure.

The injector uses a round-robin slot selection with atomic CAS for thread safety.

The `PipelineStageConfig` allows configuring `num_workers`, `num_slots`, `slot_size`, and optional `CorePinning` for dispatcher, consumer, and worker threads.

---

## 9. Step-by-Step Data Flow Trace

1. **Producer** writes uint8 measurements into `payload_buf` from Stim test data.
2. **Producer** calls `injector.submit(fid, payload, size, request_id)`.
3. **RingBufferInjector** writes RPC header (`RPCHeader`: magic, function_id, arg_len, request_id, ptp_timestamp = 24 bytes) + payload into `rx_data[slot]`.
4. **RingBufferInjector** sets `rx_flags[slot] = host_ptr` (release).
5. **Host Dispatcher** reads `rx_flags[slot]`, sees data.
6. **Host Dispatcher** parses RPC header, looks up function in the function table.
7. **Host Dispatcher** scans `idle_mask`, finds `worker_id = 2` is free.
8. **Host Dispatcher** marks bit 2 busy in `idle_mask`.
9. **Host Dispatcher** saves `inflight_slot_tags[2] = slot`.
10. **Host Dispatcher** translates `host_ptr` to `dev_ptr`, writes to `mailbox_bank[2]`.
11. **Host Dispatcher** calls `pre_launch_fn`: writes `h_ring_ptrs[0] = dev_ptr`, issues `cudaMemcpyAsync(d_trt_input, dev_ptr + 24, input_size, D2D, stream[2])`.
12. **Host Dispatcher** calls `cudaGraphLaunch(..., stream[2])`.
13. **Host Dispatcher** sets `tx_flags[slot] = 0xEEEE...` (IN_FLIGHT), then clears `rx_flags[slot] = 0` and advances to next slot.
14. **GPU DMA engine** copies input payload from ring buffer to TRT input buffer.
15. **GPU** executes TRT inference (or passthrough copy in SKIP_TRT mode).
16. **GPU DMA engine** copies TRT output to host-mapped `h_predecoder_outputs_`.
17. **GPU signal kernel** sets `ready_flags[0] = 1` (system-scope atomic release).
18. **Predecoder Worker** CAS(1, 2) on `ready_flags[0]`, wins, reads inference output.
19. **Predecoder Worker** copies output to `deferred_outputs[origin_slot]`.
20. **Predecoder Worker** computes syndrome density metrics.
21. **Predecoder Worker** calls `release_job(0)` → `ready_flags[0].store(0, release)`.
22. **Predecoder Worker** extracts `request_id` from RPC header, enqueues `PyMatchJob`.
23. **Predecoder Worker** returns `DEFERRED_COMPLETION`.
24. **Pipeline** restores bit 2 in `idle_mask` (worker free for next dispatch). Does NOT touch `tx_flags`.
25. **PyMatching Worker** pops `PyMatchJob` from queue, acquires per-thread decoder.
26. **PyMatching Worker** runs PyMatching MWPM decode over full parity check matrix.
27. **PyMatching Worker** writes `RPCResponse + DecodeResponse` into ring buffer slot.
28. **PyMatching Worker** calls `pipeline.complete_deferred(slot)` → `tx_flags[slot].store(host_addr, release)`.
29. **Consumer** scans all slots, sees `tx_flags[slot] != 0` and `!= 0xEEEE`, harvests.
30. **Consumer** calls `completion_handler(request_id, slot, success)`.
31. **Consumer** sets `slot_occupied[slot] = 0`, `__sync_synchronize()`, then clears `tx_flags[slot] = 0`. Producer may now reuse slot.

---

## 10. RPC Protocol & Ring Buffer

### 10.1 RPC Header

```cpp
struct RPCHeader {
    uint32_t magic;        // RPC_MAGIC_REQUEST
    uint32_t function_id;  // FNV-1a hash of function name
    uint32_t arg_len;      // payload length in bytes
    uint32_t request_id;   // unique request identifier
    uint64_t ptp_timestamp; // PTP timestamp (optional)
};
// sizeof(RPCHeader) == 24
#define CUDAQ_RPC_HEADER_SIZE 24u
```

### 10.2 IN_FLIGHT Sentinel

Because `cudaGraphLaunch` is asynchronous, the dispatcher clears `rx_flags[slot]` immediately after launch. Without a hold, the **producer** (FPGA sim or test) would see `rx_flags[slot]==0` and `tx_flags[slot]==0` (response not written yet) and reuse the slot, overwriting data while the GPU is still reading.

**Fix: IN_FLIGHT tag**

1. **Dispatcher**: On successful launch, write `tx_flags[current_slot].store(0xEEEEEEEEEEEEEEEEULL, release)` **before** clearing `rx_flags[current_slot]`. On launch failure, write the 0xDEAD|err value and restore the worker bit; do not write 0xEEEE. Setting `tx_data_host = nullptr` and `tx_data_dev = nullptr` in the config forces the dispatcher to use the `0xEEEE` sentinel rather than a real data address.
2. **Producer**: Reuse a slot only when **both** `rx_flags[slot]==0` **and** `tx_flags[slot]==0`. Thus the producer blocks until the consumer has harvested (tx cleared).
3. **Consumer**: When harvesting, treat only real responses: `tx_flags[slot] != 0` **and** `tx_flags[slot] != 0xEEEEEEEEEEEEEEEEULL`. Ignore 0xEEEE (in-flight). On harvest, clear `tx_flags[slot] = 0`.

**Slot lifecycle**: Idle (rx=0, tx=0) -> Written (rx=ptr, tx=0) -> In-flight (rx=0, tx=0xEEEE) -> Completed (rx=0, tx=response) -> Consumer harvests, tx=0 -> Idle.

---

## 11. Dynamic Batch Handling for ONNX Models

When building a TensorRT engine from an ONNX model with dynamic batch dimensions (dim 0 <= 0), `ai_decoder_service.cu` automatically creates an optimization profile that pins all dynamic dimensions to 1. This enables building engines from models like `predecoder_memory_d13_T13_X.onnx` which use a symbolic `batch` dimension.

---

## 12. Test Suite

A GTest-based test suite (`libs/qec/unittests/test_realtime_pipeline.cu`) validates the pipeline using `SKIP_TRT` passthrough mode (no TensorRT dependency at runtime). The tests are organized into three categories:

### 12.1 Unit Tests (8 tests)
- **AIDecoderService**: Verify SKIP_TRT buffer sizes (1600 floats = 6400 bytes), allocation, and graph capture.
- **AIPreDecoderService**: Verify mapped pinned memory allocation, `poll_next_job` / `release_job` state machine, and host-launchable graph.

### 12.2 Correctness Tests (5 tests)
Data-integrity tests that verify known payloads survive the full CUDA graph round-trip bitwise-identical (memcmp, not epsilon):
- **Zeros, Known Pattern, Random Data, Extreme Float Values**: Single-request verification with different payload patterns (including `FLT_MAX`, `NaN`, `INFINITY`).
- **Multiple Requests (5,000 iterations)**: Pushes 5,000 random 6.4 KB payloads through the pipeline and verifies bitwise identity on every one. Confirms no cross-contamination or data corruption over sustained use.

### 12.3 Integration Tests (8 tests)
- **Dispatcher lifecycle**: Shutdown semantics, stats counter accuracy, invalid RPC magic rejection, slot wraparound.
- **Single Request Round-Trip**: Full dispatcher -> graph -> poll -> verify data path.
- **Multi-Predecoder Concurrency**: 4 predecoders on 4 streams, simultaneous dispatch, per-predecoder data verification.
- **Sustained Throughput (200 requests)**: Regression test for the 128-launch-limit fix. Proves indefinite stability of the host-side dispatcher.

### 12.4 End-to-End Benchmark (test_realtime_predecoder_w_pymatching)
- Configurable surface code distance and round count: d7, d13, d13_r104, d21, d31.
- Loads Stim-generated test data (detectors, observables, parity check matrix, priors).
- Streams syndrome data at configurable rate with correctness verification (LER).
- Reports latency percentiles, throughput, backpressure stalls, syndrome density reduction.

---

## 13. Shutdown and Grace Period

- **Grace period**: After the producer stops submitting, the pipeline waits up to 5 seconds for `total_completed >= total_submitted`.
- **Consumer exit**: The consumer thread normally exits when `producer_stop && total_completed >= total_submitted`. To avoid hanging forever if some in-flight requests never complete, set a **consumer_stop** flag after the grace period; the consumer loop checks this and exits so `consumer.join()` returns and the process can print the final report and exit cleanly.
- **Dispatcher shutdown**: Set `shutdown_flag = 1` after the consumer exits, then join the dispatcher thread. The dispatcher synchronizes all worker streams before returning.
- **PyMatching thread pool**: Call `pymatch_queue.shutdown()` to unblock all waiting threads, then join all PyMatching worker threads.

---

## 14. Performance Results (d=13, T=104, 104 µs rate, 20s)

### 14.1 Decoupled Architecture (current)

Measured on Grace Blackwell (GB200) with `predecoder_memory_d13_T104_X.onnx` (FP16), 8 predecoders, 16 PyMatching workers, 16 slots:

| Metric | Value |
| :--- | :--- |
| Throughput | 9,610 req/s |
| Submitted / Completed | 192,309 / 192,309 |
| Backpressure stalls | 6,193,418 |
| p50 latency | 352.3 µs |
| Mean latency | 393.4 µs |
| p90 latency | 514.7 µs |
| p95 latency | 606.3 µs |
| p99 latency | 1,248.8 µs |
| Max latency | 3,930.0 µs |
| PyMatching decode (avg) | 224.4 µs |
| Syndrome density reduction | 98.3% |
| Pipeline LER | 0.0020 |

### 14.2 Raw TRT Baseline (trtexec)

| Mode | GPU Compute | Total Host Latency |
| :--- | :--- | :--- |
| Default | 107 µs | 119 µs |
| CUDA Graph + SpinWait | 90 µs | 99 µs |
| CUDA Graph + SpinWait + No Transfers | 88 µs | 88 µs |

---

## 15. LLM Implementation Directives (Constraints Checklist)

When generating code from this specification, the LLM **MUST** strictly adhere to the following constraints:

- [ ] **NO CUDA STREAM QUERYING**: Do not use `cudaStreamQuery()` for backpressure or completion checking. It incurs severe driver latency. Rely strictly on `idle_mask` and `ready_flags`.
- [ ] **NO WEAK ORDERING BUGS**: Do not use `volatile`. Do not use `__threadfence_system()`. You must use `cuda::std::atomic<T, cuda::thread_scope_system>` (or `<cuda/atomic>` with `thread_scope_system`) for all cross-device synchronization.
- [ ] **NO HEAD OF LINE BLOCKING**: The host dispatcher MUST NOT statically map slots to predecoders. It must dynamically allocate via `idle_mask`. The consumer MUST harvest out-of-order by scanning all active slots.
- [ ] **NO DATA LOSS**: If `idle_mask == 0` (all workers busy), the dispatcher MUST spin on the current slot (`QEC_CPU_RELAX()`). It MUST NOT advance `current_slot` until a worker is allocated and the graph is launched.
- [ ] **NO RACE CONDITIONS ON TAGS**: `inflight_slot_tags` does not need to be atomic because index `[worker_id]` is exclusively owned by the active flow once the dispatcher clears the bit in `idle_mask`, until the worker thread restores the bit.
- [ ] **READY FLAG CLAIMING**: The CPU poller MUST claim each completion exactly once using compare_exchange_strong(1, 2) on the ready flag; use relaxed memory order on CAS failure. The worker MUST clear the flag (store 0) in `release_job`.
- [ ] **IN_FLIGHT SENTINEL**: After a successful `cudaGraphLaunch`, the dispatcher MUST write `tx_flags[current_slot] = 0xEEEEEEEEEEEEEEEEULL` before clearing `rx_flags[current_slot]`. Set `tx_data_host = nullptr` and `tx_data_dev = nullptr` to force the 0xEEEE path. The producer MUST wait for both rx and tx to be 0 before reusing a slot. The consumer MUST ignore 0xEEEE and only harvest real responses (or 0xDEAD errors).
- [ ] **CONSUMER MEMORY ORDERING**: The consumer MUST set `slot_occupied[s] = 0` BEFORE calling `cudaq_host_ringbuffer_clear_slot`, with a `__sync_synchronize()` fence between them, to prevent the producer-consumer race on ARM.
- [ ] **DMA DATA MOVEMENT**: Use `cudaMemcpyAsync` (DMA engine) for data copies. Input copy is issued via `pre_launch_fn` callback before graph launch at offset `CUDAQ_RPC_HEADER_SIZE` (24 bytes). Output copy is captured inside the graph. Do not use SM-based byte-copy kernels for fixed-address transfers.
- [ ] **NO INPUT KERNEL IN GRAPH**: The captured CUDA graph must NOT contain an input-copy kernel. All input data movement is handled by the `pre_launch_fn` DMA callback issued on the worker stream before `cudaGraphLaunch`.
- [ ] **DEFERRED COMPLETION**: When the CPU stage returns `DEFERRED_COMPLETION`, the pipeline MUST release `idle_mask` but MUST NOT write `tx_flags`. The external caller MUST call `complete_deferred(slot)` to signal completion.
- [ ] **SHUTDOWN**: Use a `consumer_stop` (or equivalent) flag so the consumer thread can exit after a grace period even when `total_completed < total_submitted`; join the consumer after setting the flag so the process exits cleanly. Shut down the PyMatching queue before stopping the pipeline.
