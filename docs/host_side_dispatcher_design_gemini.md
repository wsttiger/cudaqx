# Host-Side Spin-Polling Dispatcher with Dynamic Worker Pool

## Design Specification

**Component**: `cudaq-qec` Realtime Decoding Subsystem
**Status**: Implemented
**Supersedes**: Device-side persistent kernel dispatcher (`dispatch_kernel_with_graph`) and Statically-mapped Host Dispatcher
**Target Platforms**: NVIDIA Grace Hopper (GH200), Grace Blackwell (GB200)
**Shared-Memory Model**: libcu++ `cuda::std::atomic` with `thread_scope_system`
**Last Updated**: 2026-02-26

---

## 1. System Context & Motivation

### 1.1 The Pipeline
The system performs real-time quantum error correction (QEC). An FPGA streams syndrome measurements into a host-device shared ring buffer continuously (~1 µs cadence). 
1. **Predecoding (GPU)**: TensorRT neural network inference (~70 µs for d=13 with FP16).
2. **Global Decoding (CPU)**: PyMatching (MWPM) (~11 µs for d=13 with `predecoder_memory` model, up to ~70 µs with denser residual models).

### 1.2 The Problem
The legacy architecture used a persistent GPU kernel to launch child CUDA graphs using `cudaStreamGraphFireAndForget`. This hit a hardcoded CUDA runtime limit of 128 cumulative launches, causing fatal crashes. A naive host-side port mapping FPGA slots 1:1 to GPU streams caused **Head-of-Line (HOL) blocking**: a single slow PyMatching decode would stall the sequential dispatcher, backing up the ring buffer and violating strict quantum coherence latency budgets.

### 1.3 The Solution
This document defines a **Host-Side Dispatcher with a Dynamic Worker Pool**. 
* The dispatcher runs on a dedicated CPU core.
* Predecoder streams and CPU workers act as an interchangeable pool.
* Inflight jobs are tagged with their origin slot, allowing out-of-order execution and completion.
* Synchronization relies exclusively on Grace Blackwell's NVLink-C2C hardware using libcu++ system-scope atomics.

---

## 2. Core Architecture: Dynamic Worker Pool

Instead of mapping predecoder streams statically to incoming data, the host dispatcher maintains a bitmask of available workers (`idle_mask`).

1. **Allocate**: When `rx_flags[slot]` indicates new data, the dispatcher finds the first available worker stream using a hardware bit-scan (`__builtin_ffsll`).
2. **Tag**: The dispatcher records the original `slot` in a tracking array (`inflight_slot_tags[worker_id]`) so the response can be routed correctly.
3. **Pre-launch DMA**: If a `pre_launch_fn` callback is registered on the worker, the dispatcher calls it to issue a `cudaMemcpyAsync` (DMA engine copy) of the input payload from the ring buffer to the TRT input buffer before graph launch.
4. **Dispatch**: The dispatcher launches the CUDA graph on the assigned worker's stream and clears its availability bit.
5. **Free**: When the CPU PyMatching worker finishes the job and writes the response to `tx_flags[origin_slot]`, it restores the worker's availability bit in the `idle_mask`.

---

## 3. Memory & Synchronization Model

**CRITICAL DIRECTIVE**: The ARM Neoverse architecture (Grace) is **weakly ordered**. Code generated from this document MUST NOT use `volatile`, `__threadfence_system()`, or `std::atomic_thread_fence`. 

All shared state must use **libcu++ system-scope atomics** allocated in mapped pinned memory (`cudaHostAllocMapped`).

### 3.1 Shared State Variables

| Variable | Type | Memory Location | Purpose |
| :--- | :--- | :--- | :--- |
| `rx_flags[NUM_SLOTS]` | `atomic<uint64_t, thread_scope_system>` | Mapped Pinned | FPGA writes data ptr; CPU polls (Acquire). |
| `tx_flags[NUM_SLOTS]` | `atomic<uint64_t, thread_scope_system>` | Mapped Pinned | CPU writes response; FPGA polls (Release). |
| `ready_flags[NUM_WORKERS]` | `atomic<int, thread_scope_system>` | Mapped Pinned | GPU signals TRT done; CPU polls (Release/Acquire). |
| `idle_mask` | `atomic<uint64_t, thread_scope_system>` | Host CPU Mem | Bitmask of free workers. 1 = free, 0 = busy. |
| `inflight_slot_tags[NUM_WORKERS]`| `int` (Plain array) | Host CPU Mem | Maps `worker_id` -> original FPGA `slot`. |
| `mailbox_bank[NUM_WORKERS]` | `void*` (Plain array) | Mapped Pinned | Dispatcher writes device ptr for pre-launch callback. |
| `h_ring_ptrs[NUM_WORKERS]` | `void*` (Plain array) | Mapped Pinned | Pre-launch callback writes slot device ptr for CPU worker readback. |
| `h_outputs[NUM_WORKERS]` | `void*` (Mapped Pinned) | Mapped Pinned | GPU output copied here via DMA; CPU worker reads inference results. |

---

## 4. Host Dispatcher Thread (Producer)

The dispatcher loop is a tight spin-polling loop running on a dedicated CPU core. It is implemented in `realtime/lib/daemon/dispatcher/host_dispatcher.cu` as `host_dispatcher_loop()`.

### 4.1 HostDispatchWorker Structure

Each worker in the pool has the following fields:

```cpp
struct HostDispatchWorker {
    cudaGraphExec_t graph_exec;
    cudaStream_t stream;
    uint32_t function_id;
    void (*pre_launch_fn)(void* user_data, void* slot_dev, cudaStream_t stream) = nullptr;
    void* pre_launch_data = nullptr;
};
```

The `pre_launch_fn` callback enables the dispatcher to issue a `cudaMemcpyAsync` (using the DMA copy engine) for the input payload before each graph launch, without baking application-specific logic into the generic dispatcher.

### 4.2 Dispatcher Logic (Pseudocode)
```cpp
void host_dispatcher_loop(const HostDispatcherConfig& config) {
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

**Input copy (ring buffer -> TRT input)**: Issued by the host dispatcher via `pre_launch_fn` callback as a `cudaMemcpyAsync(DeviceToDevice)` on the worker's stream *before* `cudaGraphLaunch`. The source address is dynamic (determined at dispatch time from the ring buffer slot), so it cannot be baked into the captured graph.

**Output copy (TRT output -> host-mapped outputs)**: Captured inside the CUDA graph as a `cudaMemcpyAsync(DeviceToDevice)`. Both source (`d_trt_output_`) and destination (`d_outputs_`) are fixed addresses, so this is captured at graph instantiation time.

### 5.2 Captured CUDA Graph Contents

The CUDA graph for each predecoder contains (in order):

1. **TRT inference** (`context_->enqueueV3(stream)`) -- or `passthrough_copy_kernel` if `SKIP_TRT` is set.
2. **Output DMA copy** (`cudaMemcpyAsync` D2D) -- copies TRT output to host-mapped output buffer.
3. **Signal kernel** (`predecoder_signal_ready_kernel<<<1,1>>>`) -- a single-thread kernel that performs `d_ready_flags[0].store(1, release)` to notify the CPU worker.

The graph is instantiated with `cudaGraphInstantiate(&graph_exec_, graph, 0)` for host-launch mode. The `predecoder_input_kernel` is no longer part of the graph; input data arrives via the pre-launch DMA copy.

### 5.3 Passthrough Copy Kernel (SKIP_TRT mode)

When `SKIP_TRT` is set, a vectorized passthrough kernel (`uint4` 16-byte loads/stores, 256 threads) substitutes for TRT inference for benchmarking the infrastructure overhead.

---

## 6. Worker Subsystem (Consumer)

### 6.1 Ready-Flag State Machine (Atomic Claiming)

With a single slot per predecoder (queue depth 1), the poller must **claim** each completion exactly once.

**States** (per-worker ready flag):

| Value | State      | Meaning |
| :---  | :---       | :---    |
| 0     | Idle       | Waiting for GPU, or worker has called `release_job`. |
| 1     | Ready      | GPU finished; signal kernel stored 1. |
| 2     | Processing | CPU poller claimed the job; PyMatching is running. |

**Poller**: Use `compare_exchange_strong(expected=1, desired=2, memory_order_acquire, memory_order_relaxed)`. Only the thread that wins the CAS enqueues the job. Use **relaxed on failure** so spin-polling does not add barriers that delay seeing the GPU's store(1).

**Worker**: When PyMatching finishes, call `release_job(slot_idx)` which does `ready_flags[0].store(0, release)` so the slot is Idle for the next launch.

### 6.2 Dedicated Polling/Worker Threads

Each predecoder has a dedicated polling thread that spins on `poll_next_job()` (the CAS), then runs PyMatching inline on the same thread. This avoids thread pool overhead.

### 6.3 Worker Logic (Pseudocode)
```cpp
void pymatching_worker_task(PreDecoderJob job, int worker_id,
                            AIPreDecoderService* predecoder,
                            DecoderContext* ctx,
                            WorkerPoolContext* pool_ctx) {
    // 1. Read GPU outputs from mapped pinned memory (h_outputs_)
    const int32_t* residual = static_cast<const int32_t*>(job.inference_data);

    // 2. Run PyMatching MWPM decode over spatial slices
    for (int s = 0; s < ctx->spatial_slices; ++s) {
        // ... decode each spatial slice ...
    }

    // 3. Write RPC response back to the ring buffer slot
    auto* header = static_cast<RPCResponse*>(job.ring_buffer_ptr);
    header->magic = RPC_MAGIC_RESPONSE;
    header->status = 0;
    header->result_len = sizeof(resp_data);

    // 4. Lookup origin slot and signal completion via tx_flags
    int origin_slot = job.origin_slot;
    pool_ctx->tx_flags[origin_slot].store(
        reinterpret_cast<uint64_t>(job.ring_buffer_ptr), release);

    // 5. Release GPU predecoder slot (2 -> 0)
    predecoder->release_job(job.slot_idx);

    // 6. Return worker to the dispatcher pool
    pool_ctx->idle_mask->fetch_or(1ULL << worker_id, release);
}
```

---

## 7. Out-of-Order Consumer

The consumer thread harvests completions **out-of-order** by scanning all active slots on every iteration, rather than waiting for a sequential `next_harvest` counter. This eliminates head-of-line blocking where a slow request in slot N would prevent harvesting faster completions in slot N+1.

### 7.1 Consumer Logic (Pseudocode)
```cpp
// Consumer scans all slots each iteration
while (!consumer_stop) {
    bool found_any = false;
    for (uint32_t s = 0; s < NUM_SLOTS; ++s) {
        if (slot_request[s] < 0) continue;  // no active request in this slot

        cudaq_tx_status_t status = cudaq_host_ringbuffer_poll_tx_flag(&rb, s, &err);

        if (status == CUDAQ_TX_READY) {
            int rid = slot_request[s];
            complete_ts[rid] = now();
            completed[rid] = true;
            total_completed++;

            slot_request[s] = -1;       // Reset request ID FIRST
            __sync_synchronize();       // ARM memory fence
            cudaq_host_ringbuffer_clear_slot(&rb, s);  // Then clear tx_flags
            found_any = true;
        }
    }
    if (!found_any) QEC_CPU_RELAX();
}
```

### 7.2 Consumer-Producer Race Fix

On ARM's weakly ordered memory model, the consumer must reset `slot_request[s] = -1` **before** clearing `tx_flags[s]` (via `cudaq_host_ringbuffer_clear_slot`), with a `__sync_synchronize()` fence between them. Without this ordering:
1. Consumer clears `tx_flags[s]` (slot appears free to producer)
2. Producer writes new `slot_request[s] = new_rid` 
3. Consumer's delayed `slot_request[s] = -1` clobbers the producer's write

This race caused exactly one request to get "stuck" indefinitely, eventually stalling the entire pipeline through backpressure.

---

## 8. Step-by-Step Data Flow Trace

1. **FPGA** writes INT32 measurements into `rx_data[5]`.
2. **FPGA** sets `rx_flags[5] = host_ptr`.
3. **Host Dispatcher** reads `rx_flags[5]`, sees data.
4. **Host Dispatcher** parses RPC header, looks up function in the function table.
5. **Host Dispatcher** scans `idle_mask`, finds `worker_id = 2` is free.
6. **Host Dispatcher** marks bit 2 busy in `idle_mask`.
7. **Host Dispatcher** saves `inflight_slot_tags[2] = 5`.
8. **Host Dispatcher** translates `host_ptr` to `dev_ptr`, writes to `mailbox_bank[2]`.
9. **Host Dispatcher** calls `pre_launch_fn`: writes `h_ring_ptrs[0] = dev_ptr`, issues `cudaMemcpyAsync(d_trt_input, dev_ptr + 12, input_size, D2D, stream[2])`.
10. **Host Dispatcher** calls `cudaGraphLaunch(..., stream[2])`.
11. **Host Dispatcher** sets `tx_flags[5] = 0xEEEE...` (IN_FLIGHT), then clears `rx_flags[5] = 0` and advances to `current_slot = 6`.
12. **GPU DMA engine** copies input payload from ring buffer to TRT input buffer.
13. **GPU** executes TRT inference.
14. **GPU DMA engine** copies TRT output to host-mapped `h_outputs_`.
15. **GPU signal kernel** sets `ready_flags[2] = 1` (system-scope atomic release).
16. **CPU Poller** CAS(1, 2) on `ready_flags[2]`, wins, reads `h_ring_ptrs[0]` to get ring buffer address and `h_outputs_` to get inference data.
17. **CPU Worker** runs PyMatching decode over spatial slices.
18. **CPU Worker** writes RPC response into ring buffer slot.
19. **CPU Worker** looks up `origin_slot = inflight_slot_tags[2]` (which is 5).
20. **CPU Worker** writes response address to `tx_flags[5]` (overwrites 0xEEEE).
21. **CPU Worker** calls `release_job` (`ready_flags[0].store(0, release)`), then restores bit 2 in `idle_mask`.
22. **Consumer** scans all slots, sees `tx_flags[5] != 0` and `!= 0xEEEE`, harvests.
23. **Consumer** sets `slot_request[5] = -1`, `__sync_synchronize()`, then clears `tx_flags[5] = 0`. Producer may now reuse slot 5.

---

## 9. Ring Buffer and IN_FLIGHT Sentinel

Because `cudaGraphLaunch` is asynchronous, the dispatcher clears `rx_flags[slot]` immediately after launch. Without a hold, the **producer** (FPGA sim or test) would see `rx_flags[slot]==0` and `tx_flags[slot]==0` (response not written yet) and reuse the slot, overwriting data while the GPU is still reading.

**Fix: IN_FLIGHT tag**

1. **Dispatcher**: On successful launch, write `tx_flags[current_slot].store(0xEEEEEEEEEEEEEEEEULL, release)` **before** clearing `rx_flags[current_slot]`. On launch failure, write the 0xDEAD|err value and restore the worker bit; do not write 0xEEEE. Setting `tx_data_host = nullptr` and `tx_data_dev = nullptr` in the config forces the dispatcher to use the `0xEEEE` sentinel rather than a real data address.
2. **Producer**: Reuse a slot only when **both** `rx_flags[slot]==0` **and** `tx_flags[slot]==0`. Thus the producer blocks until the consumer has harvested (tx cleared).
3. **Consumer**: When harvesting, treat only real responses: `tx_flags[slot] != 0` **and** `tx_flags[slot] != 0xEEEEEEEEEEEEEEEEULL`. Ignore 0xEEEE (in-flight). On harvest, clear `tx_flags[slot] = 0`.

**Slot lifecycle**: Idle (rx=0, tx=0) -> Written (rx=ptr, tx=0) -> In-flight (rx=0, tx=0xEEEE) -> Completed (rx=0, tx=response) -> Consumer harvests, tx=0 -> Idle.

---

## 10. Dynamic Batch Handling for ONNX Models

When building a TensorRT engine from an ONNX model with dynamic batch dimensions (dim 0 <= 0), `ai_decoder_service.cu` automatically creates an optimization profile that pins all dynamic dimensions to 1. This enables building engines from models like `predecoder_memory_d13_T13_X.onnx` which use a symbolic `batch` dimension.

---

## 11. Shutdown and Grace Period

- **Grace period**: After the producer thread exits, the main thread waits up to 5 seconds for `total_completed >= total_submitted`.
- **Consumer exit**: The consumer thread normally exits when `producer_done && total_completed >= total_submitted`. To avoid hanging forever if some in-flight requests never complete, set a **consumer_stop** flag after the grace period; the consumer loop checks this and exits so `consumer.join()` returns and the process can print the final report and exit cleanly.
- **Dispatcher shutdown**: Set `shutdown_flag = 1` after the consumer exits, then join the dispatcher thread. The dispatcher synchronizes all worker streams before returning.
- **Debug diagnostics**: If requests are stuck after the grace period, a debug dump prints per-slot rx/tx flags, slot_request state, and per-worker inflight_slot_tags and idle_mask bits.

---

## 12. Performance Results (d=13, 30 µs rate, 10s)

Measured on Grace Blackwell (GB200) with `predecoder_memory_d13_T13_X.onnx` (FP16), 16 workers, 32 slots:

| Metric | Value |
| :--- | :--- |
| Throughput | 25,331 req/s |
| Mean latency | 122.0 µs |
| p50 latency | 119.3 µs |
| p99 latency | 135.3 µs |
| Per-round (/13) | 9.4 µs/round |
| Stage A (dispatch + GPU) | 109.9 µs |
| Stage B (PyMatching) | 11.8 µs |
| Stage C (consumer lag) | 0.3 µs |
| Raw TRT inference (trtexec) | 69.5 µs |

---

## 13. LLM Implementation Directives (Constraints Checklist)

When generating code from this specification, the LLM **MUST** strictly adhere to the following constraints:

- [ ] **NO CUDA STREAM QUERYING**: Do not use `cudaStreamQuery()` for backpressure or completion checking. It incurs severe driver latency. Rely strictly on `idle_mask` and `ready_flags`.
- [ ] **NO WEAK ORDERING BUGS**: Do not use `volatile`. Do not use `__threadfence_system()`. You must use `cuda::std::atomic<T, cuda::thread_scope_system>` (or `<cuda/atomic>` with `thread_scope_system`) for all cross-device synchronization.
- [ ] **NO HEAD OF LINE BLOCKING**: The host dispatcher MUST NOT statically map slots to predecoders. It must dynamically allocate via `idle_mask`. The consumer MUST harvest out-of-order by scanning all active slots.
- [ ] **NO DATA LOSS**: If `idle_mask == 0` (all workers busy), the dispatcher MUST spin on the current slot (`QEC_CPU_RELAX()`). It MUST NOT advance `current_slot` until a worker is allocated and the graph is launched.
- [ ] **NO RACE CONDITIONS ON TAGS**: `inflight_slot_tags` does not need to be atomic because index `[worker_id]` is exclusively owned by the active flow once the dispatcher clears the bit in `idle_mask`, until the worker thread restores the bit.
- [ ] **READY FLAG CLAIMING**: The CPU poller MUST claim each completion exactly once using compare_exchange_strong(1, 2) on the ready flag; use relaxed memory order on CAS failure. The worker MUST clear the flag (store 0) in `release_job`.
- [ ] **IN_FLIGHT SENTINEL**: After a successful `cudaGraphLaunch`, the dispatcher MUST write `tx_flags[current_slot] = 0xEEEEEEEEEEEEEEEEULL` before clearing `rx_flags[current_slot]`. Set `tx_data_host = nullptr` and `tx_data_dev = nullptr` to force the 0xEEEE path. The producer MUST wait for both rx and tx to be 0 before reusing a slot. The consumer MUST ignore 0xEEEE and only harvest real responses (or 0xDEAD errors).
- [ ] **CONSUMER MEMORY ORDERING**: The consumer MUST set `slot_request[s] = -1` BEFORE calling `cudaq_host_ringbuffer_clear_slot`, with a `__sync_synchronize()` fence between them, to prevent the producer-consumer race on ARM.
- [ ] **DMA DATA MOVEMENT**: Use `cudaMemcpyAsync` (DMA engine) for data copies. Input copy is issued via `pre_launch_fn` callback before graph launch. Output copy is captured inside the graph. Do not use SM-based byte-copy kernels for fixed-address transfers.
- [ ] **SHUTDOWN**: Use a `consumer_stop` (or equivalent) flag so the consumer thread can exit after a grace period even when `total_completed < total_submitted`; join the consumer after setting the flag so the process exits cleanly.
