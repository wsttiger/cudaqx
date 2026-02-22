# Host-Side Spin-Polling Dispatcher with Dynamic Worker Pool

## Design Specification

**Component**: `cudaq-qec` Realtime Decoding Subsystem
**Status**: Approved for Implementation
**Supersedes**: Device-side persistent kernel dispatcher (`dispatch_kernel_with_graph`) and Statically-mapped Host Dispatcher
**Target Platforms**: NVIDIA Grace Hopper (GH200), Grace Blackwell (GB200)
**Shared-Memory Model**: libcu++ `cuda::std::atomic` with `thread_scope_system`
**Last Updated**: 2026-02-21

---

## 1. System Context & Motivation

### 1.1 The Pipeline
The system performs real-time quantum error correction (QEC). An FPGA streams syndrome measurements into a host-device shared ring buffer continuously (~1 µs cadence). 
1. **Predecoding (GPU)**: TensorRT neural network inference (~9 µs).
2. **Global Decoding (CPU)**: PyMatching (MWPM) (~40-300 µs, highly variable).

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
3. **Dispatch**: The dispatcher launches the CUDA graph on the assigned worker's stream and clears its availability bit.
4. **Free**: When the CPU PyMatching worker finishes the job and writes the response to `tx_flags[origin_slot]`, it restores the worker's availability bit in the `idle_mask`.

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
| `mailbox_bank[NUM_WORKERS]` | `void*` (Plain array) | Mapped Pinned | Dispatcher writes device ptr for GPU input kernel. |

---

## 4. Host Dispatcher Thread (Producer)

The dispatcher loop is a tight spin-polling loop running on a dedicated CPU core.

### 4.1 Dispatcher Logic (Pseudocode)
```cpp
#include <cuda/std/atomic>

using atomic_uint64_sys = cuda::std::atomic<uint64_t, cuda::thread_scope_system>;
using atomic_int_sys    = cuda::std::atomic<int, cuda::thread_scope_system>;

void host_dispatcher_loop(DispatcherContext& ctx) {
    size_t current_slot = 0;
    
    while (ctx.shutdown_flag->load(cuda::std::memory_order_acquire) == 0) {
        // 1. Poll incoming ring buffer
        uint64_t rx_value = ctx.rx_flags[current_slot].load(cuda::std::memory_order_acquire);
        
        if (rx_value != 0) {
            // 2. Wait for an available worker in the pool (Spin if all busy)
            uint64_t mask = ctx.idle_mask->load(cuda::std::memory_order_acquire);
            if (mask == 0) {
                QEC_CPU_RELAX();
                continue; // Do NOT advance slot. Wait for worker.
            }

            // 3. Allocate worker
            int worker_id = __builtin_ffsll(mask) - 1; 
            
            // Mark worker as busy (atomic fetch_and with inverted bit)
            ctx.idle_mask->fetch_and(~(1ULL << worker_id), cuda::std::memory_order_release);

            // 4. Tag the payload with its origin slot for out-of-order return
            ctx.inflight_slot_tags[worker_id] = current_slot;

            // 5. Translate Host Ptr to Device Ptr for the GPU Mailbox
            void* data_host = reinterpret_cast<void*>(rx_value);
            ptrdiff_t offset = (uint8_t*)data_host - ctx.rx_data_host;
            void* data_dev = (void*)(ctx.rx_data_dev + offset);
            
            ctx.h_mailbox_bank[worker_id] = data_dev;
            __sync_synchronize(); // Full barrier to ensure mailbox write is visible

            // 6. Launch graph on the assigned worker's stream
            cudaError_t err = cudaGraphLaunch(ctx.workers[worker_id].graph_exec, ctx.workers[worker_id].stream);
            if (err != cudaSuccess) {
                uint64_t error_val = (uint64_t)0xDEAD << 48 | (uint64_t)err;
                ctx.tx_flags[current_slot].store(error_val, cuda::std::memory_order_release);
                ctx.idle_mask->fetch_or(1ULL << worker_id, cuda::std::memory_order_release);
            } else {
                // 6b. Mark slot IN_FLIGHT so producer does not reuse it while GPU/workers use it
                ctx.tx_flags[current_slot].store(0xEEEEEEEEEEEEEEEEULL, cuda::std::memory_order_release);
            }

            // 7. Consume slot and advance
            ctx.rx_flags[current_slot].store(0, cuda::std::memory_order_release);
            current_slot = (current_slot + 1) % ctx.num_slots;
            
        } else {
            QEC_CPU_RELAX(); // No data, spin on current slot
        }
    }
    // Cleanup: Synchronize all streams before exit to prevent illegal memory access
    for(auto& w : ctx.workers) cudaStreamSynchronize(w.stream);
}
```

---

## 5. GPU Kernel Modifications

The predecoder GPU kernels require minimal changes, as the dynamic pooling complexity is handled entirely by the host.

1. **Input Kernel**: Reads `*mailbox_slot_ptr` (mapped pinned) to get the device pointer to the ring buffer data. It copies this to `d_trt_input`. 
2. **Output Kernel**: Copies `d_trt_output` to `h_outputs[worker_id]` (mapped pinned). 
3. **Completion Signal**: The output kernel signals the CPU polling thread by setting the ready flag:
   ```cpp
   // Device code
   d_ready_flags[worker_id].store(1, cuda::std::memory_order_release);
   ```

*(Note: `cudaGraphInstantiateFlagDeviceLaunch` MUST be removed from graph capture. Use `cudaGraphInstantiate(&graph_exec, graph, 0)`).*

---

## 6. Worker Subsystem (Consumer)

A separate CPU polling thread scans the `ready_flags` array. When a GPU graph finishes, the job is handed to a CPU thread pool for PyMatching decoding.

### 6.1 Ready-Flag State Machine (Atomic Claiming)

With a single slot per predecoder (queue depth 1), the poller must **claim** each completion exactly once. If the poller only checks `ready_flags[i]==1` and enqueues without claiming, it will enqueue the same job repeatedly until the PyMatching worker calls `release_job`, flooding the thread pool and stalling the pipeline.

**States** (per-worker ready flag):

| Value | State      | Meaning |
| :---  | :---       | :---    |
| 0     | Idle       | Waiting for GPU, or worker has called `release_job`. |
| 1     | Ready      | GPU finished; output kernel stored 1. |
| 2     | Processing | CPU poller claimed the job; PyMatching is running. |

**Poller**: Use `compare_exchange_strong(expected=1, desired=2, memory_order_acquire, memory_order_relaxed)`. Only the thread that wins the CAS enqueues the job. Use **relaxed on failure** so spin-polling does not add barriers that delay seeing the GPU's store(1).

**Worker**: When PyMatching finishes, call `release_job(slot_idx)` which does `ready_flags[0].store(0, release)` so the slot is Idle for the next launch.

### 6.2 Worker Logic (Pseudocode)
```cpp
void pymatching_worker_task(WorkerContext& ctx, int worker_id) {
    // 1. Read GPU outputs from mapped pinned memory
    // ... run PyMatching MWPM ...
    
    // 2. Lookup origin slot for out-of-order routing
    int origin_slot = ctx.inflight_slot_tags[worker_id];

    // 3. Write response back to the EXACT slot the FPGA expects
    uint64_t response_val = format_response(...);
    ctx.tx_flags[origin_slot].store(response_val, cuda::std::memory_order_release);

    // 4. Acknowledge GPU read completion (Idle for next launch)
    ctx.ready_flags[worker_id].store(0, cuda::std::memory_order_release);  // 2 -> 0

    // 5. FREE THE WORKER: Return this worker back to the dispatcher pool
    ctx.idle_mask->fetch_or((1ULL << worker_id), cuda::std::memory_order_release);
}
```

---

## 7. Step-by-Step Data Flow Trace

1. **FPGA** writes INT32 measurements into `rx_data[5]`.
2. **FPGA** sets `rx_flags[5] = host_ptr`.
3. **Host Dispatcher** reads `rx_flags[5]`, sees data.
4. **Host Dispatcher** scans `idle_mask`, finds `worker_id = 2` is free.
5. **Host Dispatcher** marks bit 2 busy in `idle_mask`.
6. **Host Dispatcher** saves `inflight_slot_tags[2] = 5`.
7. **Host Dispatcher** translates `host_ptr` to `dev_ptr`, writes to `mailbox_bank[2]`.
8. **Host Dispatcher** calls `cudaGraphLaunch(..., stream[2])`.
9. **Host Dispatcher** sets `tx_flags[5] = 0xEEEE...` (IN_FLIGHT), then clears `rx_flags[5] = 0` and advances to `current_slot = 6`.
10. **GPU** executes graph on stream 2. Finishes and sets `ready_flags[2] = 1`.
11. **CPU Poller** CAS(1, 2) on `ready_flags[2]`, wins, enqueues job once; PyMatching runs on CPU.
12. **CPU Worker** finishes PyMatching.
13. **CPU Worker** looks up `origin_slot = inflight_slot_tags[2]` (which is 5).
14. **CPU Worker** writes response to `tx_flags[5]` (overwrites 0xEEEE), then `release_job`, then restores bit 2 in `idle_mask`.
15. **Consumer** (harvest thread) sees `tx_flags[5] != 0` and `!= 0xEEEE`, harvests, then clears `tx_flags[5] = 0`. Producer may now reuse slot 5.

---

## 8. Ring Buffer and IN_FLIGHT Sentinel

Because `cudaGraphLaunch` is asynchronous, the dispatcher clears `rx_flags[slot]` immediately after launch. Without a hold, the **producer** (FPGA sim or test) would see `rx_flags[slot]==0` and `tx_flags[slot]==0` (response not written yet) and reuse the slot, overwriting data while the GPU is still reading.

**Fix: IN_FLIGHT tag**

1. **Dispatcher**: On successful launch, write `tx_flags[current_slot].store(0xEEEEEEEEEEEEEEEEULL, release)` **before** clearing `rx_flags[current_slot]`. On launch failure, write the 0xDEAD|err value and restore the worker bit; do not write 0xEEEE.
2. **Producer**: Reuse a slot only when **both** `rx_flags[slot]==0` **and** `tx_flags[slot]==0`. Thus the producer blocks until the consumer has harvested (tx cleared).
3. **Consumer**: When harvesting, treat only real responses: `tx_flags[slot] != 0` **and** `tx_flags[slot] != 0xEEEEEEEEEEEEEEEEULL`. Ignore 0xEEEE (in-flight). On harvest, clear `tx_flags[slot] = 0`.

**Slot lifecycle**: Idle (rx=0, tx=0) → Written (rx=ptr, tx=0) → In-flight (rx=0, tx=0xEEEE) → Completed (rx=0, tx=response) → Consumer harvests, tx=0 → Idle.

---

## 9. Shutdown and Grace Period

- **Grace period**: After the producer thread exits, the main thread may wait up to a bounded time (e.g. 10 s) for `total_completed >= total_submitted`.
- **Consumer exit**: The consumer thread normally exits when `producer_done && total_completed >= total_submitted`. To avoid hanging forever if some in-flight requests never complete, set a **consumer_stop** flag after the grace period; the consumer loop checks this and exits so `consumer.join()` returns and the process can print the final report and exit cleanly.
- **Diagnostic threads**: A progress reporter (submitted/completed every second) and a watchdog (stall detection every 2 s) are **optional** and should be **disabled by default** (e.g. `kEnableProgressReporter = false`, `kEnableWatchdog = false`). Enable them only when debugging stalls; otherwise they can block shutdown (e.g. watchdog not seeing `producer_done`).

---

## 10. LLM Implementation Directives (Constraints Checklist)

When generating code from this specification, the LLM **MUST** strictly adhere to the following constraints:

- [ ] **NO CUDA STREAM QUERYING**: Do not use `cudaStreamQuery()` for backpressure or completion checking. It incurs severe driver latency. Rely strictly on `idle_mask` and `ready_flags`.
- [ ] **NO WEAK ORDERING BUGS**: Do not use `volatile`. Do not use `__threadfence_system()`. You must use `cuda::std::atomic<T, cuda::thread_scope_system>` (or `<cuda/atomic>` with `thread_scope_system`) for all cross-device synchronization.
- [ ] **NO HEAD OF LINE BLOCKING**: The host dispatcher MUST NOT statically map slots to predecoders. It must dynamically allocate via `idle_mask`.
- [ ] **NO DATA LOSS**: If `idle_mask == 0` (all workers busy), the dispatcher MUST spin on the current slot (`QEC_CPU_RELAX()`). It MUST NOT advance `current_slot` until a worker is allocated and the graph is launched.
- [ ] **NO RACE CONDITIONS ON TAGS**: `inflight_slot_tags` does not need to be atomic because index `[worker_id]` is exclusively owned by the active flow once the dispatcher clears the bit in `idle_mask`, until the worker thread restores the bit.
- [ ] **READY FLAG CLAIMING**: The CPU poller MUST claim each completion exactly once using compare_exchange_strong(1, 2) on the ready flag; use relaxed memory order on CAS failure. The worker MUST clear the flag (store 0) in `release_job`.
- [ ] **IN_FLIGHT SENTINEL**: After a successful `cudaGraphLaunch`, the dispatcher MUST write `tx_flags[current_slot] = 0xEEEEEEEEEEEEEEEEULL` before clearing `rx_flags[current_slot]`. The producer MUST wait for both rx and tx to be 0 before reusing a slot. The consumer MUST ignore 0xEEEE and only harvest real responses (or 0xDEAD errors).
- [ ] **SHUTDOWN**: Use a `consumer_stop` (or equivalent) flag so the consumer thread can exit after a grace period even when `total_completed < total_submitted`; join the consumer after setting the flag so the process exits cleanly.
- [ ] **DIAGNOSTIC THREADS**: Progress reporter and watchdog threads MUST be optional and disabled by default so they do not block normal shutdown.
