# Hybrid AI Predecoder + PyMatching Global Decoder Pipeline

## Design Document

**Component**: `cudaq-qec` Realtime Decoding Subsystem  
**Status**: Implementation Complete (Test-Validated)  
**Last Updated**: 2026-02-19

---

## Table of Contents

1. [Overview](#1-overview)
2. [Problem Statement](#2-problem-statement)
3. [Architecture](#3-architecture)
4. [Component Deep-Dive](#4-component-deep-dive)
   - 4.1 [Ring Buffer & RPC Protocol](#41-ring-buffer--rpc-protocol)
   - 4.2 [GPU Persistent Dispatcher Kernel](#42-gpu-persistent-dispatcher-kernel)
   - 4.3 [AIDecoderService (Base Class)](#43-aidecoderservice-base-class)
   - 4.4 [AIPreDecoderService (Predecoder + CPU Handoff)](#44-aipredeccoderservice-predecoder--cpu-handoff)
   - 4.5 [CPU Worker Threads & PyMatching Decoder Pool](#45-cpu-worker-threads--pymatching-decoder-pool)
5. [Data Flow](#5-data-flow)
6. [Memory Architecture](#6-memory-architecture)
7. [Backpressure Protocol](#7-backpressure-protocol)
8. [Memory Ordering & Synchronization](#8-memory-ordering--synchronization)
9. [CUDA Graph Hierarchy](#9-cuda-graph-hierarchy)
10. [Pipeline Configurations](#10-pipeline-configurations)
11. [File Inventory](#11-file-inventory)
12. [Configuration Parameters](#12-configuration-parameters)
13. [Performance Benchmarking](#13-performance-benchmarking)
14. [Portability](#14-portability)
15. [Limitations & Future Work](#15-limitations--future-work)

---

## 1. Overview

This system implements a **realtime hybrid GPU/CPU pipeline** for quantum error correction (QEC) decoding on the surface code. The pipeline splits the decoding workload into two stages:

| Stage | Location | Algorithm | Data Type |
|-------|----------|-----------|-----------|
| **Predecoding** | GPU | Neural network (TensorRT, from ONNX) | INT32 |
| **Global Decoding** | CPU | PyMatching (MWPM) | float64 |

A **persistent GPU kernel** (the Dispatcher) monitors a shared ring buffer for incoming syndrome data. When data arrives, the Dispatcher launches a CUDA Graph containing a TensorRT inference pass. The neural network accepts raw measurements as INT32 tensors and produces residual detectors and a logical frame. The residual detectors are handed off to the CPU via mapped pinned memory, where a thread pool runs PyMatching MWPM decoding. Results are written back to the ring buffer and acknowledged.

The system supports multiple surface code distances via a configurable `PipelineConfig` struct: d=7, d=13, d=21, and d=31. ONNX models are compiled to TensorRT engines on first use and cached to disk as `.engine` files for fast reloading on subsequent runs.

---

## 2. Problem Statement

Surface code QEC requires decoding syndrome measurements within the coherence time of the quantum system (typically ~1 microsecond for superconducting qubits). A pure CPU decoder cannot meet this budget at scale. A pure GPU decoder lacks the flexibility to run algorithms like Minimum Weight Perfect Matching (MWPM) efficiently on GPU.

The hybrid approach exploits the strengths of each:

- **GPU**: Massively parallel neural network inference provides fast soft-decision outputs (residual detectors) that reduce the problem complexity for the global decoder.
- **CPU**: PyMatching solves the residual MWPM problem on the simplified output from the predecoder.

The critical constraint is **zero-copy, zero-allocation** on the hot path. Every buffer is pre-allocated, every kernel is pre-captured into a CUDA Graph, and every transfer uses mapped pinned memory.

---

## 3. Architecture

### System Diagram

```
 FPGA / Quantum Control (or Test Harness)
       │
       │  syndrome data (INT32 measurements)
       ▼
 ┌─────────────────────────────────────────────────────┐
 │           Ring Buffer (Mapped Pinned Memory)         │
 │  ┌──────┐ ┌──────┐ ┌──────┐        ┌──────┐       │
 │  │Slot 0│ │Slot 1│ │Slot 2│  ...   │Slot63│       │
 │  └──┬───┘ └──┬───┘ └──┬───┘        └──┬───┘       │
 │     │        │        │               │            │
 │  rx_flags[0] rx_flags[1] ...   rx_flags[63]        │
 └─────┼────────┼────────┼───────────────┼────────────┘
       │        │        │               │
       ▼        ▼        ▼               ▼
 ┌─────────────────────────────────────────────────────┐
 │          GPU Persistent Dispatcher Kernel            │
 │                                                     │
 │   Polls rx_flags[] ──► Looks up function_id         │
 │   ──► Checks backpressure ──► Launches CUDA Graph   │
 └──────────┬──────────┬──────────┬──────────┬─────────┘
            │          │          │          │
            ▼          ▼          ▼          ▼
 ┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
 │ PreDecoder 0 │ │PreDec. 1 │ │PreDec. 2 │ │PreDec. 3 │
 │ (CUDA Graph) │ │(CUDAGraph│ │(CUDAGraph│ │(CUDAGraph│
 │              │ │          │ │          │ │          │
 │  Input Kern  │ │          │ │          │ │          │
 │  ──► TRT ──► │ │   ...    │ │   ...    │ │   ...    │
 │  Output Kern │ │          │ │          │ │          │
 └──────┬───────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
        │              │            │            │
        │   (mapped pinned memory: ready_flags, outputs)
        ▼              ▼            ▼            ▼
 ┌─────────────────────────────────────────────────────┐
 │  Polling Thread (incoming_polling_loop)              │
 │  Round-robins all predecoders, dispatches to pool   │
 └──────────┬──────────────────────────────────────────┘
            │
            ▼
 ┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
 │  Worker 0    │ │ Worker 1 │ │ Worker 2 │ │ Worker 3 │
 │ (thread pool)│ │(thd pool)│ │(thd pool)│ │(thd pool)│
 │              │ │          │ │          │ │          │
 │ PyMatching 0 │ │PyMatch 1 │ │PyMatch 2 │ │PyMatch 3 │
 │ (own decoder)│ │(own dec) │ │(own dec) │ │(own dec) │
 │ Write RPC    │ │Write RPC │ │Write RPC │ │Write RPC │
 │ Set tx_flag  │ │Set tx_flg│ │Set tx_flg│ │Set tx_flg│
 └──────┬───────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
        │              │            │            │
        └──────────────┼────────────┼────────────┘
                       ▼
              tx_flags[slot] ──► FPGA
```

### Key Design Decisions

1. **CUDA Graphs everywhere** -- Both the dispatcher kernel and every predecoder instance are captured as CUDA Graphs. The dispatcher graph is instantiated with `cudaGraphInstantiateFlagDeviceLaunch`, enabling it to launch child predecoder graphs from device code via `cudaGraphLaunch(..., cudaStreamGraphFireAndForget)`.

2. **Mapped pinned memory for all CPU-GPU communication** -- `cudaHostAllocMapped` provides a single address space visible to both CPU and GPU without explicit copies. GPU writes are made visible via `__threadfence_system()`; CPU reads are ordered via `std::atomic_thread_fence(std::memory_order_acquire)`.

3. **N-deep circular queue between GPU and CPU** -- Rather than a single handoff slot, each predecoder maintains a circular buffer of depth N (default 16), allowing the GPU to pipeline multiple inferences before the CPU consumes them.

4. **Dispatcher-level backpressure** -- The dispatcher checks a predecoder's queue state *before* launching its graph. If the queue is full, the packet stays in the ring buffer and the dispatcher moves on to service other slots.

5. **ONNX model support with engine caching** -- The `AIDecoderService` accepts either a pre-built `.engine` file or an `.onnx` model. When given an ONNX file, it builds a TensorRT engine at runtime and optionally saves it to disk via the `engine_save_path` parameter. On subsequent runs, the cached `.engine` file is loaded directly, skipping the expensive autotuner phase (startup drops from ~15s to ~4s).

6. **Per-worker PyMatching decoder pool** -- Each thread pool worker gets its own pre-allocated PyMatching decoder instance via `thread_local` assignment. This eliminates mutex contention on the decode path (previous single-decoder + mutex design was ~2.4x slower).

7. **Type-agnostic I/O buffers** -- All TRT I/O buffers use `void*` rather than `float*`, supporting INT32 models natively without type casting on the GPU.

---

## 4. Component Deep-Dive

### 4.1 Ring Buffer & RPC Protocol

**Files**: `dispatch_kernel_launch.h` (protocol), test harness (allocation)

The ring buffer is the communication channel between the FPGA (or test harness) and the GPU. It consists of:

| Buffer | Type | Size | Purpose |
|--------|------|------|---------|
| `rx_flags[N]` | `volatile uint64_t*` | N slots | Non-zero = data ready; value is pointer to slot data |
| `tx_flags[N]` | `volatile uint64_t*` | N slots | Non-zero = response ready; acknowledges to FPGA |
| `rx_data` | `uint8_t*` | N x SLOT_SIZE | Slot payload area |

Each slot carries an **RPC message** in a packed wire format:

```
Request:  [RPCHeader: magic(4) | function_id(4) | arg_len(4)] [payload: arg_len bytes]
Response: [RPCResponse: magic(4) | status(4) | result_len(4)] [payload: result_len bytes]
```

The `function_id` is an FNV-1a hash of the target function name, enabling the dispatcher to route requests to different predecoder instances.

The response payload for the PyMatching pipeline is a packed `DecodeResponse`:

```c
struct __attribute__((packed)) DecodeResponse {
    int32_t total_corrections;
    int32_t converged;
};
```

### 4.2 GPU Persistent Dispatcher Kernel

**File**: `realtime/lib/daemon/dispatcher/dispatch_kernel.cu`

The dispatcher is a **persistent kernel** -- it runs for the lifetime of the system, spinning on the ring buffer. Two variants exist:

| Variant | Function | Graph Launch | Use Case |
|---------|----------|-------------|----------|
| `dispatch_kernel_device_call_only` | Direct device function calls | No | Legacy / simple RPC |
| `dispatch_kernel_with_graph` | Device function calls + CUDA Graph launch | Yes (sm_80+) | AI predecoder pipeline |

#### Dispatch Loop (Graph Variant)

```
while (!shutdown):
    rx_value = rx_flags[current_slot]
    if rx_value != 0:
        header = parse_rpc_header(rx_value)

        if header.magic is invalid:
            consume and clear slot           ← garbage data

        else:
            entry = lookup(header.function_id)

            if entry is DEVICE_CALL:
                call device function inline
                write RPC response
                set tx_flags
                consume slot

            elif entry is GRAPH_LAUNCH:
                if backpressure_check(entry):
                    skip (do NOT consume)     ← retry later
                else:
                    write mailbox
                    cudaGraphLaunch(fire-and-forget)
                    consume slot
                    (tx_flags set later by CPU)

            else:
                consume slot                  ← unknown function

        advance current_slot                  ← always advance
    KernelType::sync()
```

The `packet_consumed` flag controls whether `rx_flags[slot]` is cleared. For backpressured graph launches, the slot is left intact so the dispatcher retries on the next pass. The slot pointer **always** advances to avoid head-of-line blocking.

**Note on slot scanning**: The dispatcher only advances `current_slot` when a non-empty slot is found. When a slot is empty, it spins on that same slot. This means having many empty slots (e.g., 64 slots with only 4 in use) does not cause scanning overhead, but the dispatcher does park on a slot waiting for it to be filled.

#### Function Table Entry

Each registered function is described by a `cudaq_function_entry_t`:

```c
typedef struct {
    union {
        void *device_fn_ptr;           // DEVICE_CALL handler
        cudaGraphExec_t graph_exec;    // GRAPH_LAUNCH handler
    } handler;
    uint32_t function_id;              // FNV-1a hash
    uint8_t dispatch_mode;             // DEVICE_CALL or GRAPH_LAUNCH
    uint8_t reserved[3];
    cudaq_handler_schema_t schema;     // argument/result type descriptors

    // Graph-launch backpressure metadata:
    uint32_t mailbox_idx;              // index into global_mailbox_bank
    int *d_queue_idx;                  // → predecoder's queue tail
    volatile int *d_ready_flags;       // → predecoder's ready flags
    int *d_inflight_flag;              // → predecoder's inflight flag
} cudaq_function_entry_t;
```

#### Graph-Based Dispatch Context

The dispatcher kernel itself runs inside a CUDA Graph (`cudaq_dispatch_graph_context`), instantiated with `cudaGraphInstantiateFlagDeviceLaunch`. This is **required** for the kernel to call `cudaGraphLaunch()` from device code. The lifecycle is:

```
cudaq_create_dispatch_graph_regular()
    → cudaGraphCreate
    → cudaGraphAddKernelNode (dispatch_kernel_with_graph)
    → cudaGraphInstantiate (with DeviceLaunch flag)
    → cudaGraphUpload
    → cudaStreamSynchronize

cudaq_launch_dispatch_graph()
    → cudaGraphLaunch (from host)

cudaq_destroy_dispatch_graph()
    → cudaGraphExecDestroy + cudaGraphDestroy
```

### 4.3 AIDecoderService (Base Class)

**Files**: `ai_decoder_service.h`, `ai_decoder_service.cu`

The base class manages the TensorRT lifecycle and provides a default "autonomous" CUDA Graph that reads from a mailbox, runs inference, and writes results back to the ring buffer -- all on the GPU.

#### Constructor

```cpp
AIDecoderService(const std::string& model_path, void** device_mailbox_slot,
                 const std::string& engine_save_path = "");
```

The constructor accepts either a `.engine` file (fast deserialization) or an `.onnx` file (builds TRT engine via autotuner). When `engine_save_path` is non-empty and the model is ONNX, the built engine is serialized to disk for caching.

#### Responsibilities

- **Engine loading**: Deserializes a TensorRT `.engine` file or builds from `.onnx` via `NvOnnxParser`.
- **Engine caching**: Saves built engines to disk via `engine_save_path` for fast reload.
- **Dynamic tensor binding**: Enumerates all I/O tensors from the engine, storing metadata in `TensorBinding` structs. Supports models with multiple outputs (e.g., `residual_detectors` + `logical_frame`).
- **Buffer allocation**: Allocates persistent device buffers sized to the engine's static tensor shapes. Uses `void*` for type-agnostic I/O (INT32, FP32, etc.).
- **Graph capture**: The default `capture_graph()` creates a 3-node graph:

```
gateway_input_kernel ──► TRT enqueueV3 ──► gateway_output_kernel
```

#### Dynamic Tensor Binding

```cpp
struct TensorBinding {
    std::string name;
    void* d_buffer = nullptr;
    size_t size_bytes = 0;
    bool is_input = false;
};
std::vector<TensorBinding> all_bindings_;
```

During `setup_bindings()`, all I/O tensors are enumerated from the engine. The first input becomes `d_trt_input_`, the first output becomes `d_trt_output_` (the primary output forwarded to the CPU), and any additional outputs are allocated as auxiliary buffers in `d_aux_buffers_`.

### 4.4 AIPreDecoderService (Predecoder + CPU Handoff)

**Files**: `ai_predecoder_service.h`, `ai_predecoder_service.cu`

This derived class replaces the base class's autonomous graph with one that hands inference results off to the CPU for further processing by PyMatching.

#### Constructor

```cpp
AIPreDecoderService(const std::string& engine_path, void** device_mailbox_slot,
                    int queue_depth = 16, const std::string& engine_save_path = "");
```

#### CUDA Graph Structure

```
predecoder_input_kernel ──► TRT enqueueV3 ──► predecoder_output_kernel
```

**`predecoder_input_kernel`**:
1. Reads the current queue tail index (`d_queue_idx`).
2. Performs a defense-in-depth bounded spin on `d_ready_flags[slot]` (primary backpressure is at the dispatcher level).
3. If the slot is free: saves the FPGA ring buffer pointer to `d_ring_ptrs[slot]` and copies syndrome data to the TRT input buffer.
4. If the spin times out: sets `ring_ptr = nullptr`, causing all threads to abort safely without corrupting the queue.

**`predecoder_output_kernel`**:
1. Copies TRT output to `d_outputs[slot]` (mapped pinned memory, directly readable by CPU). Output data is `void*` (typically INT32 residual detectors).
2. Issues `__threadfence_system()` to ensure writes are visible over PCIe.
3. Sets `d_ready_flags[slot] = 1` (signals the CPU).
4. Advances `d_queue_idx` circularly.

#### N-Deep Circular Queue

Each `AIPreDecoderService` instance owns a private circular queue:

```
         GPU writes →                    ← CPU reads
    ┌───┬───┬───┬───┬───┬───┬───┬───┐
    │ 0 │ 1 │ 2 │ 3 │ 4 │...│14 │15 │   ready_flags[16]
    └───┴───┴───┴───┴───┴───┴───┴───┘
      ▲                           ▲
      │                           │
  d_queue_idx                cpu_poll_idx_
  (GPU tail)                 (CPU head)
```

| Buffer | Host Pointer | Device Pointer | Purpose |
|--------|-------------|---------------|---------|
| `h_ready_flags_` | CPU reads | `d_ready_flags_` GPU writes | 1 = job ready, 0 = slot free |
| `h_ring_ptrs_` | CPU reads | `d_ring_ptrs_` GPU writes | Original FPGA buffer address per job |
| `h_outputs_` | CPU reads | `d_outputs_` GPU writes | TRT inference output (`void*`, typically INT32) |

All three buffers are allocated with `cudaHostAllocMapped` and mapped to device pointers via `cudaHostGetDevicePointer`. The GPU writes through the device pointers; the CPU reads through the host pointers. No explicit `cudaMemcpy` is ever issued on the hot path.

#### CPU Interface

```cpp
bool poll_next_job(PreDecoderJob& out_job);
void release_job(int slot_idx);
```

`poll_next_job` checks `h_ready_flags_[cpu_poll_idx_]`. If set, it issues an acquire fence (for ARM portability), populates the `PreDecoderJob` struct with the slot index, ring buffer pointer, and a pointer into the inference output buffer, then advances the poll index.

`release_job` uses `__atomic_store_n(..., __ATOMIC_RELEASE)` to clear the flag, ensuring that all prior CPU writes (RPC response data) are visible before the GPU is allowed to reuse the slot.

### 4.5 CPU Worker Threads & PyMatching Decoder Pool

**File**: `test_realtime_predecoder_w_pymatching.cpp`

The CPU-side processing uses a **polling thread + thread pool** architecture:

1. **Polling thread** (`incoming_polling_loop`): A single dedicated thread round-robins all predecoder instances, calling `poll_next_job()` on each. When a job is found, it is dispatched to the thread pool.
2. **Thread pool** (`cudaq::qec::utils::ThreadPool`): A pool of `num_workers` threads (default 4) that execute `pymatching_worker_task` jobs concurrently.

#### PyMatching Decoder Pool

Each worker thread gets its own pre-allocated PyMatching decoder via `thread_local` assignment:

```cpp
struct DecoderContext {
    std::vector<std::unique_ptr<cudaq::qec::decoder>> decoders;
    std::atomic<int> next_decoder_idx{0};
    int z_stabilizers = 0;
    int spatial_slices = 0;

    cudaq::qec::decoder* acquire_decoder() {
        thread_local int my_idx = next_decoder_idx.fetch_add(1);
        return decoders[my_idx % decoders.size()].get();
    }
};
```

Decoders are constructed at startup from the surface code's Z parity check matrix (`H_z`) using the `cudaq-qec` plugin system:

```cpp
auto surface_code = cudaq::qec::get_code("surface_code", {{"distance", d}});
auto H_z = surface_code->get_parity_z();
for (int i = 0; i < num_workers; ++i)
    decoders.push_back(cudaq::qec::decoder::get("pymatching", H_z, pm_params));
```

The `merge_strategy` parameter is set to `"smallest_weight"` to handle parallel edges in the surface code's PCM.

#### Worker Function (`pymatching_worker_task`)

Each worker invocation:

1. **Acquires a decoder** from the pool via `ctx->acquire_decoder()` (lock-free, `thread_local`).
2. **Slices residual detectors** into `spatial_slices` groups of `z_stabilizers` each. For d=13, this is 26 slices of 84 stabilizers.
3. **Runs PyMatching** on each slice: converts INT32 residual detectors to `std::vector<double>`, calls `decoder->decode(syndrome)`.
4. **Accumulates corrections** and convergence status across all slices.
5. **Writes RPC Response**: Formats `DecodeResponse{total_corrections, converged}` into the original ring buffer slot.
6. **Releases GPU Queue Slot**: Calls `predecoder->release_job(slot_idx)`.
7. **Acknowledges to FPGA**: Sets `tx_flags[slot]` to signal completion.

#### Timing Instrumentation

The worker function includes `std::chrono::high_resolution_clock` instrumentation that accumulates PyMatching decode time and total worker time into atomic counters on the `DecoderContext`. These are reported at the end of the run as a latency breakdown.

---

## 5. Data Flow

The following traces a single syndrome packet through the entire pipeline:

```
Step  Location    Action
────  ────────    ──────────────────────────────────────────────────
 1.   Host/FPGA   Writes RPCHeader + INT32 measurements into rx_data[slot]
 2.   Host/FPGA   Sets rx_flags[slot] = pointer_to_slot_data
                  ── memory fence ──
 3.   GPU         Dispatcher reads rx_flags[slot] (non-zero → work available)
 4.   GPU         Parses RPCHeader, extracts function_id
 5.   GPU         Looks up function_id in function_table → finds entry
 6.   GPU         Backpressure check: reads entry->d_queue_idx, entry->d_ready_flags
                  If queue full: skip, leave rx_flags intact, advance slot pointer
 7.   GPU         Writes data_buffer pointer to global_mailbox_bank[mailbox_idx]
                  ── __threadfence_system() ──
 8.   GPU         cudaGraphLaunch(entry->graph_exec, cudaStreamGraphFireAndForget)
 9.   GPU         Clears rx_flags[slot] = 0, increments packet count
                  ── slot consumed ──

                  ── Inside Predecoder CUDA Graph ──
10.   GPU         predecoder_input_kernel:
                    Reads *mailbox → ring_buffer_ptr
                    Reads d_queue_idx → slot_idx
                    Copies INT32 measurements from ring buffer → d_trt_input_
11.   GPU         TRT enqueueV3: Neural network inference (INT32 → INT32)
12.   GPU         predecoder_output_kernel:
                    Copies d_trt_output_ → h_outputs_[slot_idx] (mapped pinned)
                    Saves ring_buffer_ptr → h_ring_ptrs_[slot_idx]
                    ── __threadfence_system() ──
                    Sets h_ready_flags_[slot_idx] = 1
                    Advances d_queue_idx
                  ── Graph complete ──

13.   CPU         Polling thread: h_ready_flags_[poll_idx] == 1 → job found
                  Dispatches job to thread pool
14.   CPU         Worker thread: acquires per-thread PyMatching decoder
                  ── std::atomic_thread_fence(acquire) ──
15.   CPU         Worker reads h_ring_ptrs_[poll_idx], h_outputs_[poll_idx]
16.   CPU         Worker slices residual_detectors into spatial_slices
                  For each slice: PyMatching MWPM decode → corrections
17.   CPU         Worker writes RPCResponse + DecodeResponse into ring_buffer_ptr
                  ── std::atomic_thread_fence(release) ──
18.   CPU         Worker: release_job(slot_idx) → clears h_ready_flags_[slot_idx]
                  ── __atomic_store_n(..., RELEASE) ──
19.   CPU         Worker: Sets tx_flags[ring_slot] = rx_value
                  ── FPGA/Host sees response ──
```

---

## 6. Memory Architecture

### Allocation Map

```
┌─────────────────────────────────────────────────────────────┐
│                    PINNED MAPPED MEMORY                      │
│               (cudaHostAllocMapped + cudaHostGetDevicePointer)│
│                                                             │
│  Ring Buffer:                                               │
│    rx_flags[64]          ← Host writes, GPU reads/clears    │
│    tx_flags[64]          ← CPU writes, Host reads           │
│    rx_data[64 x SLOT_SIZE] ← Host writes, GPU reads,       │
│                               CPU reads/writes              │
│                                                             │
│  Per-PreDecoder (x4):                                       │
│    h_ready_flags_[16]  ← GPU writes 1, CPU reads, CPU clears│
│    h_ring_ptrs_[16]    ← GPU writes, CPU reads              │
│    h_outputs_[16xN]    ← GPU writes (void*), CPU reads      │
│                                                             │
│  Control:                                                   │
│    shutdown_flag       ← CPU writes, GPU reads              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     DEVICE MEMORY                            │
│                                                             │
│  d_global_mailbox_bank[4]  ← Dispatcher writes, Graph reads │
│  d_function_entries[4]     ← Host copies at init, GPU reads │
│  d_stats                   ← GPU increments, Host reads     │
│                                                             │
│  Per-PreDecoder (x4):                                       │
│    d_trt_input_   (void*)  ← Input kernel writes, TRT reads │
│    d_trt_output_  (void*)  ← TRT writes, Output kernel reads│
│    d_aux_buffers_ (void*)  ← Additional TRT I/O (e.g.       │
│                               logical_frame)                 │
│    d_queue_idx_            ← GPU reads/writes (queue tail)  │
│    d_inflight_flag_        ← Dispatcher checks backpressure │
└─────────────────────────────────────────────────────────────┘
```

### Why Mapped Pinned Memory?

Traditional `cudaMemcpyAsync` requires the GPU to issue a DMA transfer on a stream, which introduces stream synchronization overhead. Mapped pinned memory (`cudaHostAllocMapped`) gives the GPU a device-accessible pointer to host memory. GPU writes travel over PCIe and become visible to the CPU (on x86, immediately; on ARM, after appropriate fencing). This eliminates all explicit copy calls from the hot path.

---

## 7. Backpressure Protocol

Backpressure prevents the GPU from overwhelming the CPU when PyMatching workers fall behind. It operates at **two levels**:

### Level 1: Dispatcher (Primary)

Before launching a predecoder graph, the dispatcher reads the predecoder's queue state directly from the function table entry:

```c
int* d_queue_idx = entry->d_queue_idx;
volatile int* d_ready_flags = entry->d_ready_flags;

int current_tail = *d_queue_idx;
if (d_ready_flags[current_tail] == 1) {
    // Queue full: skip this packet, do NOT clear rx_flags
    packet_consumed = false;
}
```

If the queue is full, the packet stays in the ring buffer. The dispatcher advances to the next slot, so **other decoders are not blocked** (no head-of-line blocking). On the next pass through the ring buffer, the dispatcher will retry the skipped slot.

### Level 2: Predecoder Input Kernel (Defense-in-Depth)

If the dispatcher's backpressure check is bypassed (e.g., backpressure pointers not wired up, or a race condition), the predecoder input kernel has a **bounded spin** as a safety net:

```c
int timeout_counter = 0;
while (d_ready_flags[slot_idx] == 1 && timeout_counter < 1000000) {
    timeout_counter++;
}

if (d_ready_flags[slot_idx] == 1) {
    ring_ptr = nullptr;  // Abort safely, don't corrupt the slot
}
```

On timeout, the kernel nullifies `ring_ptr`, which causes all threads to return without writing any data. This prevents silent corruption but means the syndrome is effectively dropped. In a correctly configured system, this path should never be reached.

---

## 8. Memory Ordering & Synchronization

The pipeline involves three independent agents (FPGA/Host, GPU, CPU) communicating through shared memory. Correctness depends on careful ordering:

### GPU → CPU (Predecoder Output → Poll)

| Agent | Operation | Ordering Primitive |
|-------|-----------|-------------------|
| GPU | Write `h_outputs_[slot]` and `h_ring_ptrs_[slot]` | (normal device writes to mapped memory) |
| GPU | `__threadfence_system()` | Ensures all prior writes are visible over PCIe |
| GPU | Write `h_ready_flags_[slot] = 1` | (the "publish" signal) |
| CPU | Read `h_ready_flags_[slot] == 1` | (volatile read) |
| CPU | `std::atomic_thread_fence(acquire)` | Prevents CPU from speculatively reading data before the flag |
| CPU | Read `h_outputs_[slot]`, `h_ring_ptrs_[slot]` | (safe: ordered after acquire) |

On x86, the acquire fence is technically a no-op (loads are not reordered with loads), but it is necessary for correctness on ARM (e.g., Grace Hopper).

### CPU → GPU (Job Release → Queue Reuse)

| Agent | Operation | Ordering Primitive |
|-------|-----------|-------------------|
| CPU | Write RPC response to ring buffer | (normal stores) |
| CPU | `__atomic_store_n(&h_ready_flags_[slot], 0, __ATOMIC_RELEASE)` | Ensures response writes are visible before flag is cleared |
| GPU | Read `d_ready_flags[slot] == 0` | (volatile read from mapped memory) |
| GPU | Overwrites `d_ring_ptrs[slot]`, `d_outputs[slot]` | (safe: flag was 0) |

### Host → GPU (Ring Buffer Signaling)

| Agent | Operation | Ordering Primitive |
|-------|-----------|-------------------|
| Host/Test | Write RPC header + payload to `rx_data[slot]` | (normal stores) |
| Host/Test | `__sync_synchronize()` / memory barrier | Full fence before flag write |
| Host/Test | Write `rx_flags[slot] = pointer` | (the "publish" signal) |
| GPU | Read `rx_flags[slot] != 0` | (volatile read from mapped memory) |

---

## 9. CUDA Graph Hierarchy

The system uses a **two-level graph hierarchy**:

```
Level 0: Dispatcher Graph (cudaq_dispatch_graph_context)
    │
    │  Instantiated with cudaGraphInstantiateFlagDeviceLaunch
    │  Contains: dispatch_kernel_with_graph (persistent kernel node)
    │
    │  Device-side cudaGraphLaunch() ──►
    │
    ├──► Level 1: PreDecoder Graph [0]
    │       predecoder_input_kernel → TRT enqueueV3 → predecoder_output_kernel
    │
    ├──► Level 1: PreDecoder Graph [1]
    │       ...
    ├──► Level 1: PreDecoder Graph [2]
    │       ...
    └──► Level 1: PreDecoder Graph [3]
            ...
```

**Level 0** must be instantiated with `cudaGraphInstantiateFlagDeviceLaunch` so that the persistent kernel running inside it can call `cudaGraphLaunch()` on **Level 1** graphs. Level 1 graphs are also instantiated with this flag and uploaded to the device. The launch mode is `cudaStreamGraphFireAndForget`, meaning the predecoder graph executes asynchronously without blocking the dispatcher.

**Requirement**: Compute capability >= sm_80 (Ampere and later). Device-side graph launch is gated by `#if __CUDA_ARCH__ >= 800`.

**Limitation**: `cudaStreamGraphFireAndForget` has a CUDA runtime limit on concurrent pending child graph launches (~128). The test limits `total_requests` to 100 to stay under this ceiling.

---

## 10. Pipeline Configurations

The test supports multiple surface code distances via the `PipelineConfig` struct:

| Config | Distance | Rounds | ONNX Model | Input Shape | Input Bytes | Residual Detectors | Z Stabilizers | Spatial Slices | Slot Size |
|--------|----------|--------|------------|-------------|-------------|-------------------|---------------|---------------|-----------|
| `d7` | 7 | 7 | `model1_d7_r7_unified_Z_batch1.onnx` | [1, 72, 7] | 2,016 | 336 | 24 | 14 | 4,096 |
| `d13` | 13 | 13 | `model1_d13_r13_unified_Z_batch1.onnx` | [1, 252, 13] | 13,104 | 2,184 | 84 | 26 | 16,384 |
| `d21` | 21 | 21 | `model1_d21_r21_unified_X_batch1.onnx` | [1, 660, 21] | 55,440 | 9,240 | 220 | 42 | 65,536 |
| `d31` | 31 | 31 | `model1_d31_r31_unified_Z_batch1.onnx` | [1, 1440, 31] | 178,560 | 29,760 | 480 | 62 | 262,144 |

All models use **INT32** tensors for both input (measurements) and output (residual detectors, logical frame).

The number of **spatial slices** is `residual_detectors / z_stabilizers`. PyMatching is called once per slice, with each slice containing one group of Z-stabilizer detector values.

Usage:

```bash
./test_realtime_predecoder_w_pymatching d7    # default
./test_realtime_predecoder_w_pymatching d13
./test_realtime_predecoder_w_pymatching d21
./test_realtime_predecoder_w_pymatching d31
```

### Engine Caching

On first run with a given configuration, the ONNX model is compiled to a TensorRT engine and saved alongside the ONNX file (e.g., `model1_d13_r13_unified_Z_batch1.engine`). Subsequent runs detect the cached engine and skip the build phase.

---

## 11. File Inventory

| File | Layer | Purpose |
|------|-------|---------|
| `realtime/include/.../cudaq_realtime.h` | API | C API header: structs, enums, function declarations |
| `realtime/include/.../dispatch_kernel_launch.h` | API | RPC protocol structs (RPCHeader, RPCResponse), FNV-1a hash |
| `realtime/lib/.../dispatch_kernel.cu` | Runtime | Persistent dispatcher kernels + graph-based dispatch context |
| `libs/qec/include/.../ai_decoder_service.h` | QEC | Base class header: TRT lifecycle, dynamic tensor bindings, engine caching |
| `libs/qec/lib/.../ai_decoder_service.cu` | QEC | Base class impl: ONNX build, engine save/load, gateway kernels, graph capture |
| `libs/qec/include/.../ai_predecoder_service.h` | QEC | Derived class header: CPU handoff queue, `QEC_CPU_RELAX` macro |
| `libs/qec/lib/.../ai_predecoder_service.cu` | QEC | Derived class impl: predecoder kernels, circular queue, poll/release |
| `libs/qec/include/.../utils/thread_pool.h` | Util | Thread pool with optional core pinning |
| `libs/qec/include/.../utils/pipeline_benchmarks.h` | Util | Reusable latency/throughput benchmarking utility |
| `libs/qec/lib/.../test_realtime_predecoder_w_pymatching.cpp` | Test | End-to-end integration test with real ONNX + PyMatching |

---

## 12. Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_SLOTS` | 64 | Ring buffer slot count (Host ↔ GPU) |
| `slot_size` | Per-config (4096 - 262144) | Max payload per slot (RPCHeader + measurements + result) |
| `num_predecoders` | 4 | Parallel predecoder instances (TRT engines) |
| `queue_depth` | 16 | N-deep circular queue per predecoder |
| `num_workers` | 4 | Thread pool size (each gets its own PyMatching decoder) |
| `total_requests` | 100 | Requests per test run (limited by CUDA graph launch ceiling) |
| Dispatcher grid | 1 block, 32 threads | Persistent kernel configuration |
| Predecoder grid | 1 block, 128 threads | Per-graph kernel configuration |
| Spin timeout | 1,000,000 iterations | Defense-in-depth backpressure in input kernel |

### Capacity Analysis

- **Total GPU→CPU queue capacity**: 4 predecoders x 16 depth = 64 slots
- **Ring buffer capacity**: 64 slots
- These are balanced: worst case, all 64 ring buffer requests could be in-flight across the predecoder queues simultaneously.
- If requests are unevenly distributed (e.g., 32 to one predecoder), that predecoder's queue fills at depth 16, and the dispatcher applies backpressure for the remaining 16.
- **Batched submission**: The test fires requests in batches of `num_predecoders` (4), waiting for each batch to complete before submitting the next. This avoids overwhelming the dispatcher and stays within CUDA graph launch limits.

---

## 13. Performance Benchmarking

### PipelineBenchmark Utility

The `PipelineBenchmark` class (`libs/qec/include/cudaq/qec/utils/pipeline_benchmarks.h`) provides reusable latency and throughput measurement for any pipeline test:

```cpp
cudaq::qec::utils::PipelineBenchmark bench("d13_r13_Z", total_requests);
bench.start();
// ... submit requests, mark_submit(i), mark_complete(i) ...
bench.stop();
bench.report();
```

It tracks per-request submit and complete timestamps, computes statistics only on completed requests, and reports:

- Min, max, mean, p50, p90, p95, p99 latencies (microseconds)
- Standard deviation
- Total wall time and throughput (req/s)
- Submitted / completed / timed-out counts

### Worker Timing Breakdown

The test also reports an average breakdown of where time is spent:

```
  Worker Timing Breakdown (avg over 100 requests):
    PyMatching decode:      164.3 us  (23.6%)
    Worker overhead:           0.4 us  ( 0.1%)
    GPU+dispatch+poll:       530.1 us  (76.3%)
    Total end-to-end:        694.8 us
    Per-round (/13):         53.4 us/round
```

### Measured Performance (representative, system-dependent)

| Config | p50 Latency | Mean Latency | Throughput | PyMatching % | Per-round |
|--------|-------------|-------------|------------|-------------|-----------|
| d=7 | 262 us | 284 us | 10,803 req/s | 12.8% | 40.6 us |
| d=13 | 658 us | 678 us | 3,467 req/s | 23.0% | 52.1 us |

### Profiling with Nsight Systems

```bash
nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
  -o d13_profile ./test_realtime_predecoder_w_pymatching d13
nsys stats d13_profile.nsys-rep
```

Key findings from profiling:
- GPU TRT inference is ~9 us/request (very fast)
- The dominant latency is in the dispatcher's slot-scanning loop and CPU polling gap
- PyMatching decode accounts for 13-23% of end-to-end latency depending on distance
- The `--cuda-graph-trace=node` flag is critical for seeing individual kernels inside CUDA graphs

---

## 14. Portability

### Architecture Support

| Feature | x86_64 | aarch64 (Grace Hopper) |
|---------|--------|----------------------|
| `QEC_CPU_RELAX()` | `_mm_pause()` | `asm volatile("yield")` |
| Acquire fence in `poll_next_job` | No-op (TSO) | Required (`std::atomic_thread_fence`) |
| Release store in `release_job` | `__atomic_store_n` | `__atomic_store_n` |
| `volatile` for mapped memory | Sufficient | Requires fences (provided) |

The `QEC_CPU_RELAX()` macro is defined in `ai_predecoder_service.h` and should be used by all polling code instead of platform-specific intrinsics.

### CUDA Compute Capability

| Feature | Minimum |
|---------|---------|
| Device-side `cudaGraphLaunch` | sm_80 (Ampere) |
| `__threadfence_system()` | sm_20+ |
| Mapped pinned memory | All CUDA devices |

---

## 15. Limitations & Future Work

1. **Linear function table lookup**: `dispatch_lookup_entry` performs a linear scan of the function table. With 4 entries this is negligible, but for larger tables a hash map or sorted binary search would be appropriate.

2. **No queue drain on shutdown**: Setting `system_stop = true` causes the worker threads to exit immediately. Jobs that the GPU has completed but the CPU hasn't polled are silently dropped. Production code should drain all queues before stopping.

3. **Dropped syndromes on timeout**: If the defense-in-depth spin timeout fires in `predecoder_input_kernel`, the syndrome is silently dropped. A production system should increment an error counter or signal the host.

4. **Static TRT shapes only**: The current implementation assumes static input/output tensor shapes. Dynamic shapes would require per-invocation shape metadata in the RPC payload and runtime TRT profile switching.

5. **Batched submission**: The test fires requests in batches of `num_predecoders` and waits for completion before the next batch. This serializes batches and underutilizes the pipeline. A pipelined submission strategy (overlapping batch N+1 submission with batch N completion) would improve throughput.

6. **Single polling thread**: The `incoming_polling_loop` is a single thread that round-robins all predecoders. At higher predecoder counts, this could become a bottleneck. A per-predecoder polling thread or lock-free MPSC queue could help.

7. **CUDA graph launch ceiling**: `cudaStreamGraphFireAndForget` has a runtime limit of ~128 concurrent pending child graph launches. The test limits `total_requests` to 100 to stay under this. Production systems with sustained high throughput may need to throttle submissions or use a different dispatch strategy.

8. **Dispatcher scanning latency**: The persistent dispatcher kernel parks on the current slot and spins until it is populated. With batched submission, there is a round-trip delay between batch completion and next-batch submission that dominates the end-to-end latency (~550 us of the ~700 us total for d=13).
