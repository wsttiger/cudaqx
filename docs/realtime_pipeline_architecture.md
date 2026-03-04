# Realtime Pipeline Architecture

## 1. Component Overview

```mermaid
classDiagram
    class RealtimePipeline {
        -Impl* impl_
        +set_gpu_stage(GpuStageFactory)
        +set_cpu_stage(CpuStageCallback)
        +set_completion_handler(CompletionCallback)
        +start()
        +stop()
        +create_injector() RingBufferInjector
        +stats() Stats
    }

    class RingBufferInjector {
        -State* state_
        +try_submit(fid, payload, size, rid) bool
        +submit(fid, payload, size, rid)
        +backpressure_stalls() uint64_t
    }

    class RingBufferManager {
        -rx_flags_ : atomic_uint64[N]
        -tx_flags_ : atomic_uint64[N]
        -rx_data_host_ : uint8_t*
        +slot_available(slot) bool
        +write_and_signal(slot, fid, payload, len)
        +poll_tx(slot, err) cudaq_tx_status_t
        +clear_slot(slot)
    }

    class HostDispatcherConfig {
        +rx_flags : atomic_uint64*
        +tx_flags : atomic_uint64*
        +idle_mask : atomic_uint64*
        +inflight_slot_tags : int*
        +h_mailbox_bank : void**
        +workers : HostDispatchWorker[]
        +function_table : cudaq_function_entry_t*
        +shutdown_flag : atomic_int*
    }

    class AIPreDecoderService {
        -h_ready_flags_ : atomic_int*
        -h_predecoder_outputs_ : void*
        -graph_exec_ : cudaGraphExec_t
        +capture_graph(stream, device_launch)
        +poll_next_job(job) bool
        +release_job(slot)
    }

    RealtimePipeline *-- RingBufferManager : owns
    RealtimePipeline *-- HostDispatcherConfig : builds
    RealtimePipeline --> RingBufferInjector : creates
    RingBufferInjector --> RingBufferManager : writes to
    HostDispatcherConfig --> AIPreDecoderService : launches graph
```

## 2. Thread Model

The pipeline spawns three categories of threads, each pinnable to a specific CPU core:

```mermaid
flowchart LR
    subgraph "Producer (main thread or FPGA DMA)"
        P["RingBufferInjector::submit()"]
    end

    subgraph "Dispatcher Thread (core 2)"
        D["host_dispatcher_loop()"]
    end

    subgraph "Worker Threads (cores 4..4+N)"
        W0["worker_loop(0)"]
        W1["worker_loop(1)"]
        Wn["worker_loop(N-1)"]
    end

    subgraph "Consumer Thread (core 3)"
        C["consumer_loop()"]
    end

    subgraph "GPU Streams"
        G0["stream[0]: CUDA Graph"]
        G1["stream[1]: CUDA Graph"]
        Gn["stream[N-1]: CUDA Graph"]
    end

    P -->|"rx_flags[slot]"| D
    D -->|"cudaGraphLaunch"| G0
    D -->|"cudaGraphLaunch"| G1
    D -->|"cudaGraphLaunch"| Gn
    G0 -->|"ready_flags[0] = 1"| W0
    G1 -->|"ready_flags[0] = 1"| W1
    Gn -->|"ready_flags[0] = 1"| Wn
    W0 -->|"tx_flags[slot]"| C
    W1 -->|"tx_flags[slot]"| C
    Wn -->|"tx_flags[slot]"| C
    C -->|"clear_slot()"| P
```

## 3. Sequence Diagram: Single Syndrome Through the Pipeline

This traces one syndrome request from submission to completion, showing every
atomic operation and the thread/device boundary crossings.

```mermaid
sequenceDiagram
    participant Prod as Producer<br/>(main thread)
    participant RB as Ring Buffer<br/>(shared memory)
    participant Disp as Dispatcher<br/>(dedicated thread)
    participant GPU as GPU Stream[w]<br/>(CUDA Graph)
    participant Work as Worker Thread[w]<br/>(CPU)
    participant Cons as Consumer<br/>(dedicated thread)
    participant App as Application<br/>(completion handler)

    Note over Prod,App: === PHASE 1: Injection ===

    Prod->>Prod: CAS next_slot (acq_rel)<br/>claim slot S
    Prod->>RB: memcpy payload → rx_data[S]
    Prod->>RB: write RPCHeader {magic, function_id}
    Prod->>RB: rx_flags[S].store(host_ptr, release)
    Prod->>Prod: slot_occupied[S] = 1<br/>slot_request[S] = request_id
    Prod->>Prod: total_submitted.fetch_add(1, release)

    Note over Prod,App: === PHASE 2: Dispatch ===

    Disp->>RB: rx_flags[S].load(acquire)<br/>sees non-zero → slot S ready
    Disp->>Disp: parse RPCHeader → function_id
    Disp->>Disp: idle_mask.load(acquire)<br/>find worker W via __builtin_ffsll
    Disp->>Disp: idle_mask.fetch_and(~(1<<W), release)<br/>mark W busy
    Disp->>Disp: inflight_slot_tags[W] = S
    Disp->>RB: h_mailbox_bank[W] = dev_ptr
    Disp->>Disp: __sync_synchronize()

    opt pre_launch_fn configured
        Disp->>GPU: pre_launch_fn: cudaMemcpyAsync<br/>DMA syndrome → TRT input buffer
    end

    Disp->>GPU: cudaGraphLaunch(graph_exec[W], stream[W])
    Disp->>RB: tx_flags[S].store(0xEEEE..., release)<br/>IN_FLIGHT sentinel
    Disp->>RB: rx_flags[S].store(0, release)<br/>free rx slot, advance

    Note over Prod,App: === PHASE 3: GPU Inference ===

    GPU->>GPU: gateway_input_kernel:<br/>copy ring buffer → TRT input
    GPU->>GPU: TRT enqueueV3:<br/>AI predecoder inference
    GPU->>GPU: cudaMemcpyAsync:<br/>TRT output → h_predecoder_outputs
    GPU->>GPU: predecoder_signal_ready_kernel:<br/>ready_flags[0].store(1, release)

    Note over Prod,App: === PHASE 4: CPU Post-Processing ===

    Work->>Work: poll_next_job():<br/>ready_flags[0].CAS(1→2, acquire)
    Work->>Work: Read h_predecoder_outputs<br/>Run PyMatching MWPM decoder
    Work->>Work: Write RPC response to ring buffer slot
    Work->>Work: release_job():<br/>ready_flags[0].store(0, release)
    Work->>RB: tx_flags[S].store(slot_host_addr, release)<br/>marks READY
    Work->>Disp: idle_mask.fetch_or(1<<W, release)<br/>worker W free again

    Note over Prod,App: === PHASE 5: Completion ===

    Cons->>RB: poll_tx(S): tx_flags[S].load(acquire)<br/>sees valid host addr → READY
    Cons->>App: completion_handler({request_id, slot, success})
    Cons->>Cons: total_completed.fetch_add(1, relaxed)
    Cons->>Cons: slot_occupied[S] = 0
    Cons->>Cons: __sync_synchronize()
    Cons->>RB: clear_slot(S):<br/>rx_flags[S] = 0, tx_flags[S] = 0
    Note over Prod: Slot S now available<br/>for next submission
```

## 4. Atomic Variables Reference

Every atomic used in the pipeline, its scope, who writes it, who reads it,
and the memory ordering used.

### Ring Buffer Flags

| Atomic | Type | Scope | Writer(s) | Reader(s) | Ordering |
|--------|------|-------|-----------|-----------|----------|
| `rx_flags[slot]` | `cuda::atomic<uint64_t, system>` | Producer ↔ Dispatcher | Producer (signal), Dispatcher (clear), Consumer (clear) | Dispatcher (poll) | store: `release`, load: `acquire` |
| `tx_flags[slot]` | `cuda::atomic<uint64_t, system>` | Dispatcher ↔ Worker ↔ Consumer | Dispatcher (IN_FLIGHT), Worker (READY/addr) | Consumer (poll) | store: `release`, load: `acquire` |

### Worker Pool Scheduling

| Atomic | Type | Scope | Writer(s) | Reader(s) | Ordering |
|--------|------|-------|-----------|-----------|----------|
| `idle_mask` | `cuda::atomic<uint64_t, system>` | Dispatcher ↔ Workers | Dispatcher (clear bit), Worker (set bit) | Dispatcher (find free worker) | fetch_and/fetch_or: `release`, load: `acquire` |

### GPU ↔ CPU Handoff (per AIPreDecoderService)

| Atomic | Type | Scope | Writer(s) | Reader(s) | Ordering |
|--------|------|-------|-----------|-----------|----------|
| `ready_flags[0]` | `cuda::atomic<int, system>` | GPU kernel ↔ Worker thread | GPU kernel (0→1), Worker (CAS 1→2), Worker (2→0) | Worker (CAS poll) | store: `release`, CAS success: `acquire`, CAS fail: `relaxed` |

### Pipeline Lifecycle

| Atomic | Type | Scope | Writer(s) | Reader(s) | Ordering |
|--------|------|-------|-----------|-----------|----------|
| `shutdown_flag` | `cuda::atomic<int, system>` | Main ↔ Dispatcher | Main thread | Dispatcher loop | store: `release`, load: `acquire` |
| `producer_stop` | `std::atomic<bool>` | Main ↔ Consumer/Injector | Main thread | Consumer, Injector | store: `release`, load: `acquire` |
| `consumer_stop` | `std::atomic<bool>` | Main ↔ Consumer/Workers | Main thread | Consumer, Workers | store: `release`, load: `acquire` |
| `total_submitted` | `std::atomic<uint64_t>` | Injector ↔ Consumer | Injector | Consumer | fetch_add: `release`, load: `acquire` |
| `total_completed` | `std::atomic<uint64_t>` | Consumer ↔ Main | Consumer | Main (stats) | fetch_add: `relaxed`, load: `relaxed` |
| `backpressure_stalls` | `std::atomic<uint64_t>` | Injector ↔ Main | Injector | Main (stats) | fetch_add: `relaxed`, load: `relaxed` |
| `started` | `std::atomic<bool>` | Main thread | start()/stop() | destructor, start() | implicit seq_cst |

### Injector Slot Claiming

| Atomic | Type | Scope | Writer(s) | Reader(s) | Ordering |
|--------|------|-------|-----------|-----------|----------|
| `next_slot` | `std::atomic<uint32_t>` | Injector-internal | try_submit (CAS) | try_submit | CAS: `acq_rel` / `relaxed` |

## 5. Ring Buffer Slot State Machine

Each of the N ring buffer slots transitions through these states. The
transitions are driven by atomic flag writes from different threads.

```mermaid
stateDiagram-v2
    [*] --> FREE : initialization

    FREE --> RX_SIGNALED : Producer writes rx_flags[S] = host_ptr
    note right of RX_SIGNALED : rx_flags ≠ 0, tx_flags = 0\nPayload + RPCHeader in rx_data[S]

    RX_SIGNALED --> IN_FLIGHT : Dispatcher reads rx_flags,\nlaunches graph,\nwrites tx_flags = 0xEEEE...,\nclears rx_flags = 0
    note right of IN_FLIGHT : rx_flags = 0, tx_flags = 0xEEEE...\nGPU processing in progress

    IN_FLIGHT --> TX_READY : Worker writes tx_flags[S] = slot_host_addr\n(after GPU done + PyMatching done)
    note right of TX_READY : rx_flags = 0, tx_flags = valid addr\nResult available for consumer

    TX_READY --> FREE : Consumer reads result,\ncalls clear_slot():\nrx_flags = 0, tx_flags = 0

    IN_FLIGHT --> TX_ERROR : cudaGraphLaunch failed\ntx_flags = 0xDEAD... | err
    TX_ERROR --> FREE : Consumer reads error,\ncalls clear_slot()
```

**`tx_flags` value encoding:**

| Value | Meaning |
|-------|---------|
| `0` | Slot is free (no pending result) |
| `0xEEEEEEEEEEEEEEEE` | IN_FLIGHT — graph launched, result not yet ready |
| `0xDEAD____XXXXXXXX` | ERROR — upper 16 bits = `0xDEAD`, lower 32 = cudaError_t |
| Any other non-zero | READY — value is host pointer to slot data containing result |

## 6. CUDA Graph Structure (per Worker)

Each worker has a pre-captured CUDA graph that executes on its dedicated stream.
The graph is instantiated once at startup and replayed for every syndrome.

```mermaid
flowchart TD
    subgraph "CUDA Graph (AIPreDecoderService)"
        A["TRT enqueueV3\n(AI predecoder inference)"] --> B["cudaMemcpyAsync\nTRT output → h_predecoder_outputs\n(host-mapped)"]
        B --> C["predecoder_signal_ready_kernel\nready_flags[0].store(1, release)"]
    end

    subgraph "Pre-Launch Callback (host-side, before graph)"
        P["pre_launch_fn:\ncudaMemcpyAsync\nring buffer slot → TRT input\n(DMA copy engine)"]
    end

    subgraph "Post-Graph (Worker Thread)"
        D["poll_next_job():\nready_flags CAS 1→2"]
        E["PyMatching MWPM decode"]
        F["Write RPC response"]
        G["release_job():\nready_flags store 0"]
        H["tx_flags[S].store(addr, release)"]
        I["idle_mask.fetch_or(1<<W, release)"]
        D --> E --> F --> G --> H --> I
    end

    P --> A
    C -.->|"GPU signals\nready_flags = 1"| D
```

## 7. Backpressure and Flow Control

The pipeline uses implicit backpressure through slot availability:

```mermaid
flowchart TD
    subgraph "Flow Control"
        Submit["Injector::try_submit()"]
        Check{"slot_available(S)?\nrx_flags=0 AND tx_flags=0"}
        CAS{"CAS next_slot\ncur → cur+1"}
        Write["Write payload + signal"]
        Stall["backpressure_stalls++\nQEC_CPU_RELAX()"]
        Retry["Retry"]

        Submit --> Check
        Check -->|yes| CAS
        Check -->|no| Stall
        CAS -->|success| Write
        CAS -->|fail (contention)| Stall
        Stall --> Retry --> Submit
    end
```

**Capacity:** With `num_slots = 32` and `num_workers = 16`, up to 32 syndromes
can be in various stages of processing simultaneously. When all 32 slots are
occupied (either waiting for dispatch, in-flight on GPU, or awaiting consumer
pickup), the injector stalls until the consumer frees a slot.

## 8. ARM Memory Ordering Considerations

The pipeline runs on NVIDIA Grace (ARM aarch64) which has a weakly-ordered
memory model. Key ordering guarantees:

1. **Producer → Dispatcher:** `rx_flags[S].store(release)` pairs with
   `rx_flags[S].load(acquire)`. The dispatcher sees all payload bytes written
   before the flag.

2. **Dispatcher → Worker (via GPU):** The CUDA graph launch is ordered by
   `cudaGraphLaunch` semantics. The `ready_flags` store inside the GPU kernel
   uses `cuda::thread_scope_system` + `memory_order_release`, paired with the
   worker's `compare_exchange_strong(acquire)`.

3. **Worker → Consumer:** `tx_flags[S].store(release)` pairs with
   `tx_flags[S].load(acquire)` in `poll_tx_flag()`. Consumer sees PyMatching
   results before the ready flag.

4. **Consumer → Producer (slot recycling):** `slot_occupied[S] = 0` followed
   by `__sync_synchronize()` (full barrier) before `clear_slot()` ensures the
   producer cannot see a free slot while the consumer is still accessing
   slot_request metadata.

```mermaid
flowchart LR
    subgraph "Release/Acquire Pairs"
        A["rx_flags store\n(release)"] -->|"paired with"| B["rx_flags load\n(acquire)"]
        C["tx_flags store\n(release)"] -->|"paired with"| D["tx_flags load\n(acquire)"]
        E["ready_flags store(1)\n(release, system scope)"] -->|"paired with"| F["ready_flags CAS\n(acquire)"]
        G["idle_mask fetch_or\n(release)"] -->|"paired with"| H["idle_mask load\n(acquire)"]
    end

    subgraph "Full Barriers"
        I["__sync_synchronize()\nbetween slot_occupied=0\nand clear_slot()"]
        J["__sync_synchronize()\nbetween mailbox_bank write\nand cudaGraphLaunch"]
    end
```
