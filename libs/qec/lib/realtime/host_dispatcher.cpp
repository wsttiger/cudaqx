/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#include "cudaq/qec/realtime/host_dispatcher.h"

#include <iostream>
#include <nvtx3/nvToolsExt.h>

namespace cudaq::qec {

void host_dispatcher_loop(const HostDispatcherConfig& config) {
    size_t current_slot = 0;
    const size_t num_slots = config.num_slots;
    const int num_workers = static_cast<int>(config.workers.size());
    uint64_t packets_dispatched = 0;

    nvtxRangePushA("Dispatcher Loop");

    while (config.shutdown_flag->load(cuda::std::memory_order_acquire) == 0) {
        uint64_t rx_value = config.rx_flags[current_slot].load(cuda::std::memory_order_acquire);

        if (rx_value != 0) {
            nvtxRangePushA("Process Slot");
            
            uint64_t mask = config.idle_mask->load(cuda::std::memory_order_acquire);
            if (mask == 0) {
                nvtxRangePushA("Wait Worker");
                QEC_CPU_RELAX();
                nvtxRangePop(); // Wait Worker
                nvtxRangePop(); // Process Slot
                continue;
            }

            int worker_id = __builtin_ffsll(static_cast<long long>(mask)) - 1;
            config.idle_mask->fetch_and(~(1ULL << worker_id), cuda::std::memory_order_release);

            config.inflight_slot_tags[worker_id] = static_cast<int>(current_slot);

            void* data_host = reinterpret_cast<void*>(rx_value);
            ptrdiff_t offset = static_cast<uint8_t*>(data_host) - config.rx_data_host;
            void* data_dev = static_cast<void*>(config.rx_data_dev + offset);

            config.h_mailbox_bank[worker_id] = data_dev;
            __sync_synchronize();

            if (config.debug_dispatch_ts) {
                config.debug_dispatch_ts[current_slot] = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            }

            nvtxRangePushA("Launch Graph");
            cudaError_t err = cudaGraphLaunch(config.workers[worker_id].graph_exec,
                                             config.workers[worker_id].stream);
            nvtxRangePop(); // Launch Graph

            if (err != cudaSuccess) {
                uint64_t error_val = (uint64_t)0xDEAD << 48 | (uint64_t)err;
                config.tx_flags[current_slot].store(error_val, cuda::std::memory_order_release);
                config.idle_mask->fetch_or(1ULL << worker_id, cuda::std::memory_order_release);
            } else {
                // Mark slot IN_FLIGHT so producer doesn't overwrite while GPU/workers use it
                config.tx_flags[current_slot].store(0xEEEEEEEEEEEEEEEEULL, cuda::std::memory_order_release);
            }

            config.rx_flags[current_slot].store(0, cuda::std::memory_order_release);
            packets_dispatched++;
            if (config.live_dispatched)
                config.live_dispatched->fetch_add(1, cuda::std::memory_order_relaxed);
            current_slot = (current_slot + 1) % num_slots;
            
            nvtxRangePop(); // Process Slot
        } else {
            QEC_CPU_RELAX();
        }
    }
    
    nvtxRangePop(); // Dispatcher Loop

    for (const auto& w : config.workers) {
        cudaStreamSynchronize(w.stream);
    }

    if (config.stats_counter) {
        *config.stats_counter = packets_dispatched;
    }
}

} // namespace cudaq::qec
