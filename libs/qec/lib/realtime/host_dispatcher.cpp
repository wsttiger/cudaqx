/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#include "cudaq/qec/realtime/host_dispatcher.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h"

#include <atomic>
#include <iostream>

namespace cudaq::qec {

void host_dispatcher_loop(const HostDispatcherConfig& config) {
    size_t current_slot = 0;
    const size_t num_slots = config.num_slots;
    const int num_entries = static_cast<int>(config.entries.size());
    uint64_t packets_dispatched = 0;

    while (!*config.shutdown_flag) {
        uint64_t rx_value = config.rx_flags_host[current_slot];

        if (rx_value == 0) {
            QEC_CPU_RELAX();
            continue;
        }

        std::atomic_thread_fence(std::memory_order_acquire);

        auto* data_host = reinterpret_cast<void*>(rx_value);
        auto* header = static_cast<const cudaq::nvqlink::RPCHeader*>(data_host);

        if (header->magic != cudaq::nvqlink::RPC_MAGIC_REQUEST) {
            config.rx_flags_host[current_slot] = 0;
            current_slot = (current_slot + 1) % num_slots;
            continue;
        }

        int entry_idx = -1;
        for (int i = 0; i < num_entries; ++i) {
            if (config.entries[i].function_id == header->function_id) {
                entry_idx = i;
                break;
            }
        }

        if (entry_idx < 0) {
            config.rx_flags_host[current_slot] = 0;
            current_slot = (current_slot + 1) % num_slots;
            continue;
        }

        const auto& entry = config.entries[entry_idx];

        // Backpressure: check if the predecoder stream is idle
        bool stream_busy = (cudaStreamQuery(entry.stream) != cudaSuccess);
        if (stream_busy) {
            current_slot = (current_slot + 1) % num_slots;
            continue;
        }

        // Backpressure: check if the predecoder queue is full
        volatile int* h_ready = entry.predecoder->get_host_ready_flags();
        volatile int* h_qidx = entry.predecoder->get_host_queue_idx();
        if (h_ready[*h_qidx] == 1) {
            current_slot = (current_slot + 1) % num_slots;
            continue;
        }

        // Translate host pointer to device pointer for the mailbox
        ptrdiff_t offset = (uint8_t*)data_host - config.rx_data_host;
        void* data_dev = (void*)(config.rx_data_dev + offset);
        config.h_mailbox_bank[entry.mailbox_idx] = data_dev;

        __sync_synchronize();

        cudaError_t err = cudaGraphLaunch(entry.graph_exec, entry.stream);
        if (err != cudaSuccess) {
            // Signal error via tx_flags (same protocol as device dispatcher)
            size_t slot_idx = ((uint8_t*)data_host - config.rx_data_host) / config.slot_size;
            uint64_t error_val = (uint64_t)0xDEAD << 48 | (uint64_t)err;
            config.tx_flags_host[slot_idx] = error_val;
        }

        config.rx_flags_host[current_slot] = 0;
        packets_dispatched++;
        current_slot = (current_slot + 1) % num_slots;
    }

    // Write stats
    if (config.stats_counter) {
        *config.stats_counter = packets_dispatched;
    }
}

} // namespace cudaq::qec
