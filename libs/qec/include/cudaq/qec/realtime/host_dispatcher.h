/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#pragma once

#include "cudaq/qec/realtime/ai_predecoder_service.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cudaq::qec {

struct HostDispatchEntry {
    uint32_t function_id;
    cudaGraphExec_t graph_exec;
    int mailbox_idx;
    AIPreDecoderService* predecoder;
    cudaStream_t stream;
};

struct HostDispatcherConfig {
    volatile uint64_t* rx_flags_host;
    volatile uint64_t* tx_flags_host;
    uint8_t* rx_data_host;
    uint8_t* rx_data_dev;
    void** h_mailbox_bank;
    size_t num_slots;
    size_t slot_size;
    std::vector<HostDispatchEntry> entries;
    volatile int* shutdown_flag;
    uint64_t* stats_counter;
};

/// Run the host-side dispatcher loop. Blocks until *config.shutdown_flag
/// becomes non-zero. Call from a dedicated thread.
void host_dispatcher_loop(const HostDispatcherConfig& config);

} // namespace cudaq::qec
