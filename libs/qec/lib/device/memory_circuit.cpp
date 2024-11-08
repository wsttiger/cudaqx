/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "memory_circuit.h"

namespace cudaq::qec {

static std::unique_ptr<std::vector<uint8_t>> rawAncillaMeasurements;
static std::unique_ptr<std::vector<uint8_t>> rawDataMeasurements;

void persistDataMeasures(uint8_t *measures, std::size_t size) {
  if (!rawDataMeasurements)
    rawDataMeasurements = std::make_unique<std::vector<uint8_t>>();

  auto &store = *rawDataMeasurements;
  store.insert(store.end(), measures, measures + size);
}

void persistAncillaMeasures(uint8_t *measures, std::size_t size) {
  if (!rawAncillaMeasurements)
    rawAncillaMeasurements = std::make_unique<std::vector<uint8_t>>();

  auto &store = *rawAncillaMeasurements;
  store.insert(store.end(), measures, measures + size);
}

std::vector<uint8_t> &getMemoryCircuitAncillaMeasurements() {
  return *rawAncillaMeasurements.get();
}

std::vector<uint8_t> &getMemoryCircuitDataMeasurements() {
  return *rawDataMeasurements.get();
}

void clearRawMeasurements() {
  (*rawDataMeasurements).clear();
  (*rawAncillaMeasurements).clear();
}

__qpu__ void __memory_circuit_stabs(
    cudaq::qview<> data, cudaq::qview<> xstab_anc, cudaq::qview<> zstab_anc,
    const code::stabilizer_round &stabilizer_round,
    const code::one_qubit_encoding &statePrep, std::size_t numRounds,
    const std::vector<std::size_t> &x_stabilizers,
    const std::vector<std::size_t> &z_stabilizers) {
  // Create the logical patch
  patch logical(data, xstab_anc, zstab_anc);

  // Prepare the initial state fault tolerantly
  statePrep({data, xstab_anc, zstab_anc});

  // Generate syndrome data
  size_t counter = 0;
  std::vector<uint8_t> measureInts((xstab_anc.size() + zstab_anc.size()) *
                                   numRounds);
  for (std::size_t round = 0; round < numRounds; round++) {
    // Run the stabilizer round, generate the syndrome measurements
    auto syndrome = stabilizer_round(logical, x_stabilizers, z_stabilizers);

    // Convert to integers for easy passing
    for (size_t i = 0; i < syndrome.size(); i++) {
      measureInts[counter] = syndrome[i];
      counter++;
    }
  }

  // Store the ancillas for analysis / decoding
  persistAncillaMeasures(measureInts.data(), measureInts.size());
}

__qpu__ void memory_circuit_mz(const code::stabilizer_round &stabilizer_round,
                               const code::one_qubit_encoding &statePrep,
                               std::size_t numData, std::size_t numAncx,
                               std::size_t numAncz, std::size_t numRounds,
                               const std::vector<std::size_t> &x_stabilizers,
                               const std::vector<std::size_t> &z_stabilizers) {

  // Allocate the data and ancilla qubits
  cudaq::qvector data(numData), xstab_anc(numAncx), zstab_anc(numAncz);

  // Persists ancilla measures
  __memory_circuit_stabs(data, xstab_anc, zstab_anc, stabilizer_round,
                         statePrep, numRounds, x_stabilizers, z_stabilizers);

  auto dataResults = mz(data);
  std::vector<uint8_t> dataInts(numData);
  for (size_t i = 0; i < numData; i++)
    dataInts[i] = dataResults[i];

  persistDataMeasures(dataInts.data(), numData);
}

__qpu__ void memory_circuit_mx(const code::stabilizer_round &stabilizer_round,
                               const code::one_qubit_encoding &statePrep,
                               std::size_t numData, std::size_t numAncx,
                               std::size_t numAncz, std::size_t numRounds,
                               const std::vector<std::size_t> &x_stabilizers,
                               const std::vector<std::size_t> &z_stabilizers) {

  // Allocate the data and ancilla qubits
  cudaq::qvector data(numData), xstab_anc(numAncx), zstab_anc(numAncz);

  // Persists ancilla measures
  __memory_circuit_stabs(data, xstab_anc, zstab_anc, stabilizer_round,
                         statePrep, numRounds, x_stabilizers, z_stabilizers);

  h(data);
  auto dataResults = mz(data);
  std::vector<uint8_t> dataInts(numData);
  for (size_t i = 0; i < numData; i++)
    dataInts[i] = dataResults[i];

  persistDataMeasures(dataInts.data(), numData);
}

} // namespace cudaq::qec
