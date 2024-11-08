/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/qec/stabilizer_utils.h"

namespace cudaq::qec {
void sortStabilizerOps(std::vector<cudaq::spin_op> &ops) {
  // Sort the stabilizers, z first, then x
  std::sort(ops.begin(), ops.end(),
            [](const cudaq::spin_op &a, const cudaq::spin_op &b) {
              auto astr = a.to_string(false);
              auto bstr = b.to_string(false);
              auto zIdxA = astr.find_first_of("Z");
              auto zIdxB = bstr.find_first_of("Z");
              if (zIdxA == std::string::npos) {
                if (zIdxB != std::string::npos)
                  return false;

                // No Z in either, must both contain a X
                auto xIdxA = astr.find_first_of("X");
                return xIdxA < bstr.find_first_of("X");
              }

              // First contains a Z
              if (zIdxB == std::string::npos)
                return true;

              return zIdxA < zIdxB;
            });
}

// Need to push into the form
// H = [ H_Z | 0   ]
//     [ 0   | H_X ]
cudaqx::tensor<uint8_t>
to_parity_matrix(const std::vector<cudaq::spin_op> &stabilizers,
                 stabilizer_type type) {
  if (stabilizers.empty())
    return cudaqx::tensor<uint8_t>();

  sortStabilizerOps(const_cast<std::vector<cudaq::spin_op> &>(stabilizers));

  if (type == stabilizer_type::XZ) {
    auto numQubits = stabilizers[0].num_qubits();
    cudaqx::tensor<uint8_t> t({stabilizers.size(), 2 * numQubits});
    // Start by counting the number of Z spin_ops
    std::size_t numZRows = 0;
    for (auto &op : stabilizers)
      if (op.to_string(false).find("Z") != std::string::npos)
        numZRows++;
      else
        break;

    // Need to shift Z bits left
    for (std::size_t row = 0; row < numZRows; row++) {
      for (std::size_t i = numQubits; i < 2 * numQubits; i++) {
        if (stabilizers[row].get_raw_data().first[0][i])
          t.at({row, i - numQubits}) = 1;
      }
    }

    auto numXRows = stabilizers.size() - numZRows;

    for (std::size_t row = 0; row < numXRows; row++) {
      for (std::size_t i = 0; i < numQubits; i++) {
        if (stabilizers[numZRows + row].get_raw_data().first[0][i])
          t.at({numZRows + row, i + numQubits}) = 1;
      }
    }

    return t;
  }

  if (type == stabilizer_type::Z) {
    auto numQubits = stabilizers[0].num_qubits();
    // Start by counting the number of Z spin_ops
    std::size_t numZRows = 0;
    for (auto &op : stabilizers)
      if (op.to_string(false).find("Z") != std::string::npos)
        numZRows++;
      else
        break;

    if (numZRows == 0)
      return cudaqx::tensor<uint8_t>();

    cudaqx::tensor<uint8_t> ret({numZRows, numQubits});
    for (std::size_t row = 0; row < numZRows; row++) {
      for (std::size_t i = numQubits; i < 2 * numQubits; i++) {
        if (stabilizers[row].get_raw_data().first[0][i])
          ret.at({row, i - numQubits}) = 1;
      }
    }

    return ret;
  }

  auto numQubits = stabilizers[0].num_qubits();
  // Start by counting the number of Z spin_ops
  std::size_t numZRows = 0;
  for (auto &op : stabilizers)
    if (op.to_string(false).find("Z") != std::string::npos)
      numZRows++;
    else
      break;

  auto numXRows = stabilizers.size() - numZRows;

  if (numXRows == 0)
    return cudaqx::tensor<uint8_t>();

  cudaqx::tensor<uint8_t> ret({numXRows, numQubits});
  for (std::size_t row = 0; row < numXRows; row++) {
    for (std::size_t i = 0; i < numQubits; i++) {
      if (stabilizers[numZRows + row].get_raw_data().first[0][i])
        ret.at({row, i}) = 1;
    }
  }

  return ret;
}

cudaqx::tensor<uint8_t> to_parity_matrix(const std::vector<std::string> &words,
                                         stabilizer_type type) {

  std::vector<cudaq::spin_op> ops;
  for (auto &os : words)
    ops.emplace_back(cudaq::spin_op::from_word(os));
  return to_parity_matrix(ops, type);
}
} // namespace cudaq::qec
