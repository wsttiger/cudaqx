/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under     *
 * the terms of the Apache License 2.0 which accompanies this distribution.     *
 ******************************************************************************/

#include "cudaq/solvers/operators/block_encoding_kernels.h"


namespace cudaq::solvers::block_encoding {

__qpu__ void prepare(cudaq::qview<> ancilla,
                     const std::vector<double> &state_prep_angles) {
  if (ancilla.size() == 0)
    return;

  ry(state_prep_angles[0], ancilla[0]);

  int angle_idx = 1;
  for (std::size_t layer = 1; layer < ancilla.size(); ++layer) {
    int num_branches = 1 << static_cast<int>(layer);

    for (int i = 0; i < num_branches; ++i) {
      for (int bit = 0; bit < static_cast<int>(layer); ++bit) {
        if (!((i >> bit) & 1))
          x(ancilla[layer - 1 - bit]);
      }

      if (layer == 1) {
        ry<cudaq::ctrl>(state_prep_angles[angle_idx], ancilla[0], ancilla[1]);
      } else if (layer == 2) {
        ry<cudaq::ctrl>(state_prep_angles[angle_idx], ancilla[0], ancilla[1],
                        ancilla[2]);
      } else if (layer == 3) {
        ry<cudaq::ctrl>(state_prep_angles[angle_idx], ancilla[0], ancilla[1],
                        ancilla[2], ancilla[3]);
      } else if (layer == 4) {
        ry<cudaq::ctrl>(state_prep_angles[angle_idx], ancilla[0], ancilla[1],
                        ancilla[2], ancilla[3], ancilla[4]);
      } else if (layer == 5) {
        ry<cudaq::ctrl>(state_prep_angles[angle_idx], ancilla[0], ancilla[1],
                        ancilla[2], ancilla[3], ancilla[4], ancilla[5]);
      } else if (layer == 6) {
        ry<cudaq::ctrl>(state_prep_angles[angle_idx], ancilla[0], ancilla[1],
                        ancilla[2], ancilla[3], ancilla[4], ancilla[5],
                        ancilla[6]);
      } else if (layer == 7) {
        ry<cudaq::ctrl>(state_prep_angles[angle_idx], ancilla[0], ancilla[1],
                        ancilla[2], ancilla[3], ancilla[4], ancilla[5],
                        ancilla[6], ancilla[7]);
      } else if (layer == 8) {
        ry<cudaq::ctrl>(state_prep_angles[angle_idx], ancilla[0], ancilla[1],
                        ancilla[2], ancilla[3], ancilla[4], ancilla[5],
                        ancilla[6], ancilla[7], ancilla[8]);
      } else if (layer == 9) {
        ry<cudaq::ctrl>(state_prep_angles[angle_idx], ancilla[0], ancilla[1],
                        ancilla[2], ancilla[3], ancilla[4], ancilla[5],
                        ancilla[6], ancilla[7], ancilla[8], ancilla[9]);
      } else if (layer == 10) {
        ry<cudaq::ctrl>(state_prep_angles[angle_idx], ancilla[0], ancilla[1],
                        ancilla[2], ancilla[3], ancilla[4], ancilla[5],
                        ancilla[6], ancilla[7], ancilla[8], ancilla[9],
                        ancilla[10]);
      }
      angle_idx++;

      for (int bit = 0; bit < static_cast<int>(layer); ++bit) {
        if (!((i >> bit) & 1))
          x(ancilla[layer - 1 - bit]);
      }
    }
  }
}

__qpu__ void unprepare(cudaq::qview<> ancilla,
                       const std::vector<double> &state_prep_angles) {
  if (ancilla.size() == 0)
    return;

  int n_ancilla = static_cast<int>(ancilla.size());
  int angle_idx = static_cast<int>(state_prep_angles.size()) - 1;

  for (int layer = n_ancilla - 1; layer >= 1; --layer) {
    int num_branches = 1 << layer;

    for (int i = num_branches - 1; i >= 0; --i) {
      for (int bit = 0; bit < layer; ++bit) {
        if (!((i >> bit) & 1))
          x(ancilla[layer - 1 - bit]);
      }

      if (layer == 1) {
        ry<cudaq::ctrl>(-state_prep_angles[angle_idx], ancilla[0], ancilla[1]);
      } else if (layer == 2) {
        ry<cudaq::ctrl>(-state_prep_angles[angle_idx], ancilla[0], ancilla[1],
                        ancilla[2]);
      } else if (layer == 3) {
        ry<cudaq::ctrl>(-state_prep_angles[angle_idx], ancilla[0], ancilla[1],
                        ancilla[2], ancilla[3]);
      } else if (layer == 4) {
        ry<cudaq::ctrl>(-state_prep_angles[angle_idx], ancilla[0], ancilla[1],
                        ancilla[2], ancilla[3], ancilla[4]);
      } else if (layer == 5) {
        ry<cudaq::ctrl>(-state_prep_angles[angle_idx], ancilla[0], ancilla[1],
                        ancilla[2], ancilla[3], ancilla[4], ancilla[5]);
      } else if (layer == 6) {
        ry<cudaq::ctrl>(-state_prep_angles[angle_idx], ancilla[0], ancilla[1],
                        ancilla[2], ancilla[3], ancilla[4], ancilla[5],
                        ancilla[6]);
      } else if (layer == 7) {
        ry<cudaq::ctrl>(-state_prep_angles[angle_idx], ancilla[0], ancilla[1],
                        ancilla[2], ancilla[3], ancilla[4], ancilla[5],
                        ancilla[6], ancilla[7]);
      } else if (layer == 8) {
        ry<cudaq::ctrl>(-state_prep_angles[angle_idx], ancilla[0], ancilla[1],
                        ancilla[2], ancilla[3], ancilla[4], ancilla[5],
                        ancilla[6], ancilla[7], ancilla[8]);
      } else if (layer == 9) {
        ry<cudaq::ctrl>(-state_prep_angles[angle_idx], ancilla[0], ancilla[1],
                        ancilla[2], ancilla[3], ancilla[4], ancilla[5],
                        ancilla[6], ancilla[7], ancilla[8], ancilla[9]);
      } else if (layer == 10) {
        ry<cudaq::ctrl>(-state_prep_angles[angle_idx], ancilla[0], ancilla[1],
                        ancilla[2], ancilla[3], ancilla[4], ancilla[5],
                        ancilla[6], ancilla[7], ancilla[8], ancilla[9],
                        ancilla[10]);
      }
      angle_idx--;

      for (int bit = 0; bit < layer; ++bit) {
        if (!((i >> bit) & 1))
          x(ancilla[layer - 1 - bit]);
      }
    }
  }

  ry(-state_prep_angles[0], ancilla[0]);
}

__qpu__ void select(cudaq::qview<> ancilla, cudaq::qview<> system,
                    const std::vector<int> &term_controls,
                    const std::vector<int> &term_ops,
                    const std::vector<int> &term_lengths,
                    const std::vector<int> &term_signs) {
  int ptr_ctrl = 0;
  int ptr_op = 0;
  int n_ancilla = ancilla.size();

  for (std::size_t i = 0; i < term_lengths.size(); ++i) {
    int n_ops = term_lengths[i];
    int sign = term_signs[i];

    for (int b = 0; b < n_ancilla; ++b) {
      int bit_val = term_controls[ptr_ctrl++];
      if (bit_val == 0)
        x(ancilla[b]);
    }

    for (int k = 0; k < n_ops; ++k) {
      int code = term_ops[ptr_op++];
      int q_idx = term_ops[ptr_op++];

      if (code == 1)
        x<cudaq::ctrl>(ancilla, system[q_idx]);
      else if (code == 2)
        y<cudaq::ctrl>(ancilla, system[q_idx]);
      else if (code == 3)
        z<cudaq::ctrl>(ancilla, system[q_idx]);
    }

    if (sign < 0) {
      if (n_ancilla == 1) {
        z(ancilla[0]);
      } else if (n_ancilla == 2) {
        z<cudaq::ctrl>(ancilla[0], ancilla[1]);
      } else if (n_ancilla == 3) {
        z<cudaq::ctrl>(ancilla[0], ancilla[1], ancilla[2]);
      } else if (n_ancilla == 4) {
        z<cudaq::ctrl>(ancilla[0], ancilla[1], ancilla[2], ancilla[3]);
      } else if (n_ancilla == 5) {
        z<cudaq::ctrl>(ancilla[0], ancilla[1], ancilla[2], ancilla[3],
                       ancilla[4]);
      } else if (n_ancilla == 6) {
        z<cudaq::ctrl>(ancilla[0], ancilla[1], ancilla[2], ancilla[3],
                       ancilla[4], ancilla[5]);
      } else if (n_ancilla == 7) {
        z<cudaq::ctrl>(ancilla[0], ancilla[1], ancilla[2], ancilla[3],
                       ancilla[4], ancilla[5], ancilla[6]);
      } else if (n_ancilla == 8) {
        z<cudaq::ctrl>(ancilla[0], ancilla[1], ancilla[2], ancilla[3],
                       ancilla[4], ancilla[5], ancilla[6], ancilla[7]);
      } else if (n_ancilla == 9) {
        z<cudaq::ctrl>(ancilla[0], ancilla[1], ancilla[2], ancilla[3],
                       ancilla[4], ancilla[5], ancilla[6], ancilla[7],
                       ancilla[8]);
      } else if (n_ancilla == 10) {
        z<cudaq::ctrl>(ancilla[0], ancilla[1], ancilla[2], ancilla[3],
                       ancilla[4], ancilla[5], ancilla[6], ancilla[7],
                       ancilla[8], ancilla[9]);
      }
    }

    int back_ptr = ptr_ctrl - 1;
    for (int b_rev = 0; b_rev < n_ancilla; ++b_rev) {
      int anc_idx = (n_ancilla - 1) - b_rev;
      int bit_val = term_controls[back_ptr--];
      if (bit_val == 0)
        x(ancilla[anc_idx]);
    }
  }
}

__qpu__ void apply(cudaq::qview<> ancilla, cudaq::qview<> system,
                   const std::vector<double> &state_prep_angles,
                   const std::vector<int> &term_controls,
                   const std::vector<int> &term_ops,
                   const std::vector<int> &term_lengths,
                   const std::vector<int> &term_signs) {
  prepare(ancilla, state_prep_angles);
  select(ancilla, system, term_controls, term_ops, term_lengths, term_signs);
  unprepare(ancilla, state_prep_angles);
}

} // namespace cudaq::solvers::block_encoding
