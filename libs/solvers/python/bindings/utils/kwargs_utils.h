/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace cudaqx {

/// @brief Return the value of given type corresponding to the provided
/// key string from the provided options `kwargs` `dict`. Return the `orVal`
/// if the key is not in the `dict`.
template <typename T>
T getValueOr(py::kwargs &options, const std::string &key, const T &orVal) {
  if (options.contains(key))
    for (auto item : options)
      if (item.first.cast<std::string>() == key)
        return item.second.cast<T>();

  return orVal;
}

inline heterogeneous_map hetMapFromKwargs(const py::kwargs &kwargs) {
  cudaqx::heterogeneous_map result;

  for (const auto &item : kwargs) {
    std::string key = py::cast<std::string>(item.first);
    auto value = item.second;

    if (py::isinstance<py::bool_>(value)) {
      result.insert(key, value.cast<bool>());
    } else if (py::isinstance<py::int_>(value)) {
      result.insert(key, value.cast<std::size_t>());
    } else if (py::isinstance<py::float_>(value)) {
      result.insert(key, value.cast<double>());
    } else if (py::isinstance<py::str>(value)) {
      result.insert(key, value.cast<std::string>());
    } else {
      throw std::runtime_error(
          "Invalid python type for mapping kwargs to a heterogeneous_map.");
    }
  }

  return result;
}
} // namespace cudaqx
