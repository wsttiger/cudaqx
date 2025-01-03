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

template <typename T>
tensor<T> toTensor(const py::array_t<T> &H) {
  py::buffer_info buf = H.request();

  // Create a vector of the array dimensions
  std::vector<std::size_t> shape;
  for (py::ssize_t d : buf.shape) {
    shape.push_back(static_cast<std::size_t>(d));
  }

  // Create a tensor and borrow the NumPy array data
  cudaqx::tensor<T> tensor_H(shape);
  tensor_H.borrow(static_cast<T *>(buf.ptr), shape);
  return tensor_H;
}

template <typename T>
py::array_t<T> fromTensor(const cudaqx::tensor<T>& tensor) {
    // Get the dimensions/shape from the tensor
    const auto& shape = tensor.shape();

    // Convert shape to vector of py::ssize_t for numpy array
    std::vector<py::ssize_t> numpy_shape;
    numpy_shape.reserve(shape.size());
    for (const auto& dim : shape) {
        numpy_shape.push_back(static_cast<py::ssize_t>(dim));
    }

    // Create numpy array with appropriate shape
    py::array_t<T> numpy_array(numpy_shape);

    // Get raw pointer to numpy array data
    auto buf = numpy_array.request();
    T* ptr = static_cast<T*>(buf.ptr);

    // Copy data from tensor to numpy array
    std::copy(tensor.data(), tensor.data() + tensor.size(), ptr);

    return numpy_array;
}

} // namespace cudaqx
