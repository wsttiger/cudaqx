/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
/// @file library_utils.h
/// @brief Provides functionality to retrieve the path of the CUDAQX library.

#include <string>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach-o/dyld.h>
#else
#include <link.h>
#endif

namespace cudaqx::__internal__ {

/// @brief Structure to hold CUDAQX library data.
struct CUDAQXLibraryData {
  std::string path; ///< The path to the CUDAQX library.
};

#if defined(__APPLE__) && defined(__MACH__)
/// @brief Retrieves the CUDAQX library path on macOS systems.
/// @param data Pointer to CUDAQXLibraryData structure to store the library
/// path.
inline static void getCUDAQXLibraryPath(CUDAQXLibraryData *data) {
  auto nLibs = _dyld_image_count();
  for (uint32_t i = 0; i < nLibs; i++) {
    auto ptr = _dyld_get_image_name(i);
    std::string libName(ptr);
    if (libName.find("cudaq-core") != std::string::npos) {
      auto casted = static_cast<CUDAQLibraryData *>(data);
      casted->path = std::string(ptr);
    }
  }
}
#else
/// @brief Callback function for dl_iterate_phdr to find CUDAQX library path on
/// non-macOS systems.
/// @param info Pointer to dl_phdr_info structure containing shared object
/// information.
/// @param size Size of the structure.
/// @param data Pointer to user-provided data (CUDAQXLibraryData in this case).
/// @return Always returns 0 to continue iteration.
inline static int getCUDAQXLibraryPath(struct dl_phdr_info *info, size_t size,
                                       void *data) {
  std::string libraryName(info->dlpi_name);
  if (libraryName.find("cudaq-solvers") != std::string::npos) {
    auto casted = static_cast<CUDAQXLibraryData *>(data);
    casted->path = std::string(info->dlpi_name);
  }
  return 0;
}
#endif

/// @brief Retrieves the path of the CUDAQX library.
/// @return A string containing the path to the CUDAQX library.
inline static std::string getCUDAQXLibraryPath() {
  __internal__::CUDAQXLibraryData data;
#if defined(__APPLE__) && defined(__MACH__)
  getCUDAQXLibraryPath(&data);
#else
  dl_iterate_phdr(__internal__::getCUDAQXLibraryPath, &data);
#endif
  return data.path;
}

} // namespace cudaqx::__internal__
