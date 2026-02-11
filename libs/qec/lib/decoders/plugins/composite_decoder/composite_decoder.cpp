/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include "common/Logger.h"
#include <memory>
#include <stdexcept>
#include <string>

namespace cudaq::qec {

/// @brief Composite decoder that chains a TensorRT pre-decoder with a global
/// decoder (e.g. pymatching).
///
/// The TRT pre-decoder (neural network) consumes the raw syndrome and outputs
/// the residual syndrome — the syndrome of errors not handled locally. That
/// residual syndrome is passed to the global decoder, which produces the
/// final correction.
///
/// Constructor parameters:
/// - "global_decoder": Type of global decoder; currently only "pymatching" is
///   supported.
/// - "onnx_path" or "onnx_load_path": Path to ONNX model for the TRT pre-decoder
///   (will build TensorRT engine). Passed to trt_decoder as "onnx_load_path".
/// - "engine_path" or "engine_load_path": Path to pre-built TensorRT engine for
///   the TRT pre-decoder. Passed to trt_decoder as "engine_load_path".
/// - Any additional params (e.g. "precision", "engine_save_path") are forwarded
///   to the trt_decoder; global-decoder-specific params (e.g. "error_rate_vec")
///   can be passed and are forwarded to the global decoder.
///
/// Decode flow: raw syndrome -> trt_decoder (outputs residual syndrome) ->
/// global_decoder(residual syndrome) -> final decoder_result.
/// The TRT pre-decoder output must have size syndrome_size when using
/// "pymatching".
class composite_decoder : public decoder {
private:
  std::unique_ptr<decoder> trt_decoder_;
  std::unique_ptr<decoder> global_decoder_;

  static cudaqx::heterogeneous_map make_trt_params(
      const cudaqx::heterogeneous_map &params) {
    cudaqx::heterogeneous_map trt_params;
    if (params.contains("onnx_path"))
      trt_params.insert("onnx_load_path", params.get<std::string>("onnx_path"));
    else if (params.contains("onnx_load_path"))
      trt_params.insert("onnx_load_path",
                        params.get<std::string>("onnx_load_path"));
    if (params.contains("engine_path"))
      trt_params.insert("engine_load_path",
                       params.get<std::string>("engine_path"));
    else if (params.contains("engine_load_path"))
      trt_params.insert("engine_load_path",
                        params.get<std::string>("engine_load_path"));
    if (params.contains("engine_save_path"))
      trt_params.insert("engine_save_path",
                        params.get<std::string>("engine_save_path"));
    if (params.contains("precision"))
      trt_params.insert("precision", params.get<std::string>("precision"));
    if (params.contains("memory_workspace"))
      trt_params.insert("memory_workspace",
                        params.get<size_t>("memory_workspace"));
    if (params.contains("use_cuda_graph"))
      trt_params.insert("use_cuda_graph", params.get<bool>("use_cuda_graph"));
    return trt_params;
  }

  static cudaqx::heterogeneous_map make_global_params(
      const cudaqx::heterogeneous_map &params,
      const std::string &global_decoder_type) {
    cudaqx::heterogeneous_map global_params;
    if (global_decoder_type == "pymatching") {
      if (params.contains("error_rate_vec"))
        global_params.insert("error_rate_vec",
                             params.get<std::vector<double>>("error_rate_vec"));
      if (params.contains("merge_strategy"))
        global_params.insert("merge_strategy",
                             params.get<std::string>("merge_strategy"));
    }
    return global_params;
  }

public:
  composite_decoder(const cudaqx::tensor<uint8_t> &H,
                    const cudaqx::heterogeneous_map &params)
      : decoder(H) {

    if (!params.contains("global_decoder"))
      throw std::runtime_error(
          "composite_decoder requires parameter 'global_decoder'");

    std::string global_decoder_type =
        params.get<std::string>("global_decoder");
    if (global_decoder_type != "pymatching")
      throw std::runtime_error(
          "composite_decoder only supports global_decoder \"pymatching\" "
          "currently; got: " +
          global_decoder_type);

    bool has_onnx =
        params.contains("onnx_path") || params.contains("onnx_load_path");
    bool has_engine =
        params.contains("engine_path") || params.contains("engine_load_path");
    if (!has_onnx && !has_engine)
      throw std::runtime_error(
          "composite_decoder requires either 'onnx_path'/'onnx_load_path' or "
          "'engine_path'/'engine_load_path' for the TRT pre-decoder");
    if (has_onnx && has_engine)
      throw std::runtime_error(
          "composite_decoder cannot specify both ONNX and engine path; use "
          "one of onnx_path/onnx_load_path or engine_path/engine_load_path");

    cudaqx::heterogeneous_map trt_params = make_trt_params(params);
    cudaqx::heterogeneous_map global_params =
        make_global_params(params, global_decoder_type);

    trt_decoder_ = decoder::get("trt_decoder", H, trt_params);
    global_decoder_ = decoder::get(global_decoder_type, H, global_params);
  }

  virtual decoder_result decode(const std::vector<float_t> &syndrome) override {
    std::vector<decoder_result> pre_results = trt_decoder_->decode_batch({syndrome});
    std::vector<std::vector<float_t>> global_in(1, pre_results[0].result);
    std::vector<decoder_result> final_results =
        global_decoder_->decode_batch(global_in);
    return final_results[0];
  }

  virtual std::vector<decoder_result> decode_batch(
      const std::vector<std::vector<float_t>> &syndromes) override {
    if (syndromes.empty())
      return {};
    std::vector<decoder_result> pre_results =
        trt_decoder_->decode_batch(syndromes);
    std::vector<std::vector<float_t>> global_syndromes;
    global_syndromes.reserve(pre_results.size());
    for (auto &r : pre_results)
      global_syndromes.push_back(std::move(r.result));
    return global_decoder_->decode_batch(global_syndromes);
  }

  virtual ~composite_decoder() = default;

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      composite_decoder,
      static std::unique_ptr<decoder> create(
          const cudaqx::tensor<uint8_t> &H,
          const cudaqx::heterogeneous_map &params) {
        return std::make_unique<composite_decoder>(H, params);
      })
};

CUDAQ_REGISTER_TYPE(composite_decoder)

} // namespace cudaq::qec
