/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/gru.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/rnn_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"

#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <popnn/Gru.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <poputil/Broadcast.hpp>
#include <poputil/Util.hpp>

namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {
namespace {
static const size_t basic_gru_cell_num_units = 3;

poplar::Tensor UnflattenWeight(const poplar::Tensor& t) {
  return t
      .reshape({t.dim(0), basic_gru_cell_num_units,
                t.dim(1) / basic_gru_cell_num_units})
      .dimShuffle({1, 0, 2});
}

// The kernel is stored as:
// [input_size + output_size, basic_gru_cell_num_units * output_size] tensor.
// This extracts the input and output weights.
std::pair<poplar::Tensor, poplar::Tensor> UnpackGruKernel(
    poplar::Tensor kernel, const size_t input_size, const size_t output_size) {
  poplar::Tensor inputWeights = UnflattenWeight(kernel.slice(0, input_size));
  poplar::Tensor outputWeights =
      UnflattenWeight(kernel.slice(input_size, input_size + output_size));
  return {inputWeights, outputWeights};
}

poplar::Tensor FlattenWeight(const poplar::Tensor& t) {
  return t.dimShuffle({1, 0, 2}).reshape({t.dim(1), t.dim(0) * t.dim(2)});
}

// Reverse of UnpackGruKernel.
poplar::Tensor PackGruKernel(poplar::Tensor input_weights,
                             poplar::Tensor output_weights) {
  return poplar::concat(FlattenWeight(input_weights),
                        FlattenWeight(output_weights));
}

class GRULayerBaseOp : public PoplarOpDef {
 protected:
  virtual std::vector<poplar::Tensor> GetOutputParams(bool training) {
    return std::vector<poplar::Tensor>{OutputTensorCount()};
  }

  virtual poputil::graphfn::Signature GetSignature(
      const std::vector<poplar::Tensor>& args, bool training) {
    poputil::graphfn::Signature signature;
    int64 total_param_count = InputTensorCount() + OutputTensorCount();
    const auto& name_list = NameList();

    for (int64 i = 0; i < total_param_count; ++i) {
      if (i < InputTensorCount()) {
        signature.push_back(poputil::graphfn::input(args[i], name_list[i]));
      } else {
        signature.push_back(poputil::graphfn::created(name_list[i]));
      }
    }
    return signature;
  }

  virtual void SetOutputTensor(DriverGraph& graph,
                               std::vector<poplar::Tensor>& args,
                               const HloInstruction* inst,
                               TensorMap& tensor_map, bool training) {
    for (int64 j = 0; j < OutputTensorCount(); ++j) {
      TF_CHECK_OK(
          AddOutputTensor(tensor_map, inst, j,
                          DriverTensor(args[j + InputTensorCount()], graph)));
    }
  }

 public:
  StatusOr<DriverTensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, ClassName());
    const HloInstruction* inst = tensor_target.tgt;
    const int64 input_index = tensor_target.input_index;

    TF_ASSIGN_OR_RETURN(popnn::gru::GruParams gru_params,
                        GetGruParameters(inst));
    TF_ASSIGN_OR_RETURN(poplar::OptionFlags gru_opts, GetGruOpts(inst, res));
    switch (input_index) {
      case 0: {
        // Allocate GRU input tensor.
        return DriverTensor(
            popnn::gru::createInput(graph, gru_params, {debug_info}, gru_opts,
                                    &res.planning_cache),
            graph);
      }
      case 1: {
        // Allocate initial state tensor.
        return DriverTensor(
            popnn::gru::createInitialState(graph, gru_params, {debug_info},
                                           gru_opts, &res.planning_cache),
            graph);
      }
      case 2: {
        // Allocate GRU weights kernel.
        poplar::Tensor input_weights;
        poplar::Tensor output_weights;
        std::tie(input_weights, output_weights) =
            popnn::gru::createWeightsKernel(graph, gru_params, {debug_info},
                                            gru_opts, &res.planning_cache);
        return DriverTensor(PackGruKernel(input_weights, output_weights),
                            graph);
      }
      case 3: {
        // Allocate GRU weights biases.
        return DriverTensor(
            popnn::gru::createWeightsBiases(graph, gru_params, {debug_info},
                                            gru_opts, &res.planning_cache),
            graph);
      }
      case 4: {
        // Allocate AUGRU seq_len
        return DriverTensor(popops::createSliceableTensor(
                                graph, poplar::INT, {gru_params.rnn.batchSize},
                                {0}, {1}, 0, name),
                            graph);
      }
      case 5: {
        // Allocate AUGRU attention
        return DriverTensor(popnn::gru::createAttention(graph, gru_params, name)
                                .dimShuffle({1, 0}),
                            graph);
      }
      default: {
        return xla::FailedPrecondition(
            "Trying to allocate %s tensor for an index out of range %d.",
            ClassName().c_str(), input_index);
      }
    }
  }

  virtual StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) {
    PoplarOpDefDebugInfo debug_info(debug_context, ClassName());
    DriverProgramSequence seq(graph, {debug_info});

    auto gru_inst = Cast<HloRNNInstruction>(inst);
    bool training = gru_inst->is_training();

    std::vector<poplar::Tensor> args;
    for (int64 i = 0; i < InputTensorCount(); ++i) {
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor input_tensor,
          FindInstructionInput(tensor_map, res, inst, i, seq, debug_info));
      args.push_back(input_tensor);
    }

    auto output_args = GetOutputParams(training);
    args.insert(args.end(), output_args.begin(), output_args.end());

    poputil::graphfn::Signature signature = GetSignature(args, training);

    TF_ASSIGN_OR_RETURN(popnn::gru::GruParams gru_params,
                        GetGruParameters(inst));
    TF_ASSIGN_OR_RETURN(poplar::OptionFlags gru_opts, GetGruOpts(inst, res));

    auto input_size = ShapeUtil::GetDimension(inst->operand(0)->shape(), 2);
    auto output_size = ShapeUtil::GetDimension(inst->operand(1)->shape(), 1);

    poplar::DebugNameAndId debug_name_and_id(debug_info);
    auto func = [&graph, &res, &inst, this, gru_params, gru_opts, input_size,
                 output_size, debug_name_and_id,
                 training](std::vector<poplar::Tensor>& args,
                           poplar::program::Sequence& prog) {
      LowerToPoplar(graph, res, inst, gru_params, gru_opts, input_size,
                    output_size, training, debug_name_and_id, args, prog);
    };

    TF_RETURN_IF_ERROR(res.graph_cache.ExecuteCached(
        inst, graph, res, seq, func, signature, args,
        gru_inst->AllocatingIndices(), gru_inst->LayoutDependencies()));

    SetOutputTensor(graph, args, inst, tensor_map, training);
    return seq;
  }

 protected:
  virtual const std::string ClassName() const = 0;

  virtual std::vector<const char*> NameList() const = 0;

  virtual int64 InputTensorCount() const = 0;

  virtual int64 OutputTensorCount() const = 0;

  virtual void LowerToPoplar(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      popnn::gru::GruParams gru_params, const poplar::OptionFlags& gru_opts,
      const int64& input_size, const int64& output_size, bool training,
      const poplar::DebugNameAndId& debug_name_and_id,
      std::vector<poplar::Tensor>& args, poplar::program::Sequence& prog) = 0;
};

class GRULayerFwdOp : public GRULayerBaseOp {
 public:
  GRULayerFwdOp() = default;

 protected:
  const std::string ClassName() const override { return "GRULayerFwdOp"; }

  std::vector<const char*> NameList() const override {
    static std::vector<const char*> name_list = {"input_seq", "input_state",
                                                 "kernel", "bias", "output"};
    return name_list;
  }

  int64 InputTensorCount() const override { return 4; }

  int64 OutputTensorCount() const override { return 1; }

  std::vector<poplar::Tensor> GetOutputParams(bool training) override {
    auto args = GRULayerBaseOp::GetOutputParams(training);
    if (training) {
      poplar::Tensor intermediates;
      args.push_back(intermediates);
    }
    return args;
  }

  poputil::graphfn::Signature GetSignature(
      const std::vector<poplar::Tensor>& args, bool training) override {
    auto signature = GRULayerBaseOp::GetSignature(args, training);

    if (training) {
      signature.push_back(poputil::graphfn::created("intermediates"));
    }
    return signature;
  }

  void SetOutputTensor(DriverGraph& graph, std::vector<poplar::Tensor>& args,
                       const HloInstruction* inst, TensorMap& tensor_map,
                       bool training) override {
    int total_param_count = InputTensorCount() + OutputTensorCount();
    GRULayerBaseOp::SetOutputTensor(graph, args, inst, tensor_map, training);
    if (training) {
      TF_CHECK_OK(
          AddOutputTensor(tensor_map, inst, OutputTensorCount(),
                          DriverTensor(args[total_param_count], graph)));
    }
  }

  void LowerToPoplar(poplar::Graph& graph, CompilerResources& res,
                     const HloInstruction* inst,
                     popnn::gru::GruParams gru_params,
                     const poplar::OptionFlags& gru_opts,
                     const int64& input_size, const int64& output_size,
                     bool training,
                     const poplar::DebugNameAndId& debug_name_and_id,
                     std::vector<poplar::Tensor>& args,
                     poplar::program::Sequence& prog) override {
    poplar::Tensor input_seq = args[0];
    poplar::Tensor input_state = args[1];
    poplar::Tensor kernel = args[2];
    poplar::Tensor biases = args[3];

    popnn::gru::GruWeights weights;
    std::tie(weights.inputWeights, weights.outputWeights) =
        UnpackGruKernel(kernel, input_size, output_size);
    weights.biases = biases;

    auto intermediates_ptr = training ? &args[5] : nullptr;

    args[4] = popnn::gru::gruFwd(
        graph, gru_params, input_state, input_seq, weights, intermediates_ptr,
        prog, {debug_name_and_id}, gru_opts, &res.planning_cache);
  }
};
REGISTER_POPLAR_OP(GRULayerFwd, GRULayerFwdOp);

class GRULayerBwdOp : public GRULayerBaseOp {
 public:
  GRULayerBwdOp() = default;

 protected:
  const std::string ClassName() const override { return "GRULayerBwdOp"; }

  std::vector<const char*> NameList() const override {
    static std::vector<const char*> name_list = {"input_seq",
                                                 "input_state",
                                                 "kernel",
                                                 "bias",
                                                 "output",
                                                 "intermediates",
                                                 "output_backprop",
                                                 "input_backprop",
                                                 "input_state_backprop",
                                                 "kernel_backprop",
                                                 "biases_backprop"};
    return name_list;
  }

  int64 InputTensorCount() const override { return 7; }

  int64 OutputTensorCount() const override { return 4; }

  void LowerToPoplar(poplar::Graph& graph, CompilerResources& res,
                     const HloInstruction* inst,
                     popnn::gru::GruParams gru_params,
                     const poplar::OptionFlags& gru_opts,
                     const int64& input_size, const int64& output_size,
                     bool training,
                     const poplar::DebugNameAndId& debug_name_and_id,
                     std::vector<poplar::Tensor>& args,
                     poplar::program::Sequence& prog) override {
    poplar::Tensor input_seq = args[0];
    poplar::Tensor input_state = args[1];
    poplar::Tensor kernel = args[2];
    poplar::Tensor biases = args[3];
    poplar::Tensor output = args[4];
    poplar::Tensor intermediates = args[5];
    poplar::Tensor output_backprop = args[6];

    popnn::gru::GruWeights weights;
    std::tie(weights.inputWeights, weights.outputWeights) =
        UnpackGruKernel(kernel, input_size, output_size);
    weights.biases = biases;

    popnn::gru::GruWeights weights_backprop;
    args[8] = popnn::gru::gruBwdWithWU(
        graph, gru_params, prog, input_state, intermediates, weights, input_seq,
        output, output_backprop, &args[7], weights_backprop,
        {debug_name_and_id}, gru_opts, &res.planning_cache);
    args[9] = PackGruKernel(weights_backprop.inputWeights,
                            weights_backprop.outputWeights);
    args[10] = weights_backprop.biases;
  }
};
REGISTER_POPLAR_OP(GRULayerBwd, GRULayerBwdOp);

class DynamicGRULayerFwdOp : public GRULayerFwdOp {
 public:
  DynamicGRULayerFwdOp() = default;

 protected:
  const std::string ClassName() const override {
    return "DynamicGRULayerFwdOp";
  }

  std::vector<const char*> NameList() const override {
    static std::vector<const char*> name_list = {
        "input_seq", "input_state", "kernel", "bias", "seq_len", "output"};
    return name_list;
  }

  int64 InputTensorCount() const override { return 5; }

  int64 OutputTensorCount() const override { return 1; }

  void LowerToPoplar(poplar::Graph& graph, CompilerResources& res,
                     const HloInstruction* inst,
                     popnn::gru::GruParams gru_params,
                     const poplar::OptionFlags& gru_opts,
                     const int64& input_size, const int64& output_size,
                     bool training,
                     const poplar::DebugNameAndId& debug_name_and_id,
                     std::vector<poplar::Tensor>& args,
                     poplar::program::Sequence& prog) override {
    poplar::Tensor input_seq = args[0];
    poplar::Tensor input_state = args[1];
    poplar::Tensor kernel = args[2];
    poplar::Tensor biases = args[3];
    poplar::Tensor seq_len = args[4];

    seq_len = seq_len.reinterpret(poplar::UNSIGNED_INT);
    gru_params.rnn.varTimeSteps = seq_len;

    popnn::gru::GruWeights weights;
    std::tie(weights.inputWeights, weights.outputWeights) =
        UnpackGruKernel(kernel, input_size, output_size);
    weights.biases = biases;

    auto intermediates_ptr = training ? &args[6] : nullptr;
    args[5] = popnn::gru::gruFwd(
        graph, gru_params, input_state, input_seq, weights, intermediates_ptr,
        prog, {debug_name_and_id}, gru_opts, &res.planning_cache);
  }
};
REGISTER_POPLAR_OP(DynamicGRULayerFwd, DynamicGRULayerFwdOp);

class DynamicGRULayerBwdOp : public GRULayerBwdOp {
 public:
  DynamicGRULayerBwdOp() = default;

 protected:
  const std::string ClassName() const override {
    return "DynamicGRULayerBwdOp";
  }

  std::vector<const char*> NameList() const override {
    static std::vector<const char*> name_list = {
        "input_seq",       "input_state",
        "kernel",          "bias",
        "seq_len",         "output",
        "intermediates",   "output_backprop",
        "input_backprop",  "input_state_backprop",
        "kernel_backprop", "biases_backprop"};
    return name_list;
  }

  int64 InputTensorCount() const override { return 8; }

  int64 OutputTensorCount() const override { return 4; }

  void LowerToPoplar(poplar::Graph& graph, CompilerResources& res,
                     const HloInstruction* inst,
                     popnn::gru::GruParams gru_params,
                     const poplar::OptionFlags& gru_opts,
                     const int64& input_size, const int64& output_size,
                     bool training,
                     const poplar::DebugNameAndId& debug_name_and_id,
                     std::vector<poplar::Tensor>& args,
                     poplar::program::Sequence& prog) override {
    poplar::Tensor input_seq = args[0];
    poplar::Tensor input_state = args[1];
    poplar::Tensor kernel = args[2];
    poplar::Tensor biases = args[3];
    poplar::Tensor seq_len = args[4];
    poplar::Tensor output = args[5];
    poplar::Tensor intermediates = args[6];
    poplar::Tensor output_backprop = args[7];

    seq_len = seq_len.reinterpret(poplar::UNSIGNED_INT);

    popnn::gru::GruWeights weights_backprop;
    popnn::gru::GruWeights weights;
    std::tie(weights.inputWeights, weights.outputWeights) =
        UnpackGruKernel(kernel, input_size, output_size);

    weights.biases = biases;

    args[9] = popnn::gru::gruBwdWithWU(
        graph, gru_params, prog, input_state, intermediates, weights, input_seq,
        seq_len, output, output_backprop, &args[8], weights_backprop,
        {debug_name_and_id}, gru_opts, &res.planning_cache);
    args[10] = PackGruKernel(weights_backprop.inputWeights,
                             weights_backprop.outputWeights);
    args[11] = weights_backprop.biases;
  }
};
REGISTER_POPLAR_OP(DynamicGRULayerBwd, DynamicGRULayerBwdOp);

class AUGRULayerFwdOp : public GRULayerFwdOp {
 public:
  AUGRULayerFwdOp() = default;

 protected:
  const std::string ClassName() const override { return "AUGRULayerFwdOp"; }

  std::vector<const char*> NameList() const override {
    static std::vector<const char*> name_list = {
        "input_seq", "input_state", "kernel", "bias",
        "seq_len",   "attention",   "output"};
    return name_list;
  }

  int64 InputTensorCount() const override { return 6; }

  void LowerToPoplar(poplar::Graph& graph, CompilerResources& res,
                     const HloInstruction* inst,
                     popnn::gru::GruParams gru_params,
                     const poplar::OptionFlags& gru_opts,
                     const int64& input_size, const int64& output_size,
                     bool training,
                     const poplar::DebugNameAndId& debug_name_and_id,
                     std::vector<poplar::Tensor>& args,
                     poplar::program::Sequence& prog) override {
    poplar::Tensor input_seq = args[0];
    poplar::Tensor input_state = args[1];
    poplar::Tensor kernel = args[2];
    poplar::Tensor biases = args[3];
    poplar::Tensor seq_len = args[4];
    poplar::Tensor attention = args[5].transpose();

    seq_len = seq_len.reinterpret(poplar::UNSIGNED_INT);

    popnn::gru::GruWeights weights;
    std::tie(weights.inputWeights, weights.outputWeights) =
        UnpackGruKernel(kernel, input_size, output_size);
    weights.biases = biases;
    auto intermediates_ptr = training ? &args[7] : nullptr;
    args[6] =
        popnn::gru::auGruFwd(graph, gru_params, input_state, input_seq, seq_len,
                             weights, intermediates_ptr, attention, prog,
                             debug_name_and_id, gru_opts, &res.planning_cache);
  }
};
REGISTER_POPLAR_OP(AUGRULayerFwd, AUGRULayerFwdOp);

class AUGRULayerBwdOp : public GRULayerBwdOp {
 public:
  AUGRULayerBwdOp() = default;

 protected:
  const std::string ClassName() const override { return "AUGRULayerBwdOp"; }

  std::vector<const char*> NameList() const override {
    static std::vector<const char*> name_list = {"input_seq",
                                                 "input_state",
                                                 "kernel",
                                                 "bias",
                                                 "seq_len",
                                                 "attention",
                                                 "output",
                                                 "intermediates",
                                                 "output_backprop",
                                                 "input_backprop",
                                                 "input_state_backprop",
                                                 "kernel_backprop",
                                                 "biases_backprop",
                                                 "attention_backprop"};
    return name_list;
  }

  int64 InputTensorCount() const override { return 9; }

  int64 OutputTensorCount() const override { return 5; }

  void LowerToPoplar(poplar::Graph& graph, CompilerResources& res,
                     const HloInstruction* inst,
                     popnn::gru::GruParams gru_params,
                     const poplar::OptionFlags& gru_opts,
                     const int64& input_size, const int64& output_size,
                     bool training,
                     const poplar::DebugNameAndId& debug_name_and_id,
                     std::vector<poplar::Tensor>& args,
                     poplar::program::Sequence& prog) override {
    poplar::Tensor input_seq = args[0];
    poplar::Tensor input_state = args[1];
    poplar::Tensor kernel = args[2];
    poplar::Tensor biases = args[3];
    poplar::Tensor seq_len = args[4];
    poplar::Tensor attention = args[5].transpose();
    poplar::Tensor output = args[6];
    poplar::Tensor intermediates = args[7];
    poplar::Tensor output_backprop = args[8];

    seq_len = seq_len.reinterpret(poplar::UNSIGNED_INT);

    popnn::gru::GruWeights weights;
    std::tie(weights.inputWeights, weights.outputWeights) =
        UnpackGruKernel(kernel, input_size, output_size);
    weights.biases = biases;

    popnn::gru::GruWeights weights_backprop;
    poplar::Tensor attention_backprop;

    args[10] = popnn::gru::auGruBwdWithWU(
        graph, gru_params, prog, input_state, intermediates, weights, input_seq,
        seq_len, output, output_backprop, &args[9], weights_backprop, attention,
        &attention_backprop, debug_name_and_id, gru_opts, &res.planning_cache);
    args[11] = PackGruKernel(weights_backprop.inputWeights,
                             weights_backprop.outputWeights);
    args[12] = weights_backprop.biases;
    args[13] = attention_backprop.transpose();
  }
};
REGISTER_POPLAR_OP(AUGRULayerBwd, AUGRULayerBwdOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
