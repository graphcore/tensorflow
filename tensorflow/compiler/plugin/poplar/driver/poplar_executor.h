/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Declares the PoplarExecutor class, which is a CPU-only implementation of
// the StreamExecutor interface. For now, this is used for testing and to
// examine the performance of host-based StreamExecutor code.
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPLAR_EXECUTOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPLAR_EXECUTOR_H_

#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <poplar/Device.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/exceptions.hpp>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_feed_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_transfer_manager.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/infeed_allocator.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/infeed_iterator.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/input_output_aliasing_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/io_thread.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/seed_generator.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/spsc_outfeed_queue.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/spsc_queue.h"
#include "tensorflow/compiler/plugin/poplar/driver/trace.pb.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/device_description.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/host/host_stream.h"
#include "tensorflow/stream_executor/host/host_timer.h"
#include "tensorflow/stream_executor/kernel_spec.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/rng.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace se = stream_executor;

namespace xla {

class HloModule;

namespace poplarplugin {

enum PoplarProgramType {
  HOST_TO_DEVICE,
  MAIN_SEQUENCE,
  DEVICE_TO_HOST,
};

class PoplarExecutable;

std::string GetRandomNumberSeedStream();
std::string GetInfeedCopyHandle(const std::string& name, int64 shape_index);
std::string GetOutfeedCopyHandle(const std::string& name, int64 shape_index);

xla::poplarplugin::PoplarXfeedManager* GetXfeedManager(int device_ordinal);

void ResetXfeedManager(int device_ordinal);

typedef std::vector<char> (*ConversionFn)(const void*, int64, int64);

using Args = tensorflow::gtl::ArraySlice<se::DeviceMemoryBase>;

using ConversionList = std::vector<ConversionFn>;

using OutfeedQueueType = SPSCOutfeedQueue<2048>;

class ModuleFilenames {
 public:
  ModuleFilenames(const HloModule& module, int64 device_hash,
                  const std::string& serialization_folder);
  std::string CachedExecutableFilename() const;
  std::string CachedEngineFilename() const;
  std::string SerializedExecutableFilename() const;
  std::string SerializedMetadataFilename() const;
  std::string Name() const;

 private:
  const std::string basename_;
  const std::string serialization_folder_;
};

class PoplarExecutor : public se::internal::StreamExecutorInterface {
 public:
  explicit PoplarExecutor();
  ~PoplarExecutor() override;

  Status Init(int device_ordinal, se::DeviceOptions) override {
    ordinal_ = device_ordinal;
    return Status::OK();
  }

  Status GetKernel(const se::MultiKernelLoaderSpec& spec,
                   se::KernelBase* kernel) override {
    return xla::Unimplemented("Not Implemented");
  }
  Status Launch(se::Stream* stream, const se::ThreadDim& thread_dims,
                const se::BlockDim& block_dims, const se::KernelBase& kernel,
                const se::KernelArgsArrayBase& args) override {
    return xla::Unimplemented("Not Implemented");
  }

  void* Allocate(uint64 size) override;
  void* GetSubBuffer(se::DeviceMemoryBase* mem, uint64 offset_bytes,
                     uint64 size_bytes) override;
  void Deallocate(se::DeviceMemoryBase* mem) override;

  void* HostMemoryAllocate(uint64 size) override { return new char[size]; }
  void HostMemoryDeallocate(void* mem) override {
    delete[] static_cast<char*>(mem);
  }
  bool HostMemoryRegister(void* mem, uint64 size) override { return true; }
  bool HostMemoryUnregister(void* mem) override { return true; }

  bool Memcpy(se::Stream* stream, void* host_dst, const se::DeviceMemoryBase&,
              uint64 size) override;
  bool Memcpy(se::Stream* stream, se::DeviceMemoryBase*, const void*,
              uint64 size) override;
  bool MemcpyDeviceToDevice(se::Stream* stream, se::DeviceMemoryBase* pop_dst,
                            const se::DeviceMemoryBase& host_src,
                            uint64 size) override;

  bool MemZero(se::Stream* stream, se::DeviceMemoryBase* location,
               uint64 size) override {
    return false;
  }
  bool Memset(se::Stream* stream, se::DeviceMemoryBase*, uint8,
              uint64 size) override {
    return false;
  }
  bool Memset32(se::Stream* stream, se::DeviceMemoryBase*, uint32,
                uint64 size) override {
    return false;
  }

  bool SynchronizeAllActivity() override;
  bool SynchronousMemZero(se::DeviceMemoryBase* location, uint64) override {
    return false;
  }

  bool SynchronousMemSet(se::DeviceMemoryBase* location, int value,
                         uint64 size) override {
    return false;
  }

  Status SynchronousMemcpy(se::DeviceMemoryBase* pop_dst, const void* host_src,
                           uint64 size) override;
  Status SynchronousMemcpy(void* host_dst, const se::DeviceMemoryBase& pop_src,
                           uint64 size) override;
  Status SynchronousMemcpyDeviceToDevice(se::DeviceMemoryBase*,
                                         const se::DeviceMemoryBase&,
                                         uint64 size) override;

  bool HostCallback(se::Stream* stream,
                    std::function<void()> callback) override;
  bool HostCallback(se::Stream* stream,
                    std::function<Status()> callback) override;

  Status AllocateEvent(se::Event* event) override {
    return xla::Unimplemented("Not implemented");
  }

  Status DeallocateEvent(se::Event* event) override {
    return xla::Unimplemented("Not implemented");
  }

  Status RecordEvent(se::Stream* stream, se::Event* event) override {
    return xla::Unimplemented("Not implemented");
  }

  Status WaitForEvent(se::Stream* stream, se::Event* event) override {
    return xla::Unimplemented("Not implemented");
  }

  se::Event::Status PollForEventStatus(se::Event* event) override {
    return se::Event::Status::kError;
  }

  bool AllocateStream(se::Stream* stream) override { return true; }
  void DeallocateStream(se::Stream* stream) override {}
  bool CreateStreamDependency(se::Stream*, se::Stream*) override;

  bool AllocateTimer(se::Timer* timer) override { return true; }
  void DeallocateTimer(se::Timer* timer) override {}
  bool StartTimer(se::Stream* stream, se::Timer* timer) override;
  bool StopTimer(se::Stream* stream, se::Timer* timer) override;

  Status BlockHostUntilDone(se::Stream* stream) override;

  int PlatformDeviceCount() override { return 1; }

  bool DeviceMemoryUsage(int64* free, int64* total) const override {
    return false;
  }

  StatusOr<std::unique_ptr<se::DeviceDescription>> CreateDeviceDescription()
      const override;

  Status EnablePeerAccessTo(StreamExecutorInterface* other) override {
    return Status::OK();
  }

  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override {
    return true;
  }

  se::SharedMemoryConfig GetDeviceSharedMemoryConfig() override {
    return se::SharedMemoryConfig::kDefault;
  }

  Status SetDeviceSharedMemoryConfig(se::SharedMemoryConfig config) override {
    return xla::Unimplemented("Shared memory not supported");
  }

  std::unique_ptr<se::internal::EventInterface> CreateEventImplementation()
      override {
    return nullptr;
  }

  std::unique_ptr<se::internal::KernelInterface> CreateKernelImplementation()
      override {
    return nullptr;
  }

  std::unique_ptr<se::internal::StreamInterface> GetStreamImplementation()
      override {
    return std::unique_ptr<se::internal::StreamInterface>(
        new se::host::HostStream());
  }

  std::unique_ptr<se::internal::TimerInterface> GetTimerImplementation()
      override {
    return std::unique_ptr<se::internal::TimerInterface>(
        new se::host::HostTimer());
  }

  // Poplar Interface
  static se::host::HostStream* AsPoplarStream(se::Stream* stream);

  std::string GetDeviceTargetName() const;

  Status ConfigurePoplarDevice(const IpuOptions&);
  Status AttachToPoplarDevice();

  bool PoplarDeviceIsAttached() const;
  bool HasPoplarTarget() const;

  const IpuOptions& GetIpuOptions() const;
  const bool IpuOptionsConfigured() const;

  const poplar::Target& GetOrCreatePoplarTarget();

  const poplar::OptionFlags& GetOptionsFlags() const { return option_flags_; }

  const poplar::OptionFlags& GetReportGraphFlags() const {
    return graph_options_;
  }

  const poplar::OptionFlags& GetReportExecutionFlags() const {
    return execution_options_;
  }

  tensorflow::CancellationManager* cancellation_manager() { return cm_.get(); }

  bool IpuTraceEventsEnabled() const {
    return current_config_.profiling().enable_ipu_trace_events();
  }

  bool CompilerReportingEnabled() const {
    return current_config_.profiling().enable_compilation_trace();
  }

  int64 ReportEventNthExecution() const {
    return current_config_.profiling().report_every_nth_execution();
  }

  bool CompilerReportingTextFormat() const {
    return current_config_.profiling().enable_poplar_reports_text();
  }

  bool CompilerReportingCborFormat() const {
    return current_config_.profiling().enable_poplar_reports_cbor();
  }

  bool IncludePoplarSerializedGraph() const {
    return current_config_.profiling().enable_poplar_graph();
  }

  int64 MaxReportSize() const {
    return current_config_.profiling().max_report_size();
  }

  std::string ReportDirectory() const {
    return current_config_.profiling().report_directory();
  }

  const IpuOptions::FloatingPointBehaviour& FloatingPointBehaviour() const {
    return current_config_.floating_point_behaviour();
  }

  const IpuOptions::VerifiedTransfers& VerifiedTransfers() const {
    return current_config_.verified_transfers();
  }

  IpuDeviceConnectionType ConnectionType() const {
    return current_config_.device_connection_type();
  }

  bool AlwaysRearrangeCopiesOnTheHost() const {
    return current_config_.speed_size_config()
        .always_rearrange_copies_on_the_host();
  }

  std::string GetSchedulerSelection() const {
    return current_config_.speed_size_config().scheduler_selection();
  }

  bool MergeInfeedCopies() const {
    return current_config_.speed_size_config().merge_infeed_io_copies();
  }

  bool DisableGraphConvCaching() const {
    return current_config_.speed_size_config()
        .disable_graph_convolution_caching();
  }

  bool DisableGraphOutlining() const {
    return current_config_.speed_size_config().disable_graph_outlining();
  }

  bool RecomputationEnabled() const {
    return current_config_.speed_size_config().allow_recompute();
  }

  poplar::OptionFlags GetConvolutionOptions() const { return conv_options_; }

  poplar::OptionFlags GetMatMulOptions() const { return matmul_options_; }

  poplar::OptionFlags GetPoolingOptions() const { return pooling_options_; }

  bool UseVerifiedTransfers() const {
    return current_config_.verified_transfers().enabled();
  }

  bool ClearMatmulPassType() const {
    return current_config_.clear_matmul_pass_type();
  }

  bool RetainControlDependencies() const {
    return current_config_.retain_control_dependencies();
  }

  bool EnableMultiSliceCombiner() const {
    return current_config_.enable_multi_slice_combiner();
  }

  bool EnableGatherSimplifier() const {
    return current_config_.enable_gather_simplifier();
  }

  bool EnableMatmulCombiner() const {
    return current_config_.enable_matmul_combiner();
  }

  bool EnableSerialization() const {
    return !current_config_.serialization_folder().empty();
  }

  const std::string& SerializationFolder() const {
    return current_config_.serialization_folder();
  }

  bool UseStableNormStatistics() const {
    return current_config_.use_stable_norm_statistics();
  }

  bool SupportsRemoteBuffers() const;

  int64 GclNumIoTiles() const { return current_config_.gcl_num_io_tiles(); }

  poplar::OptionFlags GclOptions() const { return gcl_options_; }

  bool EnableExperimentalRemoteBufferEmbedding() const {
    return current_config_.enable_experimental_remote_buffer_embedding();
  }

  int64 GetMaxAllReduceBufferSize() const {
    return current_config_.max_cross_replica_sum_buffer_size();
  }

  int64 GetMaxReduceScatterBufferSize() const {
    return current_config_.max_reduce_scatter_buffer_size();
  }

  int64 GetMaxInterIpuCopyBufferSize() const {
    return current_config_.max_inter_ipu_copies_buffer_size();
  }

  int64 GetMaxSchedulerLookaheadDepth() const {
    return std::max<int64>(1, current_config_.max_scheduler_lookahead_depth());
  }

  int64 GetMaxSchedulerSearchSpaceSize() const {
    return std::max<int64>(2,
                           current_config_.max_scheduler_search_space_size());
  }

  int64 GetMaxSendRecvClusterSize() const {
    return current_config_.max_send_recv_cluster_size();
  }

  int64 GetTriangularSolveExpanderBlockSize() const {
    // 128 is XLA default block size used in TriangularSolveExpander
    auto block_size = current_config_.triangular_solve_expander_block_size();
    return block_size <= 0 ? 128 : block_size;
  }

  IpuSelectionOrder GetSelectionOrder() const {
    return current_config_.selection_order();
  }

  void AddCompileBeginEventRecord(const std::string& module_name);

  void AddCompileEndEventRecord(const std::string& module_name,
                                const std::string& compilation_report,
                                const std::string& poplar_graph,
                                const std::string& tensor_map_json,
                                const std::string& instruction_info,
                                int64 duration);

  void DumpPoplarOutOfMemoryAllocationException(
      const std::string& module_name,
      const poplar::graph_memory_allocation_error& p_e);

  void AddHostToDeviceEventRecord(const std::string& transfer_json);

  void AddDeviceToHostEventRecord(const std::string& transfer_json);

  void AddLoadEngineEventRecord(const std::string& module_name);

  void AddExecuteEventRecord(const std::string& module_name,
                             const std::string& report);

  Status GetCompilerEvents(std::list<tensorflow::IpuTraceEvent>& out);

  StatusOr<se::DeviceMemoryBase> ExecuteEngine(
      se::StreamExecutor* executor, xla::poplarplugin::PoplarExecutable&,
      se::DeviceMemoryAllocator* allocator, const Args&);

  StatusOr<se::DeviceMemoryBase> GetTupleBufferByIndex(
      const se::DeviceMemoryBase& base, int64 value);

  bool HaveExecutableCache() const;

  Status CreateExecutableCacheDirIfMissing() const;

  Status CreateSerializedExecutableDirIfMissing() const;

  bool HaveCachedExecutable(const ModuleFilenames& filenames) const;

  ModuleFilenames GetModuleFilenames(const HloModule& module) const;

  void AboutToFreeEngine(poplar::Engine* engine);

  const int device_ordinal() const;

  static poplar::DeviceManager& GetDeviceManager();

  void CreateInfeedIterator(
      const PoplarFeedConfig& config, const std::vector<xla::Shape>& shapes,
      const tensorflow::data::IteratorContext::Params& params,
      tensorflow::FunctionLibraryRuntime* flr,
      tensorflow::data::DatasetBase* dataset);

  Status DeleteInfeedIterator(const std::string& feed_id);

  InfeedAllocator* GetInfeedAllocator();

  // Lock the outfeed queue and dequeue all the tensors from a given feed.
  // Fails if the outfeed with the given name does not exist.
  std::vector<std::vector<tensorflow::Tensor>> GetTensorsFromOutfeed(
      const std::string& feed_id, const PoplarFeedConfig_Mode& mode);

  Status RegisterOutfeeds(const OutfeedInfos& outfeed_infos);

  Status DeleteOutfeed(const std::string& feed_id);

  class HostEmbeddingInterface_ {
   public:
    virtual ~HostEmbeddingInterface_() = default;

    virtual Status EnqueueLookupIndices(int replica, const int* indices,
                                        int index_count) = 0;
    virtual Status DequeueLookupActivations(int replica, void* destination) = 0;

    virtual Status EnqueueUpdateIndices(int replica, const int* indices,
                                        int index_count) = 0;
    virtual Status EnqueueUpdateGrads(int replica, const void* grads) = 0;

    virtual StatusOr<void*> GetRow(int index) const = 0;

    virtual StatusOr<int> GetTokenCount() const = 0;

    virtual StatusOr<int> GetEncodingWidth() const = 0;

    virtual xla::StatusOr<int> GetElementSize() const = 0;
  };

  template <typename T>
  class HostEmbeddingInterface : public HostEmbeddingInterface_ {
   public:
    virtual Status DequeueLookupActivations(int replica, T* destination) = 0;
    virtual Status EnqueueUpdateGrads(int replica, const T* grads) = 0;

    Status DequeueLookupActivations(int replica, void* destination) final {
      return DequeueLookupActivations(replica, static_cast<T*>(destination));
    }

    Status EnqueueUpdateGrads(int replica, const void* grads) final {
      return EnqueueUpdateGrads(replica, static_cast<const T*>(grads));
    }
  };

  Status RegisterHostEmbedding(
      const std::string& embedding_id,
      std::unique_ptr<HostEmbeddingInterface_> embedding);

  tensorflow::Rendezvous* GetRendezvous();

  void ResetSeed(int seed);

  void SetHasCycleCounter() { has_cycle_counter_ = true; }
  static std::string GetCycleCounterStream();

  void ClearCompilationFailure();
  void NotifyCompilationFailure();

 private:
  Status CreatePoplarTarget();

  // Compute literal(s) input for ConstantOutputAllocation
  // when dealing with scalar elementwise graph.
  Status LiteralEvaluateForScalarElementwiseGraph(
      xla::poplarplugin::PoplarExecutable& executable, const Args& args,
      std::vector<std::vector<Literal>>& literal_evaluate_break_down);

  struct TensorControl {
    size_t size = 0;
    unsigned int ref_count = 0;
    bool on_device = false;
    bool in_remote_memory = false;
    std::string input_handle;
    std::string output_handle;
    ConversionFn output_convertor;
    std::vector<char> converted_data;
    char* data;

    TensorControl(size_t size_);
    ~TensorControl();
  };

  struct InputDef {
    TensorControl* tc;
    ConversionFn fn;
    bool streamed;
    bool remote_parameter;

    InputDef() {}
    InputDef(TensorControl* tc, ConversionFn fn, bool streamed,
             bool remote_parameter)
        : tc(tc),
          fn(fn),
          streamed(streamed),
          remote_parameter(remote_parameter) {}
    InputDef(const InputDef& other)
        : tc(other.tc),
          fn(other.fn),
          streamed(other.streamed),
          remote_parameter(other.remote_parameter) {}
  };
  using InputPairList = std::vector<InputDef>;
  using ArgsHandleMap = std::map<std::string, InputDef>;

  struct OutputDef {
    TensorControl* tc;
    bool streamed;

    OutputDef() {}
    OutputDef(TensorControl* tc, bool streamed) : tc(tc), streamed(streamed) {}
    OutputDef(const OutputDef& other)
        : tc(other.tc), streamed(other.streamed) {}
  };

  using OutputPairList = std::vector<OutputDef>;
  using OutputsHandleMap = std::map<std::string, OutputDef>;

  static void FlattenedDeviceMemoryList(
      InputPairList&, const xla::Shape&, void*,
      const InputOutputAliasingMap::InputInfo&, bool is_remote_parameter);

  static void FlattenedOutputDeviceMemoryList(
      OutputPairList&, const xla::Shape&, void*,
      const InputOutputAliasingMap::OutputInfo&);

  void UpdateArgsHandleMap(const Args&, se::DeviceMemoryAllocator*,
                           const xla::poplarplugin::PoplarExecutable&);

  void UpdateOutputsHandleMap(
      const xla::poplarplugin::PoplarExecutable& executable,
      const xla::Shape& shape, se::DeviceMemoryBase retbuf);

  // These classes are used to pass around information for specific output
  // allocation type
  class OutputAllocation {
   public:
    virtual se::DeviceMemoryBase GetAllocation(
        se::DeviceMemoryAllocator*, const xla::Shape&, const int64, int64&,
        const Args&, const InputOutputAliasingMap::OutputInfo&,
        const ArgsHandleMap&, const int) const = 0;

   protected:
    OutputAllocation() {}
  };

  class ConstantOutputAllocation : public OutputAllocation {
   public:
    explicit ConstantOutputAllocation(
        const std::vector<std::vector<Literal>>& constants)
        : constants_(constants) {}

    se::DeviceMemoryBase GetAllocation(
        se::DeviceMemoryAllocator*, const xla::Shape&, const int64, int64&,
        const Args&, const InputOutputAliasingMap::OutputInfo&,
        const ArgsHandleMap&, const int) const override;

   private:
    const std::vector<std::vector<Literal>>& constants_;
  };

  class RemapOutputAllocation : public OutputAllocation {
   public:
    RemapOutputAllocation(PoplarExecutor* executor,
                          const std::vector<uint64>& remap_map,
                          const InputOutputAliasingMap& io_map)
        : executor_(executor),
          remap_map_(remap_map),
          input_output_aliasing_map_(io_map) {}

    se::DeviceMemoryBase GetAllocation(
        se::DeviceMemoryAllocator*, const xla::Shape&, const int64, int64&,
        const Args&, const InputOutputAliasingMap::OutputInfo&,
        const ArgsHandleMap&, const int) const override;

   private:
    PoplarExecutor* executor_;
    const std::vector<uint64>& remap_map_;
    const InputOutputAliasingMap& input_output_aliasing_map_;
  };

  class BufferOutputAllocation : public OutputAllocation {
   public:
    BufferOutputAllocation(){};

    se::DeviceMemoryBase GetAllocation(
        se::DeviceMemoryAllocator*, const xla::Shape&, const int64, int64&,
        const Args&, const InputOutputAliasingMap::OutputInfo&,
        const ArgsHandleMap&, const int) const override;
  };

  se::DeviceMemoryBase HandleOutputBuffer(
      se::DeviceMemoryAllocator* allocator,
      const OutputAllocation& allocation_info, const xla::Shape& shape,
      const int64 output_index, int64& flat_tensor_index, const Args& args,
      const InputOutputAliasingMap::OutputInfo& output_info);

  se::DeviceMemoryBase GetOutputBuffer(
      const xla::poplarplugin::PoplarExecutable& executable,
      se::DeviceMemoryAllocator* allocator,
      const OutputAllocation& allocation_info, const xla::Shape& shape,
      const Args& args, const InputOutputAliasingMap& output_info);

  // Functions which check whether any resource variables need copying to/from
  // device
  StatusOr<bool> CheckMoveDeviceToHostRequired(const bool engine_changed);
  StatusOr<bool> CheckMoveHostToDeviceRequired(const bool engine_changed);

  // Check if there is tensor/arg of current executable on device.
  StatusOr<bool> CheckAnyArgOnDevice(const Args& args);

  // Create a new trace event object
  tensorflow::IpuTraceEvent NewTraceEvent();

  // A function used to connect device to host streams, which only copies data
  // from the 0th replica and the rest is ignored.
  void ConnectReplicatedDeviceToHost(const std::string& stream_name,
                                     TensorControl* tc);

  // Functions which move the resource variables to/from the device
  Status MoveDeviceToHost();
  Status MoveHostToDevice();

  // Functions which connect the streams to/from device
  void ConnectStreamedVariablesHostToDevice();
  void ConnectStreamedVariablesDeviceToHost();

  // Sometimes post process streamed data into the right host format
  void PostProcessStreamedVariablesDeviceToHost();

  // Takes a tensor and returns a pointer to a buffer with the data in the right
  // format
  static void* PreProcessBuffer(InputDef& id);
  // Convers the data into the right host format
  static void PostProcessBuffer(TensorControl* tc);

  // Connect stream callbacks from Send/Recv operations in the engine
  // to the corresponding host graph operations using the rendezvous mechanism.
  Status ConnectSendCallbacksToRendezvous(const SendRecvInfos& send_infos);
  Status ConnectRecvCallbacksToRendezvous(const SendRecvInfos& recv_infos);

  Status ConnectHostEmbeddingLookup(
      const HostEmbeddingInfo& lookup_info,
      HostEmbeddingInterface_* embedding_interface);
  Status ConnectHostEmbeddingUpdateToRendezvous(
      const HostEmbeddingInfo& update_info,
      HostEmbeddingInterface_* embedding_interface);

  Status DisconnectHostEmbeddingLookup(
      const HostEmbeddingInfo& lookup_info,
      HostEmbeddingInterface_* embedding_interface);
  Status DisconnectHostEmbeddingUpdate(
      const HostEmbeddingInfo& update_info,
      HostEmbeddingInterface_* embedding_interface);

  // Connect buffers provided by infeed transfer manager to Poplar
  // HostToDevice FIFO
  void ConnectInfeedsToStreamCallback(const InfeedInfos& infeed_infos);

  // Connect buffers provided by transfer manager to Poplar
  // deviceToHostFIFO()
  void ConnectOutfeedToStreamCallback(const OutfeedInfos& outfeed_infos);

  IOFunction CreateInfeedIOThreadFunction(const FeedInfo& infeed_info);
  IOFunction CreateOutfeedIOThreadFunction(const FeedInfo& outfeed_info);

  // Creates and launches the threads which send/receive data from the Poplar
  // stream callbacks.
  void LaunchIOThreads(const InfeedInfos& infeed_infos,
                       const OutfeedInfos& outfeed_infos);

  // Blocks until all the IOThreads stop.
  void StopIOThreads();

  void DeferredDeallocation();

  void ConnectSeedCallback();

  void ConnectCycleCounterCallback();

  int ordinal_;

  std::vector<std::unique_ptr<IOThread>> io_threads_;

  std::unique_ptr<tensorflow::CancellationManager> cm_;

  poplar::Engine* current_engine_;

  int64 current_replication_factor_;

  bool device_attached_;

  class IPUConfig {
   public:
    bool DeviceConfigured() const;
    bool TargetConfigured() const;
    const poplar::Target& Target();
    const poplar::Target& TargetOrDie() const;
    const poplar::Device& Device() const;
    void SetDevice(poplar::Device&& device);
    void SetDeviceAndTarget(poplar::Device&& device);
    void SetTarget(const poplar::Target& target);
    void ClearDevice();
    std::recursive_mutex& Mutex();

   private:
    absl::optional<poplar::Device> device_;
    absl::optional<poplar::Target> target_;
    std::recursive_mutex mutex_;
  };
  IPUConfig ipu_;

  int64 poplar_device_hash_;

  poplar::OptionFlags option_flags_;

  poplar::OptionFlags conv_options_;

  poplar::OptionFlags matmul_options_;

  poplar::OptionFlags pooling_options_;

  poplar::OptionFlags graph_options_;

  poplar::OptionFlags execution_options_;

  poplar::OptionFlags gcl_options_;

  std::list<TensorControl*> allocations_;

  ArgsHandleMap args_map_;
  OutputsHandleMap outputs_map_;

  bool configured_;
  IpuOptions current_config_;

  std::list<tensorflow::IpuTraceEvent> reports_;

  struct OutfeedContext {
    OutfeedContext(const FeedInfo& outfeed_info);
    OutfeedContext() = delete;

    bool Matches(const FeedInfo& outfeed_info);

    const PoplarFeedConfig config;
    const std::vector<xla::Shape> shapes;
    std::vector<tensorflow::DataType> tf_data_types;
    std::vector<tensorflow::TensorShape> tf_shapes;
    std::vector<std::vector<std::unique_ptr<OutfeedQueueType>>>
        callback_to_io_thread_queues;
    std::deque<std::vector<tensorflow::Tensor>> io_thread_output_queues;
    // Mutex to prevent TF CPU op reading from the outfeed whilst we are
    // moving a tensor from the device.
    std::recursive_mutex mutex;
  };

  // Allocator that should be used for infeeds.
  InfeedAllocator infeed_allocator;

  absl::flat_hash_map<std::string, std::unique_ptr<InfeedIterator>>
      infeed_iterators_;

  absl::flat_hash_map<std::string, std::unique_ptr<OutfeedContext>>
      outfeed_contexts_;

  absl::flat_hash_map<std::string, std::unique_ptr<HostEmbeddingInterface_>>
      host_embeddings_;

  std::mutex host_embeddings_mutex_;
  std::condition_variable host_embeddings_cv;
  bool host_embedding_registration_is_open_ = true;

  SeedGenerator seed_generator_;

  std::string ReportFileExtension() const;

  bool has_cycle_counter_;

  tensorflow::core::RefCountPtr<tensorflow::Rendezvous> rendezvous_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
