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
#include <fstream>
#include <iostream>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "tensorflow/core/common_runtime/data/standalone.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/public/session_options.h"

tensorflow::string FLAGS_graphdef;
tensorflow::string FLAGS_ret_val_input_name;
int FLAGS_count = 1000;

namespace tensorflow {
namespace data {
namespace standalone {

static void BuildRetvalNode(GraphDef& graph_def) {
  const char* const kRetValOp = "_Retval";
  NodeDef* ret_def = graph_def.add_node();
  ret_def->set_op(kRetValOp);
  ret_def->set_name("dataset");

  std::string* input_string = ret_def->add_input();
  *input_string = FLAGS_ret_val_input_name;

  AddNodeAttr("T", DT_VARIANT, ret_def);
  AddNodeAttr("index", 0, ret_def);
}

static void RunInputPipeline(const std::string& graph_as_string) {
  GraphDef graph_def;
  protobuf::TextFormat::ParseFromString(graph_as_string, &graph_def);

  BuildRetvalNode(graph_def);

  std::unique_ptr<Dataset> dataset;
  Status s = Dataset::FromGraph({}, graph_def, &dataset);

  if (!s.ok()) {
    std::cerr << "Error loading dataset from provided graph: " << s.ToString()
              << std::endl;
    return;
  }

  std::unique_ptr<Iterator> iterator;
  s = dataset->MakeIterator(&iterator);

  bool end_of_input = false;

  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  for (int num_outputs = 0; !end_of_input; ++num_outputs) {
    bool should_print_this_batch = num_outputs % FLAGS_count == 0;
    std::vector<Tensor> outputs;

    s = iterator->GetNext(&outputs, &end_of_input);
    if (!s.ok()) {
      std::cerr << "Error fetching data: " << s.ToString() << std::endl;
    }

    if (should_print_this_batch) {
      std::chrono::steady_clock::time_point end =
          std::chrono::steady_clock::now();
      std::cout << "Time to execute " << FLAGS_count << " batches: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                         start)
                       .count()
                << " milliseconds"

                << " batch number: (" << num_outputs << ")" << std::endl;

      start = std::chrono::steady_clock::now();
    }
  }
}

}  // namespace standalone
}  // namespace data
}  // namespace tensorflow

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag(
          "graph", &FLAGS_graphdef,
          "TensorFlow GraphDef file containing the input pipeline"),
      tensorflow::Flag("count", &FLAGS_count,
                       "Number of samples per displayed line"),
      tensorflow::Flag("output_node", &FLAGS_ret_val_input_name,
                       "Name of the last operation in the input pipeline, to "
                       "be added as input to a _Retval operation"),
  };

  // Parse the command line for the flags.
  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok || FLAGS_graphdef.empty()) {
    std::printf("Graph option not provided by the user!\n%s", usage.c_str());
    return 2;
  }

  if (FLAGS_ret_val_input_name.empty()) {
    std::printf("Name of node to attach return val to not found in input!\n%s",
                usage.c_str());
    return 2;
  }

  // Open the user provided graphdef, create a string from it, then parse it.
  std::ifstream input_file{FLAGS_graphdef, std::ifstream::in};

  std::string graph_string{std::istreambuf_iterator<char>(input_file),
                           std::istreambuf_iterator<char>()};

  tensorflow::data::standalone::RunInputPipeline(graph_string);
  return 0;
}
