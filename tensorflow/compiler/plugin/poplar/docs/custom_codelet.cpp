#include <cmath>
#include <limits>

#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

/*
 * A codelet to rotate a tensor 'data' which is [N, 2], by the angle (radians)
 * in the tensor 'angle', around the origin.
 */
class Rotate : public Vertex {
 public:
  Output<VectorList<float, VectorListLayout::DELTAN>> data_out;
  Input<VectorList<float, VectorListLayout::DELTAN>> data;
  Input < Vector<float> angle;

  bool compute() {
    for (unsigned i = 0; i < data.size(); ++i) {
      data_out[i][0] = cos(data[i][0]) - sin(data[i][1]);
      data_out[i][1] = sin(data[i][0]) + cos(data[i][1]);
    }
    return true;
  }
};
