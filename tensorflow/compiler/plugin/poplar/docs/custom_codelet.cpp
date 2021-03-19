#include <cmath>

#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

/*
 * A codelet to rotate a tensors 'x' and 'y', by the angle (radians) in the
 * tensor 'angle', around the origin.
 */
template <typename FPType>
class Rotate : public Vertex {
 public:
  Vector<Output<Vector<FPType>>> x_out;
  Vector<Output<Vector<FPType>>> y_out;
  Vector<Input<Vector<FPType>>> x_in;
  Vector<Input<Vector<FPType>>> y_in;
  Vector<Input<Vector<FPType>>> angle;

  bool compute() {
    for (unsigned i = 0; i < angle.size(); ++i) {
      for (unsigned j = 0; j != angle[i].size(); ++j) {
        float a = angle[i][j];
        float x = x_in[i][j];
        float y = y_in[i][j];
        x_out[i][j] = x * cos(a) - y * sin(a);
        y_out[i][j] = x * sin(a) + y * cos(a);
      }
    }
    return true;
  }
};

template class Rotate<float>;
template class Rotate<half>;
