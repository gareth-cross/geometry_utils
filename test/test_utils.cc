#include "test_utils.hpp"

using namespace Eigen;
namespace math {

std::vector<double> Range(double start, double end, double step) {
  std::vector<double> values;
  while (start < end) {
    values.push_back(start);
    start += step;
  }
  return values;
}

// This is for tests, so lazily use dynamic vector for everything.
std::vector<Eigen::VectorXd> Grid3D(double start, double end, double step) {
  const std::vector<double> tick_marks = Range(start, end, step);
  std::vector<Eigen::VectorXd> grid;
  grid.reserve(tick_marks.size() * tick_marks.size() * tick_marks.size());
  for (double x : tick_marks) {
    for (double y : tick_marks) {
      for (double z : tick_marks) {
        grid.emplace_back(Eigen::Vector3d(x, y, z));
      }
    }
  }
  return grid;
}

}  // namespace math
