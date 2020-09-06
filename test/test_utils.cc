#include "test_utils.hpp"

namespace math {

std::vector<double> Range(double start, double end, double step) {
  std::vector<double> values;
  while (start < end) {
    values.push_back(start);
    start += step;
  }
  return values;
}

}  // namespace math
